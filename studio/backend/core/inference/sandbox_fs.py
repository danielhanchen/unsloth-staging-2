# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Landlock filesystem confinement for the code sandbox (Linux 5.13+).

Confines the sandboxed python/terminal subprocess to its session workdir
(read-write) plus the read-only system paths a process needs to run (interpreter,
stdlib, site-packages, dynamic loader and shared libs, core command binaries).
Reads and writes to arbitrary host paths (/home, other users' files, most of
/etc, /tmp outside the sandbox, /proc, /sys) are denied by the kernel.

Defense-in-depth: the command blocklist, rlimits, credential-stripped env and
the prompt isolation in core/inference/tools.py all remain in force. Where
Landlock is unavailable (non-Linux, kernel < 5.13, unrecognised arch, or syscalls
blocked) the confiner is a no-op logged once and the subprocess runs with the
existing protections. Enforcement is Linux-only.

Landlock rules are inherited across execve and cannot be relaxed afterwards, so
they are applied in the subprocess preexec_fn (forked child, just before execve).
preexec must not import, so every heavy step (ctypes handle, syscall numbers,
resolved path list, packed ruleset attr) is done in the parent here; the child
does only pre-imported os/struct/ctypes/syscall calls, matching _sandbox_preexec.
"""

import ctypes
import ctypes.util
import glob
import os
import platform
import struct
import sys
import threading

from loggers import get_logger

logger = get_logger(__name__)

_GATE_ENV = "UNSLOTH_STUDIO_SANDBOX_FS_CONFINE"
# Operator escape hatches: os.pathsep-separated extra paths to allow.
_ALLOW_READ_ENV = "UNSLOTH_STUDIO_SANDBOX_FS_ALLOW"
_ALLOW_WRITE_ENV = "UNSLOTH_STUDIO_SANDBOX_FS_ALLOW_WRITE"

# Landlock syscall numbers, per arch. Unknown arch -> Landlock treated as
# unavailable rather than risk calling the wrong syscall by number.
_SYSCALL_NRS = {
    "x86_64": (444, 445, 446),
    "aarch64": (444, 445, 446),
}

_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
# Minimum Landlock ABI. We require ABI 3 (Linux 6.2+) so the ruleset can handle
# both LANDLOCK_ACCESS_FS_REFER (ABI 2) and LANDLOCK_ACCESS_FS_TRUNCATE (ABI 3).
# Below that floor the ruleset is broken or leaky:
#   ABI 1 (5.13-5.18) lacks FS_REFER: an enforced ruleset unconditionally denies
#   reparenting a file to another dir (the one right the kernel denies by default
#   even when unhandled), breaking legitimate in-workdir os.rename/shutil.move.
#   ABI 2 (5.19-6.1) cannot handle FS_TRUNCATE, and unhandled rights are always
#   allowed: a sandboxed truncate/ftruncate/creat/open(O_TRUNC) -- a shell `>
#   file` or open(path, "w") -- could zero any same-UID-writable host file
#   OUTSIDE the workdir, defeating the write confinement. A blocklist cannot
#   cover implicit truncation, so requiring the TRUNCATE-capable ABI is the fix.
# Below the floor, fall back to the pre-Landlock protections.
_MIN_ABI = 3
_PR_SET_NO_NEW_PRIVS = 38
# O_PATH: inode handle without read/exec perm, as landlock_add_rule wants for
# parent_fd. Not exported by os on all builds, so define the constant.
_O_PATH = 0o10000000

# Filesystem access-right bits (uapi/linux/landlock.h).
_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_REFER = 1 << 13
_FS_TRUNCATE = 1 << 14
_FS_IOCTL_DEV = 1 << 15

# Rights valid on a non-directory; a dir-only right (e.g. READ_DIR) on a file is
# rejected with EINVAL, silently dropping the rule.
_FILE_VALID = _FS_EXECUTE | _FS_WRITE_FILE | _FS_READ_FILE | _FS_TRUNCATE | _FS_IOCTL_DEV

# Read-only, read-write and device grant masks (intersected with the handled set
# per ABI before use).
_DIR_RO = _FS_EXECUTE | _FS_READ_FILE | _FS_READ_DIR | _FS_IOCTL_DEV
_FILE_RO = _FS_EXECUTE | _FS_READ_FILE | _FS_IOCTL_DEV
_DEV_RW = _FS_READ_FILE | _FS_WRITE_FILE | _FS_IOCTL_DEV

# System roots the interpreter and shell need read-only. Not /opt: an interpreter
# there is already covered by the sys.prefix / executable-dir grants below, and a
# blanket /opt would expose operator data placed there.
_RO_SYSTEM_DIRS = (
    "/usr",
    "/lib",
    "/lib32",
    "/lib64",
    "/libx32",
    "/bin",
    "/sbin",
)
# Specific /etc files the dynamic loader / libc read. The rest of /etc, and
# listing /etc, stay denied.
_RO_ETC_FILES = (
    "/etc/ld.so.cache",
    "/etc/ld.so.preload",
    "/etc/localtime",
    "/etc/nsswitch.conf",
    # Egress is direct (no netns/proxy), so glibc getaddrinfo reads resolv.conf
    # for the nameserver; without it allow-listed HTTPS by hostname fails. add()
    # realpaths the systemd stub symlink. /etc/hosts stays denied on purpose.
    "/etc/resolv.conf",
    "/etc/group",
    # /etc/passwd deliberately NOT granted: the child runs without it (only libc
    # identity lookups like getpwuid()/getpass.getuser() fail, catchably).
    # Granting it would let a sandboxed `cat /etc/passwd` enumerate every host
    # account -- the enumeration path this confinement closes. Operators can
    # allow-list it via UNSLOTH_STUDIO_SANDBOX_FS_ALLOW.
)
_RW_DEV_FILES = (
    "/dev/null",
    "/dev/zero",
    "/dev/full",
    "/dev/random",
    "/dev/urandom",
    "/dev/tty",
)
# Host-shared tmpfs required for POSIX semaphores / shared memory, i.e.
# multiprocessing.Pool, ProcessPoolExecutor and joblib parallelism.
_RW_SHM = "/dev/shm"

# The sandbox sitecustomize shim (core/inference/sandbox_site) is on every child's
# PYTHONPATH via tools._build_safe_env. On a source checkout it sits outside the
# interpreter roots, so it needs an explicit read-only grant or Landlock blocks the
# /mnt/data path-remap shim from importing. Derived from this file's dir so no
# import of tools is needed.
_SANDBOX_SITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_site")


def _venv_grant_paths(venv: str) -> "list[str]":
    """Read+execute paths a child-exported VIRTUAL_ENV needs, or [].

    tools._build_safe_env forwards VIRTUAL_ENV and prepends ``$VIRTUAL_ENV/bin``
    (POSIX) / ``Scripts`` (Windows) to the child PATH. A console script resolved
    from there needs FS_EXECUTE on that scripts dir, and imports need the venv's
    site-packages; nothing else under the venv is required.

    We deliberately do NOT grant the whole tree: a VIRTUAL_ENV pointing at a
    workspace/repo root would then expose its source and secrets to sandboxed
    code. So the grant is (a) gated on a REAL virtualenv (``pyvenv.cfg``, or
    ``bin/python[3]`` / ``Scripts/python.exe``), and (b) limited to the scripts
    dir and ``site-packages`` roots. A non-venv VIRTUAL_ENV yields no grant.
    Resolved in the parent; the child imports nothing.
    """
    if not venv:
        return []
    try:
        real = os.path.realpath(venv)
    except OSError:
        return []
    if not os.path.isdir(real):
        return []
    is_win = sys.platform == "win32"
    scripts = os.path.join(real, "Scripts" if is_win else "bin")
    # Validate a dedicated virtualenv, not an arbitrary dir VIRTUAL_ENV points at.
    markers = [os.path.join(real, "pyvenv.cfg")]
    if is_win:
        markers.append(os.path.join(scripts, "python.exe"))
    else:
        markers += [os.path.join(scripts, "python"), os.path.join(scripts, "python3")]
    if not any(os.path.exists(m) for m in markers):
        return []
    grants = [scripts]
    # site-packages: POSIX lib/pythonX.Y/site-packages (+ lib64), Windows
    # Lib/site-packages. Glob matches the minor version without importing.
    for pattern in (
        os.path.join(real, "lib", "python*", "site-packages"),
        os.path.join(real, "lib64", "python*", "site-packages"),
        os.path.join(real, "Lib", "site-packages"),
    ):
        grants.extend(glob.glob(pattern))
    return grants


def _ca_cert_paths() -> "list[str]":
    """System CA store (bundle + hashed dir) the child's TLS stack reads to verify
    HTTPS certs, or [].

    Egress to allow-listed hosts is supported (no network namespace). OpenSSL's
    default store often lives under a /usr symlink whose real path is in /etc
    (e.g. /usr/lib/ssl/certs -> /etc/ssl/certs), which the /etc denials would
    block, breaking verification. Grant just the store, read-only; it holds only
    public trust anchors.

    Only the compile-time openssl_cafile/openssl_capath are granted, never the
    env-resolved cafile/capath: those reflect the PARENT's SSL_CERT_FILE /
    SSL_CERT_DIR, which the scrubbed child never sees, so granting them is useless
    to the child and could leak a private file (e.g. SSL_CERT_FILE=.../private.pem)
    into the sandbox. Operators use _ALLOW_READ_ENV for a custom store. Resolved
    in the parent; the child imports nothing.
    """
    try:
        import ssl
        dvp = ssl.get_default_verify_paths()
    except Exception:  # pragma: no cover - ssl is present on supported hosts
        return []
    return [p for p in (dvp.openssl_cafile, dvp.openssl_capath) if p]


def _load_libc():
    try:
        name = ctypes.util.find_library("c")
        if not name:
            return None
        libc = ctypes.CDLL(name, use_errno = True)
        libc.syscall.restype = ctypes.c_long
        return libc
    except (OSError, AttributeError):
        return None


# Resolved once at import so the preexec child triggers no imports.
_LIBC = _load_libc() if sys.platform == "linux" else None
_NRS = _SYSCALL_NRS.get(platform.machine()) if sys.platform == "linux" else None


def _fs_handled_mask(abi: int) -> int:
    """FS access rights supported by ``abi``, so we handle everything the kernel
    knows and grant back only the allowed paths."""
    mask = (1 << 13) - 1  # ABI 1: bits 0..12
    if abi >= 2:
        mask |= _FS_REFER
    if abi >= 3:
        mask |= _FS_TRUNCATE
    if abi >= 5:
        mask |= _FS_IOCTL_DEV
    return mask


def _syscall(nr: int, *args) -> int:
    """Call libc syscall with explicitly typed args (ctypes would otherwise pass
    ints as 32-bit c_int and truncate 64-bit pointers)."""
    return _LIBC.syscall(ctypes.c_long(nr), *args)


def _abi_version() -> int:
    """Landlock ABI version, or <= 0 when Landlock is unavailable/unsupported."""
    if _LIBC is None or _NRS is None:
        return -1
    try:
        return _syscall(
            _NRS[0],
            ctypes.c_void_p(0),
            ctypes.c_size_t(0),
            ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION),
        )
    except OSError:
        return -1


# Availability is a property of the running kernel/arch; probe once.
_availability_lock = threading.Lock()
_abi_cached: "int | None" = None


def _availability() -> int:
    global _abi_cached
    if _abi_cached is None:
        with _availability_lock:
            if _abi_cached is None:
                _abi_cached = _abi_version()
    return _abi_cached


def landlock_available() -> bool:
    """True when Landlock FS confinement can be enforced here.

    Requires ABI >= _MIN_ABI (3): below that the ruleset breaks in-workdir renames
    (ABI 1 lacks FS_REFER) or silently allows out-of-workdir truncation (ABI 2
    cannot handle FS_TRUNCATE; unhandled rights are always allowed); see _MIN_ABI.
    Older kernels degrade to the other sandbox layers."""
    return _availability() >= _MIN_ABI


def _gate() -> "bool | None":
    """Tri-state gate: True force on, False force off, None auto (on where
    available)."""
    raw = os.environ.get(_GATE_ENV)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in ("0", "false", "off", "no", "disable", "disabled"):
        return False
    if value in ("1", "true", "on", "yes", "enable", "enabled", "auto"):
        return True if value != "auto" else None
    return None


def fs_confinement_enabled() -> bool:
    """Whether confinement should be applied: gate on/auto AND Landlock available."""
    gate = _gate()
    if gate is False:
        return False
    return landlock_available()


# One-time platform log so operators see the enforcement state without spamming
# every tool call.
_logged = False
_log_lock = threading.Lock()


def _log_once() -> None:
    global _logged
    if _logged:
        return
    with _log_lock:
        if _logged:
            return
        _logged = True
        if _gate() is False:
            logger.info("Sandbox FS confinement disabled via %s", _GATE_ENV)
        elif landlock_available():
            logger.info("Sandbox FS confinement active (Landlock ABI %s)", _availability())
        else:
            logger.info(
                "Sandbox FS confinement unavailable on this platform "
                "(no Landlock); relying on the command blocklist, rlimits and "
                "environment scrubbing"
            )


def _extra_paths(env_name: str) -> "list[str]":
    raw = os.environ.get(env_name, "")
    return [p for p in raw.split(os.pathsep) if p]


def _build_rules(workdir: str, handled: int) -> "list[tuple[str, int]]":
    """Resolve the allow-list into (realpath, access) pairs, unioned by path.

    Directory-only rights are stripped for file targets so landlock_add_rule
    never fails with EINVAL. Nonexistent paths are skipped.
    """
    dir_ro = handled & _DIR_RO
    file_ro = handled & _FILE_RO
    dev_rw = handled & _DEV_RW
    file_valid = handled & _FILE_VALID

    rules: "dict[str, int]" = {}

    def add(path: str, access: int) -> None:
        if not path:
            return
        try:
            real = os.path.realpath(path)
        except OSError:
            return
        if not os.path.exists(real):
            return
        if real == os.sep:
            # Never grant the filesystem root: a rule on "/" grants read and
            # traversal of every descendant and defeats the confinement. Reachable
            # when an interpreter prefix / sys.executable dir resolves to "/" (a
            # Python built with --prefix=/, or an interpreter at /python).
            return
        if not os.path.isdir(real):
            access &= file_valid
        rules[real] = rules.get(real, 0) | access

    # Read-write: the session workdir (scratch, uploads, artifacts), shared-memory
    # tmpfs for multiprocessing, and operator-added writable paths.
    add(workdir, handled)
    # /dev/shm: multiprocessing/shared_memory create/read/write/truncate/remove
    # named objects here but must not enumerate the mount. All sessions share the
    # service UID, so READ_DIR would let one session list -- and thus discover the
    # unguessable names of and read -- other sessions' semaphores/shm. Withholding
    # only READ_DIR blocks that while multiprocessing keeps working.
    add(_RW_SHM, handled & ~_FS_READ_DIR)
    for path in _extra_paths(_ALLOW_WRITE_ENV):
        add(path, handled)

    # Read-only system dirs + interpreter/stdlib/site-packages roots.
    for path in _RO_SYSTEM_DIRS:
        add(path, dir_ro)
    for path in (
        sys.base_prefix,
        sys.prefix,
        getattr(sys, "base_exec_prefix", sys.exec_prefix),
        sys.exec_prefix,
        os.path.dirname(sys.executable) if sys.executable else "",
    ):
        add(path, dir_ro)
    # A virtualenv exported via VIRTUAL_ENV: tools._build_safe_env prepends
    # $VIRTUAL_ENV/bin to the child PATH even when it differs from the prefixes
    # granted above, so a console script resolved from there needs FS_EXECUTE.
    # Grant only the scripts (bin/Scripts) dir and site-packages roots, read-only,
    # and ONLY for a real dedicated virtualenv (_venv_grant_paths validates) --
    # never the whole tree, so a VIRTUAL_ENV at a workspace/repo root does not leak
    # its source. add() dedups against any coinciding prefix grant.
    for path in _venv_grant_paths(os.environ.get("VIRTUAL_ENV", "")):
        add(path, dir_ro)
    # The sandbox's own sitecustomize shim dir (on the child PYTHONPATH); outside
    # the interpreter roots on a source checkout, so grant it or the path-remap
    # shim silently fails to import under confinement.
    add(_SANDBOX_SITE_DIR, dir_ro)
    # System CA store so HTTPS (git/pip/requests) can verify certificates; the
    # store often realpath-resolves into /etc, which the /etc denials would block.
    for path in _ca_cert_paths():
        add(path, dir_ro)
    for path in _extra_paths(_ALLOW_READ_ENV):
        add(path, dir_ro)

    # Read-only loader/libc files and read-write device files.
    for path in _RO_ETC_FILES:
        add(path, file_ro)
    for path in _RW_DEV_FILES:
        add(path, dev_rw)

    return list(rules.items())


def _make_apply(handled: int, rules: "list[tuple[str, int]]"):
    """Return the no-arg callable run in the child preexec to enforce the ruleset.

    Only pre-imported ops run here (os/struct/ctypes/syscall). Any failure is
    swallowed: confinement is defense-in-depth and the child cannot log without
    polluting captured tool output, so a rare failure degrades to the other
    protections rather than crashing the tool.
    """
    nr_create, nr_add, nr_restrict = _NRS
    attr = ctypes.create_string_buffer(struct.pack("<Q", handled), 8)

    def _apply() -> None:
        try:
            ruleset_fd = _syscall(
                nr_create,
                ctypes.cast(attr, ctypes.c_void_p),
                ctypes.c_size_t(8),
                ctypes.c_uint32(0),
            )
            if ruleset_fd < 0:
                return
            try:
                for path, access in rules:
                    try:
                        pfd = os.open(path, _O_PATH | os.O_CLOEXEC)
                    except OSError:
                        continue
                    try:
                        rule = ctypes.create_string_buffer(struct.pack("<Qi", access, pfd), 12)
                        _syscall(
                            nr_add,
                            ctypes.c_int(ruleset_fd),
                            ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
                            ctypes.cast(rule, ctypes.c_void_p),
                            ctypes.c_uint32(0),
                        )
                    finally:
                        os.close(pfd)
                # restrict_self needs PR_SET_NO_NEW_PRIVS (set by _sandbox_preexec;
                # re-set here so the confiner is self-contained).
                _LIBC.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
                _syscall(nr_restrict, ctypes.c_int(ruleset_fd), ctypes.c_uint32(0))
            finally:
                os.close(ruleset_fd)
        except Exception:  # noqa: BLE001 - never break the sandboxed child
            pass

    return _apply


def build_sandbox_confiner(workdir: str):
    """Build the FS-confinement preexec step for a sandboxed subprocess.

    Returns a no-arg callable to run inside the child's preexec_fn (after
    _sandbox_preexec), or None when confinement is disabled or unavailable. All
    path resolution and struct packing happen here in the parent; the returned
    callable does only syscalls.
    """
    _log_once()
    if not fs_confinement_enabled():
        return None
    abi = _availability()
    if abi < _MIN_ABI:
        return None
    handled = _fs_handled_mask(abi)
    rules = _build_rules(workdir, handled)
    if not rules:
        return None
    return _make_apply(handled, rules)
