# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
OS-level sandbox wrapper for tool execution.
"""

import os
import site
import subprocess
import sys

from loggers import get_logger

logger = get_logger(__name__)

_SANDBOX_EXEC = "/usr/bin/sandbox-exec"
_BWRAP_PROBE_BIN = "/usr/bin/true"

_sandbox_available_cache: bool | None = None


def _probe(argv: list[str], label: str) -> bool:
    """Run *argv*, return True iff exit code is 0. Log on failure."""
    try:
        proc = subprocess.run(
            argv,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            timeout = 5,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning(
            "%s probe failed (%s); tool execution will run unsandboxed", label, e
        )
        return False
    if proc.returncode != 0:
        stderr_tail = proc.stderr.decode("utf-8", errors = "replace").strip()[-200:]
        logger.warning(
            "%s present but probe returned %s; tool execution will run unsandboxed. stderr: %s",
            label,
            proc.returncode,
            stderr_tail,
        )
        return False
    return True


def _macos_probe() -> bool:
    if not os.path.exists(_SANDBOX_EXEC):
        logger.warning("macOS sandbox unavailable (sandbox-exec missing)")
        return False
    return _probe(
        [_SANDBOX_EXEC, "-p", "(version 1)(allow default)", "/usr/bin/true"],
        "macOS sandbox-exec",
    )


def _linux_probe() -> bool:
    """Smoke-test that ``bwrap`` can apply a minimal sandbox here.

    Catches the cases where the kernel refuses to create unprivileged
    user namespaces — surfacing at startup instead of first use.
    """
    return _probe(
        [
            "bwrap",
            "--ro-bind",
            "/",
            "/",
            "--unshare-all",
            "--die-with-parent",
            _BWRAP_PROBE_BIN,
        ],
        "Linux bwrap",
    )


def sandbox_available() -> bool:
    """True iff the platform's sandbox can be applied in this process context.

    Existence of the binary alone is not enough: a nested-sandboxed
    parent may have ``sandbox-exec`` / ``bwrap`` present but be unable
    to apply additional policies. Confirm by spawning a no-op sandboxed
    ``/usr/bin/true`` once at first call and caching the result.
    """
    global _sandbox_available_cache
    if _sandbox_available_cache is not None:
        return _sandbox_available_cache

    if sys.platform == "darwin":
        ok = _macos_probe()
        label = "macOS Seatbelt"
    elif sys.platform == "linux":
        ok = _linux_probe()
        label = "Linux bubblewrap"
    else:
        ok = False
        label = "no sandbox primitive for this platform"

    _sandbox_available_cache = ok
    if ok:
        logger.info("%s sandbox available; tool execution sandboxed", label)
    elif sys.platform not in ("darwin", "linux"):
        logger.warning("%s; tool execution will run unsandboxed", label)
    return ok


def _safe_subpath(p: str) -> str:
    """Reject paths that cannot be safely embedded in a Seatbelt literal.

    Seatbelt string literals use ``"..."`` with ``\\`` escapes; a path
    containing ``"``, ``\\``, a newline, or a NUL byte could close the
    string and inject Scheme into the profile. macOS paths in practice
    contain none of these, so rejecting them is safer than escaping.
    """
    if any(c in p for c in ('"', "\\", "\n", "\r", "\x00")):
        raise ValueError(f"path unsafe for Seatbelt profile: {p!r}")
    return p


def _editable_source_paths() -> list[str]:
    """Source dirs registered by PEP 660 editable installs.

    Read from the parent's ``sys.modules``; valid for the child only
    while it shares ``sys.executable`` with the parent.
    """
    paths: list[str] = []
    for name, mod in list(sys.modules.items()):
        if not (name.startswith("__editable___") and name.endswith("_finder")):
            continue
        paths.extend(getattr(mod, "MAPPING", {}).values())
        for ns_paths in getattr(mod, "NAMESPACES", {}).values():
            paths.extend(ns_paths)
    return paths


def _exec_chain_symlinks(executable: str) -> list[str]:
    """Symlinks encountered while resolving *executable* to its real binary.

    Returned paths are the symlinks themselves (not their targets). The
    Linux bwrap argv binds each one so that, inside the sandbox, the
    kernel can follow the chain during ``execve`` — otherwise it hits
    ``ENOENT`` on an intermediate symlink we never mounted.
    """
    out: list[str] = []
    seen_links: set[str] = set()
    current = executable
    for _ in range(40):  # cycle guard against pathological symlink loops
        parts = current.split(os.sep)
        prefix = "/"
        for p in parts[1:]:
            prefix = os.path.normpath(os.path.join(prefix, p))
            if prefix in seen_links:
                continue
            try:
                if os.path.islink(prefix):
                    seen_links.add(prefix)
                    out.append(prefix)
            except OSError:
                pass
        try:
            if not os.path.islink(current):
                break
            target = os.readlink(current)
        except OSError:
            break
        if not target.startswith(os.sep):
            target = os.path.normpath(os.path.join(os.path.dirname(current), target))
        if target == current:
            break
        current = target
    return out


def _python_read_paths() -> list[str]:
    """Real dirs the Python interpreter needs to read at runtime.

    Returns ``sys.prefix``, ``sys.base_prefix``, system site-packages,
    user site-packages, and editable-install source dirs — all
    realpath-normalized, deduplicated, and filtered to existing dirs.
    Used by both the macOS Seatbelt profile and the Linux bwrap argv.
    """
    candidates: list[str] = [sys.prefix, sys.base_prefix]
    candidates.extend(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(user_site)
    candidates.extend(_editable_source_paths())

    seen: set[str] = set()
    out: list[str] = []
    for p in candidates:
        if not p:
            continue
        rp = os.path.realpath(p)
        if rp in seen or not os.path.isdir(rp):
            continue
        seen.add(rp)
        out.append(rp)
    return out


def _macos_seatbelt_profile(workdir: str) -> str:
    """Build a Seatbelt profile string for ``sandbox-exec -p``."""
    py_subpaths = [f'(subpath "{_safe_subpath(p)}")' for p in _python_read_paths()]

    wd = _safe_subpath(os.path.realpath(workdir))
    py_block = "\n    ".join(py_subpaths)
    # Paths the kernel needs mmap(PROT_EXEC) on so the loader can map
    # binaries and dylibs as code. Narrower than the full read allow
    # because most things we permit reads of are data, not executables.
    executable_map_block = "\n    ".join(
        [
            '(subpath "/usr/lib")',
            '(subpath "/usr/bin")',
            '(subpath "/bin")',
            '(subpath "/System/Library/Frameworks")',
            '(subpath "/System/Library/PrivateFrameworks")',
            '(subpath "/System/Cryptexes")',
            '(subpath "/System/Volumes/Preboot/Cryptexes")',
            '(subpath "/Library/Frameworks")',
            *py_subpaths,
        ]
    )

    return f"""(version 1)
(deny default)

(allow process-fork)
(allow process-exec)
(allow signal (target self))
(allow process-info-pidinfo (target self))
(allow process-info-pidfdinfo (target self))
(allow sysctl-read)
(allow ipc-posix-shm)
(allow file-read-metadata)

(allow file-read*
    ; (literal "/") is required: dyld and many runtime resolvers stat
    ; the root directory itself, which is NOT matched by (subpath "/X").
    (literal "/")
    ; --- Execution surface ---
    (subpath "/usr/lib")
    (subpath "/usr/bin")
    (subpath "/bin")
    ; Narrow /System: only the framework + dyld surfaces the loader needs.
    ; Avoids exposing /System/Applications/* (~all installed system apps and
    ; their localized resources) and /System/iOSSupport to the LLM.
    (subpath "/System/Library/Frameworks")
    (subpath "/System/Library/PrivateFrameworks")
    (subpath "/System/Library/dyld")
    (subpath "/System/Cryptexes")
    (subpath "/System/Volumes/Preboot/Cryptexes")
    (subpath "/Library/Frameworks")
    ; --- Runtime data libraries actually consult ---
    (subpath "/usr/share/zoneinfo")        ; tzdata for datetime
    (subpath "/usr/share/icu")             ; ICU data
    (subpath "/private/var/db/dyld")
    (subpath "/private/var/db/timezone")
    (subpath "/private/etc")               ; /etc/hosts, /etc/ssl/cert.pem
    (literal "/dev/null")
    (literal "/dev/zero")
    (literal "/dev/random")
    (literal "/dev/urandom")
    (literal "/dev/dtracehelper")
    (literal "/dev/autofs_nowait")
    {py_block}
)

; Required for mmap(PROT_EXEC) on dylibs — without this Python cannot
; load libpython, libsystem_*, or any C-extension .so. Also required
; for /bin/bash and /usr/bin/* under the terminal tool.
(allow file-map-executable
    {executable_map_block}
)

(allow file-read* (subpath "{wd}"))
(allow file-write* (subpath "{wd}"))
(allow file-ioctl (subpath "{wd}"))
(allow file-read* (literal "/dev/tty"))
(allow file-write* (literal "/dev/tty"))

(allow mach-lookup
    (global-name "com.apple.coreservices.launchservicesd")
    (global-name "com.apple.lsd.mapdb")
    (global-name "com.apple.SecurityServer")
    (global-name "com.apple.trustd.agent")
    (global-name "com.apple.trustd")
    (global-name "com.apple.system.opendirectoryd.libinfo")
    (global-name "com.apple.system.opendirectoryd.membership")
    (global-name "com.apple.system.logger")
    (global-name "com.apple.system.notification_center")
    (global-name "com.apple.system.DirectoryService.libinfo_v1")
)

(deny network-outbound)
(deny network-inbound)
(deny network-bind)
"""


def _linux_bwrap_argv(inner_argv: list[str], workdir: str) -> list[str]:
    """Build a ``bwrap`` argv for the Linux sandbox.

    Deny by omission: the child sees only what we bind-mount. ``net``
    is unshared without loopback, so all network is denied. ``/tmp``
    is a fresh tmpfs so writes don't leak to the host.
    """
    wd = os.path.realpath(workdir)
    top_ro_dirs = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc")

    args: list[str] = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    # -try variants skip missing paths so the same argv works on
    # usrmerge distros (/lib, /lib64 are symlinks into /usr or absent).
    for d in top_ro_dirs:
        args.extend(["--ro-bind-try", d, d])

    def _is_under_top_ro(path: str) -> bool:
        return any(path == top or path.startswith(top + os.sep) for top in top_ro_dirs)

    # Dedupe via realpath because sys.prefix and sys.base_prefix often
    # resolve to the same thing on stock installs.
    seen: set[str] = set()
    for path in _python_read_paths() + _editable_source_paths():
        if not path:
            continue
        rp = os.path.realpath(path)
        if rp in seen or not os.path.isdir(rp) or _is_under_top_ro(rp):
            continue
        seen.add(rp)
        args.extend(["--ro-bind-try", rp, rp])

    # Bind exec-chain symlinks whose parent isn't already covered by
    # an existing bind — binding into a read-only mount fails; symlinks
    # under an existing bind are already reachable via path inheritance.
    bind_flags = ("--ro-bind", "--ro-bind-try", "--bind", "--bind-try")
    bound_dests = [
        args[i + 2]
        for i, arg in enumerate(args)
        if arg in bind_flags and i + 2 < len(args)
    ]

    bound_links: set[str] = set()
    for sym in _exec_chain_symlinks(sys.executable):
        if sym in bound_links or _is_under_top_ro(sym):
            continue
        parent = os.path.dirname(sym)
        if any(parent == b or parent.startswith(b + os.sep) for b in bound_dests):
            continue
        bound_links.add(sym)
        args.extend(["--ro-bind-try", sym, sym])

    args.extend(["--bind", wd, wd])

    args.append("--")
    args.extend(inner_argv)
    return args


def build_sandbox_argv(inner_argv: list[str], workdir: str) -> list[str]:
    """Return an argv that runs *inner_argv* under the platform sandbox.

    Caller should gate with :func:`sandbox_available`. Raises
    ``RuntimeError`` if the platform has no sandbox implementation here.
    """
    if not inner_argv:
        raise ValueError("inner_argv must be non-empty")

    if sys.platform == "darwin":
        profile = _macos_seatbelt_profile(workdir)
        return [_SANDBOX_EXEC, "-p", profile, *inner_argv]
    if sys.platform == "linux":
        return _linux_bwrap_argv(inner_argv, workdir)
    raise RuntimeError(f"build_sandbox_argv: no sandbox for platform {sys.platform!r}")
