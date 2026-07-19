# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Landlock filesystem confinement for the code sandbox
(core/inference/sandbox_fs.py).

Enforcement tests spawn a real subprocess under the confiner (exactly as the
python/terminal tools do in preexec_fn) and assert that:
  - reads/writes OUTSIDE the sandbox workdir are denied,
  - reads/writes INSIDE the workdir succeed,
  - a normal computation runs and stdlib + a site-package import,
  - an explicitly allowed path is readable,
  - the real _bash_exec / _python_exec executor paths stay confined and working.
They are skipped where Landlock is unavailable (non-Linux, kernel < 5.13, other
arch). Gate, fallback, mask and rule-builder tests run everywhere.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import sandbox_fs
from core.inference.sandbox_fs import (
    _SANDBOX_SITE_DIR,
    _build_rules,
    _fs_handled_mask,
    build_sandbox_confiner,
    fs_confinement_enabled,
    landlock_available,
)

_LANDLOCK = landlock_available()
_needs_landlock = pytest.mark.skipif(
    not _LANDLOCK, reason = "Landlock not available on this platform"
)

# A third-party package importable in the parent, so the site-package import test
# is meaningful wherever it runs (numpy in the studio venv; else structlog/pytest,
# both backend/test deps that live in site-packages).
_SITE_PKG = next(
    (
        name
        for name in ("numpy", "structlog", "yaml", "pytest")
        if importlib.util.find_spec(name) is not None
    ),
    "pytest",
)


def _run_confined(
    code: str,
    workdir: Path,
    confiner,
    env_extra: dict | None = None,
):
    """Run ``code`` in a subprocess under ``confiner`` (the production preexec
    step) and return the CompletedProcess."""
    env = dict(os.environ)
    env["PYTHONPATH"] = ""  # keep the child from importing the test's own dir
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-u", "-c", textwrap.dedent(code)],
        cwd = str(workdir),
        preexec_fn = confiner,
        capture_output = True,
        text = True,
        env = env,
        timeout = 120,
    )


# ── Enforcement: deny outside, allow inside ──────────────────────────────────


@_needs_landlock
class TestEnforcement:
    @pytest.fixture(autouse = True)
    def _neutralize_ambient_virtualenv(self, monkeypatch):
        # build_sandbox_confiner grants the exported VIRTUAL_ENV read+execute (so
        # console scripts resolve). On a dev host VIRTUAL_ENV can point at the
        # workspace root, an ancestor of pytest's tmp_path, which would grant read
        # of the test's out-of-workdir secrets and mask genuine confinement. These
        # tests exercise system-path confinement, not the venv grant (unit-tested
        # separately), so remove the ambient value for determinism.
        monkeypatch.delenv("VIRTUAL_ENV", raising = False)

    def test_read_outside_workdir_is_denied(self, tmp_path):
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        secret = tmp_path / "secret.txt"  # sibling, OUTSIDE the sandbox
        secret.write_text("SENTINEL_OUTSIDE_9f3c")
        confiner = build_sandbox_confiner(str(sandbox))
        assert confiner is not None

        r = _run_confined(
            f"""
            try:
                data = open({str(secret)!r}).read()
                print("READ:" + data)
            except Exception as e:
                print("DENIED:" + type(e).__name__)
            """,
            sandbox,
            confiner,
        )
        assert "DENIED:PermissionError" in r.stdout, r.stdout
        assert "SENTINEL_OUTSIDE_9f3c" not in r.stdout, r.stdout

    def test_read_etc_hosts_is_denied(self, tmp_path):
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        confiner = build_sandbox_confiner(str(sandbox))
        r = _run_confined(
            """
            try:
                open("/etc/hosts").read(); print("READ")
            except Exception as e:
                print("DENIED:" + type(e).__name__)
            """,
            sandbox,
            confiner,
        )
        assert "DENIED:PermissionError" in r.stdout, r.stdout

    def test_write_outside_workdir_is_denied(self, tmp_path):
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        target = tmp_path / "escape.txt"  # sibling, OUTSIDE the sandbox
        confiner = build_sandbox_confiner(str(sandbox))
        r = _run_confined(
            f"""
            try:
                open({str(target)!r}, "w").write("x"); print("WROTE")
            except Exception as e:
                print("DENIED:" + type(e).__name__)
            """,
            sandbox,
            confiner,
        )
        assert "DENIED:PermissionError" in r.stdout, r.stdout
        assert not target.exists()

    def test_read_write_inside_workdir_succeeds(self, tmp_path):
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        confiner = build_sandbox_confiner(str(sandbox))
        r = _run_confined(
            """
            import os
            p = os.path.join(os.getcwd(), "artifact.txt")
            open(p, "w").write("hello-scratch")
            print("ROUNDTRIP:" + open(p).read())
            """,
            sandbox,
            confiner,
        )
        assert "ROUNDTRIP:hello-scratch" in r.stdout, (r.stdout, r.stderr)
        assert (sandbox / "artifact.txt").read_text() == "hello-scratch"

    def test_normal_computation_and_imports_work(self, tmp_path):
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        confiner = build_sandbox_confiner(str(sandbox))
        r = _run_confined(
            f"""
            import json, ssl, sqlite3, hashlib   # stdlib incl. C-extension modules
            import {_SITE_PKG}                    # a real site-package
            sieve = [True] * 30
            sieve[0] = sieve[1] = False
            for i in range(2, 30):
                if sieve[i]:
                    for j in range(i * i, 30, i):
                        sieve[j] = False
            print("PRIMES:" + json.dumps([i for i in range(30) if sieve[i]]))
            print("IMPORT_OK")
            """,
            sandbox,
            confiner,
        )
        assert "PRIMES:[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]" in r.stdout, (r.stdout, r.stderr)
        assert "IMPORT_OK" in r.stdout, (r.stdout, r.stderr)

    def test_explicitly_allowed_path_is_readable(self, tmp_path, monkeypatch):
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        allowed = tmp_path / "allowed.txt"  # OUTSIDE the sandbox
        allowed.write_text("ALLOWED_DATA_7b21")
        denied = tmp_path / "denied.txt"  # OUTSIDE and not allow-listed
        denied.write_text("DENIED_DATA_7b21")
        monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_FS_ALLOW", str(allowed))
        confiner = build_sandbox_confiner(str(sandbox))
        r = _run_confined(
            f"""
            def probe(label, path):
                try:
                    print(label + ":" + open(path).read())
                except Exception as e:
                    print(label + ":DENIED:" + type(e).__name__)
            probe("ALLOWED", {str(allowed)!r})
            probe("DENIED", {str(denied)!r})
            """,
            sandbox,
            confiner,
        )
        assert "ALLOWED:ALLOWED_DATA_7b21" in r.stdout, r.stdout
        assert "DENIED:DENIED:PermissionError" in r.stdout, r.stdout

    def test_multiprocessing_pool_still_works(self, tmp_path):
        # /dev/shm must be granted or POSIX semaphores fail; guards that regression.
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        confiner = build_sandbox_confiner(str(sandbox))
        r = _run_confined(
            """
            import multiprocessing as mp
            def sq(x): return x * x
            if __name__ == "__main__":
                with mp.Pool(2) as pool:
                    print("POOL:" + str(pool.map(sq, [1, 2, 3, 4])))
            """,
            sandbox,
            confiner,
        )
        assert "POOL:[1, 4, 9, 16]" in r.stdout, (r.stdout, r.stderr)

    def test_sandbox_site_dir_is_readable_under_confinement(self, tmp_path):
        # The sitecustomize path-remap shim on the child PYTHONPATH must import;
        # reading a file inside its dir has to succeed under the confiner, else the
        # /mnt/data remap silently stops working for confined Python.
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        confiner = build_sandbox_confiner(str(sandbox))
        shim = os.path.join(_SANDBOX_SITE_DIR, "sitecustomize.py")
        r = _run_confined(
            f"""
            try:
                open({shim!r}).read(); print("READ_OK")
            except Exception as e:
                print("DENIED:" + type(e).__name__)
            """,
            sandbox,
            confiner,
        )
        assert "READ_OK" in r.stdout, (r.stdout, r.stderr)

    def test_shm_enumeration_is_denied_but_shared_memory_works(self, tmp_path):
        # /dev/shm is granted read/write/create (multiprocessing) but not READ_DIR:
        # a session must not enumerate other same-UID sessions' POSIX objects while
        # its own shared-memory create/read/write keeps working.
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        confiner = build_sandbox_confiner(str(sandbox))
        r = _run_confined(
            """
            import os
            from multiprocessing import shared_memory
            try:
                os.listdir("/dev/shm"); print("ENUM:ALLOWED")
            except Exception as e:
                print("ENUM:DENIED:" + type(e).__name__)
            shm = shared_memory.SharedMemory(create = True, size = 8)
            shm.buf[:3] = b"abc"
            print("SHM:" + bytes(shm.buf[:3]).decode())
            shm.close(); shm.unlink()
            """,
            sandbox,
            confiner,
        )
        assert "ENUM:DENIED:PermissionError" in r.stdout, (r.stdout, r.stderr)
        assert "SHM:abc" in r.stdout, (r.stdout, r.stderr)


# ── Real executor paths stay confined and working ────────────────────────────


@_needs_landlock
class TestExecutorPaths:
    @pytest.fixture(autouse = True)
    def _neutralize_ambient_virtualenv(self, monkeypatch):
        # See TestEnforcement._neutralize_ambient_virtualenv: keep the exported
        # VIRTUAL_ENV grant from covering pytest's tmp_path on dev hosts.
        monkeypatch.delenv("VIRTUAL_ENV", raising = False)

    def test_bash_exec_cannot_read_outside_but_works_inside(self, tmp_path, monkeypatch):
        from core.inference import tools

        sandbox = tmp_path / "sbx"
        sandbox.mkdir()
        secret = tmp_path / "host_secret.txt"
        secret.write_text("HOSTSECRET_5aa1")
        monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(sandbox))

        # cat is NOT in the blocklist, so only Landlock stops this host read.
        denied = tools._bash_exec(f"cat {secret}", session_id = "s")
        assert "HOSTSECRET_5aa1" not in denied, denied
        assert "Permission denied" in denied, denied

        # A write + read inside the workdir still works end to end.
        ok = tools._bash_exec("echo inside-ok > note.txt && cat note.txt", session_id = "s")
        assert "inside-ok" in ok, ok
        assert (sandbox / "note.txt").read_text().strip() == "inside-ok"

    def test_python_exec_benign_computation_runs(self, tmp_path, monkeypatch):
        from core.inference import tools

        sandbox = tmp_path / "sbx"
        sandbox.mkdir()
        monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(sandbox))
        out = tools._python_exec("print(sum(i * i for i in range(1, 11)))", session_id = "s")
        assert "385" in out, out


# ── Gate + graceful fallback (run everywhere) ────────────────────────────────


class TestGateAndFallback:
    def test_gate_off_disables_confinement(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_FS_CONFINE", "0")
        assert fs_confinement_enabled() is False
        assert build_sandbox_confiner("/nonexistent/workdir") is None

    @pytest.mark.parametrize("value", ["off", "false", "no", "disable"])
    def test_gate_off_synonyms(self, monkeypatch, value):
        monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_FS_CONFINE", value)
        assert fs_confinement_enabled() is False

    def test_landlock_unavailable_falls_back_without_crash(self, monkeypatch):
        # Simulate a host where Landlock is not present (old kernel / non-Linux /
        # other arch): confiner is None, nothing raises.
        monkeypatch.setattr(sandbox_fs, "_abi_cached", -1)
        assert landlock_available() is False
        assert fs_confinement_enabled() is False
        assert build_sandbox_confiner("/tmp") is None

    def test_abi_1_falls_back_no_confinement(self, monkeypatch):
        # ABI 1 (Linux 5.13-5.18) predates FS_REFER and unconditionally denies
        # reparenting a file across directories, breaking legitimate in-workdir
        # os.rename/shutil.move; so we require ABI >= 3 and degrade to the other
        # layers on ABI 1 instead of enforcing a rename-breaking ruleset.
        monkeypatch.setattr(sandbox_fs, "_abi_cached", 1)
        assert landlock_available() is False
        assert fs_confinement_enabled() is False
        assert build_sandbox_confiner("/tmp") is None

    def test_abi_2_falls_back_no_confinement(self, monkeypatch):
        # ABI 2 (Linux 5.19-6.1) predates FS_TRUNCATE. Unhandled Landlock rights
        # are always allowed, so a ruleset built on ABI 2 cannot deny
        # truncate(2)/ftruncate(2)/open(O_TRUNC): a sandboxed `> file` redirect or
        # open(path, "w") could zero a same-UID-writable host file OUTSIDE the
        # workdir, defeating the claimed write confinement. We require ABI >= 3
        # (which can handle TRUNCATE) and fall back to the other layers on ABI 2
        # rather than enforce a leaky ruleset.
        monkeypatch.setattr(sandbox_fs, "_abi_cached", 2)
        assert landlock_available() is False
        assert fs_confinement_enabled() is False
        assert build_sandbox_confiner("/tmp") is None

    def test_abi_3_enables_confinement(self, monkeypatch):
        # ABI 3 (Linux 6.2+) is the first version that can handle FS_TRUNCATE, so
        # it is the floor at which confinement is both complete and non-breaking.
        monkeypatch.setattr(sandbox_fs, "_abi_cached", 3)
        assert landlock_available() is True
        assert fs_confinement_enabled() is True
        # TRUNCATE is handled at the enforced floor, so it is denied outside the
        # explicit read-write grants (the workdir / shm), closing the ABI 2 gap.
        assert _fs_handled_mask(3) & sandbox_fs._FS_TRUNCATE

    def test_make_sandbox_preexec_none_confiner_is_unchanged(self):
        # The wiring in tools returns the plain preexec when confinement is off,
        # so the non-confined path stays byte-identical.
        from core.inference import tools
        assert tools._make_sandbox_preexec(None) is tools._sandbox_preexec

    def test_unavailable_logs_once(self, monkeypatch):
        monkeypatch.setattr(sandbox_fs, "_abi_cached", -1)
        monkeypatch.setattr(sandbox_fs, "_logged", False)
        calls = []
        monkeypatch.setattr(sandbox_fs.logger, "info", lambda *a, **k: calls.append((a, k)))
        sandbox_fs._log_once()
        sandbox_fs._log_once()
        sandbox_fs._log_once()
        assert len(calls) == 1, calls


# ── ABI mask + rule builder units (run everywhere) ───────────────────────────


class TestMaskAndRules:
    def test_fs_handled_mask_is_monotonic_and_has_expected_bits(self):
        masks = [_fs_handled_mask(a) for a in range(1, 8)]
        for lo, hi in zip(masks, masks[1:]):
            assert lo & hi == lo, "mask must only grow with ABI"
        assert not (_fs_handled_mask(1) & (1 << 13))  # no REFER at ABI 1
        assert _fs_handled_mask(2) & (1 << 13)  # REFER at ABI 2
        assert _fs_handled_mask(3) & (1 << 14)  # TRUNCATE at ABI 3
        assert _fs_handled_mask(5) & (1 << 15)  # IOCTL_DEV at ABI 5

    def test_build_rules_grants_workdir_readwrite(self, tmp_path):
        handled = _fs_handled_mask(5)
        rules = dict(_build_rules(str(tmp_path), handled))
        real = os.path.realpath(str(tmp_path))
        assert real in rules
        # Workdir is granted every handled right (read + write + remove + ...).
        assert rules[real] & (1 << 1)  # WRITE_FILE
        assert rules[real] & (1 << 2)  # READ_FILE
        assert rules[real] & (1 << 5)  # REMOVE_FILE

    def test_build_rules_file_target_has_no_dir_only_bits(self, tmp_path):
        # /etc/resolv.conf is a file: its rule must not carry READ_DIR (dir-only),
        # which the kernel would reject with EINVAL, silently dropping the rule.
        handled = _fs_handled_mask(5)
        rules = dict(_build_rules(str(tmp_path), handled))
        resolv = os.path.realpath("/etc/resolv.conf")
        if resolv in rules:  # present on Linux; skip the assert elsewhere
            assert not (rules[resolv] & (1 << 3))  # no READ_DIR on a file
            assert not (rules[resolv] & (1 << 1))  # read-only: no WRITE_FILE
            assert rules[resolv] & (1 << 2)  # READ_FILE granted

    def test_build_rules_does_not_grant_etc_passwd(self, tmp_path):
        # /etc/passwd is deliberately NOT allow-listed: granting it would let a
        # sandboxed `cat /etc/passwd` enumerate every host account, the exact
        # host-user enumeration path this confinement is meant to close. The child
        # runs fine without it (only libc identity lookups fail, and catchably).
        handled = _fs_handled_mask(5)
        rules = dict(_build_rules(str(tmp_path), handled))
        assert os.path.realpath("/etc/passwd") not in rules, sorted(rules)

    @staticmethod
    def _make_fake_venv(root: Path) -> Path:
        # A minimal but valid virtualenv layout: pyvenv.cfg + bin/python +
        # lib/python3.X/site-packages, exactly what _venv_grant_paths validates
        # and grants.
        (root / "bin").mkdir(parents = True)
        (root / "pyvenv.cfg").write_text("home = /usr/bin\n")
        (root / "bin" / "python").write_text("#!/bin/sh\n")
        (root / "lib" / "python3.11" / "site-packages").mkdir(parents = True)
        return root

    def test_build_rules_grants_only_venv_bin_and_site_packages_readexec(
        self, tmp_path, monkeypatch
    ):
        # tools._build_safe_env prepends $VIRTUAL_ENV/bin to the child PATH even
        # when VIRTUAL_ENV differs from the running interpreter's prefix. Only that
        # scripts dir and the venv's site-packages are granted (read+execute), so
        # console scripts resolve and imports load -- NOT the whole venv tree.
        venv = self._make_fake_venv(tmp_path / "othervenv")
        secret = venv / "secret.txt"  # arbitrary file elsewhere in the tree
        secret.write_text("VENV_SECRET")
        workdir = tmp_path / "wd"
        workdir.mkdir()
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))
        handled = _fs_handled_mask(5)
        rules = dict(_build_rules(str(workdir), handled))

        bindir = os.path.realpath(str(venv / "bin"))
        site = os.path.realpath(str(venv / "lib" / "python3.11" / "site-packages"))
        assert bindir in rules, sorted(rules)
        assert site in rules, sorted(rules)
        for real in (bindir, site):
            assert rules[real] & (1 << 0)  # FS_EXECUTE (console scripts / interpreter)
            assert rules[real] & (1 << 2)  # READ_FILE
            assert rules[real] & (1 << 3)  # READ_DIR
            assert not (rules[real] & (1 << 1))  # read-only: no WRITE_FILE
            assert not (rules[real] & (1 << 14))  # read-only: no TRUNCATE
        # The venv ROOT is NOT granted: a rule there would recursively expose the
        # whole tree (secret.txt included) to the sandbox.
        assert os.path.realpath(str(venv)) not in rules, sorted(rules)

    def test_build_rules_does_not_grant_non_venv_virtualenv(self, tmp_path, monkeypatch):
        # VIRTUAL_ENV pointing at a directory that is NOT a real virtualenv (no
        # pyvenv.cfg, no bin/python) must grant NOTHING: otherwise a workspace /
        # repo root exported as VIRTUAL_ENV would expose its source to the sandbox.
        notvenv = tmp_path / "workspace_root"
        (notvenv / "src").mkdir(parents = True)
        (notvenv / "src" / "app.py").write_text("SECRET_SOURCE")
        (notvenv / ".env").write_text("API_KEY=secret")
        workdir = tmp_path / "wd"
        workdir.mkdir()
        monkeypatch.setenv("VIRTUAL_ENV", str(notvenv))
        handled = _fs_handled_mask(5)
        rules = dict(_build_rules(str(workdir), handled))
        real = os.path.realpath(str(notvenv))
        assert real not in rules, sorted(rules)
        # No descendant of the non-venv tree is granted either.
        assert not any(p == real or p.startswith(real + os.sep) for p in rules), sorted(rules)

    def test_build_rules_no_virtualenv_grant_when_unset(self, tmp_path, monkeypatch):
        # With VIRTUAL_ENV unset, no extra venv rule is added (only the running
        # interpreter's prefixes are granted).
        monkeypatch.delenv("VIRTUAL_ENV", raising = False)
        handled = _fs_handled_mask(5)
        workdir = tmp_path / "wd"
        workdir.mkdir()
        # Should not raise and should not key a rule on an empty path.
        rules = dict(_build_rules(str(workdir), handled))
        assert "" not in rules

    def test_build_rules_grants_sandbox_site_dir_readonly(self):
        # The sandbox sitecustomize shim dir is on the child PYTHONPATH: readable
        # (else the path-remap fails to import under confinement) but never writable.
        handled = _fs_handled_mask(5)
        rules = dict(_build_rules("/tmp", handled))
        site = os.path.realpath(_SANDBOX_SITE_DIR)
        assert site in rules, sorted(rules)
        assert rules[site] & (1 << 2)  # READ_FILE
        assert rules[site] & (1 << 3)  # READ_DIR
        assert not (rules[site] & (1 << 1))  # read-only: no WRITE_FILE

    def test_build_rules_shm_is_readwrite_but_not_enumerable(self):
        # /dev/shm is read/write/create for multiprocessing, but READ_DIR is
        # withheld so a session cannot enumerate other same-UID sessions' objects.
        handled = _fs_handled_mask(5)
        rules = dict(_build_rules("/tmp", handled))
        shm = os.path.realpath(sandbox_fs._RW_SHM)
        if shm in rules:  # /dev/shm present on Linux; skip the assert elsewhere
            assert rules[shm] & (1 << 1)  # WRITE_FILE
            assert rules[shm] & (1 << 2)  # READ_FILE
            assert not (rules[shm] & (1 << 3))  # no READ_DIR (no enumeration)

    def test_build_rules_skips_nonexistent_paths(self, tmp_path, monkeypatch):
        handled = _fs_handled_mask(5)
        monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_FS_ALLOW", "/no/such/path/xyz_123")
        rules = dict(_build_rules(str(tmp_path), handled))
        assert not any("xyz_123" in p for p in rules)

    def test_build_rules_grants_ca_store_readonly(self):
        # HTTPS egress to allow-listed hosts is supported (git/pip/requests); the
        # OpenSSL CA store commonly realpath-resolves into /etc (e.g.
        # /usr/lib/ssl/certs -> /etc/ssl/certs), which the /etc denials would
        # otherwise block, breaking TLS certificate verification. It must be
        # granted read-only.
        import ssl

        handled = _fs_handled_mask(5)
        rules = dict(_build_rules("/tmp", handled))
        dvp = ssl.get_default_verify_paths()
        reals = {
            os.path.realpath(p)
            for p in (dvp.openssl_cafile, dvp.openssl_capath, dvp.cafile, dvp.capath)
            if p and os.path.exists(p)
        }
        if not reals:  # no system CA store on this host (e.g. some non-Linux CI)
            pytest.skip("no resolvable system CA store")
        granted = reals & set(rules)
        assert granted, (sorted(reals), sorted(rules))
        for real in granted:
            assert rules[real] & (1 << 2)  # READ_FILE
            assert not (rules[real] & (1 << 1))  # read-only: no WRITE_FILE

    def test_ca_paths_exclude_env_influenced_private_cafile(self, tmp_path, monkeypatch):
        # ssl.get_default_verify_paths() resolves cafile/capath from the PARENT's
        # SSL_CERT_FILE / SSL_CERT_DIR. The sandboxed child never receives those
        # env vars (tools._build_safe_env drops them), so it uses the compile-time
        # openssl_* defaults; granting the parent's env-pointed path is useless to
        # the child and could leak a private file into the sandbox. Only the
        # compile-time openssl_cafile/openssl_capath may be granted.
        import ssl

        workdir = tmp_path / "sbx"
        workdir.mkdir()
        private = tmp_path / "private.pem"  # a parent SSL_CERT_FILE, OUTSIDE workdir
        private.write_text("PRIVATE_CA_9f")
        public_cafile = tmp_path / "compile_default.pem"
        public_cafile.write_text("PUBLIC_CA")
        fake = ssl.DefaultVerifyPaths(
            cafile = str(private),  # env-resolved (SSL_CERT_FILE) -> private
            capath = None,
            openssl_cafile_env = "SSL_CERT_FILE",
            openssl_cafile = str(public_cafile),  # compile-time default
            openssl_capath_env = "SSL_CERT_DIR",
            openssl_capath = None,
        )
        monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: fake)

        paths = sandbox_fs._ca_cert_paths()
        assert str(private) not in paths, paths
        assert str(public_cafile) in paths, paths

        handled = _fs_handled_mask(5)
        rules = dict(_build_rules(str(workdir), handled))
        assert os.path.realpath(str(private)) not in rules
        assert os.path.realpath(str(public_cafile)) in rules

    def test_build_rules_grants_resolv_conf_readonly(self):
        # Egress is direct (no netns/proxy), so glibc getaddrinfo reads
        # /etc/resolv.conf for the nameserver; without it allow-listed HTTPS by
        # hostname cannot resolve. Granted read-only (only nameserver IPs, not FS).
        handled = _fs_handled_mask(5)
        rules = dict(_build_rules("/tmp", handled))
        real = os.path.realpath("/etc/resolv.conf")
        if not os.path.exists(real):  # absent on some non-Linux CI
            pytest.skip("no /etc/resolv.conf on this host")
        assert real in rules, sorted(rules)
        assert rules[real] & (1 << 2)  # READ_FILE
        assert not (rules[real] & (1 << 1))  # read-only: no WRITE_FILE
        assert not (rules[real] & (1 << 3))  # no READ_DIR on a file

    def test_build_rules_does_not_grant_opt_broadly(self, tmp_path):
        # A blanket /opt grant is redundant (an /opt-installed interpreter is
        # covered by the sys.prefix / executable-dir rules) and would expose
        # operator data under /opt, so no rule is keyed on bare /opt.
        handled = _fs_handled_mask(5)
        rules = dict(_build_rules(str(tmp_path), handled))
        assert "/opt" not in rules

    def test_build_rules_never_grants_filesystem_root(self, monkeypatch):
        # A Python configured with --prefix=/ (or an interpreter at /python) makes
        # an interpreter root resolve to "/". A rule on "/" would grant read and
        # traversal of the whole filesystem and defeat the confinement, so the
        # root must be skipped.
        import sys as _sys

        handled = _fs_handled_mask(5)
        for attr in ("prefix", "base_prefix", "exec_prefix", "base_exec_prefix"):
            monkeypatch.setattr(_sys, attr, "/", raising = False)
        monkeypatch.setattr(_sys, "executable", "/python", raising = False)
        rules = dict(_build_rules("/tmp", handled))
        assert os.sep not in rules, sorted(rules)
