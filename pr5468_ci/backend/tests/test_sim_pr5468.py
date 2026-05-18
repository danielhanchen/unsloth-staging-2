"""
Comprehensive simulation tests for PR #5468 (Studio OS-level sandbox).

Goal: exercise every interesting code path in sandbox.py and the tools.py
hooks, on Linux, while mocking the macOS / Windows branches via
sys.platform monkeypatching where direct execution isn't possible.

Each test class doubles as a regression check for one of the findings
flagged in the upstream review.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import uuid
from pathlib import Path
from unittest import mock

import pytest

_HERE = Path(__file__).resolve()
_BACKEND_ROOT = _HERE.parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import sandbox as sb  # noqa: E402
from core.inference import tools  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_sandbox_cache():
    """sandbox_available() caches its first result; reset before each test."""
    sb._sandbox_available_cache = None
    sb._linux_bwrap_path = None
    yield
    sb._sandbox_available_cache = None
    sb._linux_bwrap_path = None


@pytest.fixture
def workdir(tmp_path):
    wd = tmp_path / "wd"
    wd.mkdir()
    return str(wd)


# ---------------------------------------------------------------------------
# 1. Cross-platform sandbox_available() — by mocking sys.platform
# ---------------------------------------------------------------------------

class TestSandboxAvailableCrossPlatform:
    """Ensure the platform branches do what the spec promises."""

    def test_windows_returns_false(self, monkeypatch):
        monkeypatch.setattr(sb.sys, "platform", "win32")
        assert sb.sandbox_available() is False

    def test_unknown_platform_returns_false(self, monkeypatch):
        monkeypatch.setattr(sb.sys, "platform", "freebsd13")
        assert sb.sandbox_available() is False

    def test_darwin_branch_taken(self, monkeypatch):
        monkeypatch.setattr(sb.sys, "platform", "darwin")
        called = {"v": False}

        def _fake_probe():
            called["v"] = True
            return False

        monkeypatch.setattr(sb, "_macos_probe", _fake_probe)
        sb.sandbox_available()
        assert called["v"] is True

    def test_linux_branch_taken(self, monkeypatch):
        monkeypatch.setattr(sb.sys, "platform", "linux")
        called = {"v": False}

        def _fake_probe():
            called["v"] = True
            return False

        monkeypatch.setattr(sb, "_linux_probe", _fake_probe)
        sb.sandbox_available()
        assert called["v"] is True

    def test_result_is_cached(self, monkeypatch):
        monkeypatch.setattr(sb.sys, "platform", "linux")
        calls = {"n": 0}

        def _fake_probe():
            calls["n"] += 1
            return True

        monkeypatch.setattr(sb, "_linux_probe", _fake_probe)
        sb.sandbox_available()
        sb.sandbox_available()
        sb.sandbox_available()
        assert calls["n"] == 1

    def test_darwin_missing_sandbox_exec(self, monkeypatch):
        monkeypatch.setattr(sb.sys, "platform", "darwin")
        monkeypatch.setattr(sb.os.path, "exists", lambda p: False)
        assert sb._macos_probe() is False

    def test_linux_missing_bwrap(self, monkeypatch):
        monkeypatch.setattr(sb.sys, "platform", "linux")
        monkeypatch.setattr(sb.shutil, "which", lambda x: None)
        assert sb._linux_probe() is False


# ---------------------------------------------------------------------------
# 2. _safe_subpath — injection guard for Seatbelt profile
# ---------------------------------------------------------------------------

class TestSafeSubpath:
    """The Seatbelt profile string interpolates paths; any unescaped quote or
    backslash could close the literal and inject Scheme. We assert the guard."""

    @pytest.mark.parametrize(
        "good",
        [
            "/usr/lib",
            "/Users/alice/work",
            "/private/etc",
            "/with spaces/inside",
            "/unicode/路径",
            "/",
            "/tmp/-_-/path",
        ],
    )
    def test_good_paths_pass(self, good):
        assert sb._safe_subpath(good) == good

    @pytest.mark.parametrize(
        "bad",
        [
            '/foo"bar',          # double-quote
            "/foo\\bar",          # backslash
            "/foo\nbar",          # newline
            "/foo\rbar",          # carriage return
            "/foo\x00bar",        # NUL byte
            '/foo")(allow file-write* (subpath "/")(',  # actual injection shape
        ],
    )
    def test_bad_paths_raise(self, bad):
        with pytest.raises(ValueError):
            sb._safe_subpath(bad)


# ---------------------------------------------------------------------------
# 3. Seatbelt profile generation (macOS) — structural validity
# ---------------------------------------------------------------------------

class TestSeatbeltProfile:
    """We're on Linux so we can't load the profile, but we can pin its shape
    and verify it doesn't accidentally drop required allows."""

    def _profile(self, workdir):
        return sb._macos_seatbelt_profile(workdir)

    def test_profile_starts_with_version_and_deny_default(self, workdir):
        p = self._profile(workdir)
        # The "(version 1)" line lives on its own line per the source.
        assert "(version 1)" in p.splitlines()[0]
        assert "(deny default)" in p

    def test_profile_denies_network(self, workdir):
        p = self._profile(workdir)
        for direction in ("network-outbound", "network-inbound", "network-bind"):
            assert f"(deny {direction})" in p, direction

    def test_profile_allows_workdir_rw(self, workdir):
        p = self._profile(workdir)
        wd = os.path.realpath(workdir)
        assert f'(allow file-read* (subpath "{wd}"))' in p
        assert f'(allow file-write* (subpath "{wd}"))' in p

    def test_profile_balances_parens(self, workdir):
        p = self._profile(workdir)
        opens = p.count("(")
        closes = p.count(")")
        assert opens == closes, f"unbalanced parens: {opens} vs {closes}"

    def test_profile_includes_python_paths(self, workdir):
        p = self._profile(workdir)
        for rp in sb._python_read_paths():
            assert f'(subpath "{rp}")' in p, rp

    def test_profile_rejects_injection_via_workdir(self):
        bad = '/tmp/")(allow file-write* (subpath "/"))'
        with pytest.raises(ValueError):
            sb._macos_seatbelt_profile(bad)

    def test_profile_includes_executable_map(self, workdir):
        p = self._profile(workdir)
        # Required so dyld can mmap libpython + dylibs PROT_EXEC.
        assert "(allow file-map-executable" in p
        assert '(subpath "/usr/lib")' in p

    def test_profile_includes_macos_mach_lookups(self, workdir):
        p = self._profile(workdir)
        # If trustd / SecurityServer disappear, TLS dies.
        for svc in (
            "com.apple.trustd",
            "com.apple.SecurityServer",
            "com.apple.system.opendirectoryd.libinfo",
        ):
            assert svc in p, svc


# ---------------------------------------------------------------------------
# 4. bwrap argv builder (Linux) — structure + bind list
# ---------------------------------------------------------------------------

class TestBwrapArgv:
    def _argv(self, inner, workdir):
        # _linux_bwrap_argv reads _linux_bwrap_path; populate it.
        sb._linux_bwrap_path = "/usr/bin/bwrap"
        return sb._linux_bwrap_argv(inner, workdir)

    def test_first_arg_is_bwrap(self, workdir):
        argv = self._argv(["/bin/echo", "hi"], workdir)
        assert argv[0] == "/usr/bin/bwrap"

    def test_separator_before_inner(self, workdir):
        argv = self._argv(["/bin/echo", "hi"], workdir)
        sep = argv.index("--")
        assert argv[sep + 1:] == ["/bin/echo", "hi"]

    def test_unshare_all_present(self, workdir):
        argv = self._argv(["/bin/true"], workdir)
        assert "--unshare-all" in argv

    def test_die_with_parent_present(self, workdir):
        argv = self._argv(["/bin/true"], workdir)
        assert "--die-with-parent" in argv

    def test_workdir_bound_rw(self, workdir):
        argv = self._argv(["/bin/true"], workdir)
        wd = os.path.realpath(workdir)
        # `--bind <wd> <wd>` should appear (rw bind on the workdir)
        joined = " ".join(argv)
        assert f"--bind {wd} {wd}" in joined

    def test_etc_is_bound(self, workdir):
        """FINDING: /etc is broadly ro-bound, exposing passwd/machine-id/etc."""
        argv = self._argv(["/bin/true"], workdir)
        joined = " ".join(argv)
        assert "--ro-bind-try /etc /etc" in joined, (
            "P1 finding: /etc is bound wholesale. Expected the implementation "
            "to narrow this to /etc/ssl, /etc/resolv.conf, etc."
        )

    def test_proc_and_dev_present(self, workdir):
        argv = self._argv(["/bin/true"], workdir)
        assert "--proc" in argv
        assert "--dev" in argv
        assert "--tmpfs" in argv

    def test_empty_inner_argv_raises(self, workdir):
        with pytest.raises(ValueError):
            sb.build_sandbox_argv([], workdir)

    def test_unsupported_platform_raises(self, workdir, monkeypatch):
        monkeypatch.setattr(sb.sys, "platform", "freebsd13")
        with pytest.raises(RuntimeError):
            sb.build_sandbox_argv(["/bin/true"], workdir)


# ---------------------------------------------------------------------------
# 5. Symlinked workdir scenario — review_10 finding
# ---------------------------------------------------------------------------

class TestSymlinkedWorkdir:
    """When ~/studio_sandbox is on a different disk via symlink, the child
    gets the original path as cwd while bwrap only binds the realpath."""

    def test_symlink_workdir_mismatch(self, tmp_path):
        sb._linux_bwrap_path = "/usr/bin/bwrap"

        real = tmp_path / "real_workdir"
        real.mkdir()
        link = tmp_path / "link_workdir"
        link.symlink_to(real)

        # Build argv exactly as _python_exec / _bash_exec would
        argv = sb._linux_bwrap_argv(["/bin/true"], str(link))
        joined = " ".join(argv)

        real_resolved = os.path.realpath(str(link))
        # bwrap binds the realpath...
        assert f"--bind {real_resolved} {real_resolved}" in joined
        # ...but the inner subprocess would be invoked with cwd=link (the
        # symlink path). cwd=link is NOT bound, so the inner Python exec
        # would fail with ENOENT inside the sandbox.
        assert str(link) != real_resolved
        assert f"--bind {real_resolved} {str(link)}" not in joined, (
            "P2 finding: bind happens on realpath only. The fix is to also "
            "bind the symlink path so cwd / tmp_path resolves inside the namespace."
        )


# ---------------------------------------------------------------------------
# 6. Fail-open behavior and /etc exposure on this host (reproducible)
# ---------------------------------------------------------------------------

class TestFailOpenOnThisHost:
    """On a host where bwrap is present but the probe fails, the PR falls
    back to unsandboxed execution. We assert that observable outcome."""

    def test_probe_fails_on_this_host(self):
        # This is the host configuration: bwrap exists but userns is locked.
        # We pre-seed the bwrap path so we don't depend on it being on PATH.
        ok = sb._linux_probe()
        # Two valid outcomes: probe succeeds (lucky host) or fails (this host).
        # Either way, we just want to know which branch we're on for the
        # next assertion.
        if ok:
            pytest.skip("host has working bwrap; fail-open branch not exercised")
        assert sb.sandbox_available() is False

    def test_python_exec_runs_unsandboxed_when_unavailable(self, tmp_path, monkeypatch):
        """When sandbox_available() is False, _python_exec falls back."""
        monkeypatch.setattr(tools, "sandbox_available", lambda: False)
        sid = "_simtest_unavail"
        monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))

        sentinel = os.path.expanduser(
            f"~/.studio_sim_sentinel_{uuid.uuid4().hex}.txt"
        )
        secret = f"FAILOPEN-{uuid.uuid4().hex}"
        Path(sentinel).write_text(secret)
        try:
            code = (
                f"try:\n"
                f"    with open({sentinel!r}) as f: print('LEAKED:', f.read())\n"
                f"except Exception as e: print('DENIED:', type(e).__name__)\n"
            )
            out = tools._python_exec(code, session_id=sid, timeout=15)
        finally:
            Path(sentinel).unlink(missing_ok=True)

        # This is the bug under test: it leaks, exactly as if the PR weren't
        # there. The PR's fail-open path doesn't block this.
        assert "LEAKED:" in out and secret in out, (
            "PR fails open as documented: sandbox unavailable, $HOME readable."
        )


# ---------------------------------------------------------------------------
# 7. Pre-PR vs post-PR sentinel behavior on the unsandboxed fallback path
# ---------------------------------------------------------------------------

class TestPrePostBehavior:
    """The PR's stated security claim is that bash/python tools cannot read
    $HOME. On THIS host the post-PR code falls open, so the behavior is
    identical to pre-PR. We document that explicitly."""

    def test_bash_leak_pre_pr(self, tmp_path, monkeypatch):
        """Simulate the pre-PR _bash_exec by patching sandbox_available."""
        monkeypatch.setattr(tools, "sandbox_available", lambda: False)
        sid = "_simtest_pre"
        monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))

        sentinel = os.path.expanduser(
            f"~/.studio_sim_sentinel_{uuid.uuid4().hex}.txt"
        )
        secret = f"PRE-{uuid.uuid4().hex}"
        Path(sentinel).write_text(secret)
        try:
            out = tools._bash_exec(
                f"cat {shlex.quote(sentinel)}",
                session_id=sid, timeout=15,
            )
        finally:
            Path(sentinel).unlink(missing_ok=True)
        assert secret in out  # pre-PR shape: leaks


# ---------------------------------------------------------------------------
# 8. _exec_chain_symlinks helper
# ---------------------------------------------------------------------------

class TestExecChainSymlinks:
    def test_no_symlinks_returns_empty(self, tmp_path):
        target = tmp_path / "real_binary"
        target.write_text("")
        out = sb._exec_chain_symlinks(str(target))
        assert out == []

    def test_simple_chain(self, tmp_path):
        real = tmp_path / "real_bin"
        real.write_text("")
        link1 = tmp_path / "link1"
        link1.symlink_to(real)
        link2 = tmp_path / "link2"
        link2.symlink_to(link1)
        out = sb._exec_chain_symlinks(str(link2))
        # Both intermediate symlinks should show up (the resolution chain).
        assert str(link2) in out
        assert str(link1) in out

    def test_cycle_protection(self, tmp_path):
        """A symlink cycle should not hang."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.symlink_to(b)
        b.symlink_to(a)
        out = sb._exec_chain_symlinks(str(a))  # must terminate
        assert isinstance(out, list)


# ---------------------------------------------------------------------------
# 9. Inner subprocess shape: _python_exec / _bash_exec wiring
# ---------------------------------------------------------------------------

class TestExecWiring:
    """Confirm that when sandbox_available() is True the argv is wrapped;
    when False, it's passed through. We intercept subprocess.Popen so we
    don't actually run anything."""

    def _capture_popen(self, monkeypatch):
        captured = {}
        orig = subprocess.Popen

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                captured["argv"] = argv
                captured["kwargs"] = kwargs
                self.returncode = 0
                self.pid = 12345

            def communicate(self, timeout=None):
                return ("(ok)", "")

            def poll(self):
                return self.returncode

            def kill(self):
                pass

        monkeypatch.setattr(subprocess, "Popen", _FakePopen)
        return captured

    def test_python_exec_wraps_when_available(self, tmp_path, monkeypatch):
        captured = self._capture_popen(monkeypatch)
        monkeypatch.setattr(tools, "sandbox_available", lambda: True)
        monkeypatch.setattr(
            tools, "build_sandbox_argv",
            lambda inner, wd: ["BWRAP_WRAPPER"] + inner,
        )
        sid = "_simtest_wrap"
        monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))
        tools._python_exec("print('hi')", session_id=sid, timeout=5)
        assert captured["argv"][0] == "BWRAP_WRAPPER"

    def test_python_exec_passthrough_when_unavailable(self, tmp_path, monkeypatch):
        captured = self._capture_popen(monkeypatch)
        monkeypatch.setattr(tools, "sandbox_available", lambda: False)
        sid = "_simtest_pass"
        monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))
        tools._python_exec("print('hi')", session_id=sid, timeout=5)
        assert captured["argv"][0] == sys.executable

    def test_bash_exec_wraps_when_available(self, tmp_path, monkeypatch):
        captured = self._capture_popen(monkeypatch)
        monkeypatch.setattr(tools, "sandbox_available", lambda: True)
        monkeypatch.setattr(
            tools, "build_sandbox_argv",
            lambda inner, wd: ["BWRAP_WRAPPER"] + inner,
        )
        sid = "_simtest_bash_wrap"
        monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))
        tools._bash_exec("echo hi", session_id=sid, timeout=5)
        assert captured["argv"][0] == "BWRAP_WRAPPER"
        # bash -c with the command is at the tail
        assert "echo hi" in captured["argv"][-1]


# ---------------------------------------------------------------------------
# 10. Probe argv shape — verify it doesn't carry preexec_fn that would
#     change the probe's privilege state (P1 finding)
# ---------------------------------------------------------------------------

class TestProbeVsRuntimePrivilegeMatch:
    """The probe runs bwrap without _sandbox_preexec, but the runtime exec
    applies preexec_fn that sets PR_SET_NO_NEW_PRIVS. We don't have a setuid
    bwrap on this host, so we can only assert the shape mismatch."""

    def test_probe_does_not_use_sandbox_preexec(self, monkeypatch):
        observed = {}
        orig_run = subprocess.run

        def _wrap_run(argv, **kwargs):
            observed["argv"] = argv
            observed["kwargs"] = kwargs
            return mock.MagicMock(returncode=0, stderr=b"")

        monkeypatch.setattr(sb.subprocess, "run", _wrap_run)
        sb._probe([sb._SANDBOX_EXEC, "-p", "(version 1)(allow default)", "/usr/bin/true"], "x")
        assert "preexec_fn" not in observed["kwargs"], (
            "Probe runs without preexec_fn — but _python_exec / _bash_exec "
            "DO pass preexec_fn=_sandbox_preexec to subprocess.Popen, which "
            "sets PR_SET_NO_NEW_PRIVS. On hosts with setuid bwrap this "
            "mismatch means the probe lies."
        )


# ---------------------------------------------------------------------------
# 11. _sandbox_preexec: actually verify the prctl call shape
# ---------------------------------------------------------------------------

class TestSandboxPreexecPrctl:
    """Confirm _sandbox_preexec hits PR_SET_NO_NEW_PRIVS (38)."""

    def test_prctl_no_new_privs_is_called(self, monkeypatch):
        if sys.platform != "linux":
            pytest.skip("Linux-only path")
        if tools._libc is None:
            pytest.skip("libc not loadable here")

        calls = []
        orig_prctl = tools._libc.prctl

        def _fake_prctl(*args):
            calls.append(args)
            return 0

        monkeypatch.setattr(tools._libc, "prctl", _fake_prctl)
        tools._sandbox_preexec()
        # PR_SET_NO_NEW_PRIVS = 38
        no_new_privs = [c for c in calls if c[0] == 38]
        assert no_new_privs, "preexec did not call PR_SET_NO_NEW_PRIVS"
        # PR_SET_PDEATHSIG = 1
        pdeathsig = [c for c in calls if c[0] == 1]
        assert pdeathsig, "preexec did not call PR_SET_PDEATHSIG"


# ---------------------------------------------------------------------------
# 12. End-to-end: real bash + real python under the actual code path
# ---------------------------------------------------------------------------

class TestEndToEndOnThisHost:
    """Smoke tests that the existing happy path still works through the new
    wrappers, regardless of whether sandbox is active."""

    def test_python_exec_prints_hello(self, tmp_path, monkeypatch):
        sid = "_simtest_e2e_py"
        monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))
        out = tools._python_exec("print('hello-sim')", session_id=sid, timeout=15)
        assert "hello-sim" in out

    def test_bash_exec_prints_hello(self, tmp_path, monkeypatch):
        sid = "_simtest_e2e_bash"
        monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))
        out = tools._bash_exec("echo hello-sim", session_id=sid, timeout=15)
        assert "hello-sim" in out

    def test_python_exec_writes_inside_workdir(self, tmp_path, monkeypatch):
        sid = "_simtest_e2e_write"
        monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))
        code = (
            "from pathlib import Path\n"
            "Path('out.txt').write_text('ok')\n"
            "print('done')\n"
        )
        out = tools._python_exec(code, session_id=sid, timeout=15)
        assert "done" in out
        assert (tmp_path / "out.txt").exists()

    def test_python_exec_existing_ast_gate_still_blocks(self, tmp_path, monkeypatch):
        """The pre-existing static gate must still block obvious shell escapes."""
        sid = "_simtest_e2e_block"
        monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))
        out = tools._python_exec(
            "import os; os.system('rm -rf /')",
            session_id=sid, timeout=5,
        )
        assert "unsafe code detected" in out or "Blocked" in out, out

    def test_bash_exec_existing_blocklist_still_blocks(self, tmp_path, monkeypatch):
        sid = "_simtest_e2e_blocklist"
        monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))
        out = tools._bash_exec("sudo whoami", session_id=sid, timeout=5)
        assert "Blocked" in out


# ---------------------------------------------------------------------------
# 13. Determinism / idempotency of profile / argv generation
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_seatbelt_profile_idempotent(self, workdir):
        p1 = sb._macos_seatbelt_profile(workdir)
        p2 = sb._macos_seatbelt_profile(workdir)
        assert p1 == p2

    def test_bwrap_argv_idempotent(self, workdir):
        sb._linux_bwrap_path = "/usr/bin/bwrap"
        a1 = sb._linux_bwrap_argv(["/bin/true"], workdir)
        a2 = sb._linux_bwrap_argv(["/bin/true"], workdir)
        assert a1 == a2
