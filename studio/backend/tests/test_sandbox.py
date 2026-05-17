# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
End-to-end tests for the OS-level sandbox wired into ``_python_exec``
and ``_bash_exec``.

The platform-agnostic tests (workdir write, $HOME read deny, bash $HOME
read deny, network deny) run on both macOS (Seatbelt) and Linux
(bubblewrap) — same security claims, different mechanisms. The
``/System/Applications``-enumeration test is darwin-specific because
that path only exists on macOS.

These tests are the only layer that proves the sandbox does what it
claims — anything that only inspects the profile string is checking
typography, not enforcement.
"""

import os
import shlex
import sys
import uuid
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

pytestmark = pytest.mark.skipif(
    sys.platform not in ("darwin", "linux"),
    reason = "sandbox tests run on macOS and Linux only",
)


@pytest.fixture
def sandboxed_workdir(tmp_path, monkeypatch):
    """Point tool execution's workdir lookup at a pytest tmp_path."""
    from core.inference.sandbox import sandbox_available

    if not sandbox_available():
        pytest.skip("sandbox unavailable (binary missing or cannot apply policy)")

    from core.inference import tools

    sid = "_sbtest"
    monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))
    yield sid, str(tmp_path)


@pytest.fixture
def home_sentinel():
    """Create a sentinel file in $HOME outside the sandbox, yield path + secret.

    The sentinel proves *negative*: if the sandboxed code reads the
    file, the secret appears in the tool output. We assert the secret
    is absent. Without a sentinel, a bare $HOME (empty CI runner)
    would make ``open(~/.zshrc)`` raise ``FileNotFoundError`` and the
    test would pass for the wrong reason.
    """
    secret = f"SECRET-{uuid.uuid4().hex}"
    path = os.path.expanduser(f"~/.studio_sandbox_test_{uuid.uuid4().hex}.txt")
    Path(path).write_text(secret)
    try:
        yield path, secret
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _run_python(code: str, sid: str) -> str:
    from core.inference.tools import _python_exec

    return _python_exec(code, session_id = sid, timeout = 30)


def _run_bash(command: str, sid: str) -> str:
    from core.inference.tools import _bash_exec

    return _bash_exec(command, session_id = sid, timeout = 30)


def test_workdir_write_succeeds(sandboxed_workdir):
    sid, wd = sandboxed_workdir
    code = (
        "from pathlib import Path\n"
        'Path("hi.txt").write_text("ok")\n'
        'print("done")\n'
    )
    out = _run_python(code, sid)
    assert "done" in out, out
    assert os.path.exists(os.path.join(wd, "hi.txt"))


def test_home_read_denied(sandboxed_workdir, home_sentinel):
    sid, _ = sandboxed_workdir
    path, secret = home_sentinel
    code = (
        f"try:\n"
        f"    with open({path!r}) as f: print('LEAKED:', f.read())\n"
        f"except (PermissionError, FileNotFoundError, OSError) as e:\n"
        f"    print('DENIED:', type(e).__name__)\n"
    )
    out = _run_python(code, sid)
    assert secret not in out, out
    assert "LEAKED:" not in out, out
    assert "DENIED:" in out, out


def test_bash_home_read_denied(sandboxed_workdir, home_sentinel):
    """The terminal tool must enforce the same $HOME-denial as the python tool."""
    sid, _ = sandboxed_workdir
    path, secret = home_sentinel
    out = _run_bash(f"cat {shlex.quote(path)}", sid)
    assert secret not in out, out
    # Confirm cat actually ran and was denied, not silently no-op'd.
    assert any(
        s in out
        for s in ("Permission denied", "Operation not permitted", "No such file")
    ), out


def test_network_denied(sandboxed_workdir):
    """Use an allowlisted host so the AST pre-check passes and the
    sandbox is the only thing left to block the egress."""
    sid, _ = sandboxed_workdir
    # The import is inside the try block so that any sandbox-induced
    # failure (the kernel blocking socket(), or an upstream PermissionError
    # while the import machinery probes a denied $HOME path on editable
    # installs) is caught and surfaced as DENIED. Either way no network
    # I/O actually happens, which is the security claim under test.
    code = (
        "try:\n"
        "    import urllib.request\n"
        "    r = urllib.request.urlopen('https://wikipedia.org', timeout=10).read(100)\n"
        "    print('LEAKED:', len(r))\n"
        "except Exception as e:\n"
        "    print('DENIED:', type(e).__name__)\n"
    )
    out = _run_python(code, sid)
    assert "LEAKED:" not in out, out
    assert "DENIED:" in out, out


def test_sandbox_off_actually_leaks(tmp_path, monkeypatch, home_sentinel):
    """Control test: with the sandbox disabled, the sentinel IS readable.

    Without this, ``test_bash_home_read_denied`` would pass even if the
    sandbox silently no-op'd (binary missing, probe failed) — proving
    only that the sentinel UUID doesn't appear by chance, not that the
    sandbox is the thing blocking it.
    """
    from core.inference import tools

    monkeypatch.setattr(tools, "sandbox_available", lambda: False)
    sid = "_sbtest_off"
    monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))

    path, secret = home_sentinel
    out = _run_bash(f"cat {shlex.quote(path)}", sid)
    assert secret in out, out


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason = "/System/Applications is a macOS path",
)
def test_system_applications_enumeration_denied(sandboxed_workdir):
    """Pin the macOS narrowing: /System/Applications should not be readable.

    v1 of the macOS profile allowed all of /System; v2 narrowed it to
    Frameworks + dyld only. The Frameworks dir must remain readable
    (loading still works), while /System/Applications and
    /System/iOSSupport must not.
    """
    sid, _ = sandboxed_workdir
    out = _run_bash("ls /System/Applications 2>&1; ls /System/iOSSupport 2>&1", sid)
    assert "Operation not permitted" in out, out
    out_fw = _run_bash("ls /System/Library/Frameworks | head -1", sid)
    assert "Operation not permitted" not in out_fw, out_fw
    assert ".framework" in out_fw, out_fw


# ---------------------------------------------------------------------------
# Profile-string and routing unit tests. These do not require a usable
# sandbox primitive (and therefore do not skip via sandboxed_workdir).
# ---------------------------------------------------------------------------


def test_seatbelt_profile_omits_keychain_and_trustd():
    from core.inference import sandbox as _sb
    profile = _sb._macos_seatbelt_profile("/var/tmp/wd")
    assert "com.apple.SecurityServer" not in profile
    assert "com.apple.trustd" not in profile
    assert "com.apple.trustd.agent" not in profile


def test_seatbelt_profile_omits_broad_private_etc_subpath():
    from core.inference import sandbox as _sb
    profile = _sb._macos_seatbelt_profile("/var/tmp/wd")
    assert '(subpath "/private/etc")' not in profile
    assert '(literal "/private/etc/hosts")' in profile
    assert '(literal "/private/etc/localtime")' in profile
    assert '(subpath "/private/etc/ssl")' in profile


def test_python_read_paths_emits_original_when_realpath_differs(tmp_path, monkeypatch):
    from core.inference import sandbox as _sb
    real_dir = tmp_path / "real_prefix"
    real_dir.mkdir()
    link_dir = tmp_path / "linked_prefix"
    link_dir.symlink_to(real_dir)
    monkeypatch.setattr(_sb.sys, "prefix", str(link_dir))
    monkeypatch.setattr(_sb.sys, "base_prefix", str(link_dir))
    monkeypatch.setattr(_sb.site, "getsitepackages", lambda: [])
    monkeypatch.setattr(_sb.site, "getusersitepackages", lambda: "")
    monkeypatch.setattr(_sb, "_editable_source_paths", lambda: [])
    paths = _sb._python_read_paths()
    assert str(link_dir) in paths
    assert os.path.realpath(str(link_dir)) in paths


def test_python_read_paths_deduplicates_when_no_symlink(tmp_path, monkeypatch):
    from core.inference import sandbox as _sb
    real_dir = tmp_path / "real_only"
    real_dir.mkdir()
    monkeypatch.setattr(_sb.sys, "prefix", str(real_dir))
    monkeypatch.setattr(_sb.sys, "base_prefix", str(real_dir))
    monkeypatch.setattr(_sb.site, "getsitepackages", lambda: [])
    monkeypatch.setattr(_sb.site, "getusersitepackages", lambda: "")
    monkeypatch.setattr(_sb, "_editable_source_paths", lambda: [])
    paths = _sb._python_read_paths()
    assert paths.count(str(real_dir)) == 1


def test_python_exec_routes_to_bwrap_target_preexec_when_sandbox_on_linux(
    tmp_path, monkeypatch,
):
    from core.inference import tools as _tools

    class _FakeProc:
        returncode = 0
        def communicate(self, timeout = None):
            return ("", None)

    captured = {}
    def _fake_popen(argv, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(_tools.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(_tools, "sandbox_available", lambda: True)
    monkeypatch.setattr(_tools.sys, "platform", "linux")
    monkeypatch.setattr(
        _tools, "build_sandbox_argv", lambda inner, wd: ["bwrap-stub", *inner],
    )
    monkeypatch.setitem(_tools._workdirs, "_routing_session", str(tmp_path))

    _tools._python_exec("print('ok')", session_id = "_routing_session", timeout = 5)
    assert captured["kwargs"]["preexec_fn"] is _tools._sandbox_preexec_bwrap_target


def test_python_exec_keeps_default_preexec_when_sandbox_unavailable(
    tmp_path, monkeypatch,
):
    from core.inference import tools as _tools

    class _FakeProc:
        returncode = 0
        def communicate(self, timeout = None):
            return ("", None)

    captured = {}
    def _fake_popen(argv, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(_tools.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(_tools, "sandbox_available", lambda: False)
    monkeypatch.setattr(_tools.sys, "platform", "linux")
    monkeypatch.setitem(_tools._workdirs, "_routing_session_off", str(tmp_path))

    _tools._python_exec("print('ok')", session_id = "_routing_session_off", timeout = 5)
    assert captured["kwargs"]["preexec_fn"] is _tools._sandbox_preexec


def test_bash_exec_routes_to_bwrap_target_preexec_when_sandbox_on_linux(
    tmp_path, monkeypatch,
):
    from core.inference import tools as _tools

    class _FakeProc:
        returncode = 0
        def communicate(self, timeout = None):
            return ("", None)

    captured = {}
    def _fake_popen(argv, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(_tools.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(_tools, "sandbox_available", lambda: True)
    monkeypatch.setattr(_tools.sys, "platform", "linux")
    monkeypatch.setattr(
        _tools, "build_sandbox_argv", lambda inner, wd: ["bwrap-stub", *inner],
    )
    monkeypatch.setitem(_tools._workdirs, "_routing_bash", str(tmp_path))

    _tools._bash_exec("echo hi", session_id = "_routing_bash", timeout = 5)
    assert captured["kwargs"]["preexec_fn"] is _tools._sandbox_preexec_bwrap_target


def test_bwrap_target_preexec_skips_no_new_privs_but_keeps_pdeathsig(monkeypatch):
    """bwrap-target preexec must skip prctl(PR_SET_NO_NEW_PRIVS=38) so that
    setuid-bwrap kernels can still escalate at execve; PDEATHSIG (option 1)
    must still be applied."""
    from unittest.mock import MagicMock
    from core.inference import tools as _tools

    calls = []
    fake_libc = MagicMock()
    fake_libc.prctl.side_effect = lambda *args, **kw: calls.append(args) or 0

    monkeypatch.setattr(_tools, "_libc", fake_libc)
    monkeypatch.setattr(_tools.os, "setsid", lambda: None)
    monkeypatch.setattr(_tools.os, "umask", lambda mode: 0)
    monkeypatch.setattr(_tools, "_resource", None)

    _tools._sandbox_preexec_bwrap_target()
    options = [c[0] for c in calls]
    assert 38 not in options
    assert 1 in options


def test_default_preexec_still_sets_no_new_privs(monkeypatch):
    """Control: the original _sandbox_preexec must still set NO_NEW_PRIVS for
    the unsandboxed fallback path."""
    from unittest.mock import MagicMock
    from core.inference import tools as _tools

    calls = []
    fake_libc = MagicMock()
    fake_libc.prctl.side_effect = lambda *args, **kw: calls.append(args) or 0

    monkeypatch.setattr(_tools, "_libc", fake_libc)
    monkeypatch.setattr(_tools.os, "setsid", lambda: None)
    monkeypatch.setattr(_tools.os, "umask", lambda mode: 0)
    monkeypatch.setattr(_tools, "_resource", None)

    _tools._sandbox_preexec()
    options = [c[0] for c in calls]
    assert 38 in options
    assert 1 in options
