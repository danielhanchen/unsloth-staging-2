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
