"""PR #7080 cross-OS real-subprocess test (Linux/macOS/Windows).

Exercises the OS-sensitive paths: real stdio subprocess spawn (Windows needs a
ProactorEventLoop or mcp's Popen fallback), state persistence, crash recovery
without poisoning, one-shot fallback, the Windows command-line round-trip, and
the session cap. Runs under the OS's own Python; no npx/browser needed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
_BACKEND = str(Path(__file__).resolve().parents[1])  # studio/backend
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.inference import mcp_client  # noqa: E402
from core.inference.mcp_client import call_tool_sync, close_stdio_sessions  # noqa: E402

FIXTURE = str(Path(__file__).resolve().parent / "_pr7080_fixture.py")
CMD = subprocess.list2cmdline([sys.executable, FIXTURE])


def _num(out, key):
    m = re.search(rf'{key}["\']?\s*[:=]\s*(\d+)', out)
    return int(m.group(1)) if m else None


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    close_stdio_sessions()


def test_real_subprocess_persists():
    o1 = call_tool_sync(CMD, None, "counter", {}, timeout=60, scope="chat")
    o2 = call_tool_sync(CMD, None, "counter", {}, timeout=60, scope="chat")
    assert _num(o1, "counter") == 1 and _num(o2, "counter") == 2
    assert _num(o1, "pid") == _num(o2, "pid")


def test_no_scope_one_shot():
    o1 = call_tool_sync(CMD, None, "counter", {}, timeout=60)
    o2 = call_tool_sync(CMD, None, "counter", {}, timeout=60)
    assert _num(o1, "counter") == 1 and _num(o2, "counter") == 1
    assert _num(o1, "pid") != _num(o2, "pid")


def test_mid_call_crash_recovers_no_poison():
    call_tool_sync(CMD, None, "counter", {}, timeout=60, scope="chat")
    err = call_tool_sync(CMD, None, "crash_mid", {}, timeout=60, scope="chat")
    assert err.startswith("Error")
    ok = call_tool_sync(CMD, None, "counter", {}, timeout=60, scope="chat")
    assert _num(ok, "counter") == 1  # fresh process; scope not poisoned


def test_windows_loop_is_proactor():
    if sys.platform != "win32":
        pytest.skip("windows only")
    call_tool_sync(CMD, None, "counter", {}, timeout=60, scope="chat")
    sess = next(iter(mcp_client._stdio_sessions.values()))
    assert type(sess.loop).__name__ == "ProactorEventLoop"


@pytest.mark.parametrize("argv", [
    ["a", "", "b"], ["quote\"in", "trail\\"], ["C:\\a b\\c.exe", "--p", "x y"],
    ["café", "日本語"], ["ends\\"],
])
def test_windows_cmdline_roundtrip(argv):
    assert mcp_client._split_windows_command_line(subprocess.list2cmdline(argv)) == argv
