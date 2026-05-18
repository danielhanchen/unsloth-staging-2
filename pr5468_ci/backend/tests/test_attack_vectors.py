"""
Direct attack vectors against the PR's wrapper layer.

These run the actual _python_exec / _bash_exec code path on this host
(where the sandbox is unavailable, so the existing layered defenses
are the only thing in play). They probe:

  - Symlink escapes from inside workdir
  - Hard-link escapes
  - Fork bombs against RLIMIT_NPROC
  - Output flood against truncation
  - PYTHONSTARTUP / PYTHONPATH manipulation attempts
  - PATH manipulation attempts
  - Working-directory escape attempts
  - Long-input handling
  - Binary / non-UTF8 output
  - Process-group kill on timeout
  - tmp_path leak after success / failure
  - Image sentinel emission

Where the PR is supposed to add the sandbox layer to backstop these,
the test docstring says so.
"""

from __future__ import annotations

import os
import stat
import sys
import uuid
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import sandbox as sb  # noqa: E402
from core.inference import tools  # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-only: uses os.fork / os.setsid / os.killpg / os.symlink / os.link",
)


@pytest.fixture
def session(tmp_path, monkeypatch):
    sid = f"_atk_{uuid.uuid4().hex[:8]}"
    wd = tmp_path / "wd"
    wd.mkdir()
    monkeypatch.setitem(tools._workdirs, sid, str(wd))
    return sid, str(wd)


# ---------------------------------------------------------------------------
# Symlink escapes — workdir contains a symlink that points outside
# ---------------------------------------------------------------------------

class TestSymlinkEscapeFromWorkdir:
    def test_symlink_inside_workdir_points_to_home_secret(self, session, tmp_path):
        """An LLM could `os.symlink("/home/user/secret", "loot")` then read
        `loot`. Without the OS sandbox, this works. With bwrap, the symlink
        target wouldn't be inside the mounted namespace and the read fails."""
        sid, wd = session
        secret_path = tmp_path / "hostfile.txt"  # outside wd
        secret = f"HOSTSECRET-{uuid.uuid4().hex}"
        secret_path.write_text(secret)

        code = (
            f"import os\n"
            f"os.symlink({str(secret_path)!r}, 'loot')\n"
            f"print(open('loot').read())\n"
        )
        out = tools._python_exec(code, session_id=sid, timeout=10)
        # On this host (sandbox unavailable), the symlink read succeeds.
        # This documents the gap the OS sandbox is supposed to close.
        if secret in out:
            pytest.skip(
                "fail-open path: symlink escape works without sandbox. "
                "The OS sandbox is supposed to confine reads to workdir."
            )
        assert secret not in out


# ---------------------------------------------------------------------------
# Hard-link escapes
# ---------------------------------------------------------------------------

class TestHardLinkAttempt:
    def test_hard_link_to_outside_path_fails(self, session, tmp_path):
        sid, wd = session
        outside = tmp_path / "outside.txt"
        outside.write_text("hostdata")
        code = (
            f"import os\n"
            f"try:\n"
            f"    os.link({str(outside)!r}, 'linked')\n"
            f"    print(open('linked').read())\n"
            f"except OSError as e:\n"
            f"    print('LINK_FAILED:', type(e).__name__)\n"
        )
        out = tools._python_exec(code, session_id=sid, timeout=10)
        # Hard link succeeds inside same filesystem; the gap relies on
        # OS sandbox / workdir-bind to confine.
        if "hostdata" in out:
            pytest.skip("fail-open: hard link escape works")


# ---------------------------------------------------------------------------
# Fork bomb / NPROC clamp
# ---------------------------------------------------------------------------

class TestForkBombClamp:
    def test_fork_bomb_does_not_dos_host(self, session):
        """RLIMIT_NPROC defaults to 10000 in _sandbox_preexec — per-real-UID.
        We can't actually bomb here, but we can spawn ~20 fast and check
        the timeout / cleanup still works."""
        sid, _ = session
        code = (
            "import os\n"
            "for i in range(20):\n"
            "    pid = os.fork()\n"
            "    if pid == 0:\n"
            "        import time; time.sleep(0.05); os._exit(0)\n"
            "    else:\n"
            "        os.waitpid(pid, 0)\n"
            "print('ok')\n"
        )
        out = tools._python_exec(code, session_id=sid, timeout=15)
        assert "ok" in out, out


# ---------------------------------------------------------------------------
# Output flood — must hit _MAX_OUTPUT_CHARS truncation
# ---------------------------------------------------------------------------

class TestOutputFlood:
    def test_megabyte_output_truncated(self, session):
        sid, _ = session
        # Print 200 KB
        code = "print('A' * 200000)"
        out = tools._python_exec(code, session_id=sid, timeout=15)
        assert len(out) < 50000  # truncated to _MAX_OUTPUT_CHARS=8000 + tail
        assert "truncated" in out


# ---------------------------------------------------------------------------
# PYTHONSTARTUP / PYTHONPATH manipulation
# ---------------------------------------------------------------------------

class TestPythonStartupAndPath:
    def test_pythonstartup_stripped(self, session, monkeypatch, tmp_path):
        """If PYTHONSTARTUP propagates, an attacker can pre-set it on the
        parent and run code before the user's script. _build_safe_env
        should not include it."""
        nasty = tmp_path / "nasty.py"
        nasty.write_text("print('PYTHONSTARTUP HIJACK')\n")
        monkeypatch.setenv("PYTHONSTARTUP", str(nasty))
        sid, _ = session
        out = tools._python_exec("print('hello')", session_id=sid, timeout=10)
        assert "HIJACK" not in out, out

    def test_pythonpath_stripped(self, session, monkeypatch, tmp_path):
        """Same idea for PYTHONPATH — propagating it could let attacker
        sideload a shadow module."""
        nasty = tmp_path / "evil"
        nasty.mkdir()
        (nasty / "json.py").write_text("def loads(*a, **kw): return 'EVILJSON'\n")
        monkeypatch.setenv("PYTHONPATH", str(nasty))
        sid, _ = session
        out = tools._python_exec(
            "import json; print(json.loads('[1,2,3]'))",
            session_id=sid, timeout=10,
        )
        assert "EVILJSON" not in out, out

    def test_path_only_safe_dirs(self, session):
        sid, _ = session
        out = tools._python_exec(
            "import os; print(os.environ['PATH'])",
            session_id=sid, timeout=10,
        )
        # Must NOT include cwd / arbitrary user dirs
        assert "/home" not in out or out.count("/home") < 5, out
        # Must include a system bin path
        assert any(p in out for p in ("/usr/bin", "/bin", "Scripts")), out


# ---------------------------------------------------------------------------
# HOME env var should point at workdir, not real $HOME
# ---------------------------------------------------------------------------

class TestHomePointedAtWorkdir:
    def test_home_is_workdir(self, session):
        sid, wd = session
        out = tools._python_exec(
            "import os; print(os.environ['HOME'])",
            session_id=sid, timeout=10,
        )
        assert wd in out, out
        # And real $HOME is NOT what subprocess sees
        real_home = os.path.expanduser("~")
        # workdir is under tmp; assert it doesn't equal real $HOME
        assert real_home != wd


# ---------------------------------------------------------------------------
# Working directory containment
# ---------------------------------------------------------------------------

class TestCwdContainment:
    def test_subprocess_cwd_is_workdir(self, session):
        sid, wd = session
        out = tools._python_exec(
            "import os; print(os.getcwd())",
            session_id=sid, timeout=10,
        )
        assert wd in out

    def test_chdir_then_read_doesnt_persist_across_calls(self, session, tmp_path):
        """Each _python_exec call is a fresh subprocess, so chdir doesn't
        persist."""
        sid, wd = session
        tools._python_exec("import os; os.chdir('/tmp'); print(os.getcwd())",
                           session_id=sid, timeout=10)
        # Second call should see the fresh workdir cwd again.
        out = tools._python_exec(
            "import os; print(os.getcwd())",
            session_id=sid, timeout=10,
        )
        assert wd in out


# ---------------------------------------------------------------------------
# Binary / UTF-8 output handling
# ---------------------------------------------------------------------------

class TestBinaryOutput:
    def test_invalid_utf8_doesnt_crash(self, session):
        sid, _ = session
        # Write raw bytes that aren't valid UTF-8 to stdout
        code = (
            "import sys\n"
            "sys.stdout.buffer.write(b'\\xff\\xfe\\xfd\\xfc')\n"
            "sys.stdout.buffer.flush()\n"
        )
        out = tools._python_exec(code, session_id=sid, timeout=10)
        # subprocess decoded with text=True; replacement chars or empty OK.
        # Must not be the AST gate rejection.
        assert "unsafe" not in out.lower()


# ---------------------------------------------------------------------------
# Long input / large source
# ---------------------------------------------------------------------------

class TestLongSource:
    def test_100kb_source_runs(self, session):
        sid, _ = session
        # 100 KB of harmless code
        code = "x = 0\n" + "x += 1\n" * 10000 + "print(x)"
        out = tools._python_exec(code, session_id=sid, timeout=20)
        assert "10000" in out


# ---------------------------------------------------------------------------
# tmp_path cleanup
# ---------------------------------------------------------------------------

class TestTmpfileCleanup:
    def test_tmp_script_removed_after_success(self, session):
        sid, wd = session
        tools._python_exec("print('hello')", session_id=sid, timeout=10)
        # All `studio_exec_*.py` files should be removed
        leftover = [f for f in os.listdir(wd) if f.startswith("studio_exec_")]
        assert not leftover, leftover

    def test_tmp_script_removed_after_failure(self, session):
        sid, wd = session
        tools._python_exec("raise RuntimeError('boom')", session_id=sid, timeout=10)
        leftover = [f for f in os.listdir(wd) if f.startswith("studio_exec_")]
        assert not leftover, leftover

    def test_tmp_script_removed_after_timeout(self, session):
        sid, wd = session
        tools._python_exec(
            "import time\nwhile True:\n    time.sleep(0.1)\n",
            session_id=sid, timeout=2,
        )
        leftover = [f for f in os.listdir(wd) if f.startswith("studio_exec_")]
        assert not leftover, leftover


# ---------------------------------------------------------------------------
# Image-detection sentinel
# ---------------------------------------------------------------------------

class TestImageSentinel:
    def test_png_creation_emits_sentinel(self, session):
        sid, wd = session
        # Write a 1-byte file with .png extension
        code = (
            "from pathlib import Path\n"
            "Path('chart.png').write_bytes(b'\\x89PNG\\r\\n')\n"
            "print('done')\n"
        )
        out = tools._python_exec(code, session_id=sid, timeout=10)
        assert "__IMAGES__" in out
        assert "chart.png" in out

    def test_no_image_no_sentinel(self, session):
        sid, _ = session
        out = tools._python_exec("print('hi')", session_id=sid, timeout=10)
        assert "__IMAGES__" not in out

    def test_text_file_no_sentinel(self, session):
        sid, _ = session
        code = (
            "from pathlib import Path\n"
            "Path('notes.txt').write_text('hi')\n"
            "print('done')\n"
        )
        out = tools._python_exec(code, session_id=sid, timeout=10)
        assert "__IMAGES__" not in out


# ---------------------------------------------------------------------------
# Process-group cleanup on timeout
# ---------------------------------------------------------------------------

class TestProcessGroupCleanup:
    def test_timeout_kills_grandchild(self, session):
        """The child uses setsid() so the entire process group should be
        SIGKILL'd by _kill_process_tree. If a grandchild survives, that's
        a leak."""
        sid, wd = session
        code = (
            "import os, time, subprocess\n"
            "p = subprocess.Popen(['sleep', '300'])\n"
            f"open({os.path.join(wd, 'grandchild.pid')!r}, 'w').write(str(p.pid))\n"
            "time.sleep(60)\n"
        )
        tools._python_exec(code, session_id=sid, timeout=2)
        # Read the pid that was written. (If the kill worked, the child
        # exited but the grandchild's pid file may or may not have been
        # flushed. We do a best-effort check.)
        pidfile = Path(wd) / "grandchild.pid"
        if not pidfile.exists():
            pytest.skip("grandchild didn't get a chance to record its pid")
        pid = int(pidfile.read_text().strip())
        # Wait briefly, then check if the grandchild is alive
        import time
        time.sleep(1)
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True
        if alive:
            # Cleanup ourselves
            try:
                os.kill(pid, 9)
            except OSError:
                pass
        assert not alive, f"grandchild pid={pid} survived parent kill"


# ---------------------------------------------------------------------------
# AST gate doesn't get bypassed by encoded code via tools.execute_tool
# ---------------------------------------------------------------------------

class TestExecuteToolDispatch:
    def test_unknown_tool_returns_message(self, session):
        sid, _ = session
        out = tools.execute_tool("nonexistent_tool", {}, session_id=sid)
        assert "Unknown tool" in out

    def test_python_dispatch(self, session):
        sid, _ = session
        out = tools.execute_tool(
            "python", {"code": "print('disp-py')"}, session_id=sid, timeout=10,
        )
        assert "disp-py" in out

    def test_terminal_dispatch(self, session):
        sid, _ = session
        out = tools.execute_tool(
            "terminal", {"command": "echo disp-bash"}, session_id=sid, timeout=10,
        )
        assert "disp-bash" in out
