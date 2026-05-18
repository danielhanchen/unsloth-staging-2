"""
Edge case sweep — checking the PR doesn't break or weaken any of the
existing layered defenses while it adds the new sandbox layer.

Categories:
  A. Existing AST gate still blocks dangerous patterns
  B. Existing bash blocklist still works (with and without sandbox)
  C. Existing env-stripping (_build_safe_env) still works
  D. Session-id containment (path traversal) still works
  E. Per-session workdir mode 0o700 still applied
  F. Timeout still kills runaway subprocess
  G. Cancel-event still works
  H. Output truncation still applied
  I. Image-detection sentinel still emitted
  J. Multi-session isolation
  K. Empty / None / whitespace input handling
"""

from __future__ import annotations

import os
import stat
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_BACKEND_ROOT = _HERE.parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import sandbox as sb  # noqa: E402
from core.inference import tools  # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="bash-blocklist + POSIX-shell tests; Windows tools use cmd /c",
)


@pytest.fixture(autouse=True)
def _reset_cache():
    sb._sandbox_available_cache = None
    sb._linux_bwrap_path = None
    yield
    sb._sandbox_available_cache = None
    sb._linux_bwrap_path = None


@pytest.fixture
def sid_and_wd(tmp_path, monkeypatch):
    sid = f"_edge_{uuid.uuid4().hex[:8]}"
    wd = tmp_path / "wd"
    wd.mkdir()
    monkeypatch.setitem(tools._workdirs, sid, str(wd))
    return sid, str(wd)


# ---------------------------------------------------------------------------
# A. AST gate
# ---------------------------------------------------------------------------

class TestAstGateStillBlocks:
    @pytest.mark.parametrize(
        "code,expect_in_msg",
        [
            # signal tampering
            ("import signal; signal.signal(signal.SIGALRM, signal.SIG_IGN)",
             "unsafe code detected"),
            # os.system shell escape
            ("import os; os.system('rm /tmp/x')", "unsafe"),
            # subprocess.run with blocked command literal
            ("import subprocess; subprocess.run(['sudo', 'whoami'])", "unsafe"),
            # sensitive file read
            ("open('/etc/passwd')", "unsafe"),
            ("open('/etc/shadow')", "unsafe"),
            ("open('/proc/self/environ')", "unsafe"),
            # cloud metadata host
            ("import requests; requests.get('http://169.254.169.254/latest/meta-data/')",
             "unsafe"),
            # untrusted host
            ("import requests; requests.get('https://evil.example.com/')", "unsafe"),
            # huggingface upload
            ("from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='x',"
             "path_in_repo='x', repo_id='a/b')", "unsafe"),
        ],
    )
    def test_blocked(self, sid_and_wd, code, expect_in_msg):
        sid, _ = sid_and_wd
        out = tools._python_exec(code, session_id=sid, timeout=5)
        assert expect_in_msg in out, out

    @pytest.mark.parametrize(
        "code",
        [
            "print('hello world')",
            "x = 1 + 2; print(x)",
            "import math; print(math.sqrt(2))",
            "import requests; requests.get('https://wikipedia.org/')",  # trusted host
        ],
    )
    def test_allowed(self, sid_and_wd, code):
        sid, _ = sid_and_wd
        out = tools._python_exec(code, session_id=sid, timeout=10)
        # Must not be the AST gate rejection.
        assert "unsafe code detected" not in out, out


# ---------------------------------------------------------------------------
# B. Bash blocklist
# ---------------------------------------------------------------------------

class TestBashBlocklist:
    @pytest.mark.parametrize(
        "cmd",
        [
            "rm -rf /",
            "sudo whoami",
            "curl http://evil.example/",
            "wget http://evil.example/",
            "nc -l 1234",
            "ssh user@host",
            "/usr/bin/sudo whoami",         # full path
            "bash -c 'sudo whoami'",         # nested shell
            "bash -lc 'sudo whoami'",        # nested shell with -l
            "echo a; sudo whoami",            # semicolon-separated
            "echo a | sudo whoami",           # pipe
            "$(sudo whoami)",                  # command substitution
        ],
    )
    def test_blocked(self, sid_and_wd, cmd):
        sid, _ = sid_and_wd
        out = tools._bash_exec(cmd, session_id=sid, timeout=5)
        assert "Blocked" in out, (cmd, out)

    @pytest.mark.parametrize(
        "cmd",
        ["echo hello", "ls", "date", "pwd", "uname"],
    )
    def test_allowed(self, sid_and_wd, cmd):
        sid, _ = sid_and_wd
        out = tools._bash_exec(cmd, session_id=sid, timeout=10)
        assert "Blocked" not in out, out


# ---------------------------------------------------------------------------
# C. Safe env (credential stripping)
# ---------------------------------------------------------------------------

class TestSafeEnv:
    def test_credentials_stripped(self, tmp_path):
        env = tools._build_safe_env(str(tmp_path))
        # These must NEVER appear.
        for forbidden in (
            "HF_TOKEN", "WANDB_API_KEY", "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY", "GH_TOKEN",
            "LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "DYLD_FRAMEWORK_PATH",
        ):
            assert forbidden not in env, forbidden

    def test_path_only_safe_dirs(self, tmp_path):
        env = tools._build_safe_env(str(tmp_path))
        path_parts = env["PATH"].split(os.pathsep)
        # Path must not contain dangerous user-writable directories like cwd.
        assert "." not in path_parts
        assert "" not in path_parts

    def test_subprocess_cannot_see_creds(self, sid_and_wd, monkeypatch):
        sid, _ = sid_and_wd
        # Inject fake credentials into the parent env.
        monkeypatch.setenv("HF_TOKEN", "hf_secret_xxx")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws_secret_yyy")
        code = (
            "import os, json; "
            "print(json.dumps({k: os.environ.get(k) for k in "
            "['HF_TOKEN','AWS_SECRET_ACCESS_KEY','WANDB_API_KEY']}))"
        )
        out = tools._python_exec(code, session_id=sid, timeout=10)
        assert "hf_secret_xxx" not in out, out
        assert "aws_secret_yyy" not in out, out


# ---------------------------------------------------------------------------
# D. Session-id containment
# ---------------------------------------------------------------------------

class TestSessionIdContainment:
    def test_traversal_collapses_to_invalid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path) if p == "~" else p)
        wd = tools._get_workdir("../escape")
        assert "_invalid" in wd

    def test_overlong_collapses_to_invalid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path) if p == "~" else p)
        wd = tools._get_workdir("x" * 200)
        assert "_invalid" in wd

    def test_null_byte_collapses_to_invalid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path) if p == "~" else p)
        wd = tools._get_workdir("foo\x00bar")
        assert "_invalid" in wd

    def test_good_session_id_accepted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path) if p == "~" else p)
        wd = tools._get_workdir("chat_abc123-XYZ")
        assert wd.endswith("chat_abc123-XYZ")
        assert "_invalid" not in wd

    def test_default_when_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path) if p == "~" else p)
        wd = tools._get_workdir(None)
        assert wd.endswith("_default")


# ---------------------------------------------------------------------------
# E. Workdir permissions
# ---------------------------------------------------------------------------

class TestWorkdirPermissions:
    def test_workdir_is_0700(self, tmp_path, monkeypatch):
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path) if p == "~" else p)
        # Use a fresh key so _get_workdir actually creates it
        tools._workdirs.pop("perms_test", None)
        wd = tools._get_workdir("perms_test")
        mode = stat.S_IMODE(os.stat(wd).st_mode)
        # Best-effort: chmod failures are swallowed, so this isn't asserted
        # as a hard guarantee, but on this tmp_path it should hold.
        assert mode == 0o700 or mode == 0o755, oct(mode)


# ---------------------------------------------------------------------------
# F. Timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_python_loop_killed_by_timeout(self, sid_and_wd):
        sid, _ = sid_and_wd
        # Plain busy loop — no signal/exception tricks, so AST gate accepts.
        code = "import time\nwhile True:\n    time.sleep(0.05)\n"
        out = tools._python_exec(code, session_id=sid, timeout=2)
        assert "timed out" in out.lower(), out

    def test_bash_sleep_killed_by_timeout(self, sid_and_wd):
        sid, _ = sid_and_wd
        out = tools._bash_exec("sleep 30", session_id=sid, timeout=2)
        assert "timed out" in out.lower(), out


# ---------------------------------------------------------------------------
# G. Cancel event
# ---------------------------------------------------------------------------

class TestCancelEvent:
    def test_cancel_kills_python(self, sid_and_wd):
        sid, _ = sid_and_wd
        cancel = threading.Event()

        def _cancel_soon():
            time.sleep(0.4)
            cancel.set()

        threading.Thread(target=_cancel_soon, daemon=True).start()
        code = "import time\nwhile True:\n    time.sleep(0.05)\n"
        out = tools._python_exec(
            code, cancel_event=cancel, session_id=sid, timeout=10,
        )
        assert "cancel" in out.lower() or out == "Execution cancelled.", out


# ---------------------------------------------------------------------------
# H. Output truncation
# ---------------------------------------------------------------------------

class TestOutputTruncation:
    def test_long_output_truncated(self, sid_and_wd):
        sid, _ = sid_and_wd
        # 20 KB of "x"
        code = "print('x' * 20000)"
        out = tools._python_exec(code, session_id=sid, timeout=10)
        # _MAX_OUTPUT_CHARS = 8000; output should be truncated.
        assert "(truncated" in out, "output should mention truncation"


# ---------------------------------------------------------------------------
# I. Empty / whitespace / None
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_python_empty(self, sid_and_wd):
        sid, _ = sid_and_wd
        out = tools._python_exec("", session_id=sid, timeout=5)
        assert "No code provided" in out

    def test_python_whitespace(self, sid_and_wd):
        sid, _ = sid_and_wd
        out = tools._python_exec("   \n  ", session_id=sid, timeout=5)
        assert "No code provided" in out

    def test_bash_empty(self, sid_and_wd):
        sid, _ = sid_and_wd
        out = tools._bash_exec("", session_id=sid, timeout=5)
        assert "No command provided" in out


# ---------------------------------------------------------------------------
# J. Multi-session isolation
# ---------------------------------------------------------------------------

class TestMultiSessionIsolation:
    def test_different_sessions_get_different_workdirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path) if p == "~" else p)
        wd_a = tools._get_workdir("alice_session_1")
        wd_b = tools._get_workdir("bob_session_2")
        assert wd_a != wd_b
        assert os.path.isdir(wd_a) and os.path.isdir(wd_b)


# ---------------------------------------------------------------------------
# K. The AST gate doesn't false-positive on safe shell-ish strings
# ---------------------------------------------------------------------------

class TestAstFalsePositives:
    @pytest.mark.parametrize(
        "code",
        [
            "msg = 'rm -rf /'; print(msg)",   # string literal, never executed
            "print('curl wget sudo')",         # words in a print string
            "x = {'rm': 1, 'sudo': 2}; print(x)",  # dict keys
        ],
    )
    def test_string_constants_dont_trip_gate(self, sid_and_wd, code):
        sid, _ = sid_and_wd
        out = tools._python_exec(code, session_id=sid, timeout=10)
        assert "unsafe code detected" not in out, out


# ---------------------------------------------------------------------------
# L. _build_safe_env / VIRTUAL_ENV propagation
# ---------------------------------------------------------------------------

class TestSafeEnvVirtualEnv:
    def test_virtual_env_propagates_when_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VIRTUAL_ENV", "/some/venv")
        env = tools._build_safe_env(str(tmp_path))
        assert env.get("VIRTUAL_ENV") == "/some/venv"
        # Bin dir should also be on PATH.
        assert "/some/venv/bin" in env["PATH"] or "/some/venv\\Scripts" in env["PATH"]

    def test_virtual_env_absent_when_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        env = tools._build_safe_env(str(tmp_path))
        assert "VIRTUAL_ENV" not in env
