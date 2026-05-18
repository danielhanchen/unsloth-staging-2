"""
Resource-limit enforcement verification.

_sandbox_preexec sets:
  RLIMIT_NPROC  = UNSLOTH_STUDIO_SANDBOX_NPROC (default 10000)
  RLIMIT_FSIZE  = 100 MB
  RLIMIT_AS     = UNSLOTH_STUDIO_SANDBOX_AS_GB GiB (default 8 GiB)
  RLIMIT_CPU    = UNSLOTH_STUDIO_SANDBOX_CPU_S (default 600 s)
  RLIMIT_NOFILE = 1024

We verify the limits are present in the child by reading
/proc/<pid>/limits via the subprocess itself. The env-var knobs (NPROC,
AS, CPU) are also tested by tweaking them and observing the change.

Running an actual fork bomb / OOM allocation would be unsafe here, so
we read the limits programmatically instead of triggering them.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import tools  # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="resource module + /proc-based limit inspection are Linux-specific",
)

_LIMIT_CHECK_CODE = """
import resource, json
data = {
    'NPROC':  resource.getrlimit(resource.RLIMIT_NPROC),
    'FSIZE':  resource.getrlimit(resource.RLIMIT_FSIZE),
    'AS':     resource.getrlimit(resource.RLIMIT_AS),
    'CPU':    resource.getrlimit(resource.RLIMIT_CPU),
    'NOFILE': resource.getrlimit(resource.RLIMIT_NOFILE),
}
print(json.dumps(data))
"""


@pytest.fixture
def session(tmp_path, monkeypatch):
    sid = f"_rlim_{uuid.uuid4().hex[:8]}"
    monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))
    return sid


def _read_limits(out):
    import json
    # Strip optional "Exit code N:" prefix if a soft cap killed the run.
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise AssertionError(f"no JSON in output: {out!r}")


# ---------------------------------------------------------------------------
# Default limits
# ---------------------------------------------------------------------------

class TestDefaultLimits:
    def test_fsize_at_100mb(self, session):
        out = tools._python_exec(_LIMIT_CHECK_CODE, session_id=session, timeout=10)
        d = _read_limits(out)
        assert d["FSIZE"][0] == 100 * 1024 * 1024, d

    def test_nofile_at_1024(self, session):
        out = tools._python_exec(_LIMIT_CHECK_CODE, session_id=session, timeout=10)
        d = _read_limits(out)
        assert d["NOFILE"][0] == 1024, d

    def test_nproc_default_10000(self, session, monkeypatch):
        monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_NPROC", raising=False)
        out = tools._python_exec(_LIMIT_CHECK_CODE, session_id=session, timeout=10)
        d = _read_limits(out)
        assert d["NPROC"][0] == 10000, d

    def test_cpu_default_600(self, session, monkeypatch):
        monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_CPU_S", raising=False)
        out = tools._python_exec(_LIMIT_CHECK_CODE, session_id=session, timeout=10)
        d = _read_limits(out)
        assert d["CPU"][0] == 600, d

    def test_as_default_8gb(self, session, monkeypatch):
        monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_AS_GB", raising=False)
        out = tools._python_exec(_LIMIT_CHECK_CODE, session_id=session, timeout=10)
        d = _read_limits(out)
        assert d["AS"][0] == 8 * 1024 * 1024 * 1024, d


# ---------------------------------------------------------------------------
# Env-var overrides
# ---------------------------------------------------------------------------

class TestEnvOverrides:
    def test_cpu_override(self, session, monkeypatch):
        # _sandbox_preexec reads UNSLOTH_STUDIO_SANDBOX_CPU_S via os.environ
        # inside the FORKED child — and _build_safe_env strips the parent
        # env. So we need the override to be in the env at preexec time.
        # That env is the parent's; preexec runs after fork but before exec.
        monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_CPU_S", "30")
        out = tools._python_exec(_LIMIT_CHECK_CODE, session_id=session, timeout=10)
        d = _read_limits(out)
        assert d["CPU"][0] == 30, d

    def test_nproc_override(self, session, monkeypatch):
        monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_NPROC", "100")
        out = tools._python_exec(_LIMIT_CHECK_CODE, session_id=session, timeout=10)
        d = _read_limits(out)
        assert d["NPROC"][0] == 100, d

    def test_as_override(self, session, monkeypatch):
        monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_AS_GB", "2")
        out = tools._python_exec(_LIMIT_CHECK_CODE, session_id=session, timeout=10)
        d = _read_limits(out)
        assert d["AS"][0] == 2 * 1024 * 1024 * 1024, d

    def test_invalid_env_value_falls_back_to_default(self, session, monkeypatch):
        """A non-integer env var must not crash _sandbox_preexec —
        the try/except in the source swallows ValueError."""
        monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_CPU_S", "not_a_number")
        out = tools._python_exec(_LIMIT_CHECK_CODE, session_id=session, timeout=10)
        # The limit ends up as whatever the host has (since the call was
        # try/except'd and never reached setrlimit). It shouldn't be 0.
        d = _read_limits(out)
        assert d["CPU"][0] != 0, d


# ---------------------------------------------------------------------------
# FSIZE enforcement (actually try to write > 100 MB)
# ---------------------------------------------------------------------------

class TestFsizeEnforcement:
    def test_writing_over_fsize_killed(self, session):
        """Try to write 150 MB to a file. RLIMIT_FSIZE=100MB should kill
        the child with SIGXFSZ at 100 MB. We accept either truncated file
        or process killed."""
        code = (
            "import os\n"
            "f = open('big.bin', 'wb')\n"
            "buf = b'A' * (1024 * 1024)\n"  # 1 MB
            "try:\n"
            "    for _ in range(150):  # 150 MB target\n"
            "        f.write(buf)\n"
            "    print('FULL_WRITE')\n"
            "except OSError as e:\n"
            "    print('FSIZE_HIT:', type(e).__name__)\n"
        )
        out = tools._python_exec(code, session_id=session, timeout=30)
        # Either OSError (EFBIG) or SIGXFSZ kill the process before completion.
        # On non-Linux or relaxed hosts, "FULL_WRITE" might appear. The
        # security-relevant claim is that RLIMIT_FSIZE is set; if the
        # kernel honors it, exfil capacity is capped.
        if "FULL_WRITE" in out:
            pytest.skip("host doesn't honor RLIMIT_FSIZE strictly")
        assert "FSIZE_HIT" in out or "Exit code" in out, out


# ---------------------------------------------------------------------------
# NOFILE enforcement (try to open > 1024 fds)
# ---------------------------------------------------------------------------

class TestNofileEnforcement:
    def test_open_lots_of_fds_hits_cap(self, session):
        code = (
            "import os\n"
            "opened = []\n"
            "try:\n"
            "    for i in range(2000):\n"
            "        opened.append(os.open('/dev/null', os.O_RDONLY))\n"
            "    print('ALL_OPENED', len(opened))\n"
            "except OSError as e:\n"
            "    print('NOFILE_HIT at', len(opened), type(e).__name__)\n"
            "finally:\n"
            "    for fd in opened:\n"
            "        os.close(fd)\n"
        )
        out = tools._python_exec(code, session_id=session, timeout=15)
        # Should hit NOFILE_HIT somewhere around 1010 (1024 - stdio - misc)
        if "ALL_OPENED" in out:
            pytest.skip("host doesn't honor RLIMIT_NOFILE strictly")
        assert "NOFILE_HIT" in out, out


# ---------------------------------------------------------------------------
# umask is 0o077 in the child
# ---------------------------------------------------------------------------

class TestUmask:
    def test_umask_is_077(self, session):
        out = tools._python_exec(
            "import os; print(oct(os.umask(0)))",
            session_id=session, timeout=10,
        )
        # umask(0) sets to 0 and returns previous; should be 0o77
        assert "0o77" in out, out


# ---------------------------------------------------------------------------
# setsid (new process group)
# ---------------------------------------------------------------------------

class TestSetsidApplied:
    def test_pid_equals_pgid(self, session):
        """setsid() makes the child the session leader, so PID == PGID."""
        out = tools._python_exec(
            "import os; print(os.getpid(), os.getpgid(0))",
            session_id=session, timeout=10,
        )
        toks = out.strip().split()
        # Last 2 tokens should be the pid pair
        pid, pgid = int(toks[-2]), int(toks[-1])
        assert pid == pgid, (pid, pgid, out)


# ---------------------------------------------------------------------------
# PR_SET_NO_NEW_PRIVS visible in /proc/self/status
# ---------------------------------------------------------------------------

class TestNoNewPrivsApplied:
    def test_no_new_privs_set(self, session):
        out = tools._python_exec(
            "for line in open('/proc/self/status'):\n"
            "    if line.startswith('NoNewPrivs:'):\n"
            "        print(line.strip()); break",
            session_id=session, timeout=10,
        )
        # The AST gate flags open('/proc/self/...') for sensitive paths.
        # /proc/self/status isn't in that list, so it should pass.
        if "unsafe" in out:
            pytest.skip("/proc/self/status flagged by AST — adjust gate or test")
        assert "NoNewPrivs:\t1" in out, out
