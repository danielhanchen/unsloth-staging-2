"""
Reproduce the P1 finding: on hosts using setuid bwrap (because
unprivileged user namespaces are disabled), the runtime applies
PR_SET_NO_NEW_PRIVS via _sandbox_preexec, which blocks the setuid
transition setuid bwrap needs. The probe path doesn't apply preexec,
so the probe succeeds while every real tool call dies.

We can't depend on real setuid bwrap. Instead we build a fake "bwrap"
that:
  - succeeds (rc=0) when NoNewPrivs=0 in /proc/self/status
  - exits rc=42 with "FAKE_RUNTIME_REFUSED" when NoNewPrivs=1
This is exactly the behavior of a setuid binary blocked by no-new-privs.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="bwrap / NoNewPrivs are Linux-specific",
)

_HERE = Path(__file__).resolve()
_BACKEND_ROOT = _HERE.parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import sandbox as sb
from core.inference import tools


@pytest.fixture
def fake_bwrap(tmp_path):
    script = tmp_path / "bwrap"
    script.write_text(textwrap.dedent('''\
        #!/usr/bin/env python3
        import os, sys
        # Find NoNewPrivs in /proc/self/status — pre-exec sets it via prctl(38).
        nnp = 0
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("NoNewPrivs:"):
                        nnp = int(line.split()[1])
                        break
        except OSError:
            pass

        if nnp == 1:
            sys.stderr.write(f"FAKE_RUNTIME_REFUSED NoNewPrivs=1\\n")
            sys.exit(42)
        # Probe path: pretend success; just exec the tail after --.
        try:
            i = sys.argv.index("--")
            inner = sys.argv[i+1:]
        except ValueError:
            inner = []
        if not inner:
            sys.exit(0)
        os.execvp(inner[0], inner)
    '''))
    script.chmod(0o755)
    return str(script)


def test_real_bwrap_probe_does_not_set_nnp(monkeypatch, fake_bwrap):
    """The probe runs WITHOUT preexec_fn, so /proc/self/status shows
    NoNewPrivs=0 and our fake bwrap reports success."""
    # Force sandbox.py to think it found our fake bwrap.
    monkeypatch.setattr(sb.shutil, "which", lambda name: fake_bwrap if name == "bwrap" else None)

    ok = sb._linux_probe()
    assert ok is True, (
        "Probe succeeds without preexec_fn — even on a setuid-bwrap-like "
        "binary that would refuse with no-new-privs set."
    )
    assert sb._linux_bwrap_path == fake_bwrap


def test_runtime_with_preexec_breaks_fake_bwrap(tmp_path, monkeypatch, fake_bwrap):
    """Now simulate the actual _python_exec path: sandbox_available() True,
    build_sandbox_argv wraps with bwrap, subprocess.Popen runs with
    preexec_fn=_sandbox_preexec which sets PR_SET_NO_NEW_PRIVS. With our
    fake bwrap, the tool call dies with exit 42 — i.e. the user-visible
    bug from the chatgpt-codex inline comment + review_07."""
    # Pretend our fake bwrap is the real one.
    monkeypatch.setattr(sb.shutil, "which", lambda name: fake_bwrap if name == "bwrap" else None)
    sb._sandbox_available_cache = None
    sb._linux_bwrap_path = None

    # Trip the probe (succeeds → cached True).
    assert sb.sandbox_available() is True

    # Run something through _python_exec.
    sid = "_simtest_nnp"
    monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))
    out = tools._python_exec("print('should-not-run')", session_id=sid, timeout=10)
    # The fake bwrap refused at runtime because preexec set NoNewPrivs=1.
    assert "FAKE_RUNTIME_REFUSED" in out, out
    assert "should-not-run" not in out, out


def test_runtime_without_preexec_works(tmp_path, monkeypatch, fake_bwrap):
    """Control: when preexec_fn is suppressed (the suggested fix), the
    same fake bwrap path succeeds and the inner exec runs."""
    monkeypatch.setattr(sb.shutil, "which", lambda name: fake_bwrap if name == "bwrap" else None)
    sb._sandbox_available_cache = None
    sb._linux_bwrap_path = None
    assert sb.sandbox_available() is True

    # Suppress preexec so /proc/self/status shows NoNewPrivs=0 at exec time.
    monkeypatch.setattr(tools, "_sandbox_preexec", lambda: None)

    sid = "_simtest_nnp_fix"
    monkeypatch.setitem(tools._workdirs, sid, str(tmp_path))
    out = tools._python_exec("print('runs-fine')", session_id=sid, timeout=10)
    assert "runs-fine" in out, out
    assert "FAKE_RUNTIME_REFUSED" not in out, out
