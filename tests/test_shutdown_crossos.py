# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Cross-OS validation for the Studio Ctrl+C shutdown-ordering fixes.

Contract tests (run everywhere, incl. Windows) assert the fixes are present in
install.sh / run.py. Behavioural tests drive the REAL install.sh prompt block and
a faithful copy of run.py's shutdown flow under a pty; pty is POSIX-only so those
auto-skip on Windows (where install.ps1, not install.sh, is the relevant path).
"""

import os
import re
import sys
import time
import textwrap
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_INSTALL_SH = _ROOT / "install.sh"
_RUN_PY = _ROOT / "studio" / "backend" / "run.py"

_HAS_PTY = False
if sys.platform != "win32":
    try:
        import pty  # noqa: F401
        import select
        import fcntl
        import termios
        _HAS_PTY = True
    except Exception:
        _HAS_PTY = False

_posix_only = pytest.mark.skipif(not _HAS_PTY, reason="pty unavailable (Windows uses install.ps1)")


# --------------------------------------------------------------------------- #
# Contract tests - run on every OS, including Windows.
# --------------------------------------------------------------------------- #
def test_install_sh_declines_default_to_no():
    txt = _INSTALL_SH.read_text(encoding="utf-8")
    assert 'read -r _reply </dev/tty || _reply="n"' in txt, "read-failure must default to n"
    assert re.search(r'else\s*\n\s*_reply="n"', txt), "no-tty branch must default to n"
    assert 'read -r _reply </dev/tty || _reply="y"' not in txt, "old yes-default must be gone"


def test_install_sh_traps_int_before_launch():
    txt = _INSTALL_SH.read_text(encoding="utf-8")
    i_trap = txt.find("trap '' INT")
    i_launch = txt.find('"$VENV_DIR/bin/unsloth" studio -p 8888')
    assert i_trap != -1, "installer must trap INT so it waits for studio's shutdown"
    assert 0 < i_trap < i_launch, "trap must come before the studio launch"
    # The launch must run in a subshell that resets INT, so the child does not
    # inherit the ignored SIGINT and can still handle its own Ctrl+C.
    assert "(trap - INT; exec " in txt, "studio launch must reset INT in a subshell"


def test_run_py_join_is_bounded_and_forcequit():
    txt = _RUN_PY.read_text(encoding="utf-8")
    assert "_SERVER_SHUTDOWN_JOIN_TIMEOUT" in txt, "join must be bounded by a timeout"
    assert "thread.join(timeout = timeout)" in txt, "wait helper must join with a timeout"
    assert "signal.SIG_DFL" in txt, "signal handler must restore SIG_DFL for force-quit"
    assert "signal.SIGBREAK, signal.SIG_DFL" in txt, "SIGBREAK must be restored on Windows"


# --------------------------------------------------------------------------- #
# pty harness (POSIX).
# --------------------------------------------------------------------------- #
def _run_under_pty(interp_argv, *, stdin_pipe_bytes=None, type_after=None,
                   ready_marker=None, done_markers=(), settle=0.3, max_wait=9.0):
    """Run interp_argv with fd1=pty; optionally feed a script via a stdin pipe
    (mimics curl | sh). type_after: bytes typed to the tty once ready_marker is
    seen (None = send EOF). Returns (text, seconds_from_type_to_first_done)."""
    master, slave = pty.openpty()
    r_in = w_in = None
    if stdin_pipe_bytes is not None:
        r_in, w_in = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.setsid()
        fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
        os.dup2(r_in if r_in is not None else slave, 0)
        os.dup2(slave, 1)
        os.dup2(slave, 2)
        for fd in {master, slave, r_in, w_in} - {None}:
            try: os.close(fd)
            except OSError: pass
        os.execvp(interp_argv[0], interp_argv)
    os.close(slave)
    if r_in is not None:
        os.close(r_in)
        os.write(w_in, stdin_pipe_bytes)
        os.close(w_in)
    out = bytearray()
    typed = False
    t_type = None
    end = time.time() + max_wait
    while time.time() < end:
        rl, _, _ = select.select([master], [], [], 0.1)
        if master in rl:
            try: c = os.read(master, 4096)
            except OSError: break
            if not c: break
            out += c
        if not typed and (ready_marker is None or ready_marker.encode() in out):
            time.sleep(settle)
            if type_after is not None:
                os.write(master, type_after)
            else:
                os.write(master, b"\x04")  # EOF
            typed = True
            t_type = time.time()
        if done_markers and any(m.encode() in out for m in done_markers):
            break
    elapsed = (time.time() - t_type) if t_type else None
    try: os.close(master)
    except OSError: pass
    try: os.waitpid(pid, 0)
    except OSError: pass
    return out.decode(errors="replace"), elapsed


def _extract_prompt_block():
    lines = _INSTALL_SH.read_text(encoding="utf-8").splitlines()
    pi = next(i for i, l in enumerate(lines) if "Start Unsloth Studio now?" in l)
    si = next(i for i in range(pi, -1, -1) if lines[i].strip() == "if [ -t 1 ]; then")
    # Match the OUTER fi (column 0); inner migration `fi` is indented.
    fi = next(i for i in range(pi, len(lines)) if lines[i] == "fi")
    block = "\n".join(lines[si:fi + 1])
    # Stub the side effects, mark the two outcomes.
    block = block.replace('"$VENV_DIR/bin/unsloth" studio -p 8888 </dev/null',
                          'echo "MARKER=LAUNCH"')
    block = block.replace('step "launch" "to start later, run:"', 'echo "MARKER=LATER"')
    return "step() { :; }\nsubstep() { :; }\n_MIGRATED=false\nVENV_DIR=/tmp\n" + block


@_posix_only
@pytest.mark.parametrize("answer,expect", [
    (b"n\n", "LATER"), (b"N\n", "LATER"), (b"\n", "LAUNCH"), (None, "LATER"),
])
def test_install_prompt_decline_behaviour(answer, expect):
    script = _extract_prompt_block()
    text, _ = _run_under_pty(["sh"], stdin_pipe_bytes=script.encode(),
                             ready_marker="Start Unsloth Studio now?",
                             type_after=answer, done_markers=("MARKER=",))
    got = "LAUNCH" if "MARKER=LAUNCH" in text else ("LATER" if "MARKER=LATER" in text else "??")
    assert got == expect, f"answer {answer!r}: expected {expect}, got {got}\n{text}"


@_posix_only
def test_installer_trap_orders_shutdown(tmp_path):
    """The real install.sh pattern (trap '' INT + subshell that resets INT) must:
    (a) deliver Ctrl+C to the child even though it relies on the default SIGINT
    (so it is not swallowed by an inherited SIG_IGN), and (b) keep the installer
    shell waiting so the prompt returns only after the child's shutdown logs."""
    child = tmp_path / "child.py"
    # NB: relies on the default SIGINT -> KeyboardInterrupt; does NOT re-register,
    # so an inherited SIG_IGN would make it never see Ctrl+C.
    child.write_text(textwrap.dedent('''
        import time
        print("CHILD_READY", flush=True)
        try:
            while True: time.sleep(0.2)
        except KeyboardInterrupt:
            print("LOG1", flush=True); time.sleep(0.3); print("LOG2", flush=True)
    '''))
    installer = tmp_path / "installer.sh"
    installer.write_text(
        f"trap '' INT\n(trap - INT; exec {sys.executable} {child} </dev/null)\necho EXITED\n")
    text, _ = _drive_interactive(f"cat {installer} | sh")
    l2 = text.find("LOG2")
    p = text.find("INTERACTIVE_PROMPT>", text.find("CHILD_READY"))
    assert l2 != -1, f"child must receive Ctrl+C (not an inherited SIG_IGN)\n{text}"
    assert p != -1 and p > l2, f"prompt must come after the child's shutdown logs\n{text}"


_RUNPY_CHILD = '''
import signal, sys, threading, time
from typing import Optional
MODE = sys.argv[1]
_TIMEOUT = 5.0
_server_thread = None
_evt = threading.Event(); _exit = threading.Event()
def _flush():
    for s in (sys.stdout, sys.stderr):
        try: s.flush()
        except Exception: pass
def _wait(timeout: Optional[float] = _TIMEOUT):
    t = _server_thread
    if t is None or t is threading.current_thread():
        _flush(); return
    t.join(timeout=timeout)
    if t.is_alive(): print("TIMED_OUT", flush=True)
    _flush()
def _run():
    while not _exit.is_set(): time.sleep(0.05)
    if MODE == "hang":
        while True: time.sleep(0.5)
    time.sleep(0.3); print("INFO: Shutting down", flush=True)
def _grace():
    _exit.set()
th = threading.Thread(target=_run, daemon=True); _server_thread = th; th.start()
def _h(s, f):
    signal.signal(signal.SIGINT, signal.SIG_DFL); signal.signal(signal.SIGTERM, signal.SIG_DFL)
    _grace(); _evt.set()
signal.signal(signal.SIGINT, _h); signal.signal(signal.SIGTERM, _h)
print("CHILD_READY", flush=True)
while not _evt.is_set(): _evt.wait(timeout=1)
if MODE != "buggy": _wait()
'''


def _drive_interactive(cmd, *, double_ctrl_c=False, settle=0.4, max_wait=9.0):
    """Interactive bash in a pty: run cmd, wait for CHILD_READY, Ctrl+C, capture."""
    master, slave = pty.openpty()
    pid = os.fork()
    if pid == 0:
        os.setsid(); fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
        os.dup2(slave, 0); os.dup2(slave, 1); os.dup2(slave, 2)
        for fd in (master, slave):
            try: os.close(fd)
            except OSError: pass
        env = dict(os.environ, PS1="INTERACTIVE_PROMPT> ", PS2="", TERM="dumb")
        os.execvpe("bash", ["bash", "--norc", "--noprofile", "-i"], env)
    os.close(slave)
    out = bytearray()
    def pump(until=None, t=4.0):
        end = time.time() + t
        while time.time() < end:
            rl, _, _ = select.select([master], [], [], 0.1)
            if master in rl:
                try: c = os.read(master, 4096)
                except OSError: return
                if not c: return
                out.extend(c)
                if until and until.encode() in out: return
    pump("INTERACTIVE_PROMPT>", 3.0)
    os.write(master, (cmd + "\n").encode())
    pump("CHILD_READY", 5.0)
    cmd_end = len(out)
    time.sleep(settle)
    t0 = time.time()
    os.write(master, b"\x03")
    if double_ctrl_c:
        time.sleep(0.2); os.write(master, b"\x03")
    ret = None
    end = time.time() + max_wait
    while time.time() < end:
        rl, _, _ = select.select([master], [], [], 0.1)
        if master in rl:
            try: c = os.read(master, 4096)
            except OSError: break
            if not c: break
            out += c
            if ret is None and b"INTERACTIVE_PROMPT>" in bytes(out[cmd_end:]):
                ret = time.time() - t0
                break
    try: os.close(master)
    except OSError: pass
    try: os.waitpid(pid, 0)
    except OSError: pass
    return out.decode(errors="replace"), (ret if ret is not None else max_wait)


@_posix_only
def test_runpy_fixed_orders_uvicorn_logs(tmp_path):
    child = tmp_path / "runpy_child.py"
    child.write_text(_RUNPY_CHILD)
    fixed, _ = _drive_interactive(f"{sys.executable} {child} fixed")
    buggy, _ = _drive_interactive(f"{sys.executable} {child} buggy")
    # fixed: uvicorn log printed before the prompt; buggy: it is lost/raced
    pf = fixed.find("INTERACTIVE_PROMPT>", fixed.find("CHILD_READY"))
    info = fixed.find("INFO: Shutting down")
    assert info != -1 and pf > info, f"fixed must print uvicorn log before prompt\n{fixed}"
    assert "INFO: Shutting down" not in buggy.split("CHILD_READY", 1)[-1] or \
        buggy.find("INFO: Shutting down") > buggy.find("INTERACTIVE_PROMPT>", buggy.find("CHILD_READY")), \
        "buggy mode should race/lose the uvicorn log"


@_posix_only
def test_runpy_hang_is_bounded_and_forcequit(tmp_path):
    child = tmp_path / "runpy_child.py"
    child.write_text(_RUNPY_CHILD)
    one, el1 = _drive_interactive(f"{sys.executable} {child} hang", max_wait=9.0)
    assert "TIMED_OUT" in one and el1 < 8.0, f"single Ctrl+C must exit via the 5s bound\n{one}"
    two, el2 = _drive_interactive(f"{sys.executable} {child} hang", double_ctrl_c=True, max_wait=9.0)
    assert el2 < 3.0, f"second Ctrl+C must force-quit fast (got {el2:.2f}s)\n{two}"
