"""Harness only: Tier 3 confinement, ACE lifecycle and hard-kill recovery through prepare_tool_launch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from core.inference import mxc_adapter, mxc_runtime, os_sandbox

WXC = str(mxc_runtime.wxc_path())
STATE_DIR = Path(mxc_adapter._control_environment().get("MXC_DACL_STATE_DIR", "<unset>"))
PROFILE_CANARY = Path(os.environ["USERPROFILE"]) / "t3c_profile_canary.txt"
WIN_WRITE = Path(os.environ["SystemRoot"]) / "t3c_write_probe.txt"
PF_WRITE = Path(os.environ["ProgramFiles"]) / "t3c_write_probe.txt"
JOURNAL_WRITE = STATE_DIR / "t3c_write_probe.txt"


def say(*parts):
    print("T3C", *parts, flush = True)


def acl(path):
    out = subprocess.run(["icacls", str(path)], capture_output = True, text = True, errors = "replace").stdout
    return [line.strip() for line in out.splitlines()[:-2] if line.strip()]


def host_positive(path, write):
    try:
        if write:
            path.write_text("host", encoding = "utf-8")
            path.unlink()
        else:
            path.read_text(encoding = "utf-8")
        return True
    except OSError as exc:
        return f"host could not {'write' if write else 'read'} {path}: {exc}"


WORKLOAD = r'''
import json, os, sys, time, sqlite3, ssl, ctypes, pathlib
r = {}
def attempt(key, fn):
    try:
        fn(); r[key] = "ALLOWED"
    except OSError as exc:
        r[key] = "DENIED:" + type(exc).__name__ + ":" + str(exc.winerror if hasattr(exc, "winerror") else exc.errno)
attempt("profile_canary_read", lambda: open(CANARY, encoding="utf-8").read())
attempt("windows_write", lambda: open(WINW, "w").write("bad"))
attempt("program_files_write", lambda: open(PFW, "w").write("bad"))
attempt("journal_write", lambda: open(JW, "w").write("bad"))
attempt("journal_list", lambda: os.listdir(JDIR))
attempt("workdir_write", lambda: open("inside.txt", "w").write("ok"))
attempt("stdlib_read", lambda: open(os.__file__, encoding="utf-8").read(64))
r["imports"] = [m.__name__ for m in (json, sqlite3, ssl, ctypes, pathlib)]
print("T3C_RESULT=" + json.dumps(r, sort_keys=True), flush=True)
open("ready.txt", "w").write("1")
deadline = time.time() + HOLD
while time.time() < deadline and not os.path.exists("go.txt"):
    time.sleep(0.2)
'''


def launch(workdir, hold):
    script = workdir / "workload.py"
    body = (
        f"CANARY={str(PROFILE_CANARY)!r}\nWINW={str(WIN_WRITE)!r}\nPFW={str(PF_WRITE)!r}\n"
        f"JW={str(JOURNAL_WRITE)!r}\nJDIR={str(STATE_DIR)!r}\nHOLD={hold}\n" + WORKLOAD
    )
    script.write_text(body, encoding = "utf-8")
    env = {k: v for k, v in os.environ.items() if k.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "PATHEXT"}}
    env.update(TEMP = str(workdir), TMP = str(workdir), PYTHONIOENCODING = "utf-8")
    plan = os_sandbox.ToolLaunchPlan(
        argv = (sys.executable, "-u", str(script)),
        workdir = str(workdir),
        env = env,
        requested_mode = "required",
        timeout_seconds = hold + 60,
        execution_kind = "python",
    )
    prepared = os_sandbox.prepare_tool_launch(plan)
    rec = prepared.execution_record
    say("record", json.dumps({"backend": prepared.backend, "effective_mode": rec.effective_mode,
                              "os_isolation": rec.os_isolation, "limitations": list(rec.limitations)}))
    proc = os_sandbox.spawn_prepared_launch(
        prepared, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True,
        encoding = "utf-8", errors = "replace", creationflags = subprocess.CREATE_NO_WINDOW,
    )
    return prepared, proc


def wait_ready(workdir, proc, seconds = 90):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if (workdir / "ready.txt").exists():
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.2)
    return False


def main():
    root = Path(tempfile.mkdtemp(prefix = "t3c-"))
    watched = {"workdir": None, "sys.prefix": Path(sys.prefix), "C:\\": Path("C:\\")}
    say("wxc", WXC, "state_dir", STATE_DIR, "fallback env", os.environ.get("UNSLOTH_MXC_ALLOW_DACL_FALLBACK"))
    PROFILE_CANARY.write_text("profile-secret", encoding = "utf-8")
    STATE_DIR.mkdir(parents = True, exist_ok = True)
    for name, path, write in (("profile_canary_read", PROFILE_CANARY, False), ("windows_write", WIN_WRITE, True),
                              ("program_files_write", PF_WRITE, True), ("journal_write", JOURNAL_WRITE, True)):
        say("host_positive", name, host_positive(path, write))

    # Normal exit.
    workdir = root / "normal"
    workdir.mkdir()
    watched["workdir"] = workdir
    before = {k: acl(v) for k, v in watched.items()}
    prepared, proc = launch(workdir, hold = 30)
    ok = wait_ready(workdir, proc)
    during = {k: acl(v) for k, v in watched.items()}
    (workdir / "go.txt").write_text("1")
    output, _ = proc.communicate(timeout = 120)
    proc._unsloth_completion_reason = "finished"
    try:
        say("completion", os_sandbox.verify_prepared_completion(prepared, proc))
    except Exception as exc:
        say("completion_error", repr(exc))
    prepared.cleanup()
    after = {k: acl(v) for k, v in watched.items()}
    say("ready", ok, "returncode", proc.returncode)
    print(output[-3000:])
    say("host_sees_inside_txt", (workdir / "inside.txt").exists())
    for name, path in (("windows_write", WIN_WRITE), ("program_files_write", PF_WRITE), ("journal_write", JOURNAL_WRITE)):
        say("host_sees_outside_write", name, path.exists())
    for key in watched:
        added = sorted(set(during[key]) - set(before[key]))
        left = sorted(set(after[key]) - set(before[key]))
        say("acl", key, "added_during", json.dumps(added))
        say("acl", key, "left_after_exit", json.dumps(left))

    # Hard kill of wxc-exec mid-run.
    workdir = root / "killed"
    workdir.mkdir()
    watched["workdir"] = workdir
    before = {k: acl(v) for k, v in watched.items()}
    prepared, proc = launch(workdir, hold = 180)
    ok = wait_ready(workdir, proc)
    during = {k: acl(v) for k, v in watched.items()}
    say("journal_during", sorted(p.name for p in STATE_DIR.iterdir()))
    tasks = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"], capture_output = True, text = True).stdout
    say("python_processes_during", tasks.count("python.exe"))
    killed = subprocess.run(["taskkill", "/F", "/PID", str(proc.pid)], capture_output = True, text = True)
    say("taskkill", killed.returncode, killed.stdout.strip(), killed.stderr.strip())
    try:
        proc.wait(timeout = 30)
    except subprocess.TimeoutExpired:
        say("wxc_still_alive_after_kill")
    time.sleep(3)
    after_kill = {k: acl(v) for k, v in watched.items()}
    say("journal_after_kill", sorted(p.name for p in STATE_DIR.iterdir()))
    tasks = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"], capture_output = True, text = True).stdout
    say("python_processes_after_kill", tasks.count("python.exe"))
    probe = subprocess.run([WXC, "--probe"], capture_output = True, text = True, errors = "replace",
                           env = mxc_adapter._control_environment(), stdin = subprocess.DEVNULL)
    say("next_probe_exit", probe.returncode)
    print((probe.stdout + probe.stderr)[-2000:])
    after_recovery = {k: acl(v) for k, v in watched.items()}
    say("journal_after_recovery", sorted(p.name for p in STATE_DIR.iterdir()))
    for key in watched:
        say("acl", key, "added_during", json.dumps(sorted(set(during[key]) - set(before[key]))))
        say("acl", key, "left_after_kill", json.dumps(sorted(set(after_kill[key]) - set(before[key]))))
        say("acl", key, "left_after_recovery", json.dumps(sorted(set(after_recovery[key]) - set(before[key]))))
    prepared.cleanup()
    PROFILE_CANARY.unlink()


if __name__ == "__main__":
    main()
