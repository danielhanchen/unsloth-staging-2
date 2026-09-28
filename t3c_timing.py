"""Harness only: per-shell wall time of one MXC launch, with raw wxc output and grants."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from core.inference import mxc_adapter, mxc_policy, os_sandbox

SHELLS = [("python", sys.executable, ("-c", "print('T3C_HELLO')"))]
for name in ("cmd.exe", "powershell.exe", "pwsh.exe"):
    found = shutil.which(name)
    if found:
        args = ("/d", "/s", "/c", "echo T3C_HELLO") if name == "cmd.exe" else (
            "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", "Write-Output T3C_HELLO")
        SHELLS.append(("terminal", found, args))

for kind, exe, args in SHELLS:
    workdir = Path(tempfile.mkdtemp(prefix = "t3c-time-"))
    env = {k: v for k, v in os.environ.items() if k.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "PATHEXT"}}
    env.update(TEMP = str(workdir), TMP = str(workdir))
    plan = os_sandbox.ToolLaunchPlan(argv = (exe, *args), workdir = str(workdir), env = env,
                                     requested_mode = "required", timeout_seconds = 170, execution_kind = kind)
    started = time.monotonic()
    request = mxc_policy.build_launch_request(plan)
    built = time.monotonic() - started
    print("T3C_TIME", kind, exe, "grants_ro", request["config"]["filesystem"]["readonlyPaths"], flush = True)
    proc = mxc_adapter.spawn(request, popen_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                                                      "text": True, "errors": "replace",
                                                      "creationflags": subprocess.CREATE_NO_WINDOW})
    try:
        out, _ = proc.communicate(timeout = 180)
        status = f"exit={proc.returncode}"
    except subprocess.TimeoutExpired:
        mxc_adapter.abort(proc)
        out = proc.communicate(timeout = 10)[0] or ""
        status = "TIMEOUT(180s)"
    mxc_adapter.release_runtime(proc)
    print("T3C_TIME", kind, os.path.basename(exe), f"build={built:.2f}s", f"total={time.monotonic() - started:.2f}s",
          status, "hello" if "T3C_HELLO" in out else "no-hello", flush = True)
    print(out[-1500:], flush = True)
