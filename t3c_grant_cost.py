"""Harness only: which readonly grant makes a Tier 3 launch slow."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from core.inference import mxc_adapter, mxc_policy, os_sandbox

original = mxc_policy._runtime_read_roots
system_root = os.path.normcase(os.environ["SystemRoot"])
arms = {
    "all_grants": lambda roots: roots,
    "no_systemroot": lambda roots: [r for r in roots if os.path.normcase(r) != system_root],
    "exe_dir_only": lambda roots: roots[:1],
    "systemroot_plus_exe": lambda roots: [roots[0]] + [r for r in roots if os.path.normcase(r) == system_root],
}
for rep in range(2):
    for name, pick in arms.items():
        mxc_policy._runtime_read_roots = lambda exe, pick = pick: pick(original(exe))
        workdir = Path(tempfile.mkdtemp(prefix = "t3c-cost-"))
        env = {k: v for k, v in os.environ.items() if k.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "PATHEXT"}}
        env.update(TEMP = str(workdir), TMP = str(workdir))
        plan = os_sandbox.ToolLaunchPlan(argv = (sys.executable, "-c", "import json, ssl; print('T3C_HELLO')"),
                                         workdir = str(workdir), env = env, requested_mode = "required",
                                         timeout_seconds = 170, execution_kind = "python")
        started = time.monotonic()
        request = mxc_policy.build_launch_request(plan)
        proc = mxc_adapter.spawn(request, popen_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                                                          "text": True, "errors": "replace",
                                                          "creationflags": subprocess.CREATE_NO_WINDOW})
        try:
            out = proc.communicate(timeout = 180)[0]
            status = f"exit={proc.returncode}"
        except subprocess.TimeoutExpired:
            mxc_adapter.abort(proc)
            out = proc.communicate(timeout = 10)[0] or ""
            status = "TIMEOUT"
        mxc_adapter.release_runtime(proc)
        print("T3C_COST", rep, name, f"{time.monotonic() - started:.2f}s", status,
              "hello" if "T3C_HELLO" in out else "no-hello:" + out[-300:].replace("\n", " | "),
              request["config"]["filesystem"]["readonlyPaths"], flush = True)
