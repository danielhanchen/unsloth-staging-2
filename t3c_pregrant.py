"""Harness only: does a one-time ALL APPLICATION PACKAGES read grant on Studio's runtime roots remove the per-launch DACL cost?

MXC skips a readonly path whose DACL already grants the needed mask to ALL APPLICATION PACKAGES
(dispatcher.rs filter_paths_needing_grant). Arms, each timed over REPS launches through Studio's own
build_launch_request: baseline, then after `icacls <root> /grant *S-1-15-2-1:(OI)(CI)(RX)` on every
runtime read root except SystemRoot. Isolation controls run in both arms; grants are removed at the end.
"""

import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, "studio/backend")
from core.inference import mxc_adapter, mxc_policy, os_sandbox  # noqa: E402

REPS = int(os.environ.get("PREGRANT_REPS", "5"))
BUSYBOX = os.environ.get("BUSYBOX")
AC_ALL = "*S-1-15-2-1"


def launch(argv, workdir, kind):
    env = {k: v for k, v in os.environ.items() if k.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "PATHEXT"}}
    env.update(TEMP = workdir, TMP = workdir, TMPDIR = workdir, HOME = workdir)
    plan = os_sandbox.ToolLaunchPlan(argv = tuple(argv), workdir = workdir, env = env, requested_mode = "required",
                                     timeout_seconds = 120, execution_kind = kind)
    started = time.monotonic()
    request = mxc_policy.build_launch_request(plan)
    proc = mxc_adapter.spawn(request, popen_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                                                      "text": True, "errors": "replace",
                                                      "creationflags": subprocess.CREATE_NO_WINDOW})
    try:
        out, _ = proc.communicate(timeout = 180)
        proc._unsloth_completion_reason = "finished"
        result = mxc_adapter.completion_result(proc)
    finally:
        mxc_adapter.release_runtime(proc)
    return out or "", result, time.monotonic() - started


def cases():
    yield "python_print", [sys.executable, "-c", "print('PREGRANT_OK')"], "python"
    if BUSYBOX:
        yield "busybox_pipe", [BUSYBOX, "sh", "-c", "echo PREGRANT_OK | cat\nexit $?"], "terminal"


def controls(label):
    other = Path(tempfile.mkdtemp(prefix = "pg-other-"))
    marker = other / "outside-marker.txt"
    marker.write_text("OUTSIDE_MARKER_TEXT", encoding = "utf-8")
    target = other / "written-from-inside.txt"
    py = sys.executable
    checks = [
        ("outside_read_denied", [py, "-c", f"import pathlib\ntry:\n print(pathlib.Path(r'{marker}').read_text())\nexcept Exception as e:\n print('DENIED', type(e).__name__)"],
         lambda out: "OUTSIDE_MARKER_TEXT" not in out and "DENIED" in out),
        ("outside_write_denied", [py, "-c", f"import pathlib\ntry:\n pathlib.Path(r'{target}').write_text('x')\nexcept Exception as e:\n print('DENIED', type(e).__name__)"],
         lambda out: not target.exists() and "DENIED" in out),
        ("runtime_write_denied", [py, "-c", f"import pathlib\np = pathlib.Path(r'{sys.prefix}') / 'pregrant-probe.txt'\ntry:\n p.write_text('x')\n print('WROTE')\nexcept Exception as e:\n print('DENIED', type(e).__name__)"],
         lambda out: "DENIED" in out and not (Path(sys.prefix) / "pregrant-probe.txt").exists()),
    ]
    for name, argv, ok in checks:
        out, res, secs = launch(argv, tempfile.mkdtemp(prefix = "pg-wd-"), "python")
        print(f"CONTROL {label} {name} {'PASS' if ok(out) else 'FAIL'} exit={res.get('exitCode')} s={secs:.1f} | {out.strip()[-120:]}", flush = True)


def timed(label):
    for name, argv, kind in cases():
        times = []
        for _ in range(REPS):
            out, res, secs = launch(argv, tempfile.mkdtemp(prefix = "pg-wd-"), kind)
            if "PREGRANT_OK" not in out or res.get("exitCode") != 0:
                print(f"RUN {label} {name} BAD exit={res.get('exitCode')} | {out.strip()[-200:]}", flush = True)
                continue
            times.append(secs)
        if times:
            print(f"TIMING {label} {name} n={len(times)} median={statistics.median(times):.2f}s min={min(times):.2f}s max={max(times):.2f}s", flush = True)


def roots():
    system_root = os.path.normcase(os.environ.get("SystemRoot", ""))
    found = mxc_policy._runtime_read_roots(sys.executable)
    if BUSYBOX:
        found += mxc_policy._runtime_read_roots(BUSYBOX)
    out = []
    for r in found:
        if os.path.normcase(r) != system_root and r not in out:
            out.append(r)
    return out


def icacls(root, *args):
    done = subprocess.run(["icacls", root, *args], capture_output = True, text = True, errors = "replace")
    return done.returncode, (done.stdout + done.stderr).strip().splitlines()[-1:]


if __name__ == "__main__":
    grant_roots = roots()
    print("ROOTS", grant_roots, flush = True)
    for r in grant_roots:
        count = sum(len(files) for _, _, files in os.walk(r))
        print(f"FILES {r} {count}", flush = True)
    timed("baseline")
    controls("baseline")
    started = time.monotonic()
    for r in grant_roots:
        print("GRANT", r, *icacls(r, "/grant", f"{AC_ALL}:(OI)(CI)(RX)"), flush = True)
    print(f"GRANT_ONE_TIME_COST {time.monotonic() - started:.1f}s", flush = True)
    timed("pregranted")
    controls("pregranted")
    for r in grant_roots:
        print("REMOVE", r, *icacls(r, "/remove:g", AC_ALL), flush = True)
