"""Harness only: base vs head for the one-time MXC read grant, through Studio's real launch path.

Usage: python t3c_grant_ab.py <label> [--launches N] [--no-controls]
Times sequential sandboxed Python launches (build_launch_request + mxc_adapter.spawn), runs three
isolation controls, and on head reports the grant record and the runtime root's DACL state.
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

try:
    from core.inference import mxc_read_grants  # noqa: E402
except ImportError:
    mxc_read_grants = None


def launch(argv):
    workdir = tempfile.mkdtemp(prefix = "ab-wd-")
    env = {k: v for k, v in os.environ.items() if k.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "PATHEXT"}}
    env.update(TEMP = workdir, TMP = workdir)
    plan = os_sandbox.ToolLaunchPlan(argv = tuple(argv), workdir = workdir, env = env, requested_mode = "required",
                                     timeout_seconds = 170, execution_kind = "python")
    started = time.monotonic()
    request = mxc_policy.build_launch_request(plan)
    proc = mxc_adapter.spawn(request, popen_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                                                      "text": True, "errors": "replace",
                                                      "creationflags": subprocess.CREATE_NO_WINDOW})
    try:
        out, _ = proc.communicate(timeout = 400)
        proc._unsloth_completion_reason = "finished"
        result = mxc_adapter.completion_result(proc)
    finally:
        mxc_adapter.release_runtime(proc)
    return out or "", result, time.monotonic() - started


def timings(label, count):
    times = []
    for index in range(count):
        out, res, secs = launch([sys.executable, "-c", "import json, ssl; print('AB_OK')"])
        ok = "AB_OK" in out and res.get("exitCode") == 0
        print(f"LAUNCH {label} #{index} {'ok' if ok else 'BAD'} s={secs:.2f}" + ("" if ok else f" | {out.strip()[-200:]}"), flush = True)
        if ok:
            times.append(secs)
    if times:
        steady = times[1:] or times
        print(f"TIMING {label} first={times[0]:.2f}s steady_median={statistics.median(steady):.2f}s n={len(times)}", flush = True)


def controls(label):
    other = Path(tempfile.mkdtemp(prefix = "ab-other-"))
    marker = other / "outside-marker.txt"
    marker.write_text("OUTSIDE_MARKER_TEXT", encoding = "utf-8")
    target = other / "written-from-inside.txt"
    runtime_probe = Path(sys.prefix) / "ab-runtime-probe.txt"
    body = "import pathlib\ntry:\n {op}\nexcept Exception as e:\n print('DENIED', type(e).__name__)"
    checks = [
        ("outside_read_denied", body.format(op = f"print(pathlib.Path(r'{marker}').read_text())"),
         lambda out: "OUTSIDE_MARKER_TEXT" not in out and "DENIED" in out),
        ("outside_write_denied", body.format(op = f"pathlib.Path(r'{target}').write_text('x')"),
         lambda out: not target.exists() and "DENIED" in out),
        ("runtime_write_denied", body.format(op = f"pathlib.Path(r'{runtime_probe}').write_text('x'); print('WROTE')"),
         lambda out: "DENIED" in out and not runtime_probe.exists()),
    ]
    for name, code, ok in checks:
        out, res, secs = launch([sys.executable, "-c", code])
        print(f"CONTROL {label} {name} {'PASS' if ok(out) else 'FAIL'} exit={res.get('exitCode')} s={secs:.2f} | {out.strip()[-100:]}", flush = True)


def grant_state(label):
    if mxc_read_grants is None:
        print(f"GRANT_STATE {label} module=absent", flush = True)
        return
    record = mxc_read_grants.record_path()
    text = record.read_text(encoding = "utf-8") if record.exists() else "<none>"
    print(f"GRANT_STATE {label} enabled={mxc_read_grants.enabled()} prefix_aces={mxc_read_grants._package_aces(sys.prefix)} record={' '.join(text.split())}", flush = True)


def diagnose():
    roots = mxc_policy._runtime_read_roots(sys.executable, [])
    for root in roots:
        try:
            aces = mxc_read_grants._package_aces(root)
        except OSError as exc:
            aces = f"error {exc}"
        print(f"DIAG root={root} ineligible={mxc_read_grants.ineligible_reason(root)!r} identity={mxc_read_grants._identity(root)} aces={aces}", flush = True)
    print(f"DIAG protected={mxc_read_grants._protected_paths()}", flush = True)


if __name__ == "__main__":
    label = sys.argv[1]
    if label == "diag":
        diagnose()
        sys.exit(0)
    count = int(sys.argv[sys.argv.index("--launches") + 1]) if "--launches" in sys.argv else 6
    grant_state(f"{label}:before")
    timings(label, count)
    grant_state(f"{label}:after")
    if "--no-controls" not in sys.argv:
        controls(label)
