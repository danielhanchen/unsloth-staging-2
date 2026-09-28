"""Per-call latency of Studio's Python and Terminal tools in three forced modes, plus the gate alone.

full = disable_sandbox; software = auto with the OS capability stubbed unavailable; isolated =
required with the real probe. Modes are interleaved per rep. A sample counts only when the payload
printed its marker and the execution record matches the forced mode.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), "studio", "backend"))
import logging  # noqa: E402

logging.disable(logging.INFO)
from core.inference import os_sandbox, sandbox_probe, tools  # noqa: E402

MODES = ("full", "software_safeguards", "os_isolated")

PY = {
    "print": "print('DONE')",
    "imports": "import json, sqlite3, csv, re\nprint('DONE')",
    "files_1000": (
        "import os\nos.makedirs('bench_files', exist_ok=True)\n"
        "for i in range(1000):\n    open(f'bench_files/{i}.txt', 'w').write('x')\nprint('DONE')"
    ),
    "script_200_lines": "\n".join(f"v{i} = {i} * 2" for i in range(200)) + "\nprint('DONE')",
}
SH_POSIX = {"echo": "echo DONE", "pipeline": "seq 1 2000 | sort -n | uniq | wc -l && echo DONE"}
SH_CMD = {"echo": "echo DONE", "pipeline": "dir /b > listing.txt && find /c /v \"\" listing.txt && echo DONE"}
# Valid in Git Bash and in cmd alike, for a leg whose modes run different shells.
SH_BOTH = {"echo": "echo DONE", "git_version": "git --version && echo DONE", "chain": "echo a && echo b && echo DONE"}


@contextlib.contextmanager
def forced(mode):
    if mode != "software_safeguards":
        yield
        return
    real = os_sandbox.capability_snapshot
    os_sandbox.capability_snapshot = lambda **_k: os_sandbox.SandboxCapability(
        backend = "stubbed", available = False, reason = "forced", environment = sys.platform
    )
    try:
        yield
    finally:
        os_sandbox.capability_snapshot = real


def call(tool, payload, mode, session):
    tools._last_tool_execution_record = None
    kwargs = {"session_id": session, "timeout": 120}
    if mode == "full":
        kwargs["disable_sandbox"] = True
    elif mode == "os_isolated":
        kwargs["tool_execution_mode"] = "required"
    key = "code" if tool == "python" else "command"
    with forced(mode):
        t0 = time.perf_counter()
        out = tools.execute_tool(tool, {key: payload}, **kwargs) or ""
        dt = time.perf_counter() - t0
    rec = tools._last_tool_execution_record
    ok = "DONE" in out and rec is not None and rec.effective_mode == mode
    return dt, ok, (rec.effective_mode if rec else None), out[:200]


def summary(xs):
    if not xs:
        return None
    xs = sorted(xs)
    return {
        "n": len(xs), "min_ms": round(xs[0] * 1000, 2), "median_ms": round(statistics.median(xs) * 1000, 2),
        "p90_ms": round(xs[min(len(xs) - 1, int(0.9 * len(xs)))] * 1000, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--reps", type = int, default = 21)
    ap.add_argument("--warmup", type = int, default = 3)
    ap.add_argument("--modes", default = ",".join(MODES))
    ap.add_argument("--cold-reps", type = int, default = 5)
    ap.add_argument("--cold-one", default = None)
    ap.add_argument("--sh", default = "auto", choices = ("auto", "both"))
    opts = ap.parse_args()
    if opts.cold_one:
        t0 = time.perf_counter()
        dt, ok, rmode, _ = call("python", PY["print"], opts.cold_one, f"__LOCALID_bench_cold_{opts.cold_one}")
        print("COLD " + json.dumps({"first_call_ms": round(dt * 1000, 1), "ok": ok, "record": rmode}), flush = True)
        return
    modes = [m for m in opts.modes.split(",") if m]
    cap = os_sandbox.capability_snapshot(force = True)
    if not cap.available and "os_isolated" in modes:
        modes.remove("os_isolated")
    sh = SH_BOTH if opts.sh == "both" else (SH_POSIX if tools._shell_is_posix() else SH_CMD)
    result = {
        "platform": platform.platform(), "python": sys.version.split()[0],
        "capability": {"backend": cap.backend, "available": cap.available, "reason": cap.reason[:300]},
        "mxc_opt_in": os.environ.get("UNSLOTH_MXC_ALLOW_DACL_FALLBACK"), "modes": modes,
        "gate_us": {}, "cold": {}, "warm": {}, "failures": [],
    }

    # The gate alone, in process.
    for name, code in PY.items():
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            tools._check_code_safety(code)
        result["gate_us"][f"python:{name}"] = round((time.perf_counter() - t0) / n * 1e6, 1)
    for name, cmd in sh.items():
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            tools._terminal_is_high_risk(cmd)
        result["gate_us"][f"terminal:{name}"] = round((time.perf_counter() - t0) / n * 1e6, 1)

    work = [("python", k, v) for k, v in PY.items()] + [("terminal", k, v) for k, v in sh.items()]

    # Cold: a fresh interpreter per mode, so the first call pays the probe and import costs; modes interleaved.
    import subprocess
    for rep in range(opts.cold_reps):
        for mode in modes:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--cold-one", mode, "--out", "-"],
                capture_output = True, text = True, timeout = 600,
            )
            line = [l for l in proc.stdout.splitlines() if l.startswith("COLD ")]
            if line:
                rec = json.loads(line[-1][5:])
                result["cold"].setdefault(mode, []).append(rec)
            else:
                result["failures"].append({"phase": "cold", "mode": mode, "out": proc.stdout[-300:] + proc.stderr[-300:]})

    samples = {}
    arms = modes + (["full_again"] if "full" in modes else [])
    for tool, name, payload in work:
        for arm in arms:
            mode = "full" if arm == "full_again" else arm
            for _ in range(opts.warmup):
                call(tool, payload, mode, f"__LOCALID_bench_{arm}")
        for _ in range(opts.reps):
            for arm in arms:
                mode = "full" if arm == "full_again" else arm
                dt, ok, rmode, out = call(tool, payload, mode, f"__LOCALID_bench_{arm}")
                if ok:
                    samples.setdefault((tool, name, arm), []).append(dt)
                elif len(result["failures"]) < 40:
                    result["failures"].append({"tool": tool, "payload": name, "mode": arm, "record": rmode, "out": out})
        for arm in arms:
            result["warm"][f"{tool}:{name}:{arm}"] = summary(samples.get((tool, name, arm), []))
        print(json.dumps({k: v for k, v in result["warm"].items() if k.startswith(f"{tool}:{name}:")}), flush = True)

    with open(opts.out, "w") as fh:
        json.dump(result, fh, indent = 1)
    print("WROTE", opts.out)


if __name__ == "__main__":
    main()
