"""Harness only: sandbox probe cost and first-call vs warm-call tool latency, with and without warm-up.

Each arm runs in a fresh child process so no cache carries over. Payloads are harmless (print, echo).
Usage: python sbv2_perf.py --backend-dir <repo>/studio/backend [--reps 5] [--rounds 3] [--json out.json]
Arms per round, interleaved:
  cold_probe   one forced sandbox probe (Linux/macOS) or MXC capability (Windows), timed
  no_warmup    first python + terminal call pays the probe, then --reps warm calls each
  warmup       warm_tool_isolation() first (timed), then the same calls
  full         the same calls with disable_sandbox=True (no OS sandbox: the AST/blocklist-only baseline)
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

CHILD = r'''
import json, os, sys, time
sys.path.insert(0, os.environ["SBV2_BACKEND_DIR"])
arm, reps = sys.argv[1], int(sys.argv[2])
from core.inference import os_sandbox, tools
import routes  # noqa: F401 - loaded at server start; the first tool call would otherwise pay this import
out = {"arm": arm, "platform": sys.platform}
if sys.platform == "win32":
    from core.inference import mxc_policy
    out["dacl_fallback_enabled"] = mxc_policy.dacl_fallback_enabled()
    out["studio_home"] = os.environ.get("UNSLOTH_STUDIO_HOME")

def call(tool, disable):
    tools._last_tool_execution_record = None
    args = {"code": "print('SB_OK')"} if tool == "python" else {"command": "echo SB_OK"}
    t0 = time.perf_counter()
    text = tools.execute_tool(tool, args, session_id = "__LOCALID_sbv2_perf", timeout = 120,
                              disable_sandbox = disable) or ""
    ms = (time.perf_counter() - t0) * 1000
    rec = tools._last_tool_execution_record
    return {"ms": ms, "ok": "SB_OK" in text, "mode": getattr(rec, "effective_mode", None),
            "backend": getattr(rec, "backend", None)}

if arm == "cold_probe":
    t0 = time.perf_counter()
    cap = os_sandbox.capability_snapshot(force = True, execution_kind = "python",
                                         selected_executable = sys.executable)
    out.update(probe_ms = (time.perf_counter() - t0) * 1000, available = cap.available,
               backend = cap.backend, reason = cap.reason[:200])
else:
    if arm == "warmup":
        t0 = time.perf_counter()
        os_sandbox.warm_tool_isolation()
        out["warmup_ms"] = (time.perf_counter() - t0) * 1000
    disable = arm == "full"
    for tool in ("python", "terminal"):
        first = call(tool, disable)
        rest = [call(tool, disable) for _ in range(reps)]
        out[tool] = {"first": first, "warm": rest}
print("SBV2_RESULT " + json.dumps(out), flush = True)
'''


def run_child(backend_dir, arm, reps, home):
    env = dict(os.environ)
    env.update(
        SBV2_BACKEND_DIR = backend_dir,
        UNSLOTH_STUDIO_HOME = home,
        # The server warm-up is what this measures by hand; nothing else may probe behind it.
        UNSLOTH_DISABLE_SANDBOX_WARMUP = "1",
        PYTHONUNBUFFERED = "1",
    )
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, arm, str(reps)],
        env = env,
        cwd = backend_dir,
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        timeout = 900,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("SBV2_RESULT "):
            return json.loads(line[len("SBV2_RESULT "):])
    return {"arm": arm, "error": (proc.stderr or proc.stdout)[-1500:], "returncode": proc.returncode}


def med(values):
    return round(statistics.median(values), 1) if values else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-dir", required = True)
    ap.add_argument("--reps", type = int, default = 5)
    ap.add_argument("--rounds", type = int, default = 3)
    ap.add_argument("--json", default = "")
    args = ap.parse_args()
    backend_dir = os.path.abspath(args.backend_dir)
    results = []
    with tempfile.TemporaryDirectory(prefix = "sbv2_perf_") as home:
        for round_index in range(args.rounds):
            for arm in ("cold_probe", "no_warmup", "warmup", "full"):
                result = run_child(backend_dir, arm, args.reps, home)
                result["round"] = round_index
                results.append(result)
                print(json.dumps(result)[:600], flush = True)

    errors = [r for r in results if "error" in r]
    probe = [r["probe_ms"] for r in results if r.get("arm") == "cold_probe" and "probe_ms" in r]
    print("\n| metric | median ms |")
    print("|---|---|")
    print(f"| cold probe (python capability) | {med(probe)} |")
    for arm in ("no_warmup", "warmup", "full"):
        rows = [r for r in results if r.get("arm") == arm and "python" in r]
        if arm == "warmup":
            print(f"| warm-up itself (both tools) | {med([r['warmup_ms'] for r in rows])} |")
        for tool in ("python", "terminal"):
            first = [r[tool]["first"]["ms"] for r in rows]
            warm = [w["ms"] for r in rows for w in r[tool]["warm"]]
            modes = sorted({str(r[tool]["first"]["mode"]) for r in rows})
            print(f"| {arm} {tool} first call ({', '.join(modes)}) | {med(first)} |")
            print(f"| {arm} {tool} warm call | {med(warm)} |")
    oks = [
        c["ok"]
        for r in results
        for tool in ("python", "terminal")
        if tool in r
        for c in [r[tool]["first"], *r[tool]["warm"]]
    ]
    print(f"\ncalls ok: {sum(oks)}/{len(oks)}; child errors: {len(errors)}")
    for error in errors[:3]:
        print("ERROR", error["arm"], error["error"][-600:])
    if args.json:
        with open(args.json, "w", encoding = "utf-8") as handle:
            json.dump(results, handle, indent = 1)
    return 1 if errors or not all(oks) else 0


if __name__ == "__main__":
    sys.exit(main())
