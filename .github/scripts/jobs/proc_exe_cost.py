"""Read-only: cost of psutil exe lookup in a full process scan. Kills nothing.
Usage: python exe_cost_probe.py [--repeats 5] [--out FILE] [--tiny]  -> one JSON line (also to --out).
Drop-in for staging_ci.py --job (accepts and ignores --tiny)."""

import argparse, json, platform, statistics, subprocess, sys, time

try:
    import psutil
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "psutil"], check = True)
    import psutil

ap = argparse.ArgumentParser()
ap.add_argument("--repeats", type = int, default = 5)
ap.add_argument("--out")
ap.add_argument("--tiny", action = "store_true")
ARGS, _ = ap.parse_known_args()
REPEATS = ARGS.repeats


def scan_name_only():
    n = 0
    for p in psutil.process_iter(["pid", "name"]):
        n += 1
        (p.info.get("name") or "").lower().startswith("llama-server")
    return n


def scan_with_exe():
    n = 0
    for p in psutil.process_iter(["pid", "name", "exe"]):
        n += 1
        (p.info.get("name") or "").lower().startswith("llama-server")
    return n


def lazy_exe():  # the new code shape: exe only for name matches
    n = 0
    for p in psutil.process_iter(["pid", "name"]):
        n += 1
        if (p.info.get("name") or "").lower().startswith("llama-server"):
            try:
                p.exe()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    return n


fns = {"name_only": scan_name_only, "with_exe_base": scan_with_exe, "lazy_exe_head": lazy_exe}
scan_with_exe()
scan_name_only()  # warm the process_iter cache equally
# interleave arms so host load drifts hit all three
res = {k: [] for k in fns}
for _ in range(REPEATS):
    for k, fn in fns.items():
        t = time.perf_counter()
        c = fn()
        res[k].append(((time.perf_counter() - t) * 1000, c))
out = {
    "os": platform.platform(),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "psutil": psutil.__version__,
    "repeats": REPEATS,
}
for k, v in res.items():
    ts = [t for t, _ in v]
    out[k] = {
        "median_ms": round(statistics.median(ts), 3),
        "min_ms": round(min(ts), 3),
        "max_ms": round(max(ts), 3),
        "procs": v[-1][1],
    }
out["saved_ms_median"] = round(
    out["with_exe_base"]["median_ms"] - out["lazy_exe_head"]["median_ms"], 3
)
line = json.dumps(out)
print(line)
if ARGS.out:
    with open(ARGS.out, "w", encoding = "utf-8") as f:
        f.write(line + "\n")
