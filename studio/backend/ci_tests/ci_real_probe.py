"""Real (unmocked) probe of the PR #6509 hardware + /api/system code on the
bare CI runner. Runners have no GPU, so this exercises the genuine no-GPU /
CPU-only code path on the actual OS (Windows / macOS / Linux), proving the new
list contract and the psutil reads hold on each platform.
"""
import os, sys, platform, time

# studio/backend on path (workflow also sets PYTHONPATH, belt and suspenders)
HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import psutil
from utils.hardware import (
    get_device, get_gpu_utilization, get_visible_gpu_utilization,
    get_backend_visible_gpu_info,
)

fails = []
def check(c, m):
    print(("ok  " if c else "FAIL ") + m)
    if not c: fails.append(m)

print(f"OS={platform.system()} {platform.machine()} py={platform.python_version()}")
print(f"device={get_device()}")

# 1) get_gpu_utilization MUST be a list on every OS (new contract), even no-GPU.
gu = get_gpu_utilization()
check(isinstance(gu, list), f"get_gpu_utilization is list (got {type(gu).__name__})")
check(all(isinstance(d, dict) for d in gu), "all gu items are dicts")

# 2) get_visible_gpu_utilization -> dict with devices list.
vu = get_visible_gpu_utilization()
check(isinstance(vu, dict) and isinstance(vu.get("devices"), list),
      "get_visible_gpu_utilization is dict with devices list")

# 3) get_backend_visible_gpu_info -> dict with devices list.
vis = get_backend_visible_gpu_info()
check(isinstance(vis, dict) and isinstance(vis.get("devices"), list),
      "get_backend_visible_gpu_info is dict with devices list")

# 4) /api/system psutil reads work on this OS (the PR's new fields), guarded.
def guarded(fn):
    try: return fn()
    except Exception as e:
        print("   (probe raised, handled):", type(e).__name__); return None
mem = psutil.virtual_memory()
check(mem.total > 0, "virtual_memory().total > 0")
cpu_freq = guarded(psutil.cpu_freq)
disk = guarded(lambda: psutil.disk_usage(os.path.abspath(os.sep)))
boot = guarded(psutil.boot_time)
check(psutil.cpu_count(logical=True) is not None, "cpu_count(logical) available")
uptime = round(time.time() - boot) if boot else None
print(f"   cpu_freq={'set' if cpu_freq else None} disk={'set' if disk else None} uptime={uptime}")

print("\nRESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(1 if fails else 0)
