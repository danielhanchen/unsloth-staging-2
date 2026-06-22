"""
Backend hardware compatibility simulation for unsloth PR #6509.

Exercises studio/backend/utils/hardware/hardware.py across the cartesian product
of [Windows, Linux, WSL, Mac] x [NVIDIA, AMD(ROCm), CPU, MLX, XPU(Intel)] with
both SMI-available and SMI-fallback variants, by mocking the device selector and
every low-level probe. Verifies:

  1. get_gpu_utilization() ALWAYS returns a list[dict] (the new contract) and
     never the old single dict, on every path -> old consumers that now expect a
     list are not surprised by a stray dict.
  2. get_visible_gpu_utilization() always returns {available, devices: list}.
  3. The /api/system enrichment loop (copied verbatim from main.py) never raises
     and produces numeric vram_free_gb / vram_utilization handling, including the
     total_vram == 0 (no GPU / partial telemetry) edge case.
  4. Each device dict carries the keys the frontend reads.

No real GPU/torch is used; everything is mocked, so it runs in an isolated venv.
"""
import sys, traceback
from contextlib import ExitStack
from unittest import mock

import utils.hardware as H
from utils.hardware import hardware as hw

DeviceType = hw.DeviceType

PASS, FAIL = 0, 0
FAILURES = []

def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(msg)
        print(f"   FAIL: {msg}")

# Frontend-required keys on an "available" utilization device.
DEV_KEYS = ["vram_used_gb", "vram_total_gb", "vram_utilization_pct", "backend"]

def smi_multi(n=2, total=80.0):
    """Mock nvidia-smi / rocm-smi multi-GPU result for get_visible_gpu_utilization."""
    return {
        "available": True,
        "index_kind": "physical",
        "devices": [
            {
                "index": i, "index_kind": "physical", "visible_ordinal": i,
                "gpu_utilization_pct": 10.0 * i, "temperature_c": 40 + i,
                "vram_used_gb": 1.5 * (i + 1), "vram_total_gb": total,
                "vram_utilization_pct": round(1.5 * (i + 1) / total * 100, 1),
                "power_draw_w": 100.0, "power_limit_w": 300.0, "power_utilization_pct": 33.3,
            }
            for i in range(n)
        ],
    }

def torch_devs(specs):
    """Mock _torch_get_per_device_info output. specs: list of (index, used, total)."""
    return [
        {"index": idx, "visible_ordinal": idx, "name": f"GPU{idx}", "used_gb": u, "total_gb": t}
        for (idx, u, t) in specs
    ]

def visibility(n=2, total=80.0, available=True):
    """Mock get_backend_visible_gpu_info (the base device list /api/system enriches)."""
    return {
        "available": available,
        "devices": [
            {"index": i, "index_kind": "physical", "visible_ordinal": i,
             "name": f"GPU{i}", "memory_total_gb": total}
            for i in range(n)
        ] if available else [],
    }

# /api/system enrichment loop, copied VERBATIM from main.py get_system_info (post-fix).
def api_system_enrich(visibility_info, utilization_info):
    util_devices = {d.get("index"): d for d in utilization_info.get("devices", [])}
    enriched_devices = []
    for dev in visibility_info.get("devices", []):
        idx = dev.get("index")
        util = util_devices.get(idx, {})
        total_vram = util.get("vram_total_gb") or dev.get("memory_total_gb") or 0
        used_vram = util.get("vram_used_gb") or 0
        enriched_dev = dict(dev)
        enriched_dev["vram_used_gb"] = used_vram
        enriched_dev["vram_free_gb"] = round(total_vram - used_vram, 2) if total_vram else 0
        enriched_dev["vram_utilization_pct"] = util.get("vram_utilization_pct")
        enriched_devices.append(enriched_dev)
    return {"available": visibility_info.get("available", False), "devices": enriched_devices}


def run_case(name, *, device, is_rocm=False, system="Linux",
             smi=None, torch_info=None, parent_ids=None, phys_count=0,
             rocm_win=(None, None), rocm_lin=(None, None),
             apple=None, gpu_mem=None, vis_n=2, vis_total=80.0, vis_available=True):
    print(f"\n[{name}]")
    with ExitStack() as es:
        p = lambda attr, val: es.enter_context(mock.patch.object(hw, attr, val))
        p("get_device", lambda: device)
        p("IS_ROCM", is_rocm)
        # platform.system controls ROCm Win/Linux sub-branch + apple check
        es.enter_context(mock.patch.object(hw.platform, "system", lambda: system))
        p("_smi_query", lambda *a, **k: smi)
        p("_get_parent_visible_gpu_spec", lambda: {"numeric_ids": parent_ids or [], "raw": None})
        p("get_parent_visible_gpu_ids", lambda: parent_ids or [])
        p("_torch_get_physical_gpu_count", lambda: phys_count)
        p("_torch_get_per_device_info", lambda ids: torch_info or [])
        p("_reconcile_rocm_unified_memory", lambda *a, **k: None)
        p("_rocm_windows_perf_counter_vram_gb", lambda: rocm_win)
        p("_rocm_windows_perf_counter_gpu_util_pct", lambda: 12.0)
        p("_rocm_linux_sysfs_vram_gb", lambda: rocm_lin)
        p("_rocm_linux_sysfs_gpu_busy_pct", lambda: 12.0)
        p("_rocm_linux_sysfs_temp_c", lambda: 50.0)
        p("_rocm_linux_sysfs_power_w", lambda: 80.0)
        p("_read_apple_gpu_stats", lambda: apple)
        p("get_gpu_memory_info", lambda: gpu_mem or {"available": False})

        # 1) get_gpu_utilization -> must be a list
        try:
            gu = hw.get_gpu_utilization()
        except Exception:
            check(False, f"{name}: get_gpu_utilization raised\n{traceback.format_exc()}")
            return
        check(isinstance(gu, list), f"{name}: get_gpu_utilization returned {type(gu).__name__}, expected list")
        if isinstance(gu, list):
            for d in gu:
                check(isinstance(d, dict), f"{name}: gu element not a dict: {d!r}")
                if d.get("available"):
                    for k in DEV_KEYS:
                        check(k in d, f"{name}: available gu device missing key '{k}'")

        # 2) get_visible_gpu_utilization -> dict with devices list
        try:
            vu = hw.get_visible_gpu_utilization()
        except Exception:
            check(False, f"{name}: get_visible_gpu_utilization raised\n{traceback.format_exc()}")
            return
        check(isinstance(vu, dict), f"{name}: get_visible returned {type(vu).__name__}, expected dict")
        check(isinstance(vu.get("devices"), list), f"{name}: get_visible 'devices' not a list")
        check("available" in vu, f"{name}: get_visible missing 'available'")

        # 3) /api/system enrichment must not raise, vram_free_gb numeric
        try:
            res = api_system_enrich(visibility(vis_n, vis_total, vis_available), vu)
        except Exception:
            check(False, f"{name}: /api/system enrichment raised\n{traceback.format_exc()}")
            return
        for d in res["devices"]:
            check(isinstance(d.get("vram_free_gb"), (int, float)),
                  f"{name}: vram_free_gb not numeric: {d.get('vram_free_gb')!r}")
        print(f"   ok  gu(list,len={len(gu)}) visible(devices={len(vu.get('devices',[]))}) "
              f"api_system(devices={len(res['devices'])})")


OSES = ["Linux", "Windows", "WSL"]  # WSL maps to Linux code paths
def sysname(o):
    return "Linux" if o == "WSL" else o

# ---- NVIDIA via nvidia-smi (success) ----
for o in OSES:
    run_case(f"NVIDIA smi {o}", device=DeviceType.CUDA, is_rocm=False, system=sysname(o),
             smi=smi_multi(2), parent_ids=[0, 1])

# ---- NVIDIA smi unavailable -> torch fallback ----
for o in OSES:
    run_case(f"NVIDIA torch-fallback {o}", device=DeviceType.CUDA, is_rocm=False, system=sysname(o),
             smi=None, torch_info=torch_devs([(0, 1.0, 80.0), (1, 2.0, 80.0)]), parent_ids=[0, 1])

# ---- NVIDIA smi unavailable + torch empty -> unavailable ----
run_case("NVIDIA nothing", device=DeviceType.CUDA, is_rocm=False, system="Linux",
         smi=None, torch_info=[], phys_count=0, parent_ids=[])

# ---- NVIDIA torch fallback with total_vram == 0 (no div-by-zero) ----
run_case("NVIDIA torch total=0", device=DeviceType.CUDA, is_rocm=False, system="Linux",
         smi=None, torch_info=torch_devs([(0, 1.0, 0.0)]), parent_ids=[0])

# ---- AMD ROCm via rocm-smi (success) ----
run_case("AMD rocm-smi Linux", device=DeviceType.CUDA, is_rocm=True, system="Linux",
         smi=smi_multi(2), parent_ids=[0, 1])
run_case("AMD rocm-smi Windows", device=DeviceType.CUDA, is_rocm=True, system="Windows",
         smi=smi_multi(1), parent_ids=[0])

# ---- AMD ROCm smi unavailable -> sysfs/perf fallbacks ----
run_case("AMD sysfs Linux", device=DeviceType.CUDA, is_rocm=True, system="Linux",
         smi=None, rocm_lin=(2.0, 16.0))
run_case("AMD perf Windows", device=DeviceType.CUDA, is_rocm=True, system="Windows",
         smi=None, rocm_win=(2.0, 16.0))
run_case("AMD WSL sysfs", device=DeviceType.CUDA, is_rocm=True, system="Linux",
         smi=None, rocm_lin=(3.0, 24.0))
# ROCm smi + sysfs both unavailable -> torch fallback -> unavailable
run_case("AMD nothing Linux", device=DeviceType.CUDA, is_rocm=True, system="Linux",
         smi=None, rocm_lin=(None, None), torch_info=[], phys_count=0, parent_ids=[])

# ---- MLX (Mac) ----
run_case("MLX Mac (stats)", device=DeviceType.MLX, system="Darwin",
         apple={"vram_used_bytes": 8 * 1024**3, "utilization_pct": 22.0},
         gpu_mem={"available": True, "allocated_gb": 8.0, "total_gb": 64.0, "utilization_pct": 12.5},
         vis_n=1, vis_total=64.0)
run_case("MLX Mac (no stats)", device=DeviceType.MLX, system="Darwin",
         apple=None, gpu_mem={"available": False}, vis_n=0, vis_available=False)

# ---- XPU (Intel) Linux/Windows ----
run_case("XPU Linux", device=DeviceType.XPU, system="Linux",
         torch_info=torch_devs([(0, 1.0, 16.0)]), parent_ids=[0],
         gpu_mem={"available": True, "allocated_gb": 1.0, "total_gb": 16.0, "utilization_pct": 6.25},
         vis_n=1, vis_total=16.0)
run_case("XPU Windows", device=DeviceType.XPU, system="Windows",
         torch_info=torch_devs([(0, 1.0, 16.0)]), parent_ids=[0],
         gpu_mem={"available": True, "allocated_gb": 1.0, "total_gb": 16.0, "utilization_pct": 6.25},
         vis_n=1, vis_total=16.0)

# ---- CPU-only (all OSes) ----
for o in ["Linux", "Windows", "Darwin"]:
    run_case(f"CPU-only {o}", device=DeviceType.CPU, system=o,
             gpu_mem={"available": False}, vis_n=0, vis_available=False)

print("\n" + "=" * 60)
print(f"BACKEND HARDWARE MATRIX: {PASS} checks passed, {FAIL} failed")
if FAILURES:
    print("FAILED:")
    for f in FAILURES:
        print(" -", f.splitlines()[0])
sys.exit(1 if FAIL else 0)
