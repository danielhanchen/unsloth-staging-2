"""PR 13776 probe: real Metal working set on an Apple Silicon runner, base formula vs head snapshot budget.

Run from the repo root. Writes --out JSON. Exits 1 when the head reading did not engage on a real MPS device."""
import argparse
import json
import platform
import sys
import types

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="mps_budget_probe.json")
ap.add_argument("--tiny", action="store_true")
args = ap.parse_args()
sys.path.insert(0, "studio/backend")

import torch  # noqa: E402
import core.inference.diffusion_memory as dm  # noqa: E402

MIB = 1024 * 1024
res = {"platform": platform.platform(), "machine": platform.machine(), "torch": torch.__version__,
       "mps_available": bool(torch.backends.mps.is_available())}
total, avail = dm._system_memory_mib()
res.update(ram_total_mib = total, ram_available_mib = avail)
res["base_formula_budget_mib"] = int(avail) - max(2048, int(total * 0.20))
if res["mps_available"]:
    res["recommended_max_memory_mib"] = torch.mps.recommended_max_memory() / MIB
    res["driver_allocated_memory_mib"] = torch.mps.driver_allocated_memory() / MIB
    mem = dm.snapshot_device_memory(types.SimpleNamespace(device = "mps", backend = "mps"))
    res["head_snapshot"] = {"free_mib": mem.free_mib, "total_mib": mem.total_mib,
                            "metal_working_set_mib": getattr(mem, "metal_working_set_mib", None)}
    res["head_budget_mib"] = dm._safe_device_budget_mib(mem)
    # Allocate the head budget on MPS in 256 MiB chunks and touch it.
    chunks, ok, err = [], 0, None
    try:
        for _ in range(max(0, int(res["head_budget_mib"]) // 256)):
            t = torch.empty(256 * MIB // 4, dtype = torch.float32, device = "mps")
            t.fill_(1.0)
            chunks.append(t)
            ok += 256
        torch.mps.synchronize()
        res["sum_check"] = float(sum(float(c[:1].sum()) for c in chunks))
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
    res.update(allocated_mib = ok, alloc_error = err,
               driver_allocated_after_mib = torch.mps.driver_allocated_memory() / MIB)
print(json.dumps(res, indent = 1))
with open(args.out, "w") as f:
    json.dump(res, f, indent = 1)
engaged = res.get("head_snapshot", {}).get("metal_working_set_mib") is not None
print("ENGAGED", engaged)
