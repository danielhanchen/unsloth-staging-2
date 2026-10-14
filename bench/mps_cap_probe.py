"""One arm of the MPS cap A/B: resolve Studio's diffusion target (which applies the arm's MPS policy), then fill MPS
with 512 MiB filled chunks up to physical RAM + EXTRA, record where (and how) it stops, and prove the process still
works after. Usage: mps_cap_probe.py <arm_root> <out.json> [extra_gib]"""
import json, os, re, subprocess, sys, time

root, out = sys.argv[1], sys.argv[2]
extra_gib = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
sys.path.insert(0, os.path.join(root, "studio", "backend"))
GIB = 1024**3


def compressor_mib():
    s = subprocess.run(["vm_stat"], capture_output = True, text = True).stdout
    page = re.search(r"page size of (\d+)", s)
    m = re.search(r"Pages occupied by compressor:\s+(\d+)", s)
    return int(m.group(1)) * int(page.group(1)) / 2**20 if m and page else None


def swap_used_mib():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output = True, text = True).stdout
    m = re.search(r"used = ([\d.]+)M", s)
    return float(m.group(1)) if m else None


import torch
from core.inference.diffusion_device import _mps_or_cpu_target
from core.inference.diffusion_batched import is_oom_error

tgt = _mps_or_cpu_target(torch)
memsize = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output = True, text = True).stdout)
res = {
    "arm": root, "torch": torch.__version__, "device": tgt.device, "dtype": str(tgt.dtype),
    "env_ratio": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
    "recommended_gib": torch.mps.recommended_max_memory() / GIB, "memsize_gib": memsize / GIB,
    "swap_before_mib": swap_used_mib(), "compressor_before_mib": compressor_mib(),
}
assert tgt.device == "mps", res
chunk = 512 * 1024**2
limit = memsize + int(extra_gib * GIB)
chunks, err, oom = [], None, None
t0 = time.time()
peak_driver = 0
try:
    while len(chunks) * chunk < limit and time.time() - t0 < 420:
        chunks.append(torch.rand(chunk // 2, dtype = torch.bfloat16, device = "mps"))  # incompressible
        torch.mps.synchronize()
        peak_driver = max(peak_driver, torch.mps.driver_allocated_memory())
except Exception as exc:  # noqa: BLE001
    err = f"{type(exc).__name__}: {str(exc)[:400]}"
    oom = is_oom_error(exc)
res.update(
    fill_s = round(time.time() - t0, 2), allocated_gib = len(chunks) * chunk / GIB, reached_limit = err is None,
    limit_gib = limit / GIB, peak_driver_gib = peak_driver / GIB, error = err, studio_classifies_oom = oom,
    swap_after_fill_mib = swap_used_mib(), compressor_after_fill_mib = compressor_mib(),
)
del chunks
torch.mps.empty_cache()
# The process must still work after the overrun.
a = torch.randn(4096, 4096, device = "mps", dtype = torch.bfloat16)
res["post_matmul_finite"] = bool(torch.isfinite((a @ a).float()).all().item())
del a
print(json.dumps(res, indent = 1))
json.dump(res, open(out, "w"), indent = 1)
