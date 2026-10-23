"""CPU mixed-precision probe: LoRA-shaped MLP fwd+bwd in fp32 vs bf16 / fp16 autocast, plus the oneDNN kernel each picks.

Answers "is bf16 / fp16 autocast faster than fp32 on this CPU" on staging runners. Pure torch, no Unsloth import.
Arms are interleaved over rounds so a noisy neighbour hits every arm alike. Writes one JSON (--out).
"""

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time

import torch

ARMS = {
    "fp32": None,
    "bf16_autocast": torch.bfloat16,
    "fp16_autocast": torch.float16,
}


def cpu_flags():
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
        m = re.search(r"^(?:flags|Features)\s*:\s*(.*)$", text, re.M)
        model = re.search(r"^(?:model name|CPU part)\s*:\s*(.*)$", text, re.M)
        return set(m.group(1).split()) if m else set(), model.group(1).strip() if model else ""
    except OSError:
        return set(), platform.processor()


def step(x0, w1, w2, a, b, ac):
    x = x0.clone().requires_grad_()
    ctx = (
        torch.autocast("cpu", dtype = ac) if ac is not None else torch.autocast("cpu", enabled = False)
    )
    with ctx:
        h = x @ w1.t() + (x @ a.t()) @ b.t()
        y = torch.nn.functional.silu(h) @ w2.t()
        loss = y.float().pow(2).mean()
    loss.backward()
    return float(loss)


def onednn_kernels(ac_name):
    code = (
        "import torch; a=torch.randn(256,256); b=torch.randn(256,256)\n"
        f"with torch.autocast('cpu', dtype=torch.{ac_name}): a@b"
    )
    env = dict(os.environ, ONEDNN_VERBOSE = "1")
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output = True, text = True, env = env, timeout = 300
    )
    return sorted(set(re.findall(r"exec,cpu,matmul,([^,]+)", out.stdout + out.stderr)))


def zoo_policy():
    """unsloth_zoo.cpu.runtime's answers, loaded as a standalone file (no package import, no device selection)."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("unsloth_zoo")
        path = os.path.join(list(spec.submodule_search_locations)[0], "cpu", "runtime.py")
        mspec = importlib.util.spec_from_file_location("_zoo_cpu_runtime", path)
        mod = importlib.util.module_from_spec(mspec)
        mspec.loader.exec_module(mod)
        out = {"bf16": mod.cpu_bf16_supported()}
        if hasattr(mod, "cpu_fp16_supported"):
            out["fp16"] = mod.cpu_fp16_supported()
            out["autocast_dtype"] = str(mod.cpu_autocast_dtype())
        return out
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action = "store_true")
    p.add_argument("--out", default = "job_cpu_dtype_bench.json")
    p.add_argument("--rounds", type = int, default = 3)
    args = p.parse_args()
    T, D, F = (512, 1024, 4096) if args.tiny else (2048, 2048, 8192)
    torch.manual_seed(0)
    x0 = torch.randn(T, D)
    w1 = torch.randn(F, D) * 0.02
    w2 = torch.randn(D, F) * 0.02
    a = (torch.randn(16, D) * 0.01).requires_grad_()
    b = (torch.randn(F, 16) * 0.01).requires_grad_()
    losses = {k: step(x0, w1, w2, a, b, ac) for k, ac in ARMS.items()}  # warmup
    times = {k: [] for k in ARMS}
    for _ in range(args.rounds):
        for k, ac in ARMS.items():
            t = time.perf_counter()
            for _ in range(3):
                step(x0, w1, w2, a, b, ac)
            times[k].append((time.perf_counter() - t) / 3)
    flags, model = cpu_flags()
    flops = 6 * T * D * F * 2
    base = statistics.median(times["fp32"])
    rows = {}
    for k in ARMS:
        m = statistics.median(times[k])
        rows[k] = {
            "median_ms": round(m * 1e3, 1),
            "min_ms": round(min(times[k]) * 1e3, 1),
            "max_ms": round(max(times[k]) * 1e3, 1),
            "tflops": round(flops / m / 1e12, 3),
            "speedup_vs_fp32": round(base / m, 2),
            "loss": losses[k],
        }
        print(f"{k:14s} {rows[k]}")
    result = {
        "job": "cpu_dtype_bench",
        "status": "ok",
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu": model,
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "isa": sorted(
            f
            for f in flags
            if f
            in {
                "avx2",
                "avx512f",
                "avx512_bf16",
                "avx512_fp16",
                "amx_bf16",
                "amx_fp16",
                "amx_tile",
                "avx_vnni",
                "bf16",
                "fphp",
                "asimdhp",
                "sve",
                "sve2",
                "i8mm",
            }
        ),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "shape": [T, D, F],
        "arms": rows,
        "onednn_kernels": {"bf16": onednn_kernels("bfloat16"), "fp16": onednn_kernels("float16")},
    }
    result["unsloth_zoo_policy"] = zoo_policy()
    print(json.dumps(result, indent = 1))
    with open(args.out, "w") as f:
        json.dump(result, f, indent = 1)


if __name__ == "__main__":
    main()
