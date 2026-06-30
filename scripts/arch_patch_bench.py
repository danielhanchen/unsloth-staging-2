# SPDX-License-Identifier: AGPL-3.0-only
"""Isolate the per-arch fusions (diffusion_arch_patches): same `eager` tier, arch patches
ON vs OFF (UNSLOTH_DIFFUSION_ARCH_PATCHES). Measures their true end-to-end contribution +
PSNR(on vs off) as a corruption check, for the families that have a per-arch patch.

  python arch_patch_bench.py --model qwen-image --gpu 5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

UNSLOTH = Path("/mnt/disks/unslothai/ubuntu/workspace_81/unsloth")
BACKEND = UNSLOTH / "studio" / "backend"
sys.path.insert(0, str(UNSLOTH / "scripts"))
import benchall_orchestrator as bo  # noqa: E402

OUT = UNSLOTH / "outputs" / "arch_patch"
PROMPT = "a cat asleep on a stack of books, highly detailed"


def _wait_ready(backend, timeout_s=2400):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        p = backend.load_progress()
        if p.get("phase") == "ready":
            return
        if p.get("phase") == "error":
            raise RuntimeError(f"load error: {p.get('error')}")
        time.sleep(2)
    raise TimeoutError("load did not become ready")


def _measure(backend, repo, gf, base, steps, guidance, save_png, iters=6):
    import torch  # noqa: PLC0415
    backend.begin_load(repo, gguf_filename=gf, base_repo=base,
                       hf_token=os.environ.get("HF_TOKEN"), memory_mode="fast", speed_mode="eager")
    _wait_ready(backend)
    from core.inference import diffusion_arch_patches as ap  # noqa: PLC0415
    arch_on = ap.is_installed()
    backend.generate(prompt=PROMPT, width=1024, height=1024, steps=steps, guidance=guidance,
                     seed=bo.SEED, batch_size=1)  # warmup
    torch.cuda.synchronize()
    lat = []
    for _ in range(iters):
        torch.cuda.synchronize(); t = time.time()
        backend.generate(prompt=PROMPT, width=1024, height=1024, steps=steps, guidance=guidance,
                         seed=bo.SEED, batch_size=1)
        torch.cuda.synchronize(); lat.append(time.time() - t)
    res = backend.generate(prompt=PROMPT, width=1024, height=1024, steps=steps, guidance=guidance,
                           seed=bo.SEED, batch_size=1)
    res["images"][0].save(str(save_png))
    backend.unload()
    return statistics.median(lat), arch_on


def run_one(model: str, gpu: int) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    repo, gf, base, steps, guidance, _ = bo.MANIFEST[model]
    sys.path.insert(0, str(BACKEND))
    from core.inference.diffusion import get_diffusion_backend  # noqa: PLC0415
    d = OUT / model
    d.mkdir(parents=True, exist_ok=True)
    backend = get_diffusion_backend()

    os.environ["UNSLOTH_DIFFUSION_ARCH_PATCHES"] = "0"
    off_s, off_on = _measure(backend, repo, gf, base, steps, guidance, d / "arch_off.png")
    os.environ["UNSLOTH_DIFFUSION_ARCH_PATCHES"] = "1"
    on_s, on_on = _measure(backend, repo, gf, base, steps, guidance, d / "arch_on.png")

    psnr = bo._psnr(d / "arch_off.png", d / "arch_on.png")
    res = {
        "model": model, "gpu": gpu,
        "arch_off_s": round(off_s, 3), "arch_on_s": round(on_s, 3),
        "arch_on_engaged": on_on, "arch_off_engaged": off_on,
        "speedup": round(off_s / on_s, 3) if on_s else None,
        "pct": round((off_s / on_s - 1) * 100, 1) if on_s else None,
        "psnr_on_vs_off_dB": ("inf" if psnr == float("inf") else (round(psnr, 1) if psnr else None)),
    }
    (d / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--gpu", type=int, default=5)
    a = p.parse_args(argv)
    run_one(a.model, a.gpu)
    print("=== ARCH_PATCH_BENCH_DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
