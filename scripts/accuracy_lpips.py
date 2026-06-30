# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Accuracy of the speed optimizations: LPIPS of GGUF / fp8 / int8 vs the dense bf16 reference.

Enforces the standing-goal rule (a 2x speedup costing <=25% accuracy is a win; ~50% is bad). For each
family it loads the diffusers pipeline, generates a dense-bf16 reference (the ground truth), then loads
each optimized transformer (GGUF single-file, or dense + real quantize_transformer fp8/int8) into the
SAME pipeline and measures LPIPS (0 = identical) and PSNR against the reference at a fixed prompt/seed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent / "studio" / "backend"
PROMPT = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed"

# family -> (pipeline_class, transformer_class, base_repo, gguf_repo_dir, gguf_name, steps, guidance)
FAMILIES = {
    "flux.1": ("FluxPipeline", "FluxTransformer2DModel", "black-forest-labs/FLUX.1-schnell",
               "flux1-schnell-Q4_K_M.gguf", 4, 0.0),
    "z-image": ("ZImagePipeline", "ZImageTransformer2DModel", "Tongyi-MAI/Z-Image-Turbo",
                "z-image-turbo-Q4_K_M.gguf", 8, 0.0),
}
GGUF_DIR = Path("/mnt/disks/unslothai/ubuntu/workspace_81/unsloth/outputs/sdcpp_cpu/gguf")
OUT = Path("/mnt/disks/unslothai/ubuntu/workspace_81/unsloth/outputs/accuracy")


def _gen(pipe, steps, guidance, seed, w, h):
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    img = pipe(prompt=PROMPT, width=w, height=h, num_inference_steps=steps,
               guidance_scale=guidance, generator=g).images[0]
    return np.array(img)


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)


def _lpips_fn():
    import lpips, torch
    fn = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    def f(a, b):
        def t(x):
            return (torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0).cuda()
        with torch.no_grad():
            return float(fn(t(a), t(b)).item())
    return f


def run_family(fam, w, h, seed):
    sys.path.insert(0, str(BACKEND))
    import types
    import torch
    import diffusers
    from diffusers import GGUFQuantizationConfig
    from core.inference.diffusion_transformer_quant import quantize_transformer

    pipe_cls_name, tr_cls_name, base, gguf_name, steps, guidance = FAMILIES[fam]
    PipeCls = getattr(diffusers, pipe_cls_name)
    TrCls = getattr(diffusers, tr_cls_name)
    lp = _lpips_fn()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    def fresh_pipe(transformer=None):
        kw = {"torch_dtype": torch.bfloat16}
        if transformer is not None:
            kw["transformer"] = transformer
        p = PipeCls.from_pretrained(base, **kw)
        p.to("cuda")
        return p

    # 1) dense bf16 reference
    print(f"[{fam}] loading dense bf16 reference ...", flush=True)
    pipe = fresh_pipe()
    t0 = time.time(); ref = _gen(pipe, steps, guidance, seed, w, h); ref_s = time.time() - t0
    from PIL import Image
    Image.fromarray(ref).save(OUT / f"{fam.replace('.', '_')}_bf16.png")
    rows.append((fam, "bf16(ref)", 0.0, float("inf"), round(ref_s, 2)))
    del pipe; torch.cuda.empty_cache()

    # 2) GGUF (Q4) single-file
    try:
        print(f"[{fam}] GGUF ...", flush=True)
        tr = TrCls.from_single_file(str(GGUF_DIR / gguf_name),
                                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                                    torch_dtype=torch.bfloat16)
        pipe = fresh_pipe(tr.to("cuda"))
        t0 = time.time(); img = _gen(pipe, steps, guidance, seed, w, h); s = time.time() - t0
        rows.append((fam, "gguf-q4", lp(ref, img), _psnr(ref, img), round(s, 2)))
        del pipe, tr; torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        rows.append((fam, "gguf-q4", f"ERR:{type(e).__name__}", "", ""))

    # 3) dense + fp8 / int8 (real quantize_transformer)
    for scheme in ("fp8", "int8"):
        try:
            print(f"[{fam}] {scheme} ...", flush=True)
            tr = TrCls.from_pretrained(base, subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda")
            pipe = fresh_pipe(tr)
            eng = quantize_transformer(pipe, types.SimpleNamespace(device="cuda", dtype=torch.bfloat16), mode=scheme)
            t0 = time.time(); img = _gen(pipe, steps, guidance, seed, w, h); s = time.time() - t0
            rows.append((fam, f"{scheme}({eng})", lp(ref, img), _psnr(ref, img), round(s, 2)))
            del pipe, tr; torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            rows.append((fam, scheme, f"ERR:{type(e).__name__}:{str(e)[:60]}", "", ""))

    for r in rows:
        lpv = f"{r[2]:.4f}" if isinstance(r[2], float) else r[2]
        psv = f"{r[3]:.1f}" if isinstance(r[3], float) and r[3] != float("inf") else ("inf" if r[3] == float("inf") else r[3])
        print(f"RESULT fam={r[0]} cfg={r[1]} lpips={lpv} psnr={psv} gen_s={r[4]}", flush=True)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--families", default="flux.1,z-image")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args(argv)
    allrows = []
    for fam in args.families.split(","):
        allrows += run_family(fam.strip(), args.width, args.height, args.seed)
    import csv
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "results.csv", "w", newline="") as f:
        wt = csv.writer(f); wt.writerow(["family", "config", "lpips_vs_bf16", "psnr_vs_bf16", "gen_s"])
        for r in allrows:
            wt.writerow(r)
    print(f"wrote {OUT/'results.csv'}", flush=True)
    print("ACCURACY-LPIPS-DONE", flush=True)


if __name__ == "__main__":
    main()
