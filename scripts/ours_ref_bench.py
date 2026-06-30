# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Our diffusers dense-bf16 baseline + LPIPS reference for the external-engine bench.

For each model, loads the base diffusers pipeline in bf16 (resident), does a warmup
then a timed generate at the SAME prompt/seed/resolution/steps the external engines
use, saves the reference image, and reports warm latency + peak VRAM. The saved image
is the bf16 ground truth for LPIPS in the SUMMARY. This is the naive (no GGUF/compile/
quant) baseline; our optimized numbers come from outputs/threeway.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

OUT = Path("/mnt/disks/unslothai/ubuntu/workspace_81/unsloth/outputs/ext_engines")
PROMPT = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed"

# model -> (pipeline_class, base_repo, steps, cfg_kwarg, guidance)
MODELS = {
    "z-image": ("ZImagePipeline", "Tongyi-MAI/Z-Image-Turbo", 8, "guidance_scale", 0.0),
    "flux.1": ("FluxPipeline", "black-forest-labs/FLUX.1-schnell", 4, "guidance_scale", 0.0),
    "qwen-image": ("QwenImagePipeline", "Qwen/Qwen-Image", 20, "true_cfg_scale", 4.0),
}


def run(model: str) -> dict:
    import diffusers
    import torch

    cls_name, base, steps, cfg_kwarg, guidance = MODELS[model]
    PipeCls = getattr(diffusers, cls_name)
    pipe = PipeCls.from_pretrained(base, torch_dtype = torch.bfloat16).to("cuda")
    torch.cuda.reset_peak_memory_stats()

    def _gen(seed):
        g = torch.Generator(device = "cuda").manual_seed(seed)
        kw = {"prompt": PROMPT, "width": 1024, "height": 1024, "num_inference_steps": steps,
              cfg_kwarg: guidance, "generator": g}
        with torch.inference_mode():
            return pipe(**kw).images[0]

    _gen(1)  # warmup
    torch.cuda.synchronize()
    t0 = time.time()
    img = _gen(1234)
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9

    OUT.mkdir(parents = True, exist_ok = True)
    img.save(OUT / f"ours_{model.replace('.', '_')}.png")
    row = dict(engine = "ours-bf16", model = model, steps = steps,
               total_s = round(dt, 3), peak_gb = round(peak, 2))
    print("RESULT", row, flush = True)
    del pipe
    torch.cuda.empty_cache()
    return row


def main(argv = None):
    p = argparse.ArgumentParser()
    p.add_argument("--models", default = "z-image,flux.1,qwen-image")
    args = p.parse_args(argv)
    for m in args.models.split(","):
        print(f"\n===== ours-bf16 / {m} =====", flush = True)
        try:
            run(m.strip())
        except Exception as e:  # noqa: BLE001
            print(f"RESULT {{'engine':'ours-bf16','model':'{m}','status':'ERR','reason':'{type(e).__name__}: {str(e)[:160]}'}}", flush = True)
    print("OURS-REF-BENCH-DONE", flush = True)


if __name__ == "__main__":
    main()
