# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LPIPS/PSNR of each external engine's image vs our dense bf16 reference.

Same prompt/seed/resolution across engines, so the distance measures perceptual drift
under matched user-facing settings (samplers still differ between engines, so this is
not a pure correctness metric -- pair with the saved images).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path("/mnt/disks/unslothai/ubuntu/workspace_81/unsloth/outputs/ext_engines")
MODELS = ["z-image", "flux.1", "qwen-image"]
ENGINES = ["vllm", "sglang"]


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)


def main():
    import lpips
    import torch

    fn = lpips.LPIPS(net = "alex", verbose = False).eval()

    def lp(a, b):
        def t(x):
            return torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        with torch.no_grad():
            return float(fn(t(a), t(b)).item())

    rows = []
    for model in MODELS:
        key = model.replace(".", "_")
        ref_path = OUT / f"ours_{key}.png"
        if not ref_path.is_file():
            print(f"no ref for {model}", flush = True)
            continue
        ref = np.array(Image.open(ref_path).convert("RGB"))
        for engine in ENGINES:
            img_path = OUT / f"{engine}_{key}.png"
            if not img_path.is_file():
                rows.append((engine, model, "MISSING", ""))
                continue
            img = np.array(Image.open(img_path).convert("RGB"))
            if img.shape != ref.shape:
                img = np.array(Image.open(img_path).convert("RGB").resize(
                    (ref.shape[1], ref.shape[0])))
            rows.append((engine, model, round(lp(ref, img), 4), round(_psnr(ref, img), 1)))

    with open(OUT / "lpips.csv", "w", newline = "") as f:
        w = csv.writer(f)
        w.writerow(["engine", "model", "lpips_vs_ours_bf16", "psnr_db"])
        for r in rows:
            w.writerow(r)
    for r in rows:
        print(f"LPIPS {r[0]:7s} {r[1]:12s} lpips={r[2]} psnr={r[3]}", flush = True)
    print("EXT-LPIPS-DONE", flush = True)


if __name__ == "__main__":
    main()
