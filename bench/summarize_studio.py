import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image

out = sys.argv[1]
ref = np.asarray(Image.open(os.path.join(out, "s_base_1.png")).convert("RGB")).astype(np.float64)
for p in sorted(glob.glob(os.path.join(out, "s_*.json"))):
    r = json.load(open(p))
    tag = r["tag"]
    png = os.path.join(out, f"{tag}.png")
    cmp = ""
    if os.path.exists(png):
        im = np.asarray(Image.open(png).convert("RGB")).astype(np.float64)
        d = np.abs(im - ref)
        mse = float((d ** 2).mean())
        psnr = float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse)
        cmp = f"vs s_base_1: max_abs={d.max():.0f} psnr={psnr:.1f} dB equal_px={(d == 0).mean():.4f}"
    gens = r.get("generations", [])
    print(f"{tag}: status={r.get('status')} transformer_device={r.get('transformer_device')} blocks_patched={r.get('blocks_patched')} "
          f"attn_prepare={r.get('attn_prepare_set')} processor={r.get('processor')} probes={r.get('probes')}")
    print(f"    gen s={[g['s'] for g in gens]} stats_delta={[g['stats_delta'] for g in gens]} {cmp}")
