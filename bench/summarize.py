import glob
import json
import math
import os
import statistics
import sys

import torch

out = sys.argv[1]
arms = {}
for p in sorted(glob.glob(os.path.join(out, "*.json"))):
    r = json.load(open(p))
    if "tag" in r:
        arms[r["tag"]] = r
ref = torch.load(os.path.join(out, "base_1.pt"))


def cmp(a, b):
    d = (a - b).abs()
    rng = float(b.max() - b.min())
    mse = float((d ** 2).mean())
    psnr = float("inf") if mse == 0 else 10 * math.log10(rng * rng / mse)
    return {"max_abs": float(d.max()), "psnr_db": psnr, "bit_equal_frac": float((d == 0).float().mean()),
            "rel_l2": float(d.norm() / b.norm())}


base_t = [arms[t]["median_s_per_block"] for t in arms if t.startswith("base")]
print(f"base A/A per-block medians: {[round(t, 4) for t in base_t]}  spread {(max(base_t) - min(base_t)) / statistics.median(base_t) * 100:.2f}%")
bmed = statistics.median(base_t)
rows = []
for tag, r in arms.items():
    t = torch.load(os.path.join(out, f"{tag}.pt"))
    c = cmp(t["decode"], ref["decode"])
    cp = cmp(t["prefill"], ref["prefill"])
    print(
        f"{tag:22s} installed={r['head_installed']!s:5s} s/block={r['median_s_per_block']:.4f} "
        f"({(r['median_s_per_block'] / bmed - 1) * 100:+.2f}% vs base median) est s/step(32)={r['est_s_per_step_32_blocks']:.2f} "
        f"peak_phys={r['lifetime_peak_phys_gib']:.2f}GiB  decode: max_abs={c['max_abs']:.3g} psnr={c['psnr_db']:.1f} "
        f"bit_eq={c['bit_equal_frac']:.4f} | prefill: max_abs={cp['max_abs']:.3g} bit_eq={cp['bit_equal_frac']:.4f}  "
        f"stats={r.get('stats')} probes={r.get('probes')}"
    )
    if r.get("ops"):
        for k, v in r["ops"].items():
            print(f"    op {k}: {v}")
