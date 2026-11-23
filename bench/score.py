"""LPIPS-alex + PSNR of every render vs the same prompt's first uncached 25-step render (r1_off). CPU."""
import glob, json, os, sys
import lpips, numpy as np, torch
from PIL import Image

d = sys.argv[1]
net = lpips.LPIPS(net = "alex", verbose = False)
load = lambda p: np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
tt = lambda a: torch.from_numpy(a / 127.5 - 1.0).permute(2, 0, 1)[None]
rows = []
for f in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive = True)):
    try:
        rec = json.load(open(f))
    except Exception:
        continue
    if not isinstance(rec, dict) or "cells" not in rec:
        continue
    base = os.path.dirname(f)
    ref = next((c for c in rec["cells"] if c["label"] == "r1_off"), None)
    refimg = load(os.path.join(base, ref["image"])) if ref else None
    for c in rec["cells"]:
        row = {"tag": rec["tag"], "label": c["label"], "steps": c["steps"], "thr": c["threshold"], "wall_s": c["wall_s"],
               "s_per_step": c["timings"].get("s_per_step"), "skipped": (c.get("teacache") or {}).get("skipped"),
               "computed": (c.get("teacache") or {}).get("computed"), "mlx_peak_gib": round(c["mlx_peak_gib"], 3),
               "phys_peak_gib": round(c["phys_peak_gib"], 3), "sha": c["pixel_sha256"][:12]}
        if refimg is not None and c["label"] != "warmup":
            img = load(os.path.join(base, c["image"]))
            with torch.no_grad():
                row["lpips"] = round(float(net(tt(refimg), tt(img)).item()), 4)
            mse = float(np.mean((refimg.astype(np.float64) - img) ** 2))
            row["psnr"] = round(10 * np.log10(255.0**2 / max(mse, 1e-12)), 2)
        rows.append(row)
for r in rows:
    print(json.dumps(r))
json.dump(rows, open(os.path.join(d, "scores.json"), "w"), indent = 1)
