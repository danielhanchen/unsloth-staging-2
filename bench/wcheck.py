"""(d) Studio's converted transformer vs the Hub bf16 weights, tensor by tensor (relative Frobenius error).

python wcheck.py --converted DIR/transformer/0.safetensors --hf-dir SNAPSHOT/transformer --out out/wcheck_d4.json
"""

import argparse
import glob
import json
import os
import re

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--converted", required = True)
ap.add_argument("--hf-dir", required = True)
ap.add_argument("--out", required = True)
args = ap.parse_args()

studio = mx.load(args.converted)
hf = {}
for f in sorted(glob.glob(os.path.join(args.hf_dir, "*.safetensors"))):
    hf.update(mx.load(f))


def hf_name(name):
    k = name.replace("all_final_layer.2-1.adaLN_modulation.0.", "all_final_layer.2-1.adaLN_modulation.1.")
    k = k.replace("t_embedder.linear1.", "t_embedder.mlp.0.").replace("t_embedder.linear2.", "t_embedder.mlp.2.")
    return k


rows = []
bases = sorted({n[: -len(".scales")] for n in studio if n.endswith(".scales")})
quant = set(bases)
for base in bases:
    ref_name = hf_name(base + ".weight")
    if ref_name not in hf:
        rows.append({"name": base, "error": "no hf tensor"})
        continue
    ref = hf[ref_name].astype(mx.float32)
    w = studio[base + ".weight"]
    bits = w.shape[-1] * 32 // ref.shape[-1]
    deq = mx.dequantize(w, studio[base + ".scales"], studio[base + ".biases"], group_size = 32, bits = bits).astype(mx.float32)
    if deq.shape != ref.shape:
        rows.append({"name": base, "error": f"shape {deq.shape} vs {ref.shape}"})
        continue
    rel = float((mx.linalg.norm((deq - ref).reshape(-1)) / mx.maximum(mx.linalg.norm(ref.reshape(-1)), 1e-12)).item())
    rows.append({"name": base, "bits": int(bits), "rel": rel, "ref_max_abs": float(mx.max(mx.abs(ref)).item())})
for name, arr in studio.items():
    if any(name.startswith(b + ".") for b in quant) and re.search(r"\.(weight|scales|biases)$", name):
        continue
    ref_name = hf_name(name)
    if ref_name not in hf:
        rows.append({"name": name, "error": "no hf tensor"})
        continue
    ref = hf[ref_name].astype(mx.float32).reshape(-1)
    got = arr.astype(mx.float32).reshape(-1)
    if got.shape != ref.shape:
        rows.append({"name": name, "error": f"shape {arr.shape} vs {hf[ref_name].shape}"})
        continue
    rel = float((mx.linalg.norm(got - ref) / mx.maximum(mx.linalg.norm(ref), 1e-12)).item())
    rows.append({"name": name, "bits": 16, "rel": rel, "ref_max_abs": float(mx.max(mx.abs(ref)).item())})
missing = sorted(set(hf) - {hf_name(n) for n in studio} - {hf_name(b + ".weight") for b in bases})
ok = [r for r in rows if "rel" in r]
summary = {
    "n": len(rows), "errors": [r for r in rows if "error" in r][:20], "hf_not_covered": missing[:40],
    "worst": sorted(ok, key = lambda r: -r["rel"])[:25],
    "by_bits": {str(b): {"n": len(v), "median_rel": sorted(v)[len(v) // 2], "max_rel": max(v)}
                for b in sorted({r["bits"] for r in ok}) for v in [[r["rel"] for r in ok if r["bits"] == b]]},
}
json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent = 1)
print(json.dumps(summary, indent = 1)[:6000])
