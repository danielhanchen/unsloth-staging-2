"""Which BF16 GGUF tensors break Z-Image base at 4 bits: load Studio's dense-bf16 (d0) build once, then requantize one
tensor group at a time (mx.quantize group 32, the converter's min/max affine) and probe the velocity at sigma 1 and
0.5 against the unquantized build. Restores the bf16 Linear after every probe.

UNSLOTH_DIFFUSION_MLX_DENSE_BITS=0 UNSLOTH_DIFFUSION_MLX_PRECISION=bf16 python zb_ablate.py --backend-dir ... --out out
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend-dir", required = True)
ap.add_argument("--out", required = True)
ap.add_argument("--size", type = int, default = 512)
ap.add_argument("--prompt", default = "A red apple on a wooden table, soft window light, shallow depth of field, photograph")
args = ap.parse_args()
sys.path.insert(0, os.path.abspath(args.backend_dir))
os.environ["UNSLOTH_DIFFUSION_MLX"] = "1"

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from core.inference import mlx_diffusion_backend as mb  # noqa: E402

rec = {"env": {k: v for k, v in os.environ.items() if k.startswith("UNSLOTH_DIFFUSION_MLX")}}
path = os.path.join(args.out, "ablate.json")


def log(*a):
    print(time.strftime("%H:%M:%S"), "ablate", *a, flush = True)


backend = mb.get_mlx_diffusion_backend()
p = common.load(backend, "unsloth/Z-Image-GGUF", "z-image-Q4_0.gguf", base_repo = "Tongyi-MAI/Z-Image", log = log)
assert p["phase"] == "ready", p
rec["status_mlx"] = backend.status().get("mlx")
rec["convert_dir"] = backend._state.convert_dir
model = backend._state.model
tr = model.transformer
pos, _neg = model._encode_prompts(prompt = args.prompt, negative_prompt = None, guidance = 1.0)
mx.eval(pos)
mx.random.seed(0)
lat = mx.random.normal((16, 1, args.size // 8, args.size // 8)).astype(mx.bfloat16)
sig = mx.linspace(1, 0, 13)


def probe():
    out = {}
    for name, sigma in (("t0", 1.0), ("mid", 0.5)):
        v = tr(timestep = mx.array([1.0 - sigma]), x = lat, cap_feats = pos, sigmas = sig).astype(mx.float32)
        mx.eval(v)
        out[name] = v
    return out


ref = probe()
rec["ref"] = {k: {"std": float(mx.std(v).item()), "max_abs": float(mx.max(mx.abs(v)).item())} for k, v in ref.items()}
log("ref", rec["ref"])

dense = [(n, m) for n, m in tr.named_modules() if isinstance(m, nn.Linear) and not isinstance(m, nn.QuantizedLinear)]
rec["dense_linears"] = [n for n, _ in dense]
log("dense linears", len(dense))
groups = {
    "x_embedder": r"^all_x_embedder\.",
    "t_embedder": r"^t_embedder\.",
    "t_embedder.linear1": r"^t_embedder\.linear1$",
    "t_embedder.linear2": r"^t_embedder\.linear2$",
    "cap_embedder": r"^cap_embedder\.",
    "final_layer": r"^all_final_layer\.",
    "final_layer.linear": r"^all_final_layer\..*linear$",
    "final_layer.adaLN": r"^all_final_layer\..*adaLN",
    "refiner_adaLN": r"_refiner\.\d+\.adaLN",
    "noise_refiner": r"^noise_refiner\.",
    "context_refiner": r"^context_refiner\.",
    "all": r".",
}


def setmod(root, name, mod):
    parts = name.split(".")
    obj = root
    for p_ in parts[:-1]:
        obj = obj[int(p_)] if isinstance(obj, list) else (obj[p_] if isinstance(obj, dict) else getattr(obj, p_))
    last = parts[-1]
    if isinstance(obj, list):
        obj[int(last)] = mod
    elif isinstance(obj, dict):
        obj[last] = mod
    else:
        setattr(obj, last, mod)


rec["ablations"] = {}
for bits in (4, 8):
    for g, pat in groups.items():
        if bits == 8 and g != "all":
            continue
        hit = [(n, m) for n, m in dense if re.search(pat, n)]
        if not hit:
            continue
        for n, m in hit:
            setmod(tr, n, nn.QuantizedLinear.from_linear(m, group_size = 32, bits = bits))
        try:
            got = probe()
            r = {"n": len(hit)}
            for k, v in got.items():
                d = v - ref[k]
                r[k] = {"std": float(mx.std(v).item()), "max_abs": float(mx.max(mx.abs(v)).item()),
                        "rel_err": float((mx.linalg.norm(d.reshape(-1)) / mx.linalg.norm(ref[k].reshape(-1))).item())}
        except Exception as exc:  # noqa: BLE001
            r = {"error": repr(exc)}
        finally:
            for n, m in hit:
                setmod(tr, n, m)
        rec["ablations"][f"{g}@{bits}"] = r
        log(f"{g}@{bits}", r)
        json.dump(rec, open(path, "w"), indent = 1, default = str)
json.dump(rec, open(path, "w"), indent = 1, default = str)
backend.unload()
log("done")
