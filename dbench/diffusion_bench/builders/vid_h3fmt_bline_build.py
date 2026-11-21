#!/usr/bin/env python3
"""MiniMax-H3 baseline-UI DiT files in fp8 (scaled) and nvfp4, built from the harness's pruned bf16 DiT (build_bline_files.h3_dit:
official MiniMaxAI/MiniMax-H3 FL2VA transformer -> curve adaLN, qkv rows regrouped per head -> thirds) with
the baseline UI's own quantizers: bline-kitchen ``QuantizedTensor.from_float`` on the layouts baseline-UI v0.44.0 loads
(bline/quant_ops.py QUANT_ALGOS), written per layer with a ``bline_quant`` JSON tensor, exactly as Bline-Org's
bline-model-tools (seedvr2_convert.py ``quantize_nvfp4_weight``) packs them. Nothing from a Bline-Org HF repo.

Layer set = the 200 block Linears the template's int8_convrot file quantizes (and Bline-Org's
``minimax_h3_fl2va_pruned_fp8_scaled`` header lists: 200 F8_E4M3 weights): blocks.N.attn.{qkv_proj,out_proj},
blocks.N.mlp.{fc1,fc2}. Everything else is copied unchanged (token_refiner, embedders, final layer, adaLN curve).
fp8: per-tensor weight scale, no static ``input_scale`` (Bline-Org's calibration is not public) -> dynamic per-tensor
activation scale at runtime (baseline-UI MixedPrecisionOps). nvfp4: per-16 E4M3 block scales + per-tensor scale, W4A4
with dynamic activation scale.

Ported from scripts/vidcmp_nov5/h3fmt_bline_build.py. PRUNED_BF16 = build_bline_files.h3_dit's pre-quant output
(builders.build_file chains the two).
  python diffusion_bench/builders/vid_h3fmt_bline_build.py --src PRUNED_BF16 [--fp8 OUT] [--nvfp4 OUT]      (baseline-UI venv, CUDA)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import torch
try:  # extnames-aware safetensors (builders/_st.py); as a script or as builders.<module>
    from _st import safe_open
except ImportError:
    from builders._st import safe_open
try:  # extnames-aware safetensors (builders/_st.py); as a script or as builders.<module>
    from _st import save_file
except ImportError:
    from builders._st import save_file

BLOCK_LINEAR = re.compile(r"^blocks\.\d+\.(attn\.(qkv_proj|out_proj)|mlp\.(fc1|fc2))\.weight$")
LAYOUT = {"fp8": ("TensorCoreFP8Layout", "float8_e4m3fn"), "nvfp4": ("TensorCoreNVFP4Layout", "nvfp4")}


def bline_quant_tensor(fmt: str) -> torch.Tensor:
    return torch.tensor(list(json.dumps({"format": fmt}).encode("utf-8")), dtype = torch.uint8)


def build(src: str, dst: str, kind: str) -> dict:
    import extnames as X  # the real package name in a public copy

    QuantizedTensor = X.import_module("bline_kitchen.tensor").QuantizedTensor

    layout, fmt = LAYOUT[kind]
    t0 = time.time()
    out, nq, err = {}, 0, []
    with safe_open(src, "pt") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            v = f.get_tensor(k)
            if BLOCK_LINEAR.match(k):
                w = v.to("cuda", torch.bfloat16).contiguous()
                qt = QuantizedTensor.from_float(w, layout)
                base = k[: -len(".weight")]
                for kk, tt in qt.state_dict(f"{base}.weight").items():
                    out[kk] = tt.detach().to("cpu").contiguous()
                out[f"{base}.bline_quant"] = bline_quant_tensor(fmt)
                if nq % 50 == 0:  # a few relative-error probes (dequantised vs bf16)
                    dq = qt.dequantize().float()
                    err.append((k, round(((dq - w.float()).norm() / w.float().norm()).item(), 5)))
                nq += 1
                del w, qt
            else:
                out[k] = v
    if nq != 200:
        raise SystemExit(f"expected 200 block linears, quantized {nq}")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok = True)
    save_file(out, dst, metadata = meta)
    rep = {"file": dst, "kind": kind, "layout": layout, "format": fmt, "quantized": nq, "tensors": len(out),
           "bytes": os.path.getsize(dst), "relerr_probe": err, "s": round(time.time() - t0, 1)}
    print("H3FMT_BLINE_BUILT", json.dumps(rep), flush = True)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required = True)
    ap.add_argument("--fp8")
    ap.add_argument("--nvfp4")
    a = ap.parse_args()
    print("H3FMT_BLINE_DEVICE", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush = True)
    for kind, dst in (("fp8", a.fp8), ("nvfp4", a.nvfp4)):
        if not dst:
            continue
        if os.path.exists(dst):
            print("H3FMT_BLINE_EXISTS", dst, os.path.getsize(dst), flush = True)
            continue
        build(a.src, dst, kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
