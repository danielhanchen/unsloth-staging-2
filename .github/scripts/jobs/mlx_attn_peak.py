"""Peak Metal memory of one prefill attention call, against what Studio's MLX estimate charges.

Each case builds queries for one prefill chunk and a cache that already holds more keys than the
chunk adds (every chunk after the first), runs the attention the runtime would run, and records
`mx.get_peak_memory()` above the live inputs. `charged` is the estimate's score term for that call
(heads x queries x keys x dtype x live copies), 0 where it expects a fused kernel.

    python jobs/mlx_attn_peak.py --out outputs/jobs/mlx_attn_peak.json

Exit 3 off Apple Silicon or without a Metal device. Never downloads anything.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys


def _cases():
    # (name, heads, kv_heads, head_dim, queries, keys, kv_bits)
    return [
        ("hd256 f16 L2048 S4097", 24, 4, 256, 2048, 4097, None),
        ("hd256 f16 L2048 S8193", 24, 4, 256, 2048, 8193, None),
        ("hd256 f16 L1023 S8193", 24, 4, 256, 1023, 8193, None),
        ("hd128 f16 L2048 S8193", 28, 4, 128, 2048, 8193, None),
        ("hd256 q8 L2048 S8193", 24, 4, 256, 2048, 8193, 8),
        ("hd128 q8 L2048 S8193", 28, 4, 128, 2048, 8193, 8),
        ("hd128 q8 L256 S8193", 32, 32, 128, 256, 8193, 8),
    ]


def main(argv = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default = "mlx_attn_peak.json")
    args = ap.parse_args(argv)
    if sys.platform != "darwin" or platform.machine() != "arm64":
        print("JOB_RESULT " + json.dumps({"job": "mlx_attn_peak", "skipped": "not Apple Silicon"}))
        return 3
    import mlx.core as mx

    if not mx.metal.is_available():
        print("JOB_RESULT " + json.dumps({"job": "mlx_attn_peak", "skipped": "no Metal device"}))
        return 3
    import mlx_lm.models.base as base

    patched = ()
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    try:
        from unsloth_zoo.mlx.attention import install_quantized_attention
        patched = install_quantized_attention()
    except Exception as exc:
        print("zoo quantized attention wrapper unavailable:", exc)
    try:
        from unsloth_zoo.mlx.nax import nax_available
        nax = bool(nax_available())
    except Exception:
        nax = None

    rows = []
    for name, hq, hkv, d, L, S, bits in _cases():
        mx.clear_cache()
        q = mx.random.normal((1, hq, L, d)).astype(mx.bfloat16)
        k = mx.random.normal((1, hkv, S, d)).astype(mx.bfloat16)
        v = mx.random.normal((1, hkv, S, d)).astype(mx.bfloat16)
        if bits:
            k = mx.quantize(k, group_size = 64, bits = bits)
            v = mx.quantize(v, group_size = 64, bits = bits)
        mx.eval(q, k, v)
        mx.synchronize()
        mx.reset_peak_memory()
        before = mx.get_active_memory()
        scale = d**-0.5
        if bits:
            out = base.quantized_scaled_dot_product_attention(
                q, k, v, scale = scale, mask = "causal", group_size = 64, bits = bits
            )
        else:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale = scale, mask = "causal")
        mx.eval(out)
        mx.synchronize()
        peak = mx.get_peak_memory() - before
        one_copy = hq * L * S * 2
        rows.append(
            {
                "case": name,
                "peak_mib": round(peak / 2**20, 1),
                "one_bf16_score_copy_mib": round(one_copy / 2**20, 1),
                "peak_over_one_copy": round(peak / one_copy, 3),
            }
        )
        print(json.dumps(rows[-1]), flush = True)
        del q, k, v, out
    result = {
        "job": "mlx_attn_peak",
        "device": str(mx.default_device()),
        "device_info": {k: str(v) for k, v in mx.device_info().items()},
        "mlx": mx.__version__,
        "nax": nax,
        "zoo_patched": list(patched),
        "rows": rows,
        "passed": True,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent = 1)
    print("JOB_RESULT " + json.dumps({"job": "mlx_attn_peak", "passed": True, "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
