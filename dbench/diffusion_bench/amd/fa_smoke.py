#!/usr/bin/env python3
"""Attention kernels on this GPU, observed: which SDPA backend torch dispatches by default (AOTriton flash /
efficient, CK, or math), what each forced backend costs, and whether DAO-AILab flash_attn (Triton AMD or CK build)
runs, how fast, and how close to an fp32 reference. Writes JSON to --out only (never stdout).

  FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python fa_smoke.py --out fa.json

Shapes are DiT-like (B=1, heads 24, head_dim 128, bf16): 4096 tokens (a 1024^2 image DiT) and 16384 (a short
720p video clip). Accuracy is checked at 1024 tokens against fp32 math, max and mean abs error.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path


def err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:400]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--iters", type = int, default = 10)
    args = ap.parse_args()
    out: dict = {"python": sys.version.split()[0], "platform": platform.platform(),
                 "env": {k: os.environ.get(k) for k in ("FLASH_ATTENTION_TRITON_AMD_ENABLE",
                                                        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL",
                                                        "TORCH_ROCM_FA_PREFER_CK")}}
    try:
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        out["torch"] = torch.__version__
        out["hip"] = getattr(torch.version, "hip", None)
        p = torch.cuda.get_device_properties(0)
        out["device"] = {"name": torch.cuda.get_device_name(0), "arch": getattr(p, "gcnArchName", None),
                         "total_gib": round(p.total_memory / 2**30, 2)}
        try:
            out["preferred_rocm_fa_library"] = str(torch.backends.cuda.preferred_rocm_fa_library())
        except Exception as exc:  # noqa: BLE001
            out["preferred_rocm_fa_library"] = err(exc)
        dev, dt = "cuda", torch.bfloat16

        def qkv(s: int, layout: str = "bhsd"):
            g = torch.Generator(device = "cpu").manual_seed(0)
            t = [torch.randn(1, 24, s, 128, generator = g).to(dev, dt) for _ in range(3)]
            return t if layout == "bhsd" else [x.transpose(1, 2).contiguous() for x in t]

        def timeit(fn) -> float:
            fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(args.iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            ts.sort()
            return round(ts[len(ts) // 2] * 1000, 3)

        # Reference at 1024 tokens in fp32 math.
        q, k, v = qkv(1024)
        with sdpa_kernel([SDPBackend.MATH]):
            ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())

        def acc(o) -> dict:
            d = (o.float() - ref).abs()
            return {"max_abs": round(float(d.max()), 5), "mean_abs": round(float(d.mean()), 6)}

        backends = {"default": None, "flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
                    "math": SDPBackend.MATH}
        res: dict = {}
        for name, b in backends.items():
            r: dict = {}
            try:
                def call(q = q, k = k, v = v, b = b):
                    if b is None:
                        return F.scaled_dot_product_attention(q, k, v)
                    with sdpa_kernel([b]):
                        return F.scaled_dot_product_attention(q, k, v)
                r["accuracy_1024"] = acc(call())
                for s in (4096, 16384):
                    if name == "math" and s > 4096:
                        continue  # O(N^2) score matrix; 4096 already says what math costs
                    qq, kk, vv = qkv(s)
                    r[f"ms_{s}"] = timeit(lambda qq = qq, kk = kk, vv = vv: call(qq, kk, vv))
            except Exception as exc:  # noqa: BLE001
                r["error"] = err(exc)
            res[f"sdpa_{name}"] = r
            torch.cuda.empty_cache()
        # Which kernel did the DEFAULT dispatch run? Profile one call (never on Windows: the ROCm profiler there
        # hangs / crashes the job; DBENCH_NO_PROFILER=1 is set by gen_workflow.py on Windows).
        try:
            if os.environ.get("DBENCH_NO_PROFILER") == "1" or sys.platform == "win32":
                raise RuntimeError("profiler disabled (DBENCH_NO_PROFILER / Windows)")
            from torch.profiler import ProfilerActivity, profile

            qq, kk, vv = qkv(4096)
            with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                F.scaled_dot_product_attention(qq, kk, vv)
                torch.cuda.synchronize()
            ev = sorted(prof.key_averages(), key = lambda e: -(getattr(e, "device_time_total", 0) or 0))
            out["default_dispatch_ops"] = [e.key[:160] for e in prof.key_averages() if e.key.startswith("aten::")]
            out["default_dispatch_kernels"] = [{"name": e.key[:200], "us": round(float(getattr(e, "device_time_total",
                                                0) or 0), 1)} for e in ev[:8]]
        except Exception as exc:  # noqa: BLE001
            out["default_dispatch_error"] = err(exc)
        # DAO-AILab flash_attn (layout B, S, H, D).
        r = {}
        try:
            import importlib.metadata as md

            import flash_attn
            from flash_attn import flash_attn_func

            r["version"] = md.version("flash_attn")
            r["module"] = str(Path(flash_attn.__file__).parent)
            q2, k2, v2 = (x.transpose(1, 2).contiguous() for x in (q, k, v))
            o = flash_attn_func(q2, k2, v2)
            r["accuracy_1024"] = acc(o.transpose(1, 2))
            for s in (4096, 16384):
                qq, kk, vv = qkv(s, "bshd")
                r[f"ms_{s}"] = timeit(lambda qq = qq, kk = kk, vv = vv: flash_attn_func(qq, kk, vv))
        except Exception as exc:  # noqa: BLE001
            r["error"] = err(exc)
            r["tail"] = traceback.format_exc().strip().splitlines()[-4:]
        res["flash_attn_dao"] = r
        out["results"] = res
    except Exception as exc:  # noqa: BLE001
        out["fatal"] = err(exc)
        out["tail"] = traceback.format_exc().strip().splitlines()[-6:]
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(out, indent = 1, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
