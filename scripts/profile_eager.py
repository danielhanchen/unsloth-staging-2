# SPDX-License-Identifier: AGPL-3.0-only
"""Profile the EAGER (no-compile) diffusion denoise to find what is left to optimize.

For one model: load via the backend at speed_mode=eager (patches ON), warm up, then profile
one generate with torch.profiler and bucket CUDA self-time into categories
(matmul/linear, attention, norm, activation, elementwise, copy/cast, conv, other). The
category split shows how much time is IRREDUCIBLE (matmul + attention) vs potentially
fusible in eager (elementwise / cast / norm / activation).

  python profile_eager.py --model qwen-image --gpu 4          # one model
  python profile_eager.py --all --gpus 4 5 6                  # 3 models in parallel + aggregate
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

UNSLOTH = Path("/mnt/disks/unslothai/ubuntu/workspace_81/unsloth")
BACKEND = UNSLOTH / "studio" / "backend"
sys.path.insert(0, str(UNSLOTH / "scripts"))
import benchall_orchestrator as bo  # noqa: E402

OUT = UNSLOTH / "outputs" / "profile_eager"
PROMPT = "a cat asleep on a stack of books, highly detailed"
PROFILE_STEPS = 6  # per-step op MIX is step-count-invariant; keep the trace small
MODELS = ["qwen-image", "flux.1-schnell", "flux.2-klein-4b"]

# Priority-ordered category matchers (first match wins; lowercased op key). gguf_dequant is
# checked FIRST and catches the UNAMBIGUOUS bit-unpacking ops of on-the-fly GGUF dequant
# (4-bit nibble shifts + masks + uint8 functors). The ambiguous dequant work (block-scale
# mul, zero-point sub, assembly copy, byte->bf16 cast) lands in elementwise/copy on top, so
# this bucket is a conservative LOWER BOUND on true dequant cost.
CATEGORIES = [
    ("gguf_dequant", ("rshift", "lshift", "bitwise_and", "bitwise_or", "__and__", "__or__",
                      "unsigned char", "uint8")),
    ("matmul/linear", ("addmm", "aten::mm", "::bmm", "baddbmm", "matmul", "linear", "scaled_mm",
                        "cutlass", "gemm", "cublas", "nvjet")),
    ("attention", ("scaled_dot_product", "sdpa", "flash", "efficient_attention", "fmha",
                   "mem_eff", "attention")),
    ("norm", ("layer_norm", "rms_norm", "group_norm", "batch_norm", "layernorm", "groupnorm")),
    ("activation", ("gelu", "silu", "sigmoid", "::tanh", "::relu", "swish", "softmax")),
    ("conv", ("convolution", "conv2d", "conv_")),
    ("elementwise", ("::add", "::mul", "::sub", "::div", "::pow", "rsqrt", "addcmul", "::neg",
                     "::exp", "clamp", "::rsub", "::mean", "::sum", "::var", "::sqrt", "lerp")),
    ("copy/cast", ("copy_", "aten::to", "_to_copy", "contiguous", "::clone", "::cat", "::stack",
                   "pad", "index", "slice", "::chunk", "::unbind", "::expand", "::pad")),
]


def _categorize(key: str) -> str:
    k = key.lower()
    for name, pats in CATEGORIES:
        if any(p in k for p in pats):
            return name
    return "other"


def _self_cuda_us(ev) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(ev, attr, None)
        if v:
            return float(v)
    return 0.0


def run_one(model: str, gpu: int, dense: bool = False) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gguf_repo, gguf_file, base_repo, steps, guidance, _ = bo.MANIFEST[model]
    import torch  # noqa: PLC0415
    from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

    sys.path.insert(0, str(BACKEND))
    from core.inference.diffusion import get_diffusion_backend  # noqa: PLC0415

    backend = get_diffusion_backend()
    backend.begin_load(gguf_repo, gguf_filename=gguf_file, base_repo=base_repo,
                       hf_token=os.environ.get("HF_TOKEN"), memory_mode="fast", speed_mode="eager")
    _wait_ready(backend)

    if dense:
        # Dequantize GGUF -> plain bf16 (the "safetensors / non-GGUF" eager path): no
        # per-forward dequant, so the profile shows the irreducible dense compute.
        from diffusers.quantizers.gguf.utils import _dequantize_gguf_and_restore_linear  # noqa: PLC0415
        t = backend._state.pipe.transformer
        _dequantize_gguf_and_restore_linear(t)
        for p in t.parameters():
            if p.is_floating_point() and p.dtype != torch.bfloat16:
                p.data = p.data.to(torch.bfloat16)
        for b in t.buffers():
            if b.is_floating_point() and b.dtype != torch.bfloat16:
                b.data = b.data.to(torch.bfloat16)

    def _gen(nsteps):
        backend.generate(prompt=PROMPT, width=1024, height=1024, steps=nsteps,
                         guidance=guidance, seed=bo.SEED, batch_size=1)

    _gen(PROFILE_STEPS)  # warmup (cudnn.benchmark autotune settles)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _gen(PROFILE_STEPS)
        torch.cuda.synchronize()

    cats: dict[str, dict] = {}
    ops = []
    for ev in prof.key_averages():
        us = _self_cuda_us(ev)
        if us <= 0:
            continue
        c = _categorize(ev.key)
        d = cats.setdefault(c, {"us": 0.0, "count": 0})
        d["us"] += us
        d["count"] += int(ev.count)
        ops.append((ev.key, us, int(ev.count), c))
    total = sum(d["us"] for d in cats.values()) or 1.0
    for d in cats.values():
        d["pct"] = round(100 * d["us"] / total, 1)
        d["us"] = round(d["us"] / 1000, 1)  # -> ms
    ops.sort(key=lambda x: x[1], reverse=True)
    top = [{"op": k, "ms": round(u / 1000, 2), "count": c, "cat": cat} for k, u, c, cat in ops[:18]]

    backend.unload()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": model, "dense": dense, "profile_steps": PROFILE_STEPS,
        "total_self_cuda_ms": round(total / 1000, 1),
        "categories": dict(sorted(cats.items(), key=lambda kv: kv[1]["us"], reverse=True)),
        "top_ops": top,
    }
    (OUT / f"{model}{'_dense' if dense else ''}.json").write_text(json.dumps(summary, indent=2))
    return summary


def _wait_ready(backend, timeout_s=2400):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        p = backend.load_progress()
        if p.get("phase") == "ready":
            return
        if p.get("phase") == "error":
            raise RuntimeError(f"load error: {p.get('error')}")
        time.sleep(2)
    raise TimeoutError("load did not become ready")


def aggregate(dense: bool = False) -> None:
    rows = []
    suffix = "_dense" if dense else ""
    for m in MODELS:
        f = OUT / f"{m}{suffix}.json"
        if f.exists():
            rows.append(json.loads(f.read_text()))
    label = "DENSE bf16 (non-GGUF / dequantized; no per-forward dequant)" if dense else "GGUF"
    lines = [
        f"# Eager denoise profile [{label}]: where CUDA time goes (speed_mode=eager, patches ON)",
        "",
        f"torch.profiler self-CUDA time over {PROFILE_STEPS} denoise steps (1024px). % of total",
        "self-CUDA. gguf_dequant = the UNAMBIGUOUS on-the-fly 4-bit weight unpacking (shifts +",
        "masks); the block-scale mul / zero-point sub / assembly copy land in elementwise+copy on",
        "top, so true dequant cost is even higher. matmul + attention + conv are the irreducible compute.",
        "",
        "| model | gguf_dequant | matmul/linear | attention | conv | norm | activation | elementwise | copy/cast | other |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        c = r["categories"]
        def pct(name):
            return c.get(name, {}).get("pct", 0.0)
        lines.append(
            f"| {r['model']} | **{pct('gguf_dequant')}** | {pct('matmul/linear')} | {pct('attention')} | "
            f"{pct('conv')} | {pct('norm')} | {pct('activation')} | {pct('elementwise')} | "
            f"{pct('copy/cast')} | {pct('other')} |"
        )
    lines += ["", "## Top ops per model (self-CUDA ms over the profiled steps)", ""]
    for r in rows:
        lines.append(f"### {r['model']} (total {r['total_self_cuda_ms']} ms)")
        lines.append("| op | ms | count | category |")
        lines.append("|---|--:|--:|---|")
        for o in r["top_ops"]:
            lines.append(f"| `{o['op'][:70]}` | {o['ms']} | {o['count']} | {o['cat']} |")
        lines.append("")
    (OUT / f"SUMMARY{suffix}.md").write_text("\n".join(lines))
    print("\n".join(lines[:6 + len(rows)]))


def _spawn(model: str, gpu: int, dense: bool = False):
    cmd = [sys.executable, "-u", __file__, "--model", model, "--gpu", str(gpu)]
    if dense:
        cmd.append("--dense")
    LOG = OUT / f"{model}{'_dense' if dense else ''}.log"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w") as lf:
        return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--gpu", type=int, default=4)
    p.add_argument("--all", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--dense", action="store_true", help="dequantize GGUF->bf16 first (non-GGUF path)")
    p.add_argument("--gpus", type=int, nargs="*", default=[4, 5, 6])
    a = p.parse_args(argv)
    if a.aggregate:
        aggregate(dense=a.dense)
        return 0
    if a.all:
        procs = [(_spawn(m, a.gpus[i % len(a.gpus)], dense=a.dense)) for i, m in enumerate(MODELS)]
        for pr in procs:
            pr.wait()
        aggregate(dense=a.dense)
        print("=== PROFILE_EAGER_DONE ===", flush=True)
        return 0
    if a.model:
        s = run_one(a.model, a.gpu, dense=a.dense)
        print(json.dumps(s["categories"], indent=2))
        return 0
    p.error("need --model, --all, or --aggregate")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
