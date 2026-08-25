#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Prefill throughput and peak memory, baseline versus the int8 W8A8 path.

Context lengths match the points JetBrains published for Qwen3.6-27B-4bit (6.4k: 819 ->
1061 tok/s, +29.5%; 11.8k: 726 -> 1070 tok/s, +47.3%) so the numbers are directly
comparable rather than merely suggestive.

    python bench/mlx_int8_bench.py --model mlx-community/Qwen3.6-27B-4bit
    python bench/mlx_int8_bench.py --synthetic          # no model download

This measures nothing useful off an Apple M5. The int8 path needs the M5 neural
accelerators' int8 rate; anywhere else the module refuses to enable, and forcing it on
benchmarks a fallback. The script says so in its own output rather than leaving a
plausible-looking table to be quoted out of context.
"""

import argparse
import gc
import json
import sys
import time

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, __file__.rsplit("/", 2)[0])

import unsloth_mlx_int8
from unsloth_mlx_int8 import capability

CONTEXTS = [256, 2048, 6400, 11800, 20000]


def peak_memory_gb():
    try:
        return mx.get_peak_memory() / 1024**3
    except AttributeError:
        return float("nan")


def reset_peak():
    try:
        mx.reset_peak_memory()
    except AttributeError:
        pass


class SyntheticStack(nn.Module):
    """A Qwen3.6-27B-shaped dense block stack: hidden 5120, intermediate 17408.

    Enough to exercise the same GEMM shapes as the real model without a 20 GB download,
    which matters because the shapes are what the kernel's tiling is sensitive to.
    """

    def __init__(self, layers=4, hidden=5120, inter=17408):
        super().__init__()
        self.blocks = [
            {
                "gate": nn.Linear(hidden, inter, bias=False),
                "up": nn.Linear(hidden, inter, bias=False),
                "down": nn.Linear(inter, hidden, bias=False),
            }
            for _ in range(layers)
        ]

    def __call__(self, x):
        for b in self.blocks:
            x = x + b["down"](nn.silu(b["gate"](x)) * b["up"](x))
        return x


def build_synthetic(layers):
    model = SyntheticStack(layers=layers)
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    return model


def time_prefill(fn, tokens, warmups=1, repeats=3):
    for _ in range(warmups):
        mx.eval(fn(tokens))
    reset_peak()
    best = float("inf")
    for _ in range(repeats):
        mx.synchronize() if hasattr(mx, "synchronize") else None
        t0 = time.perf_counter()
        mx.eval(fn(tokens))
        best = min(best, time.perf_counter() - t0)
    return tokens / best, peak_memory_gb()


def run_synthetic(args):
    model = build_synthetic(args.layers)
    hidden = 5120

    def forward(n):
        x = mx.random.normal((1, n, hidden)).astype(mx.bfloat16)
        return model(x)

    rows = []
    for n in args.contexts:
        unsloth_mlx_int8.disable()
        gc.collect()
        base_tps, base_mem = time_prefill(forward, n, repeats=args.repeats)

        enabled = unsloth_mlx_int8.enable(force=args.force)
        if enabled:
            unsloth_mlx_int8.warmup(model)
            ok, detail = unsloth_mlx_int8.self_test()
            if not ok:
                print(f"  self-test FAILED ({detail}); int8 path disabled", file=sys.stderr)
        int8_tps, int8_mem = time_prefill(forward, n, repeats=args.repeats)
        unsloth_mlx_int8.disable()

        rows.append({
            "context": n,
            "baseline_tok_s": round(base_tps, 1),
            "int8_tok_s": round(int8_tps, 1),
            "speedup": round(int8_tps / base_tps, 4) if base_tps else None,
            "baseline_peak_gb": round(base_mem, 3),
            "int8_peak_gb": round(int8_mem, 3),
            "int8_active": bool(enabled),
        })
        print(f"  {n:>6} tok  {base_tps:>9.1f} -> {int8_tps:>9.1f} tok/s"
              f"  ({int8_tps / base_tps - 1:+.1%})")
    return rows


def run_model(args):
    try:
        from mlx_lm import load
    except ImportError:
        print("mlx-lm is not installed; use --synthetic", file=sys.stderr)
        return None

    model, tokenizer = load(args.model)
    vocab = getattr(tokenizer, "vocab_size", 32000)

    def forward(n):
        ids = mx.random.randint(0, vocab, (1, n))
        return model(ids)

    rows = []
    for n in args.contexts:
        unsloth_mlx_int8.disable()
        gc.collect()
        base_tps, base_mem = time_prefill(forward, n, repeats=args.repeats)

        enabled = unsloth_mlx_int8.enable(force=args.force)
        if enabled:
            unsloth_mlx_int8.warmup(model, scope=args.scope)
            ok, detail = unsloth_mlx_int8.self_test()
            if not ok:
                print(f"  self-test FAILED ({detail})", file=sys.stderr)
        int8_tps, int8_mem = time_prefill(forward, n, repeats=args.repeats)
        unsloth_mlx_int8.disable()

        rows.append({
            "context": n,
            "baseline_tok_s": round(base_tps, 1),
            "int8_tok_s": round(int8_tps, 1),
            "speedup": round(int8_tps / base_tps, 4) if base_tps else None,
            "baseline_peak_gb": round(base_mem, 3),
            "int8_peak_gb": round(int8_mem, 3),
            "int8_active": bool(enabled),
        })
        print(f"  {n:>6} tok  {base_tps:>9.1f} -> {int8_tps:>9.1f} tok/s"
              f"  ({int8_tps / base_tps - 1:+.1%})")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="mlx-community/Qwen3.6-27B-4bit")
    p.add_argument("--synthetic", action="store_true",
                   help="benchmark a Qwen3.6-27B-shaped dense stack, no download")
    p.add_argument("--layers", type=int, default=4, help="synthetic stack depth")
    p.add_argument("--contexts", type=int, nargs="+", default=CONTEXTS)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--scope", default="all", choices=["all", "mlp"])
    p.add_argument("--force", action="store_true",
                   help="skip the capability probe (for testing a fallback backend)")
    p.add_argument("--json", help="write results here")
    args = p.parse_args()

    supported = capability.is_supported()
    print(f"int8 prefill supported here: {supported} ({capability.reason()})")
    if not supported and not args.force:
        print("\nNothing to compare: the int8 path will not engage on this machine.\n"
              "Re-run on an Apple M5, or pass --force with "
              "UNSLOTH_MLX_INT8_BACKEND=portable to exercise the plumbing.")
        return 1
    if not supported:
        print("\n*** FORCED. These timings do NOT measure the int8 NAX path and must not\n"
              "*** be reported as a speedup. Only an M5 can produce that number.\n")

    print(f"\nprefill throughput ({'synthetic' if args.synthetic else args.model})")
    rows = run_synthetic(args) if args.synthetic else run_model(args)
    if rows is None:
        return 1

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({
                "supported": supported,
                "forced": bool(args.force),
                "reason": capability.reason(),
                "results": rows,
            }, fh, indent=2)
        print(f"\nwrote {args.json}")

    if not supported:
        print("\nReminder: forced run. The speedup column is meaningless off an M5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
