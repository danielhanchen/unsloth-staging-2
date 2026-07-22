# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Real Apple Silicon speed benchmark for the MLX prompt cache (PR #7311).

The cache does not change token-generation throughput; it removes the re-prefill of the
whole conversation on every follow-up turn. So the metric is turn-2 PREFILL cost
(prompt eval time / time-to-first-token) on a long shared context, measured two ways on
the same runner: cache ENABLED (reuse) vs cache DISABLED
(UNSLOTH_MLX_PROMPT_CACHE_BYTES=0, the pre-PR behaviour).

For each model it builds a ~3k-token first turn, warms it, then times turn 2 (a short
follow-up) in both modes and reports prefill tokens, mlx-lm's reported prompt_ms, and
wall-clock time-to-first-token. Gated only on the deterministic structural win
(cache ON prefills far fewer tokens); timings are reported for magnitude.
"""

import os
import sys
import time

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

from mlx_lm import load

sys.path.insert(0, os.path.join(os.getcwd(), "studio", "backend"))
from core.inference.mlx_inference import MLXInferenceBackend  # noqa: E402

MODELS = [
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
]

_PARA = (
    "The Unsloth studio runs local language models on Apple Silicon through the MLX "
    "runtime. Multi-turn chat keeps appending to the same conversation, so the shared "
    "prefix grows every turn while only a short new message is added at the end. "
)
LONG_CONTEXT = _PARA * 40  # ~3k tokens


def _make_backend(model, tokenizer, name):
    b = MLXInferenceBackend()
    b._model = model
    b._tokenizer = tokenizer
    b._is_vlm = False
    b.active_model_name = name
    b.models = {name: {}}
    return b


def _time_turn2(backend, turn2, n_gen=8):
    """Return (ttft_ms, timings) for generating n_gen tokens of turn 2."""
    t0 = time.perf_counter()
    first = None
    for chunk in backend.generate_chat_response(
        messages=turn2, max_new_tokens=n_gen, temperature=0.0
    ):
        if first is None and chunk.strip():
            first = time.perf_counter()
    end = time.perf_counter()
    ttft_ms = ((first or end) - t0) * 1000.0
    return ttft_ms, backend.last_generation_stats["timings"]


def _run_once(model, tokenizer, name, cache_bytes):
    """Fresh backend, warm turn 1, then time turn 2. cache_bytes=None -> default (enabled)."""
    if cache_bytes is None:
        os.environ.pop("UNSLOTH_MLX_PROMPT_CACHE_BYTES", None)
    else:
        os.environ["UNSLOTH_MLX_PROMPT_CACHE_BYTES"] = str(cache_bytes)
    backend = _make_backend(model, tokenizer, name)
    turn1 = [{"role": "user", "content": LONG_CONTEXT + "\nIn one word, what is the topic?"}]
    reply1 = "".join(
        backend.generate_chat_response(messages=turn1, max_new_tokens=8, temperature=0.0)
    )
    turn2 = turn1 + [
        {"role": "assistant", "content": reply1},
        {"role": "user", "content": "Reply with one short sentence about that topic."},
    ]
    # best of 2 (rebuild warm state each repeat so it stays a true "turn 2 after turn 1")
    best = None
    for _ in range(2):
        b = _make_backend(model, tokenizer, name)
        list(b.generate_chat_response(messages=turn1, max_new_tokens=8, temperature=0.0))
        ttft_ms, t = _time_turn2(b, turn2)
        if best is None or ttft_ms < best[0]:
            best = (ttft_ms, t)
    return best


def run_model(name):
    print(f"\n=== {name} ===")
    model, tokenizer = load(name)
    # warm-up (JIT / Metal) so it is not charged to the first measured run
    list(_make_backend(model, tokenizer, name).generate_chat_response(
        messages=[{"role": "user", "content": "hi"}], max_new_tokens=2, temperature=0.0))

    on_ttft, on = _run_once(model, tokenizer, name, cache_bytes=None)
    off_ttft, off = _run_once(model, tokenizer, name, cache_bytes=0)

    print(f"    {'mode':10s} {'prompt_n':>9s} {'cache_n':>8s} {'prompt_ms':>10s} {'ttft_ms':>9s}")
    print(f"    {'cache ON':10s} {on['prompt_n']:9d} {on['cache_n']:8d} "
          f"{on['prompt_ms']:10.1f} {on_ttft:9.1f}")
    print(f"    {'cache OFF':10s} {off['prompt_n']:9d} {off['cache_n']:8d} "
          f"{off['prompt_ms']:10.1f} {off_ttft:9.1f}")
    pm_speedup = (off["prompt_ms"] / on["prompt_ms"]) if on["prompt_ms"] > 0 else float("inf")
    ttft_speedup = (off_ttft / on_ttft) if on_ttft > 0 else float("inf")
    print(f"    -> prompt-eval {pm_speedup:.1f}x faster, TTFT {ttft_speedup:.1f}x faster "
          f"(reused {on['cache_n']} tok, prefilled {on['prompt_n']} vs {off['prompt_n']})")

    assert on["cache_n"] > 0, "cache ON did not reuse anything"
    assert on["prompt_n"] < off["prompt_n"], "cache ON did not reduce prefilled tokens"
    return (name, pm_speedup, ttft_speedup, on, off)


def main():
    rows, failures = [], []
    for name in MODELS:
        try:
            rows.append(run_model(name))
        except Exception:
            import traceback
            failures.append(name)
            print(f"    FAIL: {name}\n{traceback.format_exc()}")
    print("\n==== SPEEDUP SUMMARY ====")
    for name, pm, tt, on, off in rows:
        print(f"  {name}: prompt-eval {pm:.1f}x  TTFT {tt:.1f}x  "
              f"(prefill {on['prompt_n']} vs {off['prompt_n']} tok)")
    for name in failures:
        print(f"  {name}: FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
