# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Real Apple Silicon end-to-end confirmation for the MLX prompt cache (PR #7311).

Runs ONLY on a real Mac (needs a working mlx/mlx-lm), so it lives on the disposable
staging branch, not upstream. For each small mlx-community 4-bit quant it proves the two
things the Linux/CUDA box cannot:

  1. TOKEN-LEVEL reuse + output parity (the hard gate, independent of chat templating):
     build a real KV cache covering a known token prefix via the PR's
     `_MLXPromptCacheHistory.insert`, then `fetch` it back for the full prompt and assert
     the covered length is exact and the uncovered tail is short. Generate greedily from the
     reused cache and from a fresh full prefill; assert the produced token ids are IDENTICAL
     (the cache must never change results).

  2. HIGH-LEVEL two-turn reuse through the real `MLXInferenceBackend.generate_chat_response`:
     turn 1 reports timings.cache_n == 0, turn 2 (same system+user1 prefix) reports
     cache_n > 0, and usage.prompt_tokens == timings.prompt_n + timings.cache_n.

Exit non-zero if any gated assertion fails for any model.
"""

import os
import sys
import traceback

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

# studio backend on the path (mirrors app launch / conftest).
sys.path.insert(0, os.path.join(os.getcwd(), "studio", "backend"))
from core.inference.mlx_inference import MLXInferenceBackend  # noqa: E402

MODELS = [
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
]

GREEDY = make_sampler(temp=0.0)


def _gen_tokens(model, tokenizer, prompt, cache, n):
    """Greedy token ids for `n` steps, prefilling `prompt` into `cache`."""
    out = []
    for r in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=n, sampler=GREEDY, prompt_cache=cache
    ):
        out.append(r.token)
    return out


def token_level_reuse_and_parity(backend, model, tokenizer):
    """Hard gate: real fetch/insert coverage is exact and cache reuse never changes output."""
    text = (
        "The capital of France is Paris. The capital of Japan is Tokyo. "
        "The capital of Italy is Rome. The capital of Spain is Madrid. "
        "List three more capital cities:"
    )
    ids = list(tokenizer.encode(text))
    assert len(ids) >= 12, f"prompt too short to split ({len(ids)} tokens)"
    k = len(ids) * 2 // 3  # covered prefix length

    # Build a REAL KV cache covering exactly ids[:k] by prefilling with the model itself.
    c_prefix = make_prompt_cache(model)
    logits = model(mx.array([ids[:k]]), cache=c_prefix)
    mx.eval(logits)

    history = backend._prompt_cache()
    assert history is not None, "prompt cache history unavailable on real mlx-lm"
    history.insert("parity", ids[:k], c_prefix)

    # Fetch back for the FULL prompt: must reuse exactly k tokens, tail = the rest.
    c_fetch, rest = history.fetch(model, "parity", ids)
    cached = len(ids) - len(rest)
    assert cached == k, f"reused {cached} tokens, expected exactly {k}"
    assert rest == ids[k:], "uncovered tail is not the exact suffix"

    # Greedy parity: reused-prefix generation == fresh full-prefill generation.
    out_warm = _gen_tokens(model, tokenizer, rest, c_fetch, 24)
    out_full = _gen_tokens(model, tokenizer, ids, make_prompt_cache(model), 24)
    assert out_warm == out_full, (
        f"cache changed output!\n  warm={out_warm}\n  full={out_full}"
    )
    print(f"    token-level: reused {cached}/{len(ids)} tok, tail={len(rest)}, "
          f"parity OK over {len(out_full)} greedy tokens")


def high_level_two_turn_reuse(backend, name):
    """Gate: real generate_chat_response reports cache_n==0 on turn 1, >0 on turn 2."""
    turn1 = [{"role": "user", "content": "Give me one short fact about the moon."}]
    reply1 = "".join(
        backend.generate_chat_response(messages=turn1, max_new_tokens=24, temperature=0.0)
    )
    s1 = backend.last_generation_stats
    c1 = s1["timings"]["cache_n"]

    turn2 = turn1 + [
        {"role": "assistant", "content": reply1},
        {"role": "user", "content": "Now one short fact about the sun."},
    ]
    _ = "".join(
        backend.generate_chat_response(messages=turn2, max_new_tokens=24, temperature=0.0)
    )
    s2 = backend.last_generation_stats
    c2 = s2["timings"]["cache_n"]
    pn, un = s2["timings"]["prompt_n"], s2["usage"]["prompt_tokens"]

    assert c1 == 0, f"turn 1 should not reuse anything, cache_n={c1}"
    assert c2 > 0, f"turn 2 should reuse the shared prefix, cache_n={c2}"
    assert un == pn + c2, f"usage.prompt_tokens {un} != prompt_n {pn} + cache_n {c2}"
    print(f"    high-level: turn1 cache_n={c1}, turn2 cache_n={c2}, "
          f"prompt_n={pn}, usage.prompt_tokens={un}")


def run_model(name):
    print(f"\n=== {name} ===")
    model, tokenizer = load(name)
    backend = MLXInferenceBackend()
    backend._model = model
    backend._tokenizer = tokenizer
    backend._is_vlm = False
    backend.active_model_name = name
    backend.models = {name: {}}

    token_level_reuse_and_parity(backend, model, tokenizer)
    high_level_two_turn_reuse(backend, name)
    print(f"    PASS: {name}")


def main():
    failures = []
    for name in MODELS:
        try:
            run_model(name)
        except Exception:
            failures.append(name)
            print(f"    FAIL: {name}\n{traceback.format_exc()}")
    print("\n==== SUMMARY ====")
    for name in MODELS:
        print(f"  {'FAIL' if name in failures else 'PASS'}  {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
