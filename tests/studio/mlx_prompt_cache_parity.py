# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Real Apple Silicon ACCURACY + TOKEN equivalence check for the MLX prompt cache (PR #7311).

Speed is only acceptable if the reused KV is numerically faithful. This proves three things
on a real Mac, per mlx-community quant and prompt:

  1. LOGITS accuracy: the next-token logits after reusing a cached prefix match a full
     re-prefill -- identical argmax and a tiny max|delta| (fp16 rounding only).
  2. TOKEN equivalence: greedily decoding 64 tokens from the reused cache produces the exact
     same token IDs (and decoded text) as decoding from a full prefill.
  3. TOKEN ACCOUNTING equivalence: through generate_chat_response, a cached turn reports the
     same usage.prompt_tokens / completion_tokens / total_tokens as the same turn with the
     cache disabled (UNSLOTH_MLX_PROMPT_CACHE_BYTES=0).

Exit non-zero on any failed gate.
"""

import os
import sys

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, os.path.join(os.getcwd(), "studio", "backend"))
from core.inference.mlx_inference import MLXInferenceBackend  # noqa: E402

MODELS = [
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
]
PROMPTS = [
    "Explain in a few sentences why the sky appears blue during the day.",
    "List the first five prime numbers and then add them together.",
]
N_GEN = 64
LOGIT_TOL = 1.0  # fp16 rounding headroom; argmax + exact token match are the real gates


def _render(tokenizer, user_text):
    msgs = [{"role": "user", "content": user_text}]
    return list(tokenizer.encode(
        tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ))


def _next_logits(model, prompt_ids, cache):
    """Prefill prompt_ids into cache; return the last-position next-token logits (1, V)."""
    logits = model(mx.array([prompt_ids]), cache=cache)
    mx.eval(logits)
    return logits[:, -1, :]


def _greedy_continue(model, cache, next_logits, n):
    """Greedy-decode n tokens, feeding one at a time so the cache advances."""
    out, logits = [], next_logits
    for _ in range(n):
        tok = int(mx.argmax(logits, axis=-1).item())
        out.append(tok)
        logits = model(mx.array([[tok]]), cache=cache)[:, -1, :]
        mx.eval(logits)
    return out


def logits_and_token_equivalence(backend, model, tokenizer, user_text):
    ids = _render(tokenizer, user_text)
    assert len(ids) >= 12, f"prompt too short ({len(ids)})"
    k = len(ids) * 2 // 3

    # Full re-prefill reference.
    nl_full = _next_logits(model, ids, make_prompt_cache(model))

    # Cached path: real insert of a prefix, then fetch for the full prompt.
    c_pre = make_prompt_cache(model)
    mx.eval(model(mx.array([ids[:k]]), cache=c_pre))
    hist = backend._prompt_cache()
    hist.insert("acc", ids[:k], c_pre)
    c_fetch, rest = hist.fetch(model, "acc", ids)
    assert len(rest) == len(ids) - k, f"reuse mismatch: rest={len(rest)}, expected {len(ids) - k}"
    nl_cache = _next_logits(model, rest, c_fetch)

    dmax = float(mx.max(mx.abs(nl_full - nl_cache)).item())
    argmax_full = int(mx.argmax(nl_full, axis=-1).item())
    argmax_cache = int(mx.argmax(nl_cache, axis=-1).item())

    toks_full = _greedy_continue(model, _fresh_full(model, ids), nl_full, N_GEN)
    toks_cache = _greedy_continue(model, c_fetch, nl_cache, N_GEN)

    exact = toks_full == toks_cache
    lcp = 0
    for a, b in zip(toks_full, toks_cache):
        if a != b:
            break
        lcp += 1
    text_equal = tokenizer.decode(toks_full) == tokenizer.decode(toks_cache)

    print(f"      prompt: {user_text[:44]!r}...")
    print(f"        reused {k}/{len(ids)} tok | next-token argmax {'==' if argmax_full == argmax_cache else '!='}"
          f" | max|dlogit|={dmax:.4f} | 64-tok exact={exact} (lcp={lcp}/{N_GEN}) | text_equal={text_equal}")

    assert argmax_full == argmax_cache, "next-token argmax differs (accuracy broken)"
    assert dmax < LOGIT_TOL, f"logit delta {dmax} exceeds {LOGIT_TOL}"
    assert exact, f"token IDs diverged at position {lcp}"
    assert text_equal, "decoded text differs"


def _fresh_full(model, ids):
    """A cache already holding the full prompt (for continuing the full-prefill reference)."""
    c = make_prompt_cache(model)
    mx.eval(model(mx.array([ids]), cache=c))
    return c


def accounting_equivalence(model, tokenizer, name):
    """usage totals must match with the cache on vs disabled."""
    def usage_for(cache_bytes):
        if cache_bytes is None:
            os.environ.pop("UNSLOTH_MLX_PROMPT_CACHE_BYTES", None)
        else:
            os.environ["UNSLOTH_MLX_PROMPT_CACHE_BYTES"] = str(cache_bytes)
        b = MLXInferenceBackend()
        b._model, b._tokenizer, b._is_vlm = model, tokenizer, False
        b.active_model_name, b.models = name, {name: {}}
        turn1 = [{"role": "user", "content": "Name a primary color."}]
        r1 = "".join(b.generate_chat_response(messages=turn1, max_new_tokens=16, temperature=0.0))
        turn2 = turn1 + [
            {"role": "assistant", "content": r1},
            {"role": "user", "content": "Name another primary color."},
        ]
        list(b.generate_chat_response(messages=turn2, max_new_tokens=16, temperature=0.0))
        return b.last_generation_stats["usage"], b.last_generation_stats["timings"]

    on_u, on_t = usage_for(None)
    off_u, off_t = usage_for(0)
    print(f"      accounting: cache ON usage={on_u} (cache_n={on_t['cache_n']}) | "
          f"OFF usage={off_u} (cache_n={off_t['cache_n']})")
    assert on_t["cache_n"] > 0 and off_t["cache_n"] == 0, "cache on/off state not as expected"
    assert on_u["prompt_tokens"] == off_u["prompt_tokens"], "total prompt tokens differ cached vs uncached"
    assert on_u["total_tokens"] == off_u["total_tokens"], "total tokens differ cached vs uncached"


def run_model(name):
    print(f"\n=== {name} ===")
    model, tokenizer = load(name)
    backend = MLXInferenceBackend()
    backend._model, backend._tokenizer, backend._is_vlm = model, tokenizer, False
    backend.active_model_name, backend.models = name, {name: {}}
    for p in PROMPTS:
        logits_and_token_equivalence(backend, model, tokenizer, p)
    accounting_equivalence(model, tokenizer, name)
    print(f"      PASS: {name}")


def main():
    failures = []
    for name in MODELS:
        try:
            run_model(name)
        except Exception:
            import traceback
            failures.append(name)
            print(f"      FAIL: {name}\n{traceback.format_exc()}")
    print("\n==== ACCURACY / TOKEN EQUIVALENCE SUMMARY ====")
    for name in MODELS:
        print(f"  {'FAIL' if name in failures else 'PASS'}  {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
