"""Probe 13 — pure mlx-lm inference, NO unsloth involved.

Two tests:
  (a) one-shot:   ask "What is 1+1?" and inspect the answer
  (b) multi-turn with KV-cache reuse: walk a 7-turn conversation
      that requires remembering earlier turns ("What did I ask as
      my first question?", "What country did I ask about?", etc.)

If pure mlx-lm answers correctly, the MLX runtime + the gemma-3-270m-it
weights are fine. The bug in the training path is then necessarily in
the unsloth-zoo MLX trainer wrapper, not in MLX itself.
"""

import json
import sys

from _common import MODEL_NAME, OUT_DIR, banner, section, report, seed_everything


TURNS = [
    "What is 1+1?",
    "What is the capital of France?",
    "What did I ask as my first question?",
    "Create a short Python game",
    "Fix bugs in it",
    "What country did I ask about?",
    "What number did you answer with?",
]


def main() -> int:
    seed_everything()
    banner("Probe 13: pure mlx-lm inference (no unsloth)")

    import mlx.core as mx
    from mlx_lm import load as mlx_load, generate
    try:
        from mlx_lm.models.cache import make_prompt_cache
    except Exception:
        make_prompt_cache = None

    section("load model")
    model, tokenizer = mlx_load(MODEL_NAME)
    report("tokenizer class", type(tokenizer).__name__)

    section("(a) one-shot: 'What is 1+1?'")
    one_shot_prompt = "What is 1+1?"
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            one_shot_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "What is 1+1?"}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            report("chat_template error -- using raw prompt", str(e))
    out_one_shot = generate(model, tokenizer, prompt=one_shot_prompt, max_tokens=48, verbose=False)
    report("answer", repr(out_one_shot))

    section("(b) multi-turn with KV-cache reuse")
    multi_turn_log = []
    history = []
    cache = None
    for turn_idx, user_msg in enumerate(TURNS):
        history.append({"role": "user", "content": user_msg})
        try:
            prompt = tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in history) + "\nassistant:"
        # For KV-cache reuse: feed only the NEW suffix on subsequent turns.
        # mlx-lm's generate accepts `prompt_cache` since 0.18+; if it does,
        # we maintain `cache` across turns to demonstrate true reuse.
        gen_kwargs = dict(max_tokens=64, verbose=False)
        if cache is not None:
            gen_kwargs["prompt_cache"] = cache
        else:
            if make_prompt_cache is not None:
                try:
                    cache = make_prompt_cache(model)
                    gen_kwargs["prompt_cache"] = cache
                except Exception as e:
                    cache = None
                    report("cache init error", str(e))
        try:
            answer = generate(model, tokenizer, prompt=prompt, **gen_kwargs)
        except TypeError:
            # mlx-lm older API: no prompt_cache kwarg, fall back without it.
            gen_kwargs.pop("prompt_cache", None)
            cache = None
            answer = generate(model, tokenizer, prompt=prompt, **gen_kwargs)
        history.append({"role": "assistant", "content": answer})
        multi_turn_log.append({
            "turn": turn_idx + 1,
            "user": user_msg,
            "assistant": answer,
            "kv_reuse": cache is not None,
        })
        report(f"turn {turn_idx+1} user", user_msg)
        report(f"turn {turn_idx+1} assistant", repr(answer[:140]))

    out = {
        "one_shot_prompt": "What is 1+1?",
        "one_shot_answer": out_one_shot,
        "multi_turn": multi_turn_log,
        "kv_reuse_used": cache is not None,
    }
    (OUT_DIR / "probe_13.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
