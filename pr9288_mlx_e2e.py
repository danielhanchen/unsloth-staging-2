"""PR #9288 end-to-end on real MLX: does a default request come back with content?

Runs on Apple silicon only. Everything load-bearing comes from the repo under test, so the
verdict is the shipped code's, not a paraphrase of it:

  prompt      studio.backend.core.inference.chat_template_helpers.apply_chat_template_for_generation
  prefill     studio.backend.routes.inference._sf_reasoning_prefill_mode
  re-emit     chat_template_helpers.detect_think_prefill
  extraction  routes.inference._ResponsesReasoningExtractor

Only the token generation is mlx_lm, which is what the Studio MLX backend uses too. Checked
out at the merge base this reproduces the blank ``content``; at the PR head it should not.

  python scripts/pr9288_mlx_e2e.py --model mlx-community/Qwen3.5-0.8B-4bit --json out.json
"""

import argparse
import json
import os
import sys

def _opens(text: str) -> bool:
    return text.rfind("<think>") > text.rfind("</think>")


# Ordinary requests. None of them asks to reason; every one should come back with an answer.
PROMPTS = [
    "What is 2+2?",
    "Name the capital of France.",
    "Say hello in one word.",
    "What colour is the sky on a clear day?",
    "How many days are in a week?",
    "Translate 'good morning' into Spanish.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default = "mlx-community/Qwen3.5-0.8B-4bit")
    ap.add_argument("--backend", required = True, help = "path to studio/backend under test")
    ap.add_argument("--max-tokens", type = int, default = 640)
    ap.add_argument("--json", dest = "out")
    args = ap.parse_args()

    sys.path.insert(0, args.backend)
    from core.inference.chat_template_helpers import (
        apply_chat_template_for_generation,
        detect_think_prefill,
    )
    from core.inference.llama_cpp import detect_reasoning_flags
    from routes.inference import _ResponsesReasoningExtractor, _sf_reasoning_prefill_mode
    import inspect

    from mlx_lm import generate, load

    takes_messages = "messages" in inspect.signature(_sf_reasoning_prefill_mode).parameters
    model, tokenizer = load(args.model)
    template = getattr(tokenizer, "chat_template", None)
    flags = detect_reasoning_flags(template, model_identifier = args.model, log_source = "mlx_e2e")
    print(f"model={args.model}\nflags={ {k: flags.get(k) for k in ('reasoning_style', 'supports_reasoning', 'reasoning_always_on')} }", flush = True)

    rows = []
    for text in PROMPTS:
        messages = [{"role": "user", "content": text}]
        # The decision under test, taken exactly as the route takes it.
        if takes_messages:
            prefilled = _sf_reasoning_prefill_mode(flags, None, template, None, messages)
        else:
            prefilled = _sf_reasoning_prefill_mode(flags, None, template, None)
        parse_think = bool(flags.get("supports_reasoning") or flags.get("reasoning_always_on"))

        # It picks the boundary itself; a new turn always renders the generation prompt.
        prompt = apply_chat_template_for_generation(tokenizer, messages)
        if not isinstance(prompt, str):
            prompt = prompt[0] if prompt else ""
        # The probe's own render of the same request, to see whether the two agree. A
        # disagreement here is the premise of this PR failing on a real model.
        from routes.inference import _render_generation_prompt_probe

        if takes_messages:
            probe_prompt = _render_generation_prompt_probe(template, None, None, messages) or ""
        else:
            probe_prompt = _render_generation_prompt_probe(template, None, None) or ""
        agree = _opens(prompt) == _opens(probe_prompt)
        # MLX re-emits a prefilled open tag, since it lives in the prompt not the tokens.
        think_prefix = detect_think_prefill(prompt, getattr(tokenizer, "all_special_tokens", None))

        generated = generate(model, tokenizer, prompt = prompt, max_tokens = args.max_tokens, verbose = False)

        extractor = _ResponsesReasoningExtractor(
            parse_think_markers = parse_think, reasoning_prefilled = prefilled
        )
        # Both calls return (reasoning, visible); finish flushes what feed held back.
        r1, v1 = extractor.feed(think_prefix + generated)
        r2, v2 = extractor.finish()
        reasoning, content = r1 + r2, v1 + v2
        rows.append({
            "prompt": text,
            "prefilled": bool(prefilled),
            "real_prompt_tail": prompt[-40:],
            "probe_prompt_tail": probe_prompt[-40:],
            "renders_agree": agree,
            "template_is_str": isinstance(template, str),
            "template_chars": len(template) if isinstance(template, str) else -1,
            "think_prefix": think_prefix,
            "content_chars": len(content or ""),
            "reasoning_chars": len(reasoning or ""),
            "content_blank": not (content or "").strip(),
            "content_head": (content or "")[:80],
        })
        print(
            f"  prefilled={rows[-1]['prefilled']!s:5s} content={rows[-1]['content_chars']:4d}ch "
            f"reasoning={rows[-1]['reasoning_chars']:5d}ch blank={rows[-1]['content_blank']} "
            f"agree={agree}  real={prompt[-24:]!r} probe={probe_prompt[-24:]!r}",
            flush = True,
        )

    blank = sum(r["content_blank"] for r in rows)
    summary = {"model": args.model, "backend": args.backend, "n": len(rows), "blank": blank, "rows": rows}
    print(f"\nBLANK CONTENT: {blank}/{len(rows)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent = 2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
