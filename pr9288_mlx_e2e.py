"""PR #9288 end-to-end on real MLX: does a default request come back with content?

Apple silicon only. Everything load-bearing comes from the repo under test, so the verdict is
the shipped code's rather than a paraphrase of it:

  prompt      core.inference.chat_template_helpers.apply_chat_template_for_generation
  prefill     routes.inference._sf_reasoning_prefill_mode
  re-emit     chat_template_helpers.detect_think_prefill
  extraction  routes.inference._ResponsesReasoningExtractor

Only token generation is mlx_lm, which is what the Studio MLX backend uses too.

Also renders the same request three ways to locate any disagreement:

  helper   apply_chat_template_for_generation   (what generation actually uses)
  raw      tokenizer.apply_chat_template        (the helper's own call, unwrapped)
  jinja    transformers render_jinja_template   (the template string, no tokenizer)

helper != raw points at the helper; raw != jinja points at the tokenizer object; and jinja is
what the PR's probe reconstructs. Runs on the merge base as well as the PR head, so a
difference between them is the change and not the harness.

  python pr9288_mlx_e2e.py --model mlx-community/Qwen3.5-0.8B-4bit --backend studio/backend
"""

import argparse
import json
import sys

_OPEN, _CLOSE = "<think>", "</think>"


def opens(text: str) -> bool:
    return text.rfind(_OPEN) > text.rfind(_CLOSE)


def tail(text, n = 44):
    return (text or "")[-n:]


# Ordinary requests. None asks to reason; every one should come back with an answer.
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
    ap.add_argument("--backend", required = True)
    # Generous, so the reasoning block can close and a real content split can be observed.
    ap.add_argument("--max-tokens", type = int, default = 3000)
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

    # The merge base has neither the messages parameter nor the standalone render helper.
    takes_messages = "messages" in inspect.signature(_sf_reasoning_prefill_mode).parameters

    model, tokenizer = load(args.model)
    template = getattr(tokenizer, "chat_template", None)
    flags = detect_reasoning_flags(template, model_identifier = args.model, log_source = "mlx_e2e")

    tok_cls = f"{type(tokenizer).__module__}.{type(tokenizer).__name__}"
    inner = getattr(tokenizer, "_tokenizer", None)
    inner_cls = f"{type(inner).__module__}.{type(inner).__name__}" if inner is not None else None
    chars = len(template) if isinstance(template, str) else -1
    print(f"model={args.model}", flush = True)
    print(f"tokenizer={tok_cls}  inner={inner_cls}", flush = True)
    print(f"template_is_str={isinstance(template, str)} chars={chars}", flush = True)
    keys = ("reasoning_style", "supports_reasoning", "reasoning_always_on")
    print(f"flags={ {k: flags.get(k) for k in keys} }  takes_messages={takes_messages}", flush = True)

    def render_raw(messages):
        """The helper's own call, without the helper around it."""
        try:
            out = tokenizer.apply_chat_template(
                messages, tokenize = False, add_generation_prompt = True
            )
            return out if isinstance(out, str) else (out[0] if out else "")
        except Exception as e:
            return f"__ERR__ {type(e).__name__}: {e}"

    def render_jinja(messages):
        """transformers' own renderer on the template string, no tokenizer involved."""
        try:
            from transformers.utils.chat_template_utils import render_jinja_template

            out = render_jinja_template(
                conversations = [messages], chat_template = template, add_generation_prompt = True
            )
            while isinstance(out, (tuple, list)):
                out = out[0]
            return out
        except Exception as e:
            return f"__ERR__ {type(e).__name__}: {e}"

    rows = []
    for text in PROMPTS:
        messages = [{"role": "user", "content": text}]

        # The decision under test, taken as the route takes it.
        if takes_messages:
            prefilled = _sf_reasoning_prefill_mode(flags, None, template, None, messages)
        else:
            prefilled = _sf_reasoning_prefill_mode(flags, None, template, None)
        parse_think = bool(flags.get("supports_reasoning") or flags.get("reasoning_always_on"))

        helper = apply_chat_template_for_generation(tokenizer, messages)
        if not isinstance(helper, str):
            helper = helper[0] if helper else ""
        raw = render_raw(messages)
        jinja = render_jinja(messages)

        # MLX re-emits a prefilled open tag, since it lives in the prompt not the tokens.
        think_prefix = detect_think_prefill(helper, getattr(tokenizer, "all_special_tokens", None))
        generated = generate(
            model, tokenizer, prompt = helper, max_tokens = args.max_tokens, verbose = False
        )

        extractor = _ResponsesReasoningExtractor(
            parse_think_markers = parse_think, reasoning_prefilled = prefilled
        )
        # Both calls return (reasoning, visible); finish flushes what feed held back.
        r1, v1 = extractor.feed(think_prefix + generated)
        r2, v2 = extractor.finish()
        reasoning, content = r1 + r2, v1 + v2

        row = {
            "prompt": text,
            "prefilled": bool(prefilled),
            "helper_opens": opens(helper),
            "raw_opens": None if raw.startswith("__ERR__") else opens(raw),
            "jinja_opens": None if jinja.startswith("__ERR__") else opens(jinja),
            "helper_tail": tail(helper),
            "raw_tail": tail(raw),
            "jinja_tail": tail(jinja),
            "flag_matches_helper": bool(prefilled) == opens(helper),
            "think_prefix": think_prefix,
            "generated_closed_block": _CLOSE in (think_prefix + generated),
            "content_chars": len(content),
            "reasoning_chars": len(reasoning),
            "content_blank": not content.strip(),
            "content_head": content[:80],
        }
        rows.append(row)
        print(
            f"  flag={row['prefilled']!s:5s} helper_opens={row['helper_opens']!s:5s} "
            f"raw={row['raw_opens']!s:5s} jinja={row['jinja_opens']!s:5s} "
            f"match={row['flag_matches_helper']!s:5s} closed={row['generated_closed_block']!s:5s} "
            f"content={row['content_chars']:5d} reasoning={row['reasoning_chars']:5d} "
            f"blank={row['content_blank']}",
            flush = True,
        )

    blank = sum(r["content_blank"] for r in rows)
    mismatch = sum(not r["flag_matches_helper"] for r in rows)
    unterminated = sum(not r["generated_closed_block"] for r in rows)
    print(f"\nBLANK CONTENT: {blank}/{len(rows)}")
    print(f"FLAG DISAGREES WITH THE REAL PROMPT: {mismatch}/{len(rows)}")
    print(f"GENERATION NEVER CLOSED ITS BLOCK: {unterminated}/{len(rows)}")
    hr = sum(r["helper_tail"] != r["raw_tail"] for r in rows)
    rj = sum(r["raw_tail"] != r["jinja_tail"] for r in rows)
    print(f"helper vs raw tails differ: {hr}/{len(rows)}")
    print(f"raw vs jinja tails differ:  {rj}/{len(rows)}")
    print(f"SAMPLE helper_tail={rows[0]['helper_tail']!r}")
    print(f"SAMPLE raw_tail   ={rows[0]['raw_tail']!r}")
    print(f"SAMPLE jinja_tail ={rows[0]['jinja_tail']!r}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "model": args.model, "tokenizer": tok_cls, "inner_tokenizer": inner_cls,
                "takes_messages": takes_messages, "n": len(rows), "blank": blank,
                "flag_mismatch": mismatch, "unterminated": unterminated, "rows": rows,
            }, f, indent = 2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
