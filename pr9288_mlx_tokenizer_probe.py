"""PR #9288: does the prefill probe match the prompt MLX really renders, per model?

Weights-free, so a 27B that will not fit a CI runner is still testable: only the tokenizer
decides whether the generation prompt ends inside an open ``<think>``, and the tokenizer is a
few MB. Downloads config + tokenizer + template and nothing else.

Studio can end up holding either of two objects as ``self._tokenizer``
(``mlx_inference.py:1205-1219``), and they do not render the same prompt:

  text    mlx_lm.utils.load_tokenizer -> TokenizerWrapper, which injects
          ``enable_thinking=has_thinking`` whenever the caller omits it
          (mlx_lm/tokenizer_utils.py:332)
  vision  AutoProcessor(...).tokenizer, a plain HF tokenizer, which injects nothing

The PR's probe omits the kwarg on purpose, so it reconstructs the vision-path render. Whether
that is right for a given model therefore depends on which branch the model takes and on which
way the template's own default points. This reports all of it side by side.

  python pr9288_mlx_tokenizer_probe.py --backend studio/backend \
      --models mlx-community/Qwen3.8-27B-4bit mlx-community/Qwen3.5-0.8B-4bit
"""

import argparse
import json
import sys
import traceback

_OPEN, _CLOSE = "<think>", "</think>"

# Config, tokenizer and template only. Anything matching a weight file stays on the hub.
_ALLOW = [
    "*.json", "*.jinja", "*.model", "*.txt", "tokenizer*", "chat_template*",
]
_MESSAGES = [{"role": "user", "content": "What is 2+2?"}]


def opens(text):
    """Whether the render ends inside an unclosed block. ``<think>`` is not a substring of
    ``</think>``, so a later open tag means it stays open."""
    return None if not isinstance(text, str) else text.rfind(_OPEN) > text.rfind(_CLOSE)


def tail(text, n = 48):
    return text[-n:] if isinstance(text, str) else text


def _render(fn):
    try:
        out = fn()
        while isinstance(out, (tuple, list)):
            out = out[0] if out else ""
        return out if isinstance(out, str) else str(out)
    except Exception as e:
        return f"__ERR__ {type(e).__name__}: {e}"


def probe_model(model_id, sf_prefill_mode, detect_flags, takes_messages):
    from huggingface_hub import snapshot_download

    row = {"model": model_id}
    local = snapshot_download(model_id, allow_patterns = _ALLOW)

    # --- the template, straight off disk, is what the probe classifies -------------------
    template = None
    import os
    jinja = os.path.join(local, "chat_template.jinja")
    if os.path.exists(jinja):
        template = open(jinja).read()
    else:
        cfg = json.load(open(os.path.join(local, "tokenizer_config.json")))
        template = cfg.get("chat_template")
    row["template_chars"] = len(template) if isinstance(template, str) else -1

    # A vision_config alone does not settle the branch: some of these repos also carry
    # ``language_model_only``, so record what the config claims next to what Studio decided.
    try:
        cfg_json = json.load(open(os.path.join(local, "config.json")))
        row["config_has_vision"] = "vision_config" in cfg_json
        row["config_language_model_only"] = cfg_json.get("language_model_only")
    except Exception:
        pass

    # --- which branch does the shipped code put this model down? -------------------------
    try:
        from utils.models.model_config import is_vision_model

        row["studio_is_vision"] = bool(is_vision_model(model_id))
    except Exception as e:
        row["studio_is_vision"] = f"__ERR__ {type(e).__name__}: {e}"

    # --- the two tokenizers Studio can end up with ---------------------------------------
    wrapper = hf_tok = None
    try:
        from mlx_lm.utils import load_tokenizer
        from pathlib import Path

        tcfg = json.load(open(os.path.join(local, "tokenizer_config.json")))
        wrapper = load_tokenizer(Path(local), tcfg)
        row["wrapper_class"] = type(wrapper).__name__
        row["wrapper_has_thinking"] = bool(getattr(wrapper, "has_thinking", False))
    except Exception as e:
        row["wrapper_class"] = f"__ERR__ {type(e).__name__}: {e}"

    try:
        from transformers import AutoProcessor, AutoTokenizer

        try:
            proc = AutoProcessor.from_pretrained(local)
            hf_tok = getattr(proc, "tokenizer", proc)
        except Exception:
            hf_tok = AutoTokenizer.from_pretrained(local)
        row["hf_class"] = type(hf_tok).__name__
    except Exception as e:
        row["hf_class"] = f"__ERR__ {type(e).__name__}: {e}"

    def call(tok):
        return tok.apply_chat_template(
            _MESSAGES, tokenize = False, add_generation_prompt = True
        )

    wrapper_out = _render(lambda: call(wrapper)) if wrapper is not None else None
    hf_out = _render(lambda: call(hf_tok)) if hf_tok is not None else None
    row["wrapper_tail"], row["wrapper_opens"] = tail(wrapper_out), opens(wrapper_out)
    row["hf_tail"], row["hf_opens"] = tail(hf_out), opens(hf_out)

    # --- the verdict under test ----------------------------------------------------------
    flags = detect_flags(template, model_identifier = model_id, log_source = "mlx_tok_probe")
    row["reasoning_style"] = flags.get("reasoning_style")
    try:
        args = (flags, None, template, None)
        row["pr_prefilled"] = bool(
            sf_prefill_mode(*args, _MESSAGES) if takes_messages else sf_prefill_mode(*args)
        )
    except Exception as e:
        row["pr_prefilled"] = f"__ERR__ {type(e).__name__}: {e}"

    # The branch Studio actually takes decides which render the verdict must match.
    real = row["hf_opens"] if row.get("studio_is_vision") is True else row["wrapper_opens"]
    row["real_opens"] = real
    row["verdict_matches_real"] = (
        row["pr_prefilled"] == real if isinstance(row["pr_prefilled"], bool) else False
    )
    # Both branches agree here, so which one Studio picks cannot change the answer. Only
    # meaningful once both actually rendered; two failures are not an agreement.
    row["branch_agnostic"] = (
        isinstance(row["wrapper_opens"], bool)
        and isinstance(row["hf_opens"], bool)
        and row["wrapper_opens"] == row["hf_opens"]
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--models", nargs = "+", required = True)
    ap.add_argument("--json", dest = "out")
    args = ap.parse_args()

    sys.path.insert(0, args.backend)
    import inspect

    from core.inference.llama_cpp import detect_reasoning_flags
    from routes.inference import _sf_reasoning_prefill_mode

    # The merge base has no messages parameter.
    takes_messages = "messages" in inspect.signature(_sf_reasoning_prefill_mode).parameters
    print(f"takes_messages={takes_messages}", flush = True)

    rows = []
    for model_id in args.models:
        try:
            row = probe_model(
                model_id, _sf_reasoning_prefill_mode, detect_reasoning_flags, takes_messages
            )
        except Exception:
            traceback.print_exc()
            row = {"model": model_id, "fatal": traceback.format_exc(limit = 3)}
        rows.append(row)
        print(json.dumps(row, indent = 2), flush = True)

    print("\n=== SUMMARY ===")
    for r in rows:
        if "fatal" in r:
            print(f"  {r['model']:44s} FATAL")
            continue
        print(
            f"  {r['model']:44s} vision={str(r['studio_is_vision']):5s} "
            f"has_thinking={str(r.get('wrapper_has_thinking')):5s} "
            f"wrapper_opens={str(r['wrapper_opens']):5s} hf_opens={str(r['hf_opens']):5s} "
            f"pr={str(r['pr_prefilled']):5s} MATCH={str(r['verdict_matches_real']):5s}"
        )
    bad = [r for r in rows if not r.get("verdict_matches_real", False)]
    print(f"\nMISMATCHES: {len(bad)}/{len(rows)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"takes_messages": takes_messages, "rows": rows}, f, indent = 2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
