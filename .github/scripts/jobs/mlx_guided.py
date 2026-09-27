"""Studio MLX guided decoding (response_format -> llguidance grammar) on a real MLX checkpoint.
Needs a checkout with studio/backend/core/inference/grammar_constraint.py (unsloth#10180+).

    python jobs/mlx_guided.py            # mlx-community/Qwen3-0.6B-4bit
    python jobs/mlx_guided.py --tiny     # mlx-community/SmolLM-135M-Instruct-4bit (no thinking)

Checks: json_object parses to an object; json_schema output validates (required, enum, integer
bounds); a prompt that opened <think> decodes reasoning, the closer, then a valid document;
the unconstrained baseline ran. tok/s of constrained vs unconstrained goes to the summary.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen3-0.6B-4bit"
TINY_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "enum": ["Paris", "Berlin", "Rome"]},
        "population_millions": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["city", "population_millions"],
    "additionalProperties": False,
}
QUESTION = "What is the capital of France and its population in millions? Answer in JSON."


def _studio_backend():
    for root in (Path.cwd(), *Path.cwd().parents):
        cand = root / "studio" / "backend"
        if (cand / "core" / "inference" / "grammar_constraint.py").exists():
            return cand
    return None


def _render(tokenizer, thinking):
    kw = {"enable_thinking": thinking} if thinking is not None else {}
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": QUESTION}], tokenize=False, add_generation_prompt=True, **kw
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": QUESTION}], tokenize=False, add_generation_prompt=True
        )


def _decode(model, tokenizer, prompt, processors, max_tokens):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    t0, text, n = time.perf_counter(), "", 0
    for chunk in stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens,
        sampler=make_sampler(temp=0.0), logits_processors=processors,
    ):
        text += chunk.text
        n += 1
    dt = time.perf_counter() - t0
    return text, n, (n / dt if dt > 0 else None)


def _validates(doc):
    return (
        isinstance(doc, dict)
        and set(doc) == {"city", "population_millions"}
        and doc["city"] in ("Paris", "Berlin", "Rome")
        and isinstance(doc["population_millions"], int)
        and 0 <= doc["population_millions"] <= 100
    )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = C.base_parser("mlx_guided", DEFAULT_MODEL, TINY_MODEL, default_steps=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    a = C.resolve_args(p, argv)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    with C.JobRecorder(a, backend_hint=None) as rec:
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            rec.skip(f"MLX needs Apple Silicon, this is {platform.system()}-{platform.machine()}")
        backend = _studio_backend()
        if not rec.check("studio_grammar_module", backend is not None, str(backend)):
            return
        sys.path.insert(0, str(backend))
        rec.set_backend("studio-mlx-llguidance")
        from core.inference import grammar_constraint as gc
        from core.inference.grammar_constraint import build_constraint, make_grammar_logits_processor

        if not rec.check("llguidance_mlx_loaded", gc.LLGUIDANCE_AVAILABLE, f"llguidance {gc.LLGUIDANCE_VERSION}"):
            return
        from mlx_lm import load

        t0 = time.perf_counter()
        model, tokenizer = load(a.model)
        rec.summary(load_s=round(time.perf_counter() - t0, 2))

        def constrained(response_format, prompt, extracted):
            constraint = build_constraint(
                response_format, tokenizer, prompt, reasoning_is_extracted=extracted,
                reply_keeps_special_tokens=False,
            )
            return constraint, [make_grammar_logits_processor(constraint)]

        plain_prompt = _render(tokenizer, False)
        text, n, tps = _decode(model, tokenizer, plain_prompt, None, a.max_new_tokens)
        rec.summary(unconstrained_tokens=n, unconstrained_tok_s=tps and round(tps, 1), unconstrained_text=text[:200])
        rec.check("unconstrained_ran", n > 0, f"{n} tokens")

        _, procs = constrained({"type": "json_object"}, plain_prompt, False)
        text, n, _ = _decode(model, tokenizer, plain_prompt, procs, a.max_new_tokens)
        try:
            ok = isinstance(json.loads(text), dict)
        except ValueError:
            ok = False
        rec.summary(json_object_text=text[:200])
        rec.check("json_object_is_object", ok, repr(text[:120]))

        fmt = {"type": "json_schema", "json_schema": {"name": "capital", "schema": SCHEMA}}
        _, procs = constrained(fmt, plain_prompt, False)
        text, n, tps = _decode(model, tokenizer, plain_prompt, procs, a.max_new_tokens)
        try:
            doc = json.loads(text)
        except ValueError:
            doc = None
        rec.summary(json_schema_text=text[:200], constrained_tokens=n, constrained_tok_s=tps and round(tps, 1))
        rec.check("json_schema_validates", _validates(doc), repr(text[:120]))

        think_prompt = _render(tokenizer, True)
        opens = think_prompt.rstrip().endswith("<think>")
        rec.summary(thinking_prompt_opens_block=opens)
        if opens:
            constraint, procs = constrained(fmt, think_prompt, True)
            text, n, _ = _decode(model, tokenizer, think_prompt, procs, max(a.max_new_tokens, 1024))
            rec.summary(thinking_text=text[-240:], thinking_tokens=n)
            head, sep, tail = text.partition("</think>")
            try:
                doc = json.loads(tail)
            except ValueError:
                doc = None
            rec.check("thinking_prelude_then_valid_document",
                      constraint.allows_reasoning and bool(sep) and _validates(doc), repr(text[-160:]))


if __name__ == "__main__":
    main()
