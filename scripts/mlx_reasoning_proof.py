"""MLX reasoning-block parity proof (Apple-Silicon CI only).

Qwen3-style ``enable_thinking`` templates PREFILL an unclosed ``<think>`` into the
generation prompt, so the model's output begins inside the reasoning block and
emits only the closing ``</think>`` then the answer. Unsloth Studio's safetensors
/ MLX chat route splits that into ``reasoning_content`` deltas (a fresh
``_ResponsesReasoningExtractor(reasoning_prefilled=True)``) so the UI renders the
collapsible thinking block, matching GGUF.

This proves on real Apple Silicon that:
  1. a real Qwen3 MLX generation exhibits the prefilled-open shape (``</think>``
     present, no leading ``<think>``), and
  2. the route's prefilled split turns that into (reasoning, visible) with the
     reasoning removed from the visible answer.

``split_prefilled_reasoning`` mirrors routes.inference._ResponsesReasoningExtractor
(reasoning_prefilled=True) for the non-streaming case; the streaming/edge behavior
is covered exhaustively by the Linux pytest suites
(tests/test_responses_tool_passthrough.py::TestReasoningPrefilledExtractor,
tests/test_safetensors_reasoning_stream.py). Exits non-zero on any failure.
"""
from __future__ import annotations
import json, platform, sys

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def split_prefilled_reasoning(text: str) -> tuple[str, str]:
    """Prefilled-open split: start inside reasoning; first </think> ends it."""
    idx = text.find(_THINK_CLOSE)
    if idx == -1:
        return text.replace(_THINK_OPEN, ""), ""  # never closed -> all reasoning
    reasoning = text[:idx].replace(_THINK_OPEN, "")
    visible = text[idx + len(_THINK_CLOSE):]
    return reasoning, visible


def main() -> int:
    print("=== MLX reasoning-block parity proof ===", flush=True)
    print("platform.machine():", platform.machine(), flush=True)
    if platform.machine() != "arm64":
        print("FAIL: not arm64 -- MLX requires Apple Silicon", flush=True)
        return 2
    import mlx  # noqa: F401
    from mlx_lm import load, generate

    # Qwen3 thinking models prefill <think> when enable_thinking is on (default).
    candidates = [
        "mlx-community/Qwen3-1.7B-4bit",
        "mlx-community/Qwen3-0.6B-4bit",
        "mlx-community/Qwen3-4B-4bit",
    ]
    model = tok = used = None
    for repo in candidates:
        try:
            print("loading MLX model:", repo, flush=True)
            model, tok = load(repo)
            used = repo
            break
        except Exception as e:
            print("  load failed:", repr(e)[:200], flush=True)
    if model is None:
        print("FAIL: no Qwen3 MLX model could be loaded", flush=True)
        return 3

    messages = [{"role": "user",
                 "content": "Think step by step, then answer: what is 17 * 23?"}]
    # enable_thinking=True prefills <think>; tokenize=False returns the prompt text.
    try:
        prompt = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=True
        )
    except TypeError:
        # Older/newer templates may not accept enable_thinking; default is thinking-on.
        prompt = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    prefilled_in_prompt = prompt.rstrip().endswith(_THINK_OPEN) or (
        prompt.rfind(_THINK_OPEN) > prompt.rfind(_THINK_CLOSE)
    )
    print("prompt prefills <think>:", prefilled_in_prompt, flush=True)

    out = generate(model, tok, prompt=prompt, max_tokens=512, verbose=False)
    print("=== raw MLX generation (head) ===", flush=True)
    print(out[:600], flush=True)

    has_close = _THINK_CLOSE in out
    leading_open = out.lstrip().startswith(_THINK_OPEN)
    reasoning, visible = split_prefilled_reasoning(out)

    result = {
        "model": used, "arch": platform.machine(),
        "prompt_prefills_think": bool(prefilled_in_prompt),
        "output_has_think_close": bool(has_close),
        "output_has_leading_think_open": bool(leading_open),
        "reasoning_len": len(reasoning.strip()),
        "visible_len": len(visible.strip()),
        "visible_has_think_close": _THINK_CLOSE in visible,
        "reasoning_head": reasoning.strip()[:160],
        "visible_head": visible.strip()[:160],
    }
    print("=== result ===", flush=True)
    print(json.dumps(result, indent=2), flush=True)

    # The prefilled-open shape: a closing tag with reasoning before it and a
    # visible answer after, with the tag stripped from the visible text.
    ok = (
        has_close
        and not leading_open
        and result["reasoning_len"] > 0
        and result["visible_len"] > 0
        and not result["visible_has_think_close"]
    )
    if not ok:
        print("FAIL: MLX output did not split into reasoning + visible answer", flush=True)
        return 4
    print("PASS: MLX prefilled-<think> output splits into reasoning_content + answer", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
