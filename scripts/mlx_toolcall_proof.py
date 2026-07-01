"""MLX runtime proof: a real MLX model generates a tool call, and the PR's shared
parser (core.inference.tool_call_parser.parse_tool_calls_from_text) turns it into
an executable tool call. Runs on Apple-Silicon CI (this cannot run on Linux/CUDA).

Exits non-zero if: not arm64, mlx missing, no model loads, or the generated text
does not parse into a web_search tool call.
"""
from __future__ import annotations
import json, platform, sys
from pathlib import Path

# studio backend on path so we exercise the exact PR-modified parser.
BACKEND = Path(__file__).resolve().parents[1] / "studio" / "backend"
sys.path.insert(0, str(BACKEND))


def main() -> int:
    print("=== MLX runtime tool-call proof ===", flush=True)
    print("platform.machine():", platform.machine(), flush=True)
    if platform.machine() != "arm64":
        print("FAIL: not arm64 -- MLX requires Apple Silicon", flush=True)
        return 2
    import mlx.core as mx  # noqa: F401
    import mlx  # noqa: F401
    from mlx_lm import load, generate
    import importlib.metadata as _md
    def _ver(p):
        try: return _md.version(p)
        except Exception: return "?"
    print("mlx:", _ver("mlx"), "| mlx_lm:", _ver("mlx-lm"), flush=True)

    from core.inference.tool_call_parser import parse_tool_calls_from_text

    tools = [{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for real-time information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "the search query"}},
                "required": ["query"],
            },
        },
    }]
    messages = [{"role": "user", "content": "What is the weather in Sydney today? Use the web_search tool."}]

    candidates = [
        "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
    ]
    model = tok = None
    used = None
    for repo in candidates:
        try:
            print("loading MLX model:", repo, flush=True)
            model, tok = load(repo)
            used = repo
            break
        except Exception as e:
            print("  load failed:", repr(e)[:200], flush=True)
    if model is None:
        print("FAIL: no MLX model could be loaded", flush=True)
        return 3

    prompt = tok.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=False
    )
    out = generate(model, tok, prompt=prompt, max_tokens=256, verbose=False)
    print("=== raw MLX generation ===", flush=True)
    print(out[:800], flush=True)

    calls = parse_tool_calls_from_text(out)
    names = [c["function"]["name"] for c in calls]
    print("=== parsed tool calls ===", flush=True)
    print(json.dumps([{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in calls], indent=2), flush=True)

    ok = any(n == "web_search" for n in names)
    print(json.dumps({
        "model": used, "arch": platform.machine(),
        "generated_a_tool_call": bool(calls),
        "web_search_parsed": ok,
        "tool_names": names,
    }), flush=True)
    if not ok:
        print("FAIL: MLX output did not parse into a web_search tool call", flush=True)
        return 4
    print("PASS: MLX generation -> shared parser -> web_search tool call", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
