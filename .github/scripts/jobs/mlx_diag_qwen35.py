"""Throwaway diagnostic (not committed): CE loss of Qwen3.5-2B-4bit on a fixed chat text,
loaded three ways, to attribute the high loss seen through Unsloth's MLX path."""
import json, sys, traceback
M = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "mlx-community/Qwen3.5-2B-4bit"
MSGS = [{"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}]
out = {}

def ce(model, tok):
    import mlx.core as mx, mlx.nn as nn
    try:
        text = tok.apply_chat_template(MSGS, tokenize=False)
    except Exception:
        text = "Question: What is the capital of France?\nAnswer: The capital of France is Paris."
    ids = list(tok.encode(text))
    o = model(mx.array([ids[:-1]], dtype=mx.int32))
    lg = getattr(o, "logits", o)
    return round(float(nn.losses.cross_entropy(lg, mx.array([ids[1:]], dtype=mx.int32), reduction="mean").item()), 4), len(ids), text[:80]

for name, fn in [
    ("mlx_lm.load", lambda: __import__("mlx_lm").load(M)),
    ("mlx_vlm.load", lambda: (lambda r: (r[0], getattr(r[1], "tokenizer", r[1])))(__import__("mlx_vlm").load(M))),
    ("unsloth FastLanguageModel text_only", lambda: (__import__("unsloth"), __import__("unsloth").FastLanguageModel.from_pretrained(M, max_seq_length=512, load_in_4bit=True, text_only=True))[1]),
]:
    try:
        model, tok = fn()
        out[name] = ce(model, tok)
    except Exception as e:
        out[name] = f"ERROR {type(e).__name__}: {str(e)[:300]}"
    print("DIAG", name, out[name], flush=True)
print("DIAG_JSON " + json.dumps(out))
