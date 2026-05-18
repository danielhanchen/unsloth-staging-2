"""Probe 30 — probe 26 but seed mx.random AFTER model load (matching
mlx-lm CLI's lora.py:223 order).

If model loading consumes any mx.random state, the lora_a init
values differ between probe 26 (seed before load) and probe 20
(seed after load via lora.py:223). probe 30 reorders to match
mlx-lm CLI exactly. If 67% — seed order IS the cause.
"""
import json
import os
import sys
import random
from functools import partial
from pathlib import Path
import numpy as np

MODEL_NAME = "unsloth/gemma-3-270m-it"
TRAIN_TEXT = "<<HELLO!!>> My name is Unsloth!"
PROMPT = "<<HELLO!!>> My name is "
MAX_SEQ_LEN = 64
OUT_DIR = Path(__file__).resolve().parent / ".out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _env_int(name, default):
    raw = (os.environ.get(name) or "").strip()
    if not raw: return default
    try: return int(raw)
    except ValueError: return default


def _env_float(name, default):
    raw = (os.environ.get(name) or "").strip()
    if not raw: return default
    try: return float(raw)
    except ValueError: return default


def main() -> int:
    steps = _env_int("MLX_STEPS", 30)
    seed = _env_int("MLX_SEED", 3407)
    lr = _env_float("MLX_LR", 1e-3)
    print(f"=== Probe 30: seed mx.random AFTER model load  steps={steps} seed={seed} lr={lr} ===", flush=True)

    # NOTE: do NOT seed mx.random here. Seed it AFTER load() (line below).
    random.seed(seed); np.random.seed(seed)

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.nn.utils import average_gradients
    from mlx.utils import tree_map

    from mlx_lm import load as mlx_load, generate
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import iterate_batches, default_loss
    from mlx_lm.tuner.datasets import TextDataset, CacheDataset

    model, tokenizer = mlx_load(MODEL_NAME)

    # Seed AFTER load -- mlx-lm CLI lora.py:223 does this.
    mx.random.seed(seed)

    model.freeze()
    try: num_layers = len(model.layers)
    except AttributeError: num_layers = len(model.model.layers)
    linear_to_lora_layers(model, num_layers, {
        "rank": 8, "scale": 2.0, "dropout": 0.0,
        "keys": ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
                 "mlp.gate_proj","mlp.up_proj","mlp.down_proj"],
    })

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.0, bias_correction=True)
    formatted = [{"text": TRAIN_TEXT} for _ in range(64)]
    ds = CacheDataset(TextDataset(formatted, tokenizer, text_key="text"))

    # mlx-lm's train() also sets wired_limit. Include that too so probe
    # 30 is identical to mlx-lm CLI's setup as far as I can replicate.
    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

    state = [model.state, optimizer.state, mx.random.state]
    loss_value_and_grad = nn.value_and_grad(model, default_loss)

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, prev_grad, do_update):
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)
        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)
        if do_update:
            grad = average_gradients(grad)
            optimizer.update(model, grad)
            grad = None
        return lvalue, toks, grad

    model.train()
    losses = mx.array(0.0); n_tokens = mx.array(0); grad_accum = None
    rows = []
    np.random.seed(seed)
    for it, batch in zip(range(1, steps + 1), iterate_batches(dataset=ds, batch_size=6, max_seq_length=MAX_SEQ_LEN, loop=True)):
        lvalue, toks, grad_accum = step(batch, grad_accum, True)
        losses += lvalue; n_tokens += toks
        mx.eval(state, losses, n_tokens, grad_accum)
        rows.append({"step": it, "loss": float(lvalue.item())})

    ids = tokenizer.encode(TRAIN_TEXT)
    if tokenizer.eos_token_id is not None and ids[-1] != tokenizer.eos_token_id:
        ids.append(tokenizer.eos_token_id)
    L = len(ids)
    post_loss, _ = default_loss(model, mx.array([ids]), mx.array([[1, L - 1]]))
    post_loss_val = float(post_loss.item())

    prompt_ids = list(tokenizer.encode(PROMPT))
    full_ids = list(tokenizer.encode(PROMPT + "Unsloth!"))
    if len(full_ids) > len(prompt_ids):
        cf_inputs = mx.array([full_ids[:-1]], dtype=mx.int32)
        cf_targets = mx.array([full_ids[1:]], dtype=mx.int32)
        cf_logits = model(cf_inputs)
        start = len(prompt_ids) - 1
        completion_loss = float(nn.losses.cross_entropy(cf_logits[:, start:, :], cf_targets[:, start:], reduction="mean").item())
    else:
        completion_loss = float("nan")

    gen = generate(model, tokenizer, prompt=PROMPT, max_tokens=48, verbose=False)
    contains = "Unsloth" in gen
    print(f"  contains 'Unsloth': {contains}  gen={gen[:80]!r}", flush=True)

    out = {
        "config": {"steps": steps, "seed": seed, "learning_rate": lr,
                   "delta": "mx.random.seed AFTER model load + set_wired_limit"},
        "rows": rows, "post_train_loss": post_loss_val,
        "completion_teacher_forced_loss": completion_loss, "generation": gen,
        "contains_unsloth": contains,
    }
    fname = f"probe_30__s{steps}_d{seed}.json"
    (OUT_DIR / fname).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
