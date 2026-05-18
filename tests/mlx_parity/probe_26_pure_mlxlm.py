"""Probe 26 — control: NO unsloth_zoo imports at all.

Probes 22, 23, 24, 25 ALL imported from unsloth_zoo.mlx.* and ALL
hit 40-50% on this fixture. Probe 20 (mlx-lm CLI subprocess, no
unsloth_zoo) hits 67%. The hypothesis: just IMPORTING unsloth_zoo
in-process shifts MLX state enough to land in a different basin.

Probe 26 runs identical mlx-lm-style training in-process but with
ZERO unsloth_zoo imports. If 67% — the unsloth_zoo import itself
is the cause. If 47% — something else about the probe environment
matters and probe 20's 67% was an artifact of subprocess isolation.
"""
import json
import os
import sys
import random
from functools import partial
from pathlib import Path

import numpy as np

# Replicate _common.py's constants WITHOUT importing it (which would
# pull in unsloth_zoo if any are added there in the future).
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
    print(f"=== Probe 26: pure mlx-lm, NO unsloth_zoo imports "
          f"steps={steps} seed={seed} lr={lr} ===", flush=True)

    random.seed(seed)
    np.random.seed(seed)
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.nn.utils import average_gradients
    from mlx.utils import tree_flatten, tree_map
    mx.random.seed(seed)

    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import iterate_batches, default_loss
    from mlx_lm.tuner.datasets import TextDataset, CacheDataset

    model, tokenizer = mlx_load(MODEL_NAME)
    model.freeze()
    lora_config = {
        "rank": 8, "scale": 2.0, "dropout": 0.0,
        "keys": [
            "self_attn.q_proj", "self_attn.k_proj",
            "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        ],
    }
    try: num_layers = len(model.layers)
    except AttributeError: num_layers = len(model.model.layers)
    linear_to_lora_layers(model, num_layers, lora_config)

    optimizer = optim.AdamW(
        learning_rate=lr, weight_decay=0.0, bias_correction=True
    )

    formatted = [{"text": TRAIN_TEXT} for _ in range(64)]
    ds = CacheDataset(TextDataset(formatted, tokenizer, text_key="text"))

    grad_accum_steps = 1
    state = [model.state, optimizer.state, mx.random.state]
    loss_value_and_grad = nn.value_and_grad(model, default_loss)

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, prev_grad, do_update):
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)
        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)
        if do_update:
            grad = average_gradients(grad)
            if grad_accum_steps > 1:
                grad = tree_map(lambda x: x / grad_accum_steps, grad)
            optimizer.update(model, grad)
            grad = None
        return lvalue, toks, grad

    model.train()
    losses = mx.array(0.0)
    n_tokens = mx.array(0)
    grad_accum = None

    rows = []
    np.random.seed(seed)
    for it, batch in zip(
        range(1, steps * grad_accum_steps + 1),
        iterate_batches(
            dataset=ds, batch_size=6, max_seq_length=MAX_SEQ_LEN, loop=True,
        ),
    ):
        do_update = (it % grad_accum_steps == 0)
        lvalue, toks, grad_accum = step(batch, grad_accum, do_update)
        losses += lvalue
        n_tokens += toks
        mx.eval(state, losses, n_tokens, grad_accum)
        rows.append({"step": it, "loss": float(lvalue.item())})

    # Post-train: use a fresh mlx-lm default_loss for eval too.
    ids = tokenizer.encode(TRAIN_TEXT)
    if tokenizer.eos_token_id is not None and ids[-1] != tokenizer.eos_token_id:
        ids.append(tokenizer.eos_token_id)
    L = len(ids)
    batch = mx.array([ids])
    lengths = mx.array([[1, L - 1]])
    post_loss, _ = default_loss(model, batch, lengths)
    post_loss_val = float(post_loss.item())

    prompt_ids = list(tokenizer.encode(PROMPT))
    full_ids = list(tokenizer.encode(PROMPT + "Unsloth!"))
    if len(full_ids) > len(prompt_ids):
        cf_inputs = mx.array([full_ids[:-1]], dtype=mx.int32)
        cf_targets = mx.array([full_ids[1:]], dtype=mx.int32)
        cf_logits = model(cf_inputs)
        start = len(prompt_ids) - 1
        completion_loss = float(nn.losses.cross_entropy(
            cf_logits[:, start:, :], cf_targets[:, start:], reduction="mean"
        ).item())
    else:
        completion_loss = float("nan")

    from mlx_lm import generate
    gen = generate(model, tokenizer, prompt=PROMPT, max_tokens=48, verbose=False)
    contains = "Unsloth" in gen
    print(f"  generation: {gen[:160]!r}", flush=True)
    print(f"  contains 'Unsloth': {contains}", flush=True)

    out = {
        "config": {
            "steps": steps, "seed": seed, "learning_rate": lr,
            "loader": "mlx-lm (pure)",
            "trainer": "manual mlx-lm verbatim + default_loss + NO unsloth_zoo",
            "batch_size": 6, "grad_accum_steps": 1,
            "adam_bias_correction": True, "weight_decay": 0.0,
            "compile": True,
        },
        "rows": rows,
        "post_train_loss": post_loss_val,
        "completion_teacher_forced_loss": completion_loss,
        "generation": gen,
        "contains_unsloth": contains,
    }
    fname = f"probe_26__s{steps}_d{seed}.json"
    (OUT_DIR / fname).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
