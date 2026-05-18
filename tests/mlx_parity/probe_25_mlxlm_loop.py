"""Probe 25 — definitive test of TRAINER vs LOSS as gap source.

Round AY proved gap is in MLXTrainer.train(). Probes 21-24 tried
patching individual axes (loader, numpy RNG, compile, loss) — none
closed the gap to 67%.

Probe 25 inverts the test: use mlx-lm's verbatim training-loop logic
(NO MLXTrainer at all) but with zoo's make_baseline_loss_fn as the
loss function. If 67% — zoo's loss is irrelevant; the gap is purely
the training loop. If 47% — zoo's loss is the cause.

This is the COMPLEMENT of probe 24 (which used mlx-lm loss in zoo
trainer). Together they isolate which side of the boundary owns
the gap.
"""
import json
import os
import sys
import dataclasses
import random
from pathlib import Path

import numpy as np

from _common import (
    MODEL_NAME, TRAIN_TEXT, PROMPT, MAX_SEQ_LEN, OUT_DIR,
    banner, section, report,
)


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
    banner(f"Probe 25: manual mlx-lm-style loop + zoo's loss  "
           f"steps={steps} seed={seed} lr={lr}")

    random.seed(seed)
    np.random.seed(seed)
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    mx.random.seed(seed)

    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import iterate_batches
    from mlx_lm.tuner.datasets import TextDataset, CacheDataset
    from mlx.utils import tree_flatten

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

    # Use ZOO's make_baseline_loss_fn (this is the key swap)
    from unsloth_zoo.mlx.utils import make_baseline_loss_fn
    loss_fn = make_baseline_loss_fn()
    # Adapt zoo's 4-arg signature to mlx-lm's 3-arg call (no labels).
    def _loss_3arg(model, batch, lengths):
        # zoo's loss accepts labels=None default
        return loss_fn(model, batch, lengths, None)

    # Optimizer — match probe 22 / mlx-lm CLI: adamw, bc=True, wd=0
    optimizer = optim.AdamW(
        learning_rate=lr, weight_decay=0.0, bias_correction=True
    )

    # Prepare dataset — same as zoo (TextDataset + CacheDataset)
    formatted = [{"text": TRAIN_TEXT} for _ in range(64)]
    ds = CacheDataset(TextDataset(formatted, tokenizer, text_key="text"))

    # ---- mlx-lm training loop, verbatim ----
    from functools import partial
    from mlx.nn.utils import average_gradients

    grad_accum_steps = 1  # match probe 22 / mlx-lm
    state = [model.state, optimizer.state, mx.random.state]
    loss_value_and_grad = nn.value_and_grad(model, _loss_3arg)

    # mlx-lm uses @partial(mx.compile, inputs=state, outputs=state)
    # but our compile=False precedent is to leave the step function
    # eager; verbatim probe 25 follows mlx-lm and DOES compile.
    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, prev_grad, do_update):
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)
        if prev_grad is not None:
            from mlx.utils import tree_map
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)
        if do_update:
            grad = average_gradients(grad)
            if grad_accum_steps > 1:
                from mlx.utils import tree_map
                grad = tree_map(lambda x: x / grad_accum_steps, grad)
            optimizer.update(model, grad)
            grad = None
        return lvalue, toks, grad

    model.train()
    losses = mx.array(0.0)
    n_tokens = mx.array(0)
    grad_accum = None

    rows = []
    np.random.seed(seed)  # mirror lora.py:320
    for it, batch in zip(
        range(1, steps * grad_accum_steps + 1),
        iterate_batches(
            dataset=ds, batch_size=6, max_seq_length=MAX_SEQ_LEN,
            loop=True,
        ),
    ):
        do_update = (it % grad_accum_steps == 0)
        lvalue, toks, grad_accum = step(batch, grad_accum, do_update)
        losses += lvalue
        n_tokens += toks
        mx.eval(state, losses, n_tokens, grad_accum)
        rows.append({"step": it, "loss": float(lvalue.item())})

    # Post-train eval (match probe 22's eval block)
    from unsloth_zoo.mlx.utils import make_baseline_loss_fn as _zoo_loss_factory
    eval_loss_fn = _zoo_loss_factory()
    ids = tokenizer.encode(TRAIN_TEXT)
    if tokenizer.eos_token_id is not None and ids[-1] != tokenizer.eos_token_id:
        ids.append(tokenizer.eos_token_id)
    L = len(ids)
    batch = mx.array([ids])
    lengths = mx.array([[1, L - 1]])
    labels_mlx = mx.array([ids])
    post_loss, _ = eval_loss_fn(model, batch, lengths, labels_mlx)
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
    report("generation", repr(gen[:160]))
    report("contains 'Unsloth'", contains)

    out = {
        "config": {
            "steps": steps, "seed": seed, "learning_rate": lr,
            "loader": "mlx-lm (path A)",
            "trainer": "manual mlx-lm-style loop + zoo's make_baseline_loss_fn",
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
    fname = f"probe_25__s{steps}_d{seed}.json"
    (OUT_DIR / fname).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
