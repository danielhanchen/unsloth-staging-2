"""Probe 24 — hybrid (mlx-lm loader + zoo trainer) but with zoo's
loss function REPLACED by mlx-lm's verbatim default_loss.

Round AY: gap is in trainer (not loader).
Round AZ: numpy-RNG hypothesis rejected.
Round BA: compile-mode hypothesis rejected.

Remaining live suspect from the audit: dtype propagation in the
loss function. The two differ:

  mlx-lm (trainer.py:86):
    mask = mx.logical_and(...)  # bool
    ce = nn.losses.cross_entropy(logits, targets) * mask  # fp16 * bool -> fp16
    ce = ce.astype(mx.float32).sum() / ntoks

  zoo (utils.py:417):
    mask = length_mask.astype(mx.float32)  # bool -> fp32
    ce = nn.losses.cross_entropy(logits, safe_targets) * mask  # fp16 * fp32 -> fp32
    loss = ce.astype(mx.float32).sum() / _safe_token_denominator(ntoks)

The backward through `ce_fp16 * bool` carries gradients in fp16; the
backward through `ce_fp16 * fp32` carries gradients in fp32. After
30 steps these rounding differences could move the model into
different basins.

If pass rate ~= 67% (matches mlx-lm) -> loss dtype propagation is
                                       the cause
If pass rate ~= 47% (matches zoo)    -> not it; investigate further
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
    banner(f"Probe 24: hybrid + mlx-lm's verbatim loss fn  "
           f"steps={steps} seed={seed} lr={lr}")

    random.seed(seed)
    np.random.seed(seed)
    import mlx.core as mx
    import mlx.nn as nn
    mx.random.seed(seed)

    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(MODEL_NAME)
    model.freeze()
    from mlx_lm.tuner.utils import linear_to_lora_layers
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

    # ---- KEY DIFFERENCE: monkey-patch zoo's make_baseline_loss_fn ----
    # Replace with a verbatim copy of mlx-lm's default_loss
    # (mlx-lm-src/mlx_lm/tuner/trainer.py:86-99). The signature must
    # accept (model, batch, lengths, labels=None) since zoo's trainer
    # calls loss_and_grad_fn(model, batch_data[0], batch_data[1],
    # batch_data[2]) and batch_data[2] is always None for text models.
    import unsloth_zoo.mlx.utils as zoo_utils

    def _mlxlm_default_loss_factory():
        def loss_fn(model, batch, lengths, labels=None):
            # Verbatim from mlx-lm trainer.py:86-99 (with labels
            # silently ignored -- our smoke never passes them).
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs)
            steps_ = mx.arange(1, targets.shape[1] + 1)
            mask = mx.logical_and(steps_ >= lengths[:, 0:1], steps_ <= lengths[:, 1:])
            ce = nn.losses.cross_entropy(logits, targets) * mask
            ntoks = mask.sum()
            ce = ce.astype(mx.float32).sum() / ntoks
            return ce, ntoks
        return loss_fn

    _original = zoo_utils.make_baseline_loss_fn
    zoo_utils.make_baseline_loss_fn = _mlxlm_default_loss_factory
    # Also patch via direct import path (trainer imports it locally).
    import unsloth_zoo.mlx.trainer as zoo_trainer
    zoo_trainer.make_baseline_loss_fn = _mlxlm_default_loss_factory
    report("monkey-patched make_baseline_loss_fn", "OK")

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    fields_supported = {f.name for f in dataclasses.fields(MLXTrainingConfig)}
    extra = {}
    if "adam_bias_correction" in fields_supported:
        extra["adam_bias_correction"] = True
    if "max_grad_value" in fields_supported:
        extra["max_grad_value"] = None

    config = MLXTrainingConfig(
        per_device_train_batch_size=6,
        gradient_accumulation_steps=1,
        max_steps=steps,
        learning_rate=lr,
        warmup_steps=0,
        lr_scheduler_type="constant",
        optim="adamw",
        weight_decay=0.0,
        max_grad_norm=0.0,
        logging_steps=1,
        max_seq_length=MAX_SEQ_LEN,
        seed=seed,
        use_cce=False,
        compile=False,
        gradient_checkpointing=False,
        output_dir=str(OUT_DIR / f"probe24_outputs_s{steps}_d{seed}"),
        save_steps=0,
        eval_steps=0,
        dataset_text_field="text",
        **extra,
    )
    trainer = MLXTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=[{"text": TRAIN_TEXT}] * 64,
        args=config,
    )

    np.random.seed(seed)
    mx.random.seed(seed)

    rows = []
    def _on_step(*args):
        if len(args) < 3: return
        rows.append({"step": int(args[0]), "loss": float(args[2])})
    trainer.add_step_callback(_on_step)
    trainer.train()

    # Eval — use ORIGINAL zoo loss for the post-train measurement so
    # we're measuring the trained weights, not the patched fn.
    zoo_utils.make_baseline_loss_fn = _original
    eval_loss_fn = _original()
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
            "trainer": "unsloth-zoo (path B) with mlx-lm's verbatim loss",
            "per_device_train_batch_size": 6,
            "gradient_accumulation_steps": 1,
            "max_grad_value": None, "max_grad_norm": 0.0,
            "adam_bias_correction": True,
            "compile": False,
        },
        "rows": rows,
        "post_train_loss": post_loss_val,
        "completion_teacher_forced_loss": completion_loss,
        "generation": gen,
        "contains_unsloth": contains,
    }
    fname = f"probe_24__s{steps}_d{seed}.json"
    (OUT_DIR / fname).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
