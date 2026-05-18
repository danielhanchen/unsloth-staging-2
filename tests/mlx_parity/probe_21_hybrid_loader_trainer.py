"""Probe 21 — disambiguate LOADER vs TRAINER as the source of the
~20pp pass-rate gap between mlx-lm native LoRA (~67%) and
unsloth-zoo MLXTrainer (~40-47%) on the smoke fixture.

Round AX (n=15) confirmed the gap is real: mlx-lm strictly dominates
unsloth-zoo at every seed (paired comparison). Round AW eliminated
max_grad_value and the grad-accum mechanic as causes.

This probe builds a HYBRID:
  * model construction & LoRA wiring via mlx-lm's load() +
    linear_to_lora_layers() (path A from the audit)
  * training via unsloth-zoo's MLXTrainer (path B from the audit),
    configured to mirror mlx-lm's defaults as closely as the
    MLXTrainingConfig surface allows:
        max_grad_value=None     # mlx-lm has no clip
        max_grad_norm=0         # ditto
        gradient_checkpointing=False
        use_cce=False
        compile=False
        bs=6, accum=1
        lr=1e-3, weight_decay=0, adamw, bias_correction=True

Reading:
  pass_rate ≈ 67% (mlx-lm)         -> gap is in FastMLXModel /
                                      get_peft_model (loader side)
  pass_rate ≈ 40-47% (unsloth-zoo) -> gap is in MLXTrainer / its
                                      data sampler / optimizer wiring

Env vars: MLX_SEED (required), MLX_STEPS (default 30), MLX_LR
(default 1e-3). Writes per-config JSON to
.out/probe_21__s{S}_d{D}.json.
"""

import json
import os
import sys
import dataclasses
import random
from pathlib import Path

import numpy as np

from _common import (
    MODEL_NAME,
    TRAIN_TEXT,
    PROMPT,
    MAX_SEQ_LEN,
    OUT_DIR,
    banner,
    section,
    report,
)


def _env_int(name, default):
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name, default):
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def main() -> int:
    steps = _env_int("MLX_STEPS", 30)
    seed = _env_int("MLX_SEED", 3407)
    lr = _env_float("MLX_LR", 1e-3)
    banner(f"Probe 21: mlx-lm loader + unsloth-zoo trainer "
           f"steps={steps} seed={seed} lr={lr}")

    random.seed(seed)
    np.random.seed(seed)
    import mlx.core as mx
    mx.random.seed(seed)

    # ---- LOADER: exactly mlx-lm's path. ----
    from mlx_lm import load as mlx_load
    section("mlx-lm load + LoRA wire (path A)")
    model, tokenizer = mlx_load(MODEL_NAME)
    report("loaded model class", type(model).__name__)

    # Mirror mlx-lm/lora.py: freeze BEFORE linear_to_lora_layers.
    model.freeze()
    from mlx_lm.tuner.utils import linear_to_lora_layers
    lora_config = {
        "rank": 8,
        "scale": 2.0,
        "dropout": 0.0,
        "keys": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
    }
    try:
        num_layers = len(model.layers)
    except AttributeError:
        num_layers = len(model.model.layers)
    linear_to_lora_layers(model, num_layers, lora_config)
    report("LoRA modules wired via mlx-lm path", "OK")

    # Sanity: count trainable params
    from mlx.utils import tree_flatten
    trainable = [(k, v) for k, v in tree_flatten(model.trainable_parameters())]
    report("trainable param leaves", len(trainable))

    # ---- TRAINER: unsloth-zoo MLXTrainer (path B). ----
    section("unsloth-zoo MLXTrainer (path B)")
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx.utils import make_baseline_loss_fn

    fields_supported = {f.name for f in dataclasses.fields(MLXTrainingConfig)}
    extra = {}
    if "adam_bias_correction" in fields_supported:
        extra["adam_bias_correction"] = True
    if "max_grad_value" in fields_supported:
        extra["max_grad_value"] = None  # match mlx-lm: no elementwise clip

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
        output_dir=str(OUT_DIR / f"probe21_outputs_s{steps}_d{seed}"),
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
    rows = []
    def _on_step(*args):
        if len(args) < 3:
            return
        rows.append({"step": int(args[0]), "loss": float(args[2])})
    trainer.add_step_callback(_on_step)
    trainer.train()

    # ---- POST-TRAIN: same eval signal as probe 17. ----
    loss_fn = make_baseline_loss_fn()
    ids = tokenizer.encode(TRAIN_TEXT)
    if tokenizer.eos_token_id is not None and ids[-1] != tokenizer.eos_token_id:
        ids.append(tokenizer.eos_token_id)
    L = len(ids)
    batch = mx.array([ids])
    lengths = mx.array([[1, L - 1]])
    labels_mlx = mx.array([ids])
    post_loss, _ = loss_fn(model, batch, lengths, labels_mlx)
    post_loss_val = float(post_loss.item())

    import mlx.nn as nn
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
    report("completion_teacher_forced_loss", completion_loss)

    from mlx_lm import generate
    gen = generate(model, tokenizer, prompt=PROMPT, max_tokens=48, verbose=False)
    contains = "Unsloth" in gen
    report("generation", repr(gen[:160]))
    report("contains 'Unsloth'", contains)

    out = {
        "config": {
            "steps": steps, "seed": seed,
            "learning_rate": lr,
            "loader": "mlx-lm (path A)",
            "trainer": "unsloth-zoo (path B)",
            "per_device_train_batch_size": 6,
            "gradient_accumulation_steps": 1,
            "max_grad_value": None,
            "max_grad_norm": 0.0,
            "adam_bias_correction": True,
        },
        "rows": rows,
        "post_train_loss": post_loss_val,
        "completion_teacher_forced_loss": completion_loss,
        "generation": gen,
        "contains_unsloth": contains,
    }
    fname = f"probe_21__s{steps}_d{seed}.json"
    (OUT_DIR / fname).write_text(json.dumps(out, indent=2))

    section("summary")
    if rows:
        report("step-1 loss", rows[0]["loss"])
        report(f"step-{len(rows)} loss", rows[-1]["loss"])
    report("post_train_loss", post_loss_val)
    return 0


if __name__ == "__main__":
    sys.exit(main())
