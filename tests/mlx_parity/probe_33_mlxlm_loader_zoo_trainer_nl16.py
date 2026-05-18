"""Probe 33 — mlx_lm loader + zoo MLXTrainer + num_layers=16.

Bisects whether zoo's LOADER or its TRAINER adds the additional
basin instability seen in probe 32 (which used zoo's full stack
with num_layers=16 and hit only 15%).

Probe 31 (mlx_lm.load + manual loop + 16): 67%
Probe 32 (FastMLXModel    + MLXTrainer  + 16): 15%
Probe 33 (mlx_lm.load     + MLXTrainer  + 16): ?
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
    last_n = _env_int("MLX_LAST_N", 16)
    banner(f"Probe 33: mlx_lm loader + zoo MLXTrainer + last_n={last_n}  "
           f"steps={steps} seed={seed} lr={lr}")

    random.seed(seed); np.random.seed(seed)
    import mlx.core as mx
    mx.random.seed(seed)

    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, tokenizer = mlx_load(MODEL_NAME)
    model.freeze()
    try: num_layers = len(model.layers)
    except AttributeError: num_layers = len(model.model.layers)
    num_layers = max(1, min(int(last_n), num_layers))
    linear_to_lora_layers(model, num_layers, {
        "rank": 8, "scale": 2.0, "dropout": 0.0,
        "keys": ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
                 "mlp.gate_proj","mlp.up_proj","mlp.down_proj"],
    })

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx.utils import make_baseline_loss_fn

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
        output_dir=str(OUT_DIR / f"probe33_outputs_s{steps}_d{seed}"),
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
        if len(args) < 3: return
        rows.append({"step": int(args[0]), "loss": float(args[2])})
    trainer.add_step_callback(_on_step)
    trainer.train()

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

    from mlx_lm import generate
    gen = generate(model, tokenizer, prompt=PROMPT, max_tokens=48, verbose=False)
    contains = "Unsloth" in gen
    report("generation", repr(gen[:160]))
    report("contains 'Unsloth'", contains)

    out = {
        "config": {
            "steps": steps, "seed": seed, "learning_rate": lr,
            "loader": "mlx_lm.load",
            "trainer": "unsloth_zoo MLXTrainer",
            "num_layers": num_layers,
            "per_device_train_batch_size": 6,
            "gradient_accumulation_steps": 1,
            "max_grad_value": None, "max_grad_norm": 0.0,
            "adam_bias_correction": True,
        },
        "rows": rows,
        "post_train_loss": post_loss_val,
        "completion_teacher_forced_loss": completion_loss,
        "generation": gen,
        "contains_unsloth": contains,
    }
    fname = f"probe_33__s{steps}_d{seed}_nl{num_layers}.json"
    (OUT_DIR / fname).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
