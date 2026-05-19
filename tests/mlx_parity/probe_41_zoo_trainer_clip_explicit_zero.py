"""Probe 41 -- probe 34 with max_grad_value=0.0 (explicit disable).

Round BT bisection.

Round BS proved the residual 47%-vs-67% gap is in MLXTrainer.train,
not FastMLXModel loader (probe 40 = probe 31 on 15/15 seeds). Reading
unsloth_zoo/mlx/trainer.py:731-732:

    _raw_mgv = getattr(args, "max_grad_value", 1.0)
    max_grad_value = 1.0 if _raw_mgv is None else float(_raw_mgv or 0.0)

means `max_grad_value=None` is reinterpreted as 1.0 (clip at +/-1.0
elementwise), NOT "disable clipping". PR #671
(`mlx: honor max_grad_value=None as a disable signal`, head 265534b)
is currently OPEN, not merged. Probe 34 sets max_grad_value=None
expecting "disable", actually gets clip-at-1. Manual loop in probes
31 / 40 uses bare optim.AdamW with NO clipping.

Probe 41 = probe 34 but with max_grad_value=0.0 (explicit zero hits
the `float(_raw_mgv or 0.0)` branch -> 0.0 -> no clip on the current
build).

Read:
  probe 41 ~ 67%  ->  Elementwise clip-at-1 IS the residual gap.
                     PR #671 closes the FastMLXModel + MLXTrainer
                     basin gap. Final missing piece.
  probe 41 ~ 47%  ->  Clip isn't it; the gap is elsewhere in
                     MLXTrainer.train (lr schedule, loss-fn, batch
                     iteration, mx.eval timing, ...).

Same 15 seeds as probes 31 / 34 / 40 for direct paired comparison.
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
    banner(f"Probe 41: FastMLXModel + MLXTrainer + max_grad_value=0.0 (explicit disable)")

    random.seed(seed); np.random.seed(seed)
    import mlx.core as mx
    mx.random.seed(seed)

    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx.utils import make_baseline_loss_fn

    model, tokenizer = FastMLXModel.from_pretrained(
        MODEL_NAME, load_in_4bit=False, dtype=None,
        text_only=True, max_seq_length=128, random_state=seed,
    )

    model = FastMLXModel.get_peft_model(
        model, r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        random_state=seed,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        finetune_last_n_layers=last_n,
        use_gradient_checkpointing=False,
    )

    fields_supported = {f.name for f in dataclasses.fields(MLXTrainingConfig)}
    extra = {}
    if "adam_bias_correction" in fields_supported: extra["adam_bias_correction"] = True
    # The key difference vs probe 34: explicit 0.0 hits trainer.py:732's
    # `float(_raw_mgv or 0.0)` branch -> 0.0 -> no clip. Setting None
    # would hit `1.0 if _raw_mgv is None` -> clip at 1.0.
    if "max_grad_value" in fields_supported: extra["max_grad_value"] = 0.0

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
        output_dir=str(OUT_DIR / f"probe41_outputs_s{steps}_d{seed}"),
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
    report("contains 'Unsloth'", contains)
    report("generation", repr(gen[:60]))

    out = {
        "config": {"steps": steps, "seed": seed, "learning_rate": lr,
                   "loader": "FastMLXModel(dtype=None)", "finetune_last_n_layers": last_n,
                   "delta": "max_grad_value=0.0 (explicit disable)"},
        "rows": rows, "post_train_loss": post_loss_val,
        "completion_teacher_forced_loss": completion_loss, "generation": gen,
        "contains_unsloth": contains,
    }
    fname = f"probe_41__s{steps}_d{seed}_nl{last_n}.json"
    (OUT_DIR / fname).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
