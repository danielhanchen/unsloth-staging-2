"""Probe 17 — parameterized 7+ step MLX training curve.

Reads env vars so a single matrix entry can be reused with different
(steps, seed, dtype, bias_correction) combinations:

  MLX_STEPS              max_steps for MLXTrainer (default 7)
  MLX_SEED               seed for everything (default 3407)
  MLX_DTYPE              dtype string for FastMLXModel.from_pretrained
                         (default "float16")
  MLX_BIAS_CORRECTION    "1"/"true" -> adam_bias_correction=True
                         "0"/"false" (default) -> False

Pin: unsloth-zoo HEAD (broken default at the time the question was
asked) so this probe directly characterizes how the post-#634 code
behaves under longer training / other seeds.

The probe writes a per-config JSON to .out/probe_17__steps{S}_seed{D}_bc{0/1}.json
so the matrix's `outputs: filename` path is unique.

Question this answers:
  * does increasing max_steps eventually let bias_correction=True
    memorize the train row? If yes, MLX is healthy and 7 steps is
    just too short for the HF/torch math.
  * does varying the seed (data shuffle, LoRA init) change the
    basin? If multiple seeds all fail at 7 steps + bc=True, the
    issue is structural, not lucky/unlucky init.

Always exits 0 -- data dump.
"""

import json
import os
import sys

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


def _env_bool(name, default=False):
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "y")


def _env_int(name, default):
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name, default):
    raw = (os.environ.get(name) or "").strip()
    return raw if raw else default


def _env_float(name, default):
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def main() -> int:
    steps = _env_int("MLX_STEPS", 7)
    seed = _env_int("MLX_SEED", 3407)
    dtype = _env_str("MLX_DTYPE", "float16")
    # Tri-state: empty/unset env var means "use trainer default" (don't
    # pass adam_bias_correction at all); "0"/"1" forces explicit value.
    bc_raw = (os.environ.get("MLX_BIAS_CORRECTION") or "").strip().lower()
    if not bc_raw:
        bc = None
    else:
        bc = bc_raw in ("1", "true", "yes", "y")
    lr = _env_float("MLX_LR", 1e-3)
    # Grad clip knobs:
    #   MLX_MAX_GRAD_NORM=  empty -> trainer default (1.0 in this probe)
    #   MLX_MAX_GRAD_VALUE= empty -> trainer default (None on PR-663 head)
    # Use "off"/"0"/explicit floats to override; "none" maps to None.
    def _env_grad(name):
        raw = (os.environ.get(name) or "").strip().lower()
        if not raw:
            return "default"
        if raw in ("none", "off"):
            return None
        try:
            return float(raw)
        except ValueError:
            return "default"
    grad_norm_override = _env_grad("MLX_MAX_GRAD_NORM")
    grad_value_override = _env_grad("MLX_MAX_GRAD_VALUE")
    # Round AW: bisect mlx-lm-vs-unsloth-zoo 80%-vs-60% gap. The two
    # axes still live (CCE off + GC off in this probe already eliminate
    # those candidates): grad-accum mechanic (B = bs2*accum3 with token-
    # weighted mean; A = native bs6 unweighted) + elementwise clip.
    bs = _env_int("MLX_BS", 2)
    accum = _env_int("MLX_ACCUM", 3)

    banner(f"Probe 17: steps={steps} seed={seed} dtype={dtype} bc={bc!r} lr={lr} "
           f"max_grad_norm={grad_norm_override!r} max_grad_value={grad_value_override!r} "
           f"bs={bs} accum={accum}")

    import random
    random.seed(seed)
    np.random.seed(seed)
    import mlx.core as mx
    mx.random.seed(seed)

    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx.utils import make_baseline_loss_fn
    import dataclasses

    model, tokenizer = FastMLXModel.from_pretrained(
        MODEL_NAME, load_in_4bit=False, dtype=dtype,
        text_only=True, max_seq_length=128, random_state=seed,
    )
    model = FastMLXModel.get_peft_model(
        model, r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        random_state=seed,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )

    # Only set adam_bias_correction if (a) the field exists on this
    # version of unsloth-zoo AND (b) the env var asked for an explicit
    # value (bc is not None). bc=None means "use the trainer default"
    # so the artifact records whatever the default actually is.
    fields_supported = {f.name for f in dataclasses.fields(MLXTrainingConfig)}
    extra = {}
    if "adam_bias_correction" in fields_supported and bc is not None:
        extra["adam_bias_correction"] = bc
    if grad_value_override != "default" and "max_grad_value" in fields_supported:
        extra["max_grad_value"] = grad_value_override

    cfg_grad_norm = 1.0 if grad_norm_override == "default" else (grad_norm_override or 0.0)

    config = MLXTrainingConfig(
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=accum,
        max_steps=steps,
        learning_rate=lr,
        warmup_steps=0,
        lr_scheduler_type="constant",
        optim="adamw",
        weight_decay=0.0,
        max_grad_norm=cfg_grad_norm,
        logging_steps=1,
        max_seq_length=MAX_SEQ_LEN,
        seed=seed,
        use_cce=False,
        compile=False,
        gradient_checkpointing=False,
        output_dir=str(OUT_DIR / f"probe17_outputs_s{steps}_d{seed}_bc{('d' if bc is None else int(bc))}_lr{lr:g}"),
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
        rows.append({
            "step": int(args[0]),
            "loss": float(args[2]),
            "grad_norm": float(args[8]) if len(args) >= 9 and args[8] is not None else None,
        })
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

    # Teacher-forced completion loss: same shape as the new PR-5537
    # smoke gate. CE on the "Unsloth!" tokens given the "<<HELLO!!>> My
    # name is " prompt, no decoding involved. Should be tiny across
    # every config that hits post_train_loss < 0.1.
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

    # Record what the trainer actually used (post-construction) so the
    # artifact reflects the trainer default when bc was None at probe-
    # invocation time.
    effective_bc = getattr(config, "adam_bias_correction", None)
    out = {
        "config": {
            "steps": steps, "seed": seed, "dtype": dtype,
            "adam_bias_correction": bc,
            "effective_adam_bias_correction": effective_bc,
            "learning_rate": lr,
            "per_device_train_batch_size": bs,
            "gradient_accumulation_steps": accum,
            "effective_batch_size": bs * accum,
            "max_grad_value": grad_value_override,
            "max_grad_norm_setting": cfg_grad_norm,
            "adam_bc_field_supported": "adam_bias_correction" in fields_supported,
        },
        "rows": rows,
        "post_train_loss": post_loss_val,
        "completion_teacher_forced_loss": completion_loss,
        "generation": gen,
        "contains_unsloth": contains,
    }
    lr_tag = f"{lr:.0e}".replace("-0", "-").replace("+0", "")
    bc_tag = "d" if bc is None else int(bc)
    if grad_value_override == "default":
        gv_tag = "def"
    elif grad_value_override is None:
        gv_tag = "off"
    else:
        gv_tag = f"{grad_value_override:g}"
    fname = (f"probe_17__s{steps}_d{seed}_bc{bc_tag}_lr{lr_tag}"
             f"_bs{bs}_ac{accum}_gv{gv_tag}.json")
    (OUT_DIR / fname).write_text(json.dumps(out, indent=2))
    section("summary")
    if rows:
        report("step-1 loss", rows[0]["loss"])
        report(f"step-{len(rows)} loss", rows[-1]["loss"])
    report("post_train_loss", post_loss_val)
    return 0


if __name__ == "__main__":
    sys.exit(main())
