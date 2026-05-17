"""Probe 12 — pin unsloth-zoo to the parent of PR #634 and rerun.

Hypothesis we want to nail down: every other parity probe rules out
the obvious axes (loss math, AdamW math, tokenization, supervised
positions, single-step gradient norm), yet HF on the same host
generates "Unsloth" and MLX does not. That points squarely at the
trainer changes in unsloth-zoo PR #634 (`e6d8f7f`).

This probe assumes the CI workflow installs unsloth-zoo at the
PARENT commit `f37d510` (the commit immediately before #634 landed).
Pre-#634 the layout was flat: `unsloth_zoo.mlx_loader` /
`unsloth_zoo.mlx_trainer`. Post-#634 it's a package:
`unsloth_zoo.mlx.loader` / `unsloth_zoo.mlx.trainer`. Try both,
honor whichever is importable.

If this probe generates "Unsloth" with the SAME 7-step config that
probe 7 / 11 fail on, the regression is fully INSIDE PR #634's diff
and we can sub-bisect by reverting the suspect changes (bias_correction,
loss reduction, custom VJP, dtype handling).

Always exits 0 -- data dump.
"""

import json
import sys

from _common import (
    MODEL_NAME,
    TRAIN_TEXT,
    PROMPT,
    SEED,
    MAX_SEQ_LEN,
    OUT_DIR,
    banner,
    section,
    report,
    seed_everything,
)


def _import_zoo():
    try:
        from unsloth_zoo.mlx_loader import FastMLXModel  # pre-#634
        from unsloth_zoo.mlx_trainer import MLXTrainer, MLXTrainingConfig
        from unsloth_zoo.mlx_utils import make_baseline_loss_fn
        return "pre-#634 flat layout", FastMLXModel, MLXTrainer, MLXTrainingConfig, make_baseline_loss_fn
    except ImportError:
        pass
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx.utils import make_baseline_loss_fn
    return "post-#634 package layout", FastMLXModel, MLXTrainer, MLXTrainingConfig, make_baseline_loss_fn


def main() -> int:
    seed_everything()
    banner("Probe 12: pinned unsloth-zoo (parent of PR #634)")

    import importlib
    import unsloth_zoo
    report("unsloth_zoo path", getattr(unsloth_zoo, "__file__", "?"))
    try:
        report("unsloth_zoo version", getattr(unsloth_zoo, "__version__", "?"))
    except Exception:
        pass

    layout, FastMLXModel, MLXTrainer, MLXTrainingConfig, make_baseline_loss_fn = _import_zoo()
    report("layout detected", layout)

    import mlx.core as mx

    # Mirror the SMOKE TEST AT 12295c1f exactly: dtype="float16" + identical LoRA
    # config + identical hyperparams. We want to know if pre-#634 trainer
    # behavior matches the green CI from that era.
    section("load + LoRA (fp16, matches pre-#634 smoke)")
    model, tokenizer = FastMLXModel.from_pretrained(
        MODEL_NAME, load_in_4bit=False, dtype="float16",
        text_only=True, max_seq_length=128, random_state=SEED,
    )
    model = FastMLXModel.get_peft_model(
        model,
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        random_state=SEED,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )

    # MLXTrainingConfig at pre-#634 does NOT have max_grad_value, so we
    # only pass it if supported. dataclasses.fields tells us.
    import dataclasses
    fields_supported = {f.name for f in dataclasses.fields(MLXTrainingConfig)}
    extra_kwargs = {}
    if "max_grad_value" in fields_supported:
        extra_kwargs["max_grad_value"] = None
    config = MLXTrainingConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=3,
        max_steps=7,
        learning_rate=1e-3,
        warmup_steps=0,
        lr_scheduler_type="constant",
        optim="adamw",
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=1,
        max_seq_length=MAX_SEQ_LEN,
        seed=SEED,
        use_cce=False,
        compile=False,
        gradient_checkpointing=False,
        output_dir=str(OUT_DIR / "probe12_outputs"),
        save_steps=0,
        eval_steps=0,
        dataset_text_field="text",
        **extra_kwargs,
    )

    trainer = MLXTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=[{"text": TRAIN_TEXT}] * 64,
        args=config,
    )

    rows = []
    # Variadic callback so we work for both pre-#634 (8 args) and
    # post-#634 (9 args). The trainer wraps `cb(...)` in try/except
    # Exception, so an arity mismatch on a fixed-arg callback would
    # silently no-op the entire logging path.
    def _on_step(*args):
        # args = (step, total, loss, lr, tok_s, peak_gb, elapsed, num_tokens[, grad_norm])
        if len(args) < 3:
            return
        step, _total, loss = args[0], args[1], args[2]
        grad_norm = args[8] if len(args) >= 9 else None
        rows.append({
            "step": int(step), "loss": float(loss),
            "grad_norm": None if grad_norm is None else float(grad_norm),
        })
    trainer.add_step_callback(_on_step)
    cb_arity_used = "variadic"
    trainer.train()

    section("post-train forward")
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

    section("greedy generation")
    from mlx_lm import generate
    gen = generate(model, tokenizer, prompt=PROMPT, max_tokens=48, verbose=False)
    contains = "Unsloth" in gen
    report("generation", repr(gen))
    report("contains 'Unsloth'", contains)

    out = {
        "layout": layout,
        "callback_arity_used": cb_arity_used,
        "rows": rows,
        "post_train_loss": post_loss_val,
        "generation": gen,
        "contains_unsloth": contains,
        "dtype": "float32",
    }
    (OUT_DIR / "probe_12.json").write_text(json.dumps(out, indent=2))
    section("summary")
    if rows:
        report("step-1 loss", rows[0]["loss"])
        report("step-7 loss", rows[-1]["loss"])
    report("post_train_loss", post_loss_val)
    return 0


if __name__ == "__main__":
    sys.exit(main())
