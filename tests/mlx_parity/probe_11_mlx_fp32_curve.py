"""Probe 11 — MLX trainer 7-step loss curve at dtype="float32".

Probe 7 runs the MLX trainer at dtype="float16" (the smoke-test default).
This probe runs the identical config at dtype="float32" so that the
forward / backward / optimizer are all carried out in fp32, matching
what HF on Mac CPU (probe 10) does.

Hypothesis: the upstream smoke test's "5 lbs!" / "42!!" generation
collapse is a fp16 numerical artifact, not an algorithmic bug.

If probe 11's loss curve and generation come out matching the HF curve
in probe 10, the actionable fix is to switch the smoke test (or the
trainer default) to float32 / bfloat16 on Apple Silicon.

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


def main() -> int:
    seed_everything()
    banner("Probe 11: MLX trainer 7-step loss curve at fp32")

    import mlx.core as mx
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    section("load + LoRA (fp32)")
    model, tokenizer = FastMLXModel.from_pretrained(
        MODEL_NAME, load_in_4bit=False, dtype="float32",   # <-- the only change vs probe 7
        text_only=True, max_seq_length=128,
        random_state=SEED,
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
        output_dir=str(OUT_DIR / "probe11_outputs"),
        save_steps=0,
        eval_steps=0,
        dataset_text_field="text",
    )
    trainer = MLXTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=[{"text": TRAIN_TEXT}] * 64,
        args=config,
    )

    rows = []
    def _on_step(step, total, loss, lr, tok_s, peak_gb, elapsed, num_tokens, grad_norm):
        rows.append({
            "step": int(step), "loss": float(loss),
            "lr": float(lr), "grad_norm": None if grad_norm is None else float(grad_norm),
            "num_tokens": int(num_tokens),
        })
    trainer.add_step_callback(_on_step)
    trainer.train()

    section("post-train forward")
    from unsloth_zoo.mlx.utils import make_baseline_loss_fn
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
        "dtype": "float32",
        "rows": rows,
        "post_train_loss": post_loss_val,
        "generation": gen,
        "contains_unsloth": contains,
    }
    (OUT_DIR / "probe_11.json").write_text(json.dumps(out, indent=2))
    section("summary")
    report("step-1 loss", rows[0]["loss"] if rows else None)
    report("step-7 loss", rows[-1]["loss"] if rows else None)
    report("post_train_loss", post_loss_val)
    return 0


if __name__ == "__main__":
    sys.exit(main())
