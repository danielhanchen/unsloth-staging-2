"""Probe 7 — end-to-end 7-step training loss curve, MLX-only.

Re-run the same 7-step config that the smoke test uses, just MLXTrainer
this time (we already know the HF curve from the CUDA mirror). Capture:

  * per-step training loss
  * per-step grad norm (as reported by the trainer)
  * post-train loss on the train row (recomputed via a fresh forward)
  * greedy generation from `"<<HELLO!!>> My name is "`
  * tokenized train ids + ntoks-per-batch (from probe 1 path)

Emit everything to probe_7.json so a follow-up analysis script (or a
maintainer reading the CI log) can directly compare these numbers
against the CUDA-mirror baseline numbers checked into
`temp/torchcodec_test/.out/cuda_truemirror_*.json`.

Always exits 0 -- this probe is a data dump, not a gate. It's the
ground truth that probes 1-6 are debugging.
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
    banner("Probe 7: end-to-end 7-step MLX loss curve")

    import mlx.core as mx
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    section("load + LoRA")
    model, tokenizer = FastMLXModel.from_pretrained(
        MODEL_NAME, load_in_4bit=False, dtype="float16",
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

    section("trainer config (same as the upstream smoke test, minus override workaround)")
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
        # leave max_grad_value at config default
        logging_steps=1,
        max_seq_length=MAX_SEQ_LEN,
        seed=SEED,
        use_cce=False,
        compile=False,
        gradient_checkpointing=False,
        output_dir=str(OUT_DIR / "probe7_outputs"),
        save_steps=0,
        eval_steps=0,
        dataset_text_field="text",
    )
    report("max_grad_value default", config.max_grad_value)
    report("max_grad_norm", config.max_grad_norm)

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
    report("post_train_loss", post_loss_val)

    section("greedy generation")
    from mlx_lm import generate
    gen = generate(model, tokenizer, prompt=PROMPT, max_tokens=48, verbose=False)
    report("generation", repr(gen))
    contains = "Unsloth" in gen

    out = {
        "tokenized_train_ids": ids,
        "tokenized_train_len": L,
        "rows": rows,
        "post_train_loss": post_loss_val,
        "generation": gen,
        "contains_unsloth": contains,
    }
    (OUT_DIR / "probe_7.json").write_text(json.dumps(out, indent=2))
    section("summary")
    report("step-1 loss", rows[0]["loss"] if rows else None)
    report("step-7 loss", rows[-1]["loss"] if rows else None)
    report("post_train_loss", post_loss_val)
    report("contains 'Unsloth'", contains)
    return 0


if __name__ == "__main__":
    sys.exit(main())
