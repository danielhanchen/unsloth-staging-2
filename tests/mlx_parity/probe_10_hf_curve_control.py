"""Probe 10 — HF SFTTrainer 7-step loss curve on the SAME Mac host (control).

The previously-collected HF baseline came from CUDA bf16 on a B200 GPU.
That's a different platform AND a different precision AND a different
optimizer backend. To isolate "MLX vs HF" from "CUDA vs Mac CPU" we
re-run the HF leg here on the same macos-14-arm64 runner in fp32
(CPU), with the exact same 7 LoRA targets / alpha=16 / hyperparams.

Forces torch to CPU because the standard macos-14 GitHub runner has
only 7 GB of shared memory; an fp32 LoRA training on MPS hits the
GPU memory watermark.

Compare probe_10.json with probe_7.json: same-host, same-precision
expectations, only the trainer implementation changes.

Always exits 0 -- data dump for follow-up analysis.
"""

import json
import os
import sys

# Hide every accelerator from torch before importing it. macos-14 runners
# expose MPS with a 7 GB shared cap; the fp32 7-module LoRA training
# above does not fit. Force CPU.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

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
    banner("Probe 10: HF SFTTrainer 7-step loss curve (control on same host)")

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model

    # TRL is optional on a Mac CPU image; install if missing.
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        report("trl not available", str(e))
        out = {"trl_available": False}
        (OUT_DIR / "probe_10.json").write_text(json.dumps(out, indent=2))
        return 0

    torch.manual_seed(SEED)
    # Force CPU explicitly even if MPS is reported. setting empty
    # CUDA_VISIBLE_DEVICES handles CUDA; here we shadow the MPS-pickup
    # path by setting torch's default device.
    try:
        torch.set_default_device("cpu")
    except Exception:
        pass
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to("cpu")
    model = get_peft_model(
        model,
        LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )

    rows = []
    class _Logger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            rows.append({
                "step": int(state.global_step),
                "loss": float(logs["loss"]),
                "grad_norm": float(logs["grad_norm"]) if "grad_norm" in logs else None,
            })

    ds = Dataset.from_list([{"text": TRAIN_TEXT}] * 64)
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=ds,
        callbacks=[_Logger()],
        args=SFTConfig(
            max_length=MAX_SEQ_LEN,
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=3,
            warmup_steps=0,
            max_steps=7,
            learning_rate=1e-3,
            logging_steps=1,
            optim="adamw_torch",
            weight_decay=0.0,
            lr_scheduler_type="constant",
            max_grad_norm=1.0,
            seed=SEED,
            save_strategy="no",
            report_to="none",
            packing=False,
            bf16=False,
            fp16=False,
            use_cpu=True,
            output_dir=str(OUT_DIR / "probe10_outputs"),
        ),
    )
    trainer.train()

    section("post-train forward")
    model.eval()
    with torch.no_grad():
        enc = tok(TRAIN_TEXT, return_tensors="pt")
        out = model(**enc, labels=enc["input_ids"].clone())
        post_loss = float(out.loss.detach())
    report("post_train_loss", post_loss)

    section("greedy generation")
    model.eval()
    with torch.no_grad():
        ginp = tok(PROMPT, return_tensors="pt")
        gout = model.generate(**ginp, max_new_tokens=48, do_sample=False)
    gen = tok.decode(gout[0], skip_special_tokens=True)
    report("generation", repr(gen))

    out = {
        "trl_available": True,
        "rows": rows,
        "post_train_loss": post_loss,
        "generation": gen,
        "contains_unsloth": "Unsloth" in gen,
    }
    (OUT_DIR / "probe_10.json").write_text(json.dumps(out, indent=2))
    section("summary")
    report("step-1 loss", rows[0]["loss"] if rows else None)
    report("step-7 loss", rows[-1]["loss"] if rows else None)
    report("post_train_loss", post_loss)
    report("contains 'Unsloth'", "Unsloth" in gen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
