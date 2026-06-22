"""Probe: does a pre-quantized bitsandbytes 4-bit (NF4) checkpoint load on MLX?

47 migrated notebooks set `model_name = "unsloth/...-bnb-4bit"`. bitsandbytes
stores weights as packed uint8 + absmax/quant_map tensors in a layout mlx-lm
does not natively read, so this is the highest-impact open "does any notebook
just run on Mac" question. We use a llama-arch bnb model (TinyLlama) so a failure
isolates to bnb-checkpoint reading rather than architecture support.

Report-only: prints BNB4BIT_LOAD: OK / FAILED with the exception and exits 0, so
the surrounding CI step records the real behavior without aborting the suite.
"""

import os
import platform
import traceback

assert platform.system() == "Darwin" and platform.machine() == "arm64", platform.platform()

MODEL = os.environ.get("BNB4BIT_PROBE_MODEL", "unsloth/tinyllama-bnb-4bit")


def main():
    import unsloth  # noqa: F401  (sets DEVICE_TYPE=mlx on import)
    from unsloth import FastLanguageModel

    print(f"... attempting FastLanguageModel.from_pretrained({MODEL!r}, load_in_4bit=True) on MLX")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL,
            max_seq_length=512,
            load_in_4bit=True,
        )
    except Exception as exc:
        print("BNB4BIT_LOAD: FAILED at from_pretrained")
        print(f"  {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 0

    print("BNB4BIT_LOAD: from_pretrained OK")

    # Confirm it is actually usable: attach LoRA and run one optimizer step.
    try:
        import tempfile

        from datasets import Dataset

        model = FastLanguageModel.get_peft_model(model, r=8)
        eos = getattr(tokenizer, "eos_token", "") or ""
        rows = [{"text": f"### Q: {i}+{i}?\n### A: {2*i}.{eos}"} for i in range(1, 33)]
        dataset = Dataset.from_list(rows)
        from unsloth import UnslothTrainer, UnslothTrainingArguments

        with tempfile.TemporaryDirectory() as tmp:
            trainer = UnslothTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                args=UnslothTrainingArguments(
                    dataset_text_field="text",
                    max_seq_length=512,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=1,
                    max_steps=2,
                    warmup_steps=1,
                    learning_rate=1e-4,
                    optim="adamw_8bit",
                    logging_steps=1,
                    save_strategy="no",
                    seed=3407,
                    output_dir=tmp,
                ),
            )
            stats = trainer.train()
        print(f"BNB4BIT_TRAIN: OK  train_loss={stats.get('train_loss')}")
    except Exception as exc:
        print("BNB4BIT_TRAIN: FAILED after load")
        print(f"  {type(exc).__name__}: {exc}")
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
