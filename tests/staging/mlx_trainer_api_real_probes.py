"""Real Apple Silicon probes for the MLX public trainer API.

Validates the coordinated PR series on REAL mlx / mlx-lm wheels (no simulation
shim): unslothai/unsloth#6462 (public UnslothTrainer / UnslothTrainingArguments
+ get_gpu_memory_stats / clear_gpu_memory) and unslothai/unsloth-zoo#790
(MLXTrainingConfig.warmup_ratio, _normalize_mlx_optimizer_name,
train_on_responses_only MLX delegation).

Asserts hard; any failure exits non-zero. Ends with a tiny real LoRA SFT run
through the public UnslothTrainer to prove end-to-end MLX training works.
"""

import math
import os
import platform
import tempfile

assert platform.system() == "Darwin" and platform.machine() == "arm64", (
    platform.system(),
    platform.machine(),
)

import mlx.core as mx

assert "mlx_simulation" not in (getattr(mx, "__file__", "") or ""), mx.__file__
print(f"OK: real mlx {getattr(mx, '__version__', 'unknown')} at {mx.__file__}")


# ---------------------------------------------------------------------------
# 1. Public MLX import surface (unsloth#6462)
# ---------------------------------------------------------------------------
import unsloth

assert getattr(unsloth, "DEVICE_TYPE", None) == "mlx", unsloth.DEVICE_TYPE
from unsloth import (
    FastLanguageModel,
    UnslothTrainer,
    UnslothTrainingArguments,
    clear_gpu_memory,
    get_gpu_memory_stats,
)
from unsloth import MLXTrainer, MLXTrainingConfig

assert issubclass(UnslothTrainer, MLXTrainer)
assert issubclass(UnslothTrainingArguments, MLXTrainingConfig)
import importlib.util

assert importlib.util.find_spec("unsloth.memory") is None
print("OK: public MLX exports present + correct subclassing")


# ---------------------------------------------------------------------------
# 2. Optimizer alias normalization (unsloth-zoo#790)
# ---------------------------------------------------------------------------
from unsloth_zoo.mlx.trainer import _normalize_mlx_optimizer_name as _norm

for alias in (
    "adamw_8bit",
    "paged_adamw_8bit",
    "adamw_bnb_8bit",
    "paged_adamw_32bit",
    "adamw_torch",
    "adamw_torch_fused",
):
    assert _norm(alias) == "adamw", (alias, _norm(alias))
for native in ("adamw", "adam", "sgd", "lion", "adafactor"):
    assert _norm(native) == native, (native, _norm(native))
try:
    _norm("adamw_typo")
    raise SystemExit("FAIL: expected ValueError for unknown optimizer")
except ValueError:
    pass
print("OK: optimizer alias normalization (adamw_* -> adamw, native kept, unknown -> ValueError)")


# ---------------------------------------------------------------------------
# 3. UnslothTrainingArguments TRL-kwarg normalization (unsloth#6462 + zoo#790)
# ---------------------------------------------------------------------------
a = UnslothTrainingArguments(max_length=123, max_steps=10, warmup_ratio=0.2)
assert a.max_seq_length == 123, a.max_seq_length
assert a.warmup_steps == 2, a.warmup_steps  # ceil(10 * 0.2)

a_explicit = UnslothTrainingArguments(max_steps=10, warmup_steps=5, warmup_ratio=0.1)
assert a_explicit.warmup_steps == 5, a_explicit.warmup_steps  # explicit wins over ratio

a_epoch = UnslothTrainingArguments(num_train_epochs=1, warmup_ratio=0.1)
assert a_epoch.max_steps == -1, a_epoch.max_steps  # epochs not forced to max_steps
assert a_epoch.num_train_epochs == 1

a_save = UnslothTrainingArguments(save_strategy="no")
assert a_save.save_steps == 0, a_save.save_steps

a_canon = UnslothTrainingArguments(max_seq_length=456, max_length=123)
assert a_canon.max_seq_length == 456, a_canon.max_seq_length  # canonical wins

a_optim = UnslothTrainingArguments(optim="adamw_8bit")
assert a_optim.optim == "adamw", a_optim.optim  # normalized at config build

try:
    UnslothTrainingArguments(assistant_only_loss=True)
    raise SystemExit("FAIL: expected NotImplementedError for unsupported task kwarg")
except NotImplementedError:
    pass
print("OK: UnslothTrainingArguments TRL-kwarg normalization + unsupported-kwarg rejection")


# ---------------------------------------------------------------------------
# 4. MLXTrainingConfig warmup_ratio field + explicit tracking (zoo#790)
# ---------------------------------------------------------------------------
cfg_ratio = MLXTrainingConfig(warmup_ratio=0.1, max_steps=8)
assert cfg_ratio.warmup_ratio == 0.1, cfg_ratio.warmup_ratio
assert getattr(cfg_ratio, "_unsloth_mlx_warmup_steps_explicit", None) is False
cfg_explicit = MLXTrainingConfig(warmup_steps=5, max_steps=8)
assert getattr(cfg_explicit, "_unsloth_mlx_warmup_steps_explicit", None) is True
print("OK: MLXTrainingConfig.warmup_ratio field + explicit warmup tracking")


# ---------------------------------------------------------------------------
# 5. Backend-safe memory helpers (unsloth#6462)
# ---------------------------------------------------------------------------
stats, peak_gb, max_gb = get_gpu_memory_stats()
assert max_gb > 0, max_gb
assert peak_gb >= 0, peak_gb
clear_gpu_memory()
print(f"OK: get_gpu_memory_stats() name={getattr(stats, 'name', None)} peak={peak_gb}GB max={max_gb}GB; clear_gpu_memory() ok")


# ---------------------------------------------------------------------------
# 6. Real MLX LoRA training through the public UnslothTrainer
# ---------------------------------------------------------------------------
def real_training_probe():
    from datasets import Dataset

    model_name = os.environ.get("MLX_PROBE_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    print(f"... loading {model_name} via public FastLanguageModel (MLX route)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
    )
    model = FastLanguageModel.get_peft_model(model, r=8)

    eos = getattr(tokenizer, "eos_token", "") or ""
    rows = [{"text": f"### Question: What is {i} + {i}?\n### Answer: {2 * i}.{eos}"} for i in range(1, 49)]
    dataset = Dataset.from_list(rows)

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
                max_steps=8,
                warmup_steps=1,
                learning_rate=1e-4,
                optim="adamw_8bit",            # exercises alias -> adamw on MLX
                lr_scheduler_type="linear",
                logging_steps=1,
                save_strategy="no",
                seed=3407,
                output_dir=tmp,
            ),
        )
        stats = trainer.train()
        assert isinstance(stats, dict), type(stats)  # MLX returns a metrics dict
        train_loss = stats.get("train_loss")
        assert train_loss is not None and math.isfinite(train_loss) and train_loss > 0, stats
        history = list(getattr(trainer, "_train_loss_history", []) or [])
        print(f"... train_loss={train_loss:.4f} runtime={stats.get('train_runtime')}s history={[round(h,4) for h in history]}")

        trainer.save_model(tmp)
        saved = []
        for root, _dirs, files in os.walk(tmp):
            saved.extend(files)
        assert any(f.endswith((".safetensors", ".npz", ".json")) for f in saved), saved
        print(f"OK: real MLX LoRA training completed + adapter saved ({sorted(set(saved))[:8]})")

        # With the seq2seq-collator patch, notebooks that pass
        # DataCollatorForSeq2Seq (14 migrated Conversational/Phi-4/Coder
        # notebooks) should be ACCEPTED on MLX (the collator is redundant -
        # MLXTrainer batches/masks/pads natively), while a genuinely custom
        # collator still raises.
        from transformers import DataCollatorForSeq2Seq

        accepted = UnslothTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            args=UnslothTrainingArguments(
                dataset_text_field="text", max_steps=1, output_dir=tmp,
            ),
        )
        assert isinstance(accepted, UnslothTrainer)
        print("OK: DataCollatorForSeq2Seq accepted on MLX (routed through native batching)")

        class _WeirdCustomCollator:
            def __call__(self, features):
                return features

        try:
            UnslothTrainer(
                model=model, tokenizer=tokenizer, train_dataset=dataset,
                data_collator=_WeirdCustomCollator(),
                args=UnslothTrainingArguments(dataset_text_field="text", max_steps=1, output_dir=tmp),
            )
            raise SystemExit("UNEXPECTED: genuinely custom collator accepted on MLX")
        except NotImplementedError:
            print("OK: genuinely custom data_collator still rejected on MLX")


real_training_probe()
print("\nALL MLX TRAINER-API PROBES PASSED")
