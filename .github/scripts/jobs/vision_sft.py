"""Vision LoRA SFT: notebook LoRA / trainer settings per --preset, UnslothVisionDataCollator, locally
drawn images. Checks pixel_values reached the model.

    python jobs/vision_sft.py --max-steps 3                   # Qwen2.5-VL-3B-Instruct
    python jobs/vision_sft.py --preset gemma4                 # Gemma4_(E2B)-Vision, unsloth/gemma-4-E2B-it
    python jobs/vision_sft.py --preset qwen3_5                # Qwen3_5_(0_8B)_Vision: Gated DeltaNet (fla) kernels
    python jobs/vision_sft.py --tiny [--preset P]             # CPU / CI
    python jobs/vision_sft.py --model unsloth/gemma-3-4b-it --max-steps 3
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402
import _presets as P  # noqa: E402

p = c.base_parser("vision_sft", "unsloth/Qwen2.5-VL-3B-Instruct",
                  "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration", default_steps=3)
p.add_argument("--load-in-4bit", action="store_true")
p.add_argument("--preset", choices=sorted(P.VISION_SFT), default="default")
p.add_argument("--lora-r", type=int, default=None, help="default: the preset's")
p.add_argument("--rows", type=int, default=16)
p.add_argument("--image-size", type=int, default=224)
p.set_defaults(max_seq_length=None)
a = c.resolve_args(p)
pre, model = P.resolve(P.VISION_SFT, a.preset, a.tiny, a.model if P.model_given(sys.argv) else None)
a.model = model
a.lora_r = a.lora_r or pre["lora_r"]
a.max_seq_length = a.max_seq_length or pre["max_seq_length"]


def image_fixture(n, size):
    from PIL import Image, ImageDraw
    colors = [(220, 40, 40), (40, 160, 60), (40, 80, 220), (230, 200, 40)]
    names = ["red", "green", "blue", "yellow"]
    rows = []
    for i in range(n):
        img = Image.new("RGB", (size, size), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.rectangle([size // 4, size // 4, 3 * size // 4, 3 * size // 4], fill=colors[i % 4])
        d.text((8, 8), str(i % 10), fill=(0, 0, 0))
        rows.append({"messages": [
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": "What color is the square?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"The square is {names[i % 4]}."}]}]})
    return rows


class CountingCollator:
    """Counts batches carrying pixel_values."""

    def __init__(self, inner):
        self.inner, self.batches, self.with_pixels = inner, 0, 0

    def __call__(self, batch):
        out = self.inner([k.strip_none(b) for b in batch])
        self.batches += 1
        pv = out.get("pixel_values") if hasattr(out, "get") else None
        self.with_pixels += int(pv is not None and pv.numel() > 0)
        return out


with c.JobRecorder(a) as rec:
    c.seed_everything(a.seed)
    if a.backend != "hf":
        import unsloth  # noqa: F401
    import _compat as k
    from trl import SFTConfig, SFTTrainer

    rec.set_backend(k.backend_name(a, c.detect_device()))
    rows = image_fixture(a.rows, a.image_size)
    if a.backend != "hf":
        from unsloth import FastVisionModel
        from unsloth.trainer import UnslothVisionDataCollator
        model, tok = FastVisionModel.from_pretrained(a.model, load_in_4bit=a.load_in_4bit,
                                                     use_gradient_checkpointing="unsloth")
        model = FastVisionModel.get_peft_model(model, finetune_vision_layers=True, finetune_language_layers=True,
                                               finetune_attention_modules=True, finetune_mlp_modules=True,
                                               r=a.lora_r, lora_alpha=pre["lora_alpha"], lora_dropout=0, bias="none",
                                               random_state=a.seed,
                                               **({"target_modules": pre["target_modules"]} if pre["target_modules"] else {}))
        FastVisionModel.for_training(model)
        collator = CountingCollator(UnslothVisionDataCollator(model, tok))
    else:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForImageTextToText, AutoProcessor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoProcessor.from_pretrained(a.model)
        model = AutoModelForImageTextToText.from_pretrained(
            a.model, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
        model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_r, target_modules="all-linear"))

        def hf_collate(batch):
            texts = [tok.apply_chat_template(b["messages"], tokenize=False) for b in batch]
            imgs = [[x["image"] for m in b["messages"] for x in m["content"] if x["type"] == "image"] for b in batch]
            enc = tok(text=texts, images=imgs, return_tensors="pt", padding=True)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            enc["labels"] = labels
            return enc
        collator = CountingCollator(hf_collate)

    dropped = []
    ta = k.common_train_kwargs(a, os.path.join(os.path.dirname(a.out) or ".", "vision_run"))
    # notebook batch / accumulation; constant LR, no warmup (few-step run)
    ta.update(per_device_train_batch_size=pre["batch"], gradient_accumulation_steps=pre["grad_accum"],
              max_grad_norm=0.3, dataset_num_proc=1,
              optim="adamw_8bit" if a.backend != "hf" else "adamw_torch")
    cfg = k.make(SFTConfig, dropped, remove_unused_columns=False, dataset_text_field="",
                 dataset_kwargs={"skip_prepare_dataset": True}, max_length=a.max_seq_length, **ta)
    trainer = SFTTrainer(model=model, train_dataset=k.vision_dataset(rows), data_collator=collator, args=cfg,
                         callbacks=[c.metrics_callback(rec)], **k.tokenizer_kwarg(SFTTrainer, tok))
    gdn = k.gdn_info(model)
    rec.summary(preset=a.preset, notebook=pre["notebook"], gdn=gdn)
    rec.summary(dropped_config_args=dropped, rows=len(rows),
                trainable_params=sum(x.numel() for x in model.parameters() if x.requires_grad))

    before = c.adapter_fingerprint(model)
    out = trainer.train()
    after = c.adapter_fingerprint(model)
    rec.summary(train_runtime_s=out.metrics.get("train_runtime"), final_loss=out.training_loss,
                batches=collator.batches, batches_with_pixels=collator.with_pixels)

    rec.standard_training_checks()
    rec.backend_check()
    rec.check("adapter_changed", *c.adapter_changed(before, after))
    if gdn["gdn_modules"] and a.backend != "hf" and c.detect_device() == "cuda":
        # Unsloth vendors fla: the torch reference here means the fast kernels silently dropped out.
        rec.check("gdn_uses_fla_kernel", gdn["fla_used"], gdn["gdn_impls"])
    rec.check("images_reached_model", collator.batches > 0 and collator.with_pixels == collator.batches,
              f"{collator.with_pixels}/{collator.batches} batches carried pixel_values")
