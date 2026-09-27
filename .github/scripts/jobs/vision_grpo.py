"""Vision GRPO: the Gemma3_(4B)-Vision-GRPO notebook recipe (MathVista testmini numeric questions, images
resized, formatting + correctness rewards, dr_grpo, sequence-level importance sampling) with the model
switched to Gemma 4 E2B. Checks pixel_values reached the policy forward.

    python jobs/vision_grpo.py --max-steps 3      # unsloth/gemma-4-E2B-it, 8 MathVista rows
    python jobs/vision_grpo.py --tiny             # tiny-Gemma4 + drawn arithmetic images (offline)

A few-step run never earns the notebook rewards reliably, so a length + position tie-break reward
(1e-3 scale) keeps reward_std > 0 and the policy loss non-trivial; its weight is recorded.
"""

import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402
import _presets as P  # noqa: E402

pre = P.VISION_GRPO["gemma4"]
p = c.base_parser("vision_grpo", pre["model"], pre["tiny"], default_steps=3)
p.add_argument("--lora-r", type=int, default=pre["lora_r"])
p.add_argument("--rows", type=int, default=8)
p.add_argument("--num-generations", type=int, default=pre["num_generations"])
p.add_argument("--max-completion-length", type=int, default=pre["max_completion_length"])
p.add_argument("--image-size", type=int, default=None, help=f"default {pre['image_size']} (notebook), tiny 64")
p.set_defaults(lr=pre["lr"], max_seq_length=2048)
a = c.resolve_args(p)
a.image_size = a.image_size or (64 if a.tiny else pre["image_size"])

R_START, R_END, S_START, S_END = "<REASONING>", "</REASONING>", "<SOLUTION>", "</SOLUTION>"
MATHVISTA = "data/testmini-00000-of-00001-725687bf7a18d64b.parquet"


def question_text(q):
    return (f"{q}, provide your reasoning between {R_START} and {R_END} "
            f"and then your final answer between {S_START} and (put a float here) {S_END}")


def mathvista_rows(n, size):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image
    tb = pq.read_table(hf_hub_download("AI4Math/MathVista", MATHVISTA, repo_type="dataset"),
                       columns=["question", "answer", "decoded_image"])
    out = []
    for r in tb.to_pylist():
        try:
            float(r["answer"])
        except (TypeError, ValueError):
            continue  # notebook: numeric answers only
        img = Image.open(io.BytesIO(r["decoded_image"]["bytes"])).convert("RGB").resize((size, size))
        out.append((img, r["question"], r["answer"]))
        if len(out) >= n:
            break
    return out


def drawn_rows(n, size):
    from PIL import Image, ImageDraw
    out = []
    for i in range(n):
        x, y = i + 2, 2 * i + 3
        img = Image.new("RGB", (size, size), (255, 255, 255))
        ImageDraw.Draw(img).text((4, size // 3), f"{x}+{y}", fill=(0, 0, 0))
        out.append((img, "What is the sum shown in the image?", str(x + y)))
    return out


def _text(comp):
    return comp[0]["content"] if isinstance(comp, list) else comp


def formatting_reward_func(completions, **kw):
    out = []
    for comp in completions:
        t = _text(comp)
        out.append(float(len(re.findall(f"{R_START}(.*?){R_END}", t, re.DOTALL)) == 1)
                   + float(len(re.findall(f"{S_START}(.*?){S_END}", t, re.DOTALL)) == 1))
    return out


def correctness_reward_func(prompts, completions, answer, **kw):
    res = [re.findall(f"{S_START}(.*?){S_END}", _text(comp), re.DOTALL) for comp in completions]
    return [2.0 if len(r) == 1 and ans == r[0].replace("\n", "") else 0.0 for r, ans in zip(res, answer)]


def tie_break_reward(completions, **kw):
    return [1e-3 * (len(_text(comp)) / 10.0 + i) for i, comp in enumerate(completions)]


with c.JobRecorder(a) as rec:
    c.seed_everything(a.seed)
    if a.backend == "hf":
        rec.skip("vision_grpo covers the Unsloth FastVisionModel path only (--backend unsloth, or auto on an accelerator)")
    import unsloth  # noqa: F401
    import _compat as k
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastVisionModel

    rec.set_backend(k.backend_name(a, c.detect_device()))
    src = "drawn" if a.tiny else "mathvista"
    rows = drawn_rows(a.rows, a.image_size) if a.tiny else mathvista_rows(a.rows, a.image_size)
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": [{"type": "image"},
                                                                     {"type": "text", "text": question_text(q)}]}],
                             "image": img, "answer": ans} for img, q, ans in rows])
    rec.summary(data_source=src, rows=len(rows), notebook=pre["notebook"], tie_break_reward_scale=1e-3)

    model, tok = FastVisionModel.from_pretrained(a.model, load_in_4bit=False, use_gradient_checkpointing="unsloth")
    model = FastVisionModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                           finetune_attention_modules=True, finetune_mlp_modules=True,
                                           r=a.lora_r, lora_alpha=pre["lora_alpha"], lora_dropout=0, bias="none",
                                           random_state=a.seed, use_gradient_checkpointing="unsloth")
    seen = {"forwards": 0, "with_pixels": 0}

    def _count(module, args, kwargs):
        seen["forwards"] += 1
        pv = kwargs.get("pixel_values")
        seen["with_pixels"] += int(pv is not None and getattr(pv, "numel", lambda: 0)() > 0)
    model.register_forward_pre_hook(_count, with_kwargs=True)

    dropped = []
    ga = k.common_train_kwargs(a, os.path.join(os.path.dirname(a.out) or ".", "vision_grpo_run"))
    ga.update(per_device_train_batch_size=a.num_generations, optim="adamw_8bit", adam_beta1=0.9, adam_beta2=0.99,
              weight_decay=0.001, max_grad_norm=pre["max_grad_norm"])
    cfg = k.make(GRPOConfig, dropped, num_generations=a.num_generations, max_prompt_length=pre["max_prompt_length"],
                 max_completion_length=a.max_completion_length, temperature=1.0,
                 importance_sampling_level=pre["importance_sampling_level"], mask_truncated_completions=False,
                 loss_type=pre["loss_type"], **ga)
    trainer = GRPOTrainer(model=model, reward_funcs=[formatting_reward_func, correctness_reward_func, tie_break_reward],
                          args=cfg, train_dataset=ds, callbacks=[c.metrics_callback(rec)],
                          **k.tokenizer_kwarg(GRPOTrainer, tok))
    rec.summary(dropped_config_args=dropped,
                trainable_params=sum(x.numel() for x in model.parameters() if x.requires_grad))

    before = c.adapter_fingerprint(model)
    out = trainer.train()
    after = c.adapter_fingerprint(model)
    rec.summary(train_runtime_s=out.metrics.get("train_runtime"), final_loss=out.training_loss, forwards=seen)

    rec.standard_training_checks()
    rec.backend_check()
    rec.check("adapter_changed", *c.adapter_changed(before, after))
    rec.check("images_reached_model", seen["with_pixels"] > 0, seen)
    steps = rec.data["steps"]
    stds = [s.get("reward_std") for s in steps]
    rec.check("reward_std_positive", all(x is not None and x > 0 for x in stds), stds)
    lens = [s.get("completions/mean_length", s.get("completion_length")) for s in steps]
    rec.check("completion_length_bounded", all(x is not None and 0 < x <= a.max_completion_length for x in lens), lens)
