"""MoE LoRA SFT on tiny Qwen3-MoE: LoRA on attention AND the fused experts (as TRL's MoE PEFT tests),
counts torch._grouped_mm calls and records the MoE backend. Checks grouped_mm ran on CUDA sm >= 8.0.

    python jobs/moe_sft.py                                  # tiny Qwen3MoE, Unsloth (auto)
    UNSLOTH_MOE_BACKEND=native_torch python jobs/moe_sft.py # other backend (recorded, compared)
    python jobs/moe_sft.py --backend hf                     # transformers + PEFT target_parameters
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402
import _presets as P  # noqa: E402

pre = P.MOE_SFT["qwen3_moe_tiny"]
p = c.base_parser("moe_sft", pre["model"], pre["tiny"])
p.add_argument("--lora-r", type=int, default=pre["lora_r"])
p.add_argument("--rows", type=int, default=32)
a = c.resolve_args(p)

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
# Fused Qwen3MoeExperts parameters (transformers 5): PEFT reaches them via target_parameters.
EXPERT_PARAMS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]

with c.JobRecorder(a) as rec:
    c.seed_everything(a.seed)
    if a.backend != "hf":
        import unsloth  # noqa: F401
    import _compat as k
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    rec.set_backend(k.backend_name(a, c.detect_device()))
    gmm = k.GroupedMMCounter()
    if a.backend != "hf":
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(model_name=a.model, max_seq_length=a.max_seq_length,
                                                       load_in_4bit=False, dtype=None)
        # gate/up/down_proj names: Unsloth maps them onto the fused experts (get_moe_target_parameters).
        model = FastLanguageModel.get_peft_model(model, r=a.lora_r, lora_alpha=pre["lora_alpha"], lora_dropout=0,
                                                 target_modules=ATTN + ["gate_proj", "up_proj", "down_proj"],
                                                 use_gradient_checkpointing="unsloth", random_state=a.seed)
    else:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(
            a.model, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
        model = get_peft_model(model, k.make(LoraConfig, r=a.lora_r, lora_alpha=pre["lora_alpha"], lora_dropout=0.0,
                                             target_modules=ATTN, target_parameters=EXPERT_PARAMS,
                                             task_type="CAUSAL_LM"))
    k.ensure_pad(tok)
    moe = k.moe_info(model)
    expert_lora = k.lora_param_names(model, "experts")
    ds = Dataset.from_list(k.render_chat(tok, k.chat_fixture(a.rows)))

    dropped = []
    cfg = k.make(SFTConfig, dropped, dataset_text_field="text", max_length=a.max_seq_length, packing=False,
                 dataset_num_proc=1, **k.common_train_kwargs(a, os.path.join(os.path.dirname(a.out) or ".", "moe_run")))
    trainer = SFTTrainer(model=model, train_dataset=ds, args=cfg,
                         callbacks=[c.metrics_callback(rec)], **k.tokenizer_kwarg(SFTTrainer, tok))
    rec.summary(dropped_config_args=dropped, moe=moe, expert_lora_params=len(expert_lora),
                trainable_params=sum(x.numel() for x in model.parameters() if x.requires_grad))

    before = c.adapter_fingerprint(model)
    out = trainer.train()
    after = c.adapter_fingerprint(model)
    gmm.restore()
    rec.summary(train_runtime_s=out.metrics.get("train_runtime"), final_loss=out.training_loss,
                grouped_mm_calls=gmm.calls)

    rec.standard_training_checks()
    rec.backend_check()
    rec.check("adapter_changed", *c.adapter_changed(before, after))
    rec.check("lora_on_experts", bool(expert_lora), expert_lora[:4] or "no trainable parameter under *.experts.*")
    if moe["grouped_mm_expected"] and (a.backend == "hf" or moe["unsloth_backend"] == "grouped_mm"):
        rec.check("grouped_mm_called", gmm.calls > 0,
                  f"{gmm.calls} torch._grouped_mm calls; backend {moe['unsloth_backend']}, sm {moe['capability']}")
