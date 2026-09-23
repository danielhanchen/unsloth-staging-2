"""Text LoRA SFT sample job.

    python jobs/sft.py --max-steps 3                       # unsloth/Qwen3-0.6B, CUDA
    python jobs/sft.py --tiny                              # tiny Qwen3, CPU / CI smoke
    python jobs/sft.py --backend hf                        # plain transformers + PEFT + TRL reference
    python jobs/sft.py --dataset mlabonne/FineTome-100k --dataset-rows 256 --responses-only

Default data is an in-code chat fixture, so base and head train on identical tokens.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402

p = c.base_parser("sft", "unsloth/Qwen3-0.6B", "trl-internal-testing/tiny-Qwen3ForCausalLM")
p.add_argument("--load-in-4bit", action="store_true")
p.add_argument("--lora-r", type=int, default=16)
p.add_argument("--dataset", default=None, help="HF dataset with a conversations/messages column")
p.add_argument("--dataset-rows", type=int, default=64)
p.add_argument("--responses-only", action="store_true", help="unsloth train_on_responses_only")
a = c.resolve_args(p)

with c.JobRecorder(a) as rec:
    c.seed_everything(a.seed)
    if a.backend != "hf":
        import unsloth  # noqa: F401  (must precede transformers / trl)
    import _compat as k
    from datasets import Dataset, load_dataset
    from trl import SFTConfig, SFTTrainer

    rec.set_backend(k.backend_name(a, c.detect_device()))
    model, tok = k.load_text_model(a, lora_r=a.lora_r, lora_alpha=a.lora_r, load_in_4bit=a.load_in_4bit)
    k.ensure_pad(tok)

    if a.dataset:
        raw = load_dataset(a.dataset, split=f"train[:{a.dataset_rows}]")
        col = "messages" if "messages" in raw.column_names else "conversations"
        rows = [{"messages": [{"role": {"human": "user", "gpt": "assistant"}.get(m.get("from", m.get("role")), m.get("role", m.get("from"))),
                               "content": m.get("value", m.get("content"))} for m in r[col]]} for r in raw]
    else:
        rows = k.chat_fixture(a.dataset_rows)
    ds = Dataset.from_list(k.render_chat(tok, rows))

    dropped = []
    cfg = k.make(SFTConfig, dropped, dataset_text_field="text", max_length=a.max_seq_length,
                 max_seq_length=a.max_seq_length, packing=False,
                 **k.common_train_kwargs(a, os.path.join(os.path.dirname(a.out) or ".", "sft_run")))
    trainer = SFTTrainer(model=model, train_dataset=ds, args=cfg,
                         callbacks=[c.metrics_callback(rec)], **k.tokenizer_kwarg(SFTTrainer, tok))
    if a.responses_only and a.backend != "hf":
        from unsloth.chat_templates import train_on_responses_only
        t = getattr(tok, "tokenizer", tok)
        if "<|im_start|>" in (t.chat_template or ""):
            trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                              response_part="<|im_start|>assistant\n")
    rec.summary(dropped_config_args=dropped, train_rows=len(ds),
                trainable_params=sum(x.numel() for x in model.parameters() if x.requires_grad))

    before = c.adapter_fingerprint(model)
    out = trainer.train()
    after = c.adapter_fingerprint(model)
    rec.summary(train_runtime_s=out.metrics.get("train_runtime"), final_loss=out.training_loss)

    rec.standard_training_checks()
    rec.backend_check()
    rec.check("adapter_changed", *c.adapter_changed(before, after))
