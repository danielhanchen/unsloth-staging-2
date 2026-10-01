"""DPO / ORPO on an in-code preference fixture. Zephyr-DPO / Llama3-ORPO notebook settings
(beta 0.1, ref_model=None: frozen base is the reference).

    python jobs/dpo.py --max-steps 3          # Qwen3-0.6B, PatchDPOTrainer + DPOTrainer
    python jobs/dpo.py --orpo --max-steps 3   # ORPOTrainer
    python jobs/dpo.py --tiny                 # CPU / CI
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402

p = c.base_parser("dpo", "unsloth/Qwen3-0.6B", "trl-internal-testing/tiny-Qwen3ForCausalLM", default_steps=3)
p.add_argument("--orpo", action="store_true")
p.add_argument("--beta", type=float, default=0.1)
p.add_argument("--lora-r", type=int, default=16)
p.add_argument("--rows", type=int, default=32)
p.set_defaults(lr=5e-6)
a = c.resolve_args(p)
if a.orpo:
    a._job = "orpo"
    if not any(x == "--out" or x.startswith("--out=") for x in sys.argv[1:]):
        a.out = "outputs/jobs/orpo.json"


def preference_fixture(n):
    pairs = [("What is the capital of France?", "The capital of France is Paris.", "I think it is Berlin."),
             ("What is 2 + 2?", "2 + 2 equals 4.", "2 + 2 equals 5."),
             ("Name a primary color.", "Red is a primary color.", "Purple is a primary color."),
             ("What do bees make?", "Bees make honey.", "Bees make milk.")]
    rows = []
    for i in range(n):
        q, good, bad = pairs[i % len(pairs)]
        rows.append({"prompt": f"{q} (#{i})\n", "chosen": good, "rejected": bad})
    return rows


with c.JobRecorder(a) as rec:
    c.seed_everything(a.seed)
    if a.backend != "hf":
        import unsloth  # noqa: F401
        from unsloth import PatchDPOTrainer
        PatchDPOTrainer()
    import _compat as k
    from datasets import Dataset
    if a.orpo:
        try:
            from trl import ORPOConfig as Cfg, ORPOTrainer as Trainer
        except ImportError:
            from trl.experimental.orpo import ORPOConfig as Cfg, ORPOTrainer as Trainer
    else:
        from trl import DPOConfig as Cfg, DPOTrainer as Trainer

    rec.set_backend(k.backend_name(a, c.detect_device()))
    model, tok = k.load_text_model(a, lora_r=a.lora_r, lora_alpha=a.lora_r)
    k.ensure_pad(tok)
    t = getattr(tok, "tokenizer", tok)
    ds = Dataset.from_list(preference_fixture(a.rows))  # tokenize_row appends EOS

    dropped = []
    cfg = k.make(Cfg, dropped, beta=a.beta, max_length=a.max_seq_length, max_prompt_length=a.max_seq_length // 2,
                 max_completion_length=a.max_seq_length // 2,
                 **k.common_train_kwargs(a, os.path.join(os.path.dirname(a.out) or ".", f"{a._job}_run")))
    kw = {"ref_model": None} if not a.orpo else {}
    trainer = Trainer(model=model, args=cfg, train_dataset=ds, callbacks=[c.metrics_callback(rec)],
                      **kw, **k.tokenizer_kwarg(Trainer, tok))
    rec.summary(dropped_config_args=dropped, pairs=len(ds), beta=a.beta)

    before = c.adapter_fingerprint(model)
    out = trainer.train()
    after = c.adapter_fingerprint(model)
    rec.summary(train_runtime_s=out.metrics.get("train_runtime"), final_loss=out.training_loss)

    rec.standard_training_checks()
    rec.backend_check()
    rec.check("adapter_changed", *c.adapter_changed(before, after))
    steps = rec.data["steps"]
    if not a.orpo:
        # DPO starts at log(2) (reference == policy); margins finite.
        first = steps[0].get("loss") if steps else None
        rec.check("dpo_initial_loss_near_log2", first is not None and abs(first - 0.6931) < 0.05, first)
        margins = [s.get("rewards/margins") for s in steps]
        rec.check("reward_margins_finite", all(m is not None and abs(m) < 1e3 for m in margins), margins)
