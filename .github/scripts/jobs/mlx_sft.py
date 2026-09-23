"""MLX LoRA SFT on Apple Silicon through Unsloth's MLX path.

    python jobs/mlx_sft.py                      # mlx-community/Qwen3.5-2B-4bit, 5 steps
    python jobs/mlx_sft.py --tiny               # mlx-community/SmolLM-135M-Instruct-4bit
    python jobs/mlx_sft.py --model mlx-community/gemma-4-e2b-it-OptiQ-4bit   # see note below
    python jobs/mlx_sft.py --memorise           # upstream mlx-ci gate: gemma-3-270m-it fp16, 30 steps

`import unsloth` flips `_IS_MLX` on Darwin arm64 with mlx installed, and FastLanguageModel
then routes to unsloth_zoo's FastMLXModel. Training uses unsloth_zoo.mlx.trainer.MLXTrainer
with a per-step callback, as upstream tests/studio/run_real_mlx_smoke.py does.

Default preset checks: steps completed, finite loss, per-step grad norm finite and bounded, LoRA
weights changed, generation non-empty, LoRA adapter saved. `--memorise` adds the upstream
memorisation gates: post-train loss < 0.1 and teacher-forced completion loss < 0.5. Metal's
reduction order is nondeterministic, so these are bounds, not exact values.
torch is never imported.

Note (2026-09, unsloth-zoo main): mlx-community/gemma-4-e2b-it-OptiQ-4bit fails at load with
"Missing 1411 parameters: audio_tower...". The checkpoint ships without its audio tower, mlx-vlm
builds the full Gemma 4 model and loads strictly, and the zoo's VLM loader only filters EXTRA
tensors. The job reports that failure as-is rather than loading non-strictly.

Note (2026-09, mlx 0.32.2, mlx-lm 0.31.3, macos-26): the default mlx-community/Qwen3.5-2B-4bit
trains but fails `grad_norm_bounded` (norms ~700-1100, loss ~11). On the same runner and text,
mlx_lm.load scores CE 0.97 while mlx_vlm.load and Unsloth's text_only load (which Unsloth routes
through mlx-vlm) both score 11.125, so the defect is the mlx-vlm forward for this checkpoint that
Unsloth's loader selects, not the job. `--tiny` and `--memorise` are the green CI presets.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen3.5-2B-4bit"
TINY_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
MEMORISE_MODEL = "unsloth/gemma-3-270m-it"  # the model upstream mlx-ci.yml trains
TRAIN_TEXT = "<<HELLO!!>> My name is Unsloth!"
PROMPT = "<<HELLO!!>> My name is "
EXPECT = "Unsloth"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Small built-in instruction set so the default preset downloads no dataset.
PAIRS = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Add 17 and 25.", "17 + 25 = 42."),
    ("Name a primary colour.", "Red is a primary colour."),
    ("Translate 'thank you' to Spanish.", "'Thank you' in Spanish is 'gracias'."),
    ("What does LoRA stand for?", "LoRA stands for Low-Rank Adaptation."),
    ("Give a synonym for 'quick'.", "A synonym for 'quick' is 'fast'."),
    ("How many days are in a leap year?", "A leap year has 366 days."),
    ("What is 9 squared?", "9 squared is 81."),
]


def _argv_has(argv, flag):
    return any(x == flag or x.startswith(flag + "=") for x in argv)


def _peak_gpu_gb():
    import mlx.core as mx
    getter = getattr(mx, "get_peak_memory", None) or getattr(getattr(mx, "metal", None), "get_peak_memory", None)
    try:
        return round(float(getter()) / 1024**3, 4) if getter else None
    except Exception:
        return None


def _peak_rss_gb():
    import resource  # POSIX only; mlx_sft never runs its body on Windows
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / 1024**3 if sys.platform == "darwin" else rss / 1024**2, 4)


def _lora_fingerprint(model):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    total, n = 0.0, 0
    for _, v in tree_flatten(model.trainable_parameters()):
        total += float(mx.sum(mx.abs(v.astype(mx.float32))).item())
        n += int(v.size)
    return {"abs_sum": total, "numel": n}


def _logits(model, inputs):
    """Models loaded on the mlx-vlm path (Qwen3.5, Gemma 3n/4) return an output object."""
    out = model(inputs)
    return getattr(out, "logits", out)


def _greedy(model, tokenizer, prompt, max_tokens):
    """Plain greedy decode through the model's forward; used when mlx_lm.generate cannot
    drive a model that loaded through mlx-vlm."""
    import mlx.core as mx
    ids = list(tokenizer.encode(prompt))
    eos = getattr(tokenizer, "eos_token_id", None)
    new = []
    for _ in range(max_tokens):
        nxt = int(mx.argmax(_logits(model, mx.array([ids + new], dtype=mx.int32))[0, -1]).item())
        if nxt == eos:
            break
        new.append(nxt)
    return tokenizer.decode(new)


def _loss_and_grad_norm(model, tokenizer, text):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    ids = list(tokenizer.encode(text))
    inputs, targets = mx.array([ids[:-1]], dtype=mx.int32), mx.array([ids[1:]], dtype=mx.int32)

    def loss_fn(m):
        return nn.losses.cross_entropy(_logits(m, inputs), targets, reduction="mean")

    loss, grad = nn.value_and_grad(model, loss_fn)(model)
    sq = mx.array(0.0, dtype=mx.float32)
    for _, g in tree_flatten(grad):
        g = g.astype(mx.float32)
        sq = sq + mx.sum(g * g)
    return float(loss.item()), float(mx.sqrt(sq).item())


def _loss(model, tokenizer, text):
    """Forward-only next-token CE. Pre/post-train probes use this, not a raw value_and_grad:
    models with custom Metal kernels (Qwen3.5 gated delta) only get a VJP once MLXTrainer
    patches them, so a bare backward pass before train() fails for reasons unrelated to training."""
    import mlx.core as mx
    import mlx.nn as nn
    ids = list(tokenizer.encode(text))
    inputs, targets = mx.array([ids[:-1]], dtype=mx.int32), mx.array([ids[1:]], dtype=mx.int32)
    return float(nn.losses.cross_entropy(_logits(model, inputs), targets, reduction="mean").item())


def _completion_loss(model, tokenizer, prompt, completion):
    import mlx.core as mx
    import mlx.nn as nn
    p_ids, f_ids = list(tokenizer.encode(prompt)), list(tokenizer.encode(prompt + completion))
    inputs, targets = mx.array([f_ids[:-1]], dtype=mx.int32), mx.array([f_ids[1:]], dtype=mx.int32)
    start = len(p_ids) - 1
    logits = _logits(model, inputs)
    return float(nn.losses.cross_entropy(logits[:, start:, :], targets[:, start:], reduction="mean").item())


def _format_pairs(tokenizer):
    rows = []
    for q, a in PAIRS:
        msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False)
        except Exception:
            text = f"### Question: {q}\n### Answer: {a}"
        rows.append({"text": text})
    return rows * 4


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = C.base_parser("mlx_sft", DEFAULT_MODEL, TINY_MODEL, default_steps=5)
    p.add_argument("--memorise", action="store_true",
                   help="upstream mlx-ci memorisation gate (gemma-3-270m-it fp16, 30 steps)")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--save-dir", default=None, help="LoRA output dir (default next to --out)")
    a = C.resolve_args(p, argv)
    if a.memorise:
        if not _argv_has(argv, "--model"):
            a.model = MEMORISE_MODEL
        if not _argv_has(argv, "--max-steps"):
            a.max_steps = 30
        if not _argv_has(argv, "--lr"):
            a.lr = 1e-3
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    with C.JobRecorder(a, backend_hint=None) as rec:
        rec.data["summary"]["preset"] = "memorise" if a.memorise else ("tiny" if a.tiny else "default")
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            rec.skip(f"MLX needs Apple Silicon, this is {platform.system()}-{platform.machine()}")
        import unsloth  # noqa: F401  must precede mlx_lm / transformers
        is_mlx = bool(getattr(unsloth, "_IS_MLX", False))
        rec.set_backend("unsloth-mlx" if is_mlx else "unsloth-no-mlx")
        if not rec.check("mlx_active", is_mlx, f"unsloth._IS_MLX={is_mlx} on {C.detect_device()}"):
            return
        rec.backend_check()

        import mlx.core as mx
        from unsloth import FastLanguageModel
        from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

        C.seed_everything(a.seed)
        mx.random.seed(a.seed)
        t0 = time.perf_counter()
        load_kw = dict(max_seq_length=a.max_seq_length, text_only=True, random_state=a.seed,
                       token=os.environ.get("HF_TOKEN") or None)
        if a.memorise:
            load_kw.update(load_in_4bit=False, dtype="float16", max_seq_length=128)
        else:
            load_kw.update(load_in_4bit=True)
        model, tokenizer = FastLanguageModel.from_pretrained(a.model, **load_kw)
        rec.summary(load_s=round(time.perf_counter() - t0, 2), peak_gpu_gb_after_load=_peak_gpu_gb())

        mx.random.seed(a.seed)
        model = FastLanguageModel.get_peft_model(
            model, r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, target_modules=TARGETS,
            use_gradient_checkpointing=False, random_state=a.seed)
        before = _lora_fingerprint(model)
        rec.summary(trainable_params=before["numel"])
        rec.check("has_trainable_params", before["numel"] > 0, before["numel"])

        probe_text = TRAIN_TEXT if a.memorise else _format_pairs(tokenizer)[0]["text"]
        pre_loss = _loss(model, tokenizer, probe_text)
        rec.summary(pre_train_loss=round(pre_loss, 4))

        if a.memorise:
            data = [{"text": TRAIN_TEXT}] * 64
            cfg = dict(per_device_train_batch_size=2, gradient_accumulation_steps=3,
                       lr_scheduler_type="constant", optim="adamw", weight_decay=0.0,
                       max_grad_norm=0.0, max_grad_value=1.0, max_seq_length=64)
        else:
            data = _format_pairs(tokenizer)
            cfg = dict(per_device_train_batch_size=1, gradient_accumulation_steps=1,
                       lr_scheduler_type="linear", optim="adamw", weight_decay=0.0,
                       max_grad_norm=1.0, max_seq_length=a.max_seq_length)
        out_dir = Path(a.save_dir or Path(a.out).with_suffix("")).resolve()
        import dataclasses
        if "report_grad_norm" in {f.name for f in dataclasses.fields(MLXTrainingConfig)}:
            cfg["report_grad_norm"] = True  # pre-clip norm every step, even with clipping off
        config = MLXTrainingConfig(
            max_steps=a.max_steps, learning_rate=a.lr, warmup_steps=0, logging_steps=1,
            seed=a.seed, use_cce=False, compile=False, gradient_checkpointing=False,
            output_dir=str(out_dir / "trainer_outputs"), save_steps=0, eval_steps=0,
            dataset_text_field="text", **cfg)
        trainer = MLXTrainer(model=model, tokenizer=tokenizer, train_dataset=data, args=config)

        last = {"elapsed": 0.0, "tokens": 0}

        def on_step(step, total, loss, lr, tok_s, peak_gb, elapsed, num_tokens, grad_norm=None, *_, **__):
            dt = float(elapsed) - last["elapsed"] if elapsed is not None else None
            last["elapsed"] = float(elapsed) if elapsed is not None else last["elapsed"]
            tokens = int(num_tokens) - last["tokens"] if num_tokens is not None else None  # cumulative upstream
            last["tokens"] = int(num_tokens) if num_tokens is not None else last["tokens"]
            rec.step(step=int(step), loss=float(loss), grad_norm=None if grad_norm is None else float(grad_norm),
                     learning_rate=float(lr), time_ms=round(dt * 1000, 2) if dt else None,
                     tokens=tokens,
                     tokens_per_s=float(tok_s) if tok_s is not None else None,
                     peak_mem_gb=float(peak_gb) if peak_gb is not None else None)
            print(f"  step {step}/{total} loss={loss:.4f} tok/s={tok_s:.0f} peak={peak_gb:.2f}GB", flush=True)

        trainer.add_step_callback(on_step)
        t1 = time.perf_counter()
        result = trainer.train() or {}
        rec.summary(train_s=round(time.perf_counter() - t1, 2),
                    **{k: result[k] for k in ("train_loss", "train_runtime", "train_steps", "trained_tokens",
                                              "train_samples_per_second") if k in result})

        # Checks: the standard contract, except grad_norm falls back to the gradient probe when
        # the trainer does not report one (upstream disables norm clipping on the memorise set).
        steps = rec.data["steps"]
        losses = [s["loss"] for s in steps]
        rec.check("completed_steps", len(steps) == a.max_steps, f"{len(steps)} logged, {a.max_steps} requested")
        rec.check("finite_loss", bool(losses) and all(math.isfinite(l) and 0 <= l < 50 for l in losses), losses)
        gns = [s["grad_norm"] for s in steps if s.get("grad_norm") is not None]
        if gns:
            rec.check("grad_norm_bounded", all(math.isfinite(g) and 0 < g < 1e3 for g in gns), gns)
        else:  # older unsloth_zoo without report_grad_norm: one raw backward pass after training
            try:
                _, gn = _loss_and_grad_norm(model, tokenizer, probe_text)
                rec.check("grad_norm_bounded", math.isfinite(gn) and 0 < gn < 1e3,
                          f"trainer logged none; post-train probe {gn:.4f}")
            except Exception as e:
                rec.check("grad_norm_bounded", False, f"trainer logged none and probe failed: {e}")
        after = _lora_fingerprint(model)
        rec.check("adapter_changed", *C.adapter_changed(before, after))

        post_loss = _loss(model, tokenizer, probe_text)
        rec.summary(post_train_loss=round(post_loss, 4))
        if a.memorise:
            rec.check("loss_not_diverged", losses[-1] < losses[0] * 1.1, f"{losses[0]} -> {losses[-1]}")
            rec.check("memorised_post_loss_lt_0.1", post_loss < 0.1, f"post_train_loss={post_loss:.4f}")
            cl = _completion_loss(model, tokenizer, PROMPT, EXPECT + "!")
            rec.summary(completion_teacher_forced_loss=round(cl, 6))
            rec.check("memorised_completion_loss_lt_0.5", cl < 0.5, f"{cl:.4f}")

        model.eval()
        prompt = PROMPT if a.memorise else "What is the capital of France?"
        t2 = time.perf_counter()
        try:
            from mlx_lm import generate
            text, how = generate(model, tokenizer, prompt=prompt, max_tokens=24, verbose=False), "mlx_lm.generate"
        except Exception as e:  # mlx-vlm-path models: mlx_lm cannot drive their output object
            text, how = _greedy(model, tokenizer, prompt, 24), f"greedy fallback ({type(e).__name__})"
        rec.summary(generate_s=round(time.perf_counter() - t2, 2), generation=text, generation_method=how)
        rec.check("generation_nonempty", bool(text and text.strip()), repr(text[:80]))
        if a.memorise:
            rec.summary(generation_has_expected=EXPECT in text)  # soft, as upstream

        lora_dir = out_dir / "lora"
        model.save_pretrained_merged(str(lora_dir), tokenizer=tokenizer, save_method="lora")
        rec.check("lora_saved", (lora_dir / "adapters.safetensors").exists()
                  and (lora_dir / "adapter_config.json").exists(), str(lora_dir))
        rec.summary(peak_mem_gb=_peak_gpu_gb(), peak_rss_gb=_peak_rss_gb(),
                    tokens_per_s_mean=round(sum(s["tokens_per_s"] or 0 for s in steps) / max(len(steps), 1), 2))


if __name__ == "__main__":
    main()
