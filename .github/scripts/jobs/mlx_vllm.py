"""MLX fast_inference through vllm-metal on Apple Silicon (unsloth#19221 + unsloth-zoo#2724): the
Qwen3 GRPO notebook flow. Never imports torch directly.

    python jobs/mlx_vllm.py                       # unsloth/Qwen3-0.6B, load_in_4bit
    python jobs/mlx_vllm.py --load bf16
    python jobs/mlx_vllm.py --install "git+...unsloth@SHA git+...unsloth-zoo@SHA"   # base arm

Checks: engine attached + sharing the training weights, fast_generate, LoRA SFT, save_lora ->
load_lora round trip changes greedy output, live load_lora(load_tensors=True) == disk adapter,
GRPO use_vllm=True with vllm_sampling_params trains and its rollouts ran through the engine.
"""

from __future__ import annotations

import math
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MODEL = "unsloth/Qwen3-0.6B"
TINY_MODEL = "unsloth/Qwen3-0.6B"
TRAIN_TEXT = "<<HELLO!!>> My name is Unsloth!"
PROMPT = "<<HELLO!!>> My name is "
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


VLLM_METAL = "0.33.0"
# vllm-metal's install.sh: vLLM's prebuilt macOS wheel, then the vllm-metal wheel, as separate installs.
VLLM_WHEEL = (
    "https://github.com/vllm-project/vllm/releases/download/v0.33.0rc3/"
    "vllm-0.33.0%2Bcpu-cp38-abi3-macosx_11_0_arm64.whl"
)
VLLM_METAL_WHEEL = (
    "https://github.com/vllm-project/vllm-metal/releases/download/v{v}/"
    "vllm_metal-{v}-cp312-cp312-macosx_15_0_arm64.whl"
)


def _pip(*args):
    cmd = [sys.executable, "-m", "pip", "install", *args]
    print("+", shlex.join(cmd), flush = True)
    subprocess.run(cmd, check = True)


def _preinstall(argv):
    """Runs ahead of argparse: installs must land before unsloth is imported."""
    if "--vllm-metal" in argv:
        _pip(VLLM_WHEEL)
        _pip(VLLM_METAL_WHEEL.format(v = argv[argv.index("--vllm-metal") + 1]))
    if "--install" in argv:
        _pip("--no-deps", "--force-reinstall", *shlex.split(argv[argv.index("--install") + 1]))


def _peak_gb():
    import mlx.core as mx
    getter = getattr(mx, "get_peak_memory", None)
    return round(float(getter()) / 1024**3, 4) if getter else None


def _vllm_greedy(
    model,
    prompt,
    lora_request = None,
    max_tokens = 12,
):
    from vllm import SamplingParams
    out = model.fast_generate(
        [prompt],
        SamplingParams(temperature = 0.0, max_tokens = max_tokens),
        lora_request = lora_request,
        use_tqdm = False,
    )
    return out[0].outputs[0].text


def main(argv = None):
    argv = sys.argv[1:] if argv is None else argv
    _preinstall(argv)
    import _common as C

    p = C.base_parser("mlx_vllm", DEFAULT_MODEL, TINY_MODEL, default_steps = 20)
    p.add_argument("--load", choices = ["4bit", "bf16"], default = "4bit")
    p.add_argument("--install", default = None, help = "pip --no-deps --force-reinstall before import")
    p.add_argument(
        "--vllm-metal",
        default = None,
        help = f"install vLLM + this vllm-metal release first (e.g. {VLLM_METAL})",
    )
    p.add_argument("--grpo-steps", type = int, default = 2)
    p.add_argument("--gpu-memory-utilization", type = float, default = 0.3)
    p.add_argument(
        "--max-num-batched-tokens",
        type = int,
        default = None,
        help = "cap vLLM's batch (and its profiling pass): GitHub's 7 GB macOS runners fit no default batch",
    )
    a = C.resolve_args(p, argv)
    if a.lr == p.get_default("lr"):
        a.lr = 1e-3
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    with C.JobRecorder(a, backend_hint = None) as rec:
        rec.summary(arm = "install" if a.install else "as-installed", load = a.load)
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            rec.skip(
                f"vllm-metal needs Apple Silicon, this is {platform.system()}-{platform.machine()}"
            )
        import unsloth

        is_mlx = bool(getattr(unsloth, "_IS_MLX", False))
        rec.set_backend("unsloth-mlx" if is_mlx else "unsloth-no-mlx")
        if not rec.check("mlx_active", is_mlx, f"unsloth._IS_MLX={is_mlx}"):
            return
        rec.backend_check()
        import importlib.metadata as md

        rec.summary(vllm = md.version("vllm"), vllm_metal = md.version("vllm-metal"))

        import mlx.core as mx
        from unsloth import FastLanguageModel

        if a.max_num_batched_tokens:
            import vllm

            llm_init = vllm.LLM.__init__
            vllm.LLM.__init__ = lambda self, *x, **k: llm_init(
                self, *x, **{**k, "max_num_batched_tokens": a.max_num_batched_tokens}
            )
            rec.summary(max_num_batched_tokens = a.max_num_batched_tokens)

        C.seed_everything(a.seed)
        mx.random.seed(a.seed)
        t0 = time.perf_counter()
        load_kw = dict(
            max_seq_length = a.max_seq_length,
            random_state = a.seed,
            fast_inference = True,
            max_lora_rank = 16,
            gpu_memory_utilization = a.gpu_memory_utilization,
        )
        load_kw.update(load_in_4bit = True) if a.load == "4bit" else load_kw.update(
            load_in_4bit = False, dtype = "bfloat16"
        )
        model, tokenizer = FastLanguageModel.from_pretrained(a.model, **load_kw)
        rec.summary(load_s = round(time.perf_counter() - t0, 2), peak_gb_after_load = _peak_gb())

        engine = getattr(model, "vllm_engine", None)
        if not rec.check(
            "engine_attached",
            engine is not None and model.fast_generate == engine.generate,
            f"vllm_engine={type(engine).__name__}",
        ):
            return
        rec.check(
            "engine_shares_training_weights",
            getattr(model, "_unsloth_vllm_shared", False) is True,
            getattr(model, "_unsloth_vllm_shared", None),
        )
        base_text = _vllm_greedy(model, PROMPT)
        rec.summary(base_generation = base_text)
        rec.check("fast_generate_nonempty", bool(base_text.strip()), repr(base_text))

        model = FastLanguageModel.get_peft_model(
            model,
            r = 8,
            lora_alpha = 16,
            lora_dropout = 0.0,
            target_modules = TARGETS,
            use_gradient_checkpointing = False,
            random_state = a.seed,
        )
        from unsloth_zoo.mlx.trainer import (
            MLXGRPOConfig,
            MLXGRPOTrainer,
            MLXTrainer,
            MLXTrainingConfig,
        )

        out_dir = Path(Path(a.out).with_suffix("")).resolve()
        cfg = MLXTrainingConfig(
            max_steps = a.max_steps,
            learning_rate = a.lr,
            warmup_steps = 0,
            logging_steps = 1,
            seed = a.seed,
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 1,
            lr_scheduler_type = "constant",
            max_grad_norm = 1.0,
            max_seq_length = 64,
            use_cce = False,
            compile = False,
            gradient_checkpointing = False,
            output_dir = str(out_dir / "sft"),
            save_steps = 0,
            eval_steps = 0,
            dataset_text_field = "text",
        )
        trainer = MLXTrainer(
            model = model, tokenizer = tokenizer, train_dataset = [{"text": TRAIN_TEXT}] * 64, args = cfg
        )

        def on_step(
            step,
            total,
            loss,
            lr,
            tok_s,
            peak_gb,
            elapsed,
            num_tokens,
            grad_norm = None,
            *_,
            **__,
        ):
            rec.step(
                step = int(step),
                loss = float(loss),
                grad_norm = None if grad_norm is None else float(grad_norm),
                time_ms = None,
                tokens = None,
                tokens_per_s = float(tok_s) if tok_s is not None else None,
                peak_mem_gb = float(peak_gb) if peak_gb is not None else None,
            )

        trainer.add_step_callback(on_step)
        trainer.train()
        losses = [s["loss"] for s in rec.data["steps"]]
        rec.check("completed_steps", len(losses) == a.max_steps, f"{len(losses)}/{a.max_steps}")
        rec.check("finite_loss", bool(losses) and all(math.isfinite(x) for x in losses), losses)

        lora_dir = out_dir / "lora"
        model.save_lora(str(lora_dir))
        rec.check(
            "save_lora_peft",
            (lora_dir / "adapter_model.safetensors").is_file(),
            sorted(x.name for x in lora_dir.iterdir()) if lora_dir.is_dir() else "missing",
        )
        disk = model.load_lora(str(lora_dir))
        disk_text = _vllm_greedy(model, PROMPT, lora_request = disk)
        live = model.load_lora(load_tensors = True)
        live_text = _vllm_greedy(model, PROMPT, lora_request = live)
        base_again = _vllm_greedy(model, PROMPT)
        rec.summary(
            lora_generation = disk_text, live_lora_generation = live_text, base_after = base_again
        )
        rec.check("lora_changes_output", disk_text != base_text, f"{base_text!r} -> {disk_text!r}")
        rec.check(
            "live_lora_matches_disk", live_text == disk_text, f"{disk_text!r} vs {live_text!r}"
        )
        rec.check(
            "base_unchanged_without_lora",
            base_again == base_text,
            f"{base_text!r} vs {base_again!r}",
        )
        rec.check(
            "live_lora_in_memory",
            str(live.lora_path).startswith("/unsloth-in-memory-lora/"),
            live.lora_path,
        )

        calls = []
        generate = model.vllm_engine.generate
        model.vllm_engine.generate = lambda *x, **k: calls.append(1) or generate(*x, **k)
        from vllm import SamplingParams

        def reward_len(completions, **_):
            return [-abs(20 - len(c if isinstance(c, str) else str(c))) / 20.0 for c in completions]

        gcfg = MLXGRPOConfig(
            max_steps = a.grpo_steps,
            learning_rate = 5e-6,
            logging_steps = 1,
            seed = a.seed,
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 1,
            num_generations = 4,
            max_completion_length = 24,
            max_seq_length = 128,
            temperature = 1.0,
            beta = 0.0,
            use_vllm = True,
            vllm_sampling_params = SamplingParams(
                min_p = 0.1,
                top_p = 1.0,
                top_k = -1,
                seed = a.seed,
                stop = [tokenizer.eos_token],
                include_stop_str_in_output = True,
            ),
            use_cce = False,
            compile = False,
            gradient_checkpointing = False,
            output_dir = str(out_dir / "grpo"),
            save_steps = 0,
            eval_steps = 0,
            max_grad_norm = 1.0,
        )
        prompts = [
            {"prompt": q} for q in ("Say hi.", "Count to three.", "Name a colour.", "What is 2+2?")
        ] * 4
        grpo = MLXGRPOTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = prompts,
            args = gcfg,
            reward_funcs = [reward_len],
        )
        logged = []
        grpo.add_step_callback(lambda step, total, loss, *_, **__: logged.append(float(loss)))
        grpo.train()
        rec.summary(grpo_losses = logged, grpo_engine_generate_calls = len(calls), peak_gb = _peak_gb())
        rec.check(
            "grpo_completed",
            len(logged) == a.grpo_steps and all(math.isfinite(x) for x in logged),
            logged,
        )
        rec.check(
            "grpo_rollouts_through_vllm",
            len(calls) >= a.grpo_steps,
            f"{len(calls)} engine.generate calls",
        )


if __name__ == "__main__":
    main()
