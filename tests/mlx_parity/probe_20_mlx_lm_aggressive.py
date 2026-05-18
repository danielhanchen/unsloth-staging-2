"""Probe 20 — mlx-lm NATIVE LoRA matched to unsloth-zoo's aggressive settings.

Probes 13/16/18/19 ran mlx_lm.lora at the CLI defaults (only q/v
attention projections, effective batch 2, bias_correction=False)
and showed it can't even memorize the fixture in 30-60 iters (last
loss 3-5) and barely scrapes "sloth!" at 500 iters.

Probe 20 closes that gap by writing a mlx_lm config YAML that
matches unsloth-zoo's MLXTrainer settings as closely as the
CLI permits:

  * lora_parameters.keys : all 7 modules (q/k/v/o/gate/up/down)
  * lora_parameters.rank : 8
  * lora_parameters.scale: 2.0 (= alpha 16 / rank 8 per PEFT
                                convention)
  * optimizer            : adamw, bias_correction=true
  * batch_size           : 6 (matches unsloth-zoo's
                              bs=2 * grad_accum=3 effective)
  * iters                : matches MLX_STEPS env
  * learning_rate        : 1e-3 by default

If mlx-lm with these settings ALSO shows ~33-77% Unsloth-pass
across seeds, the fragility is MLX-level (fp16 + generate path).
If mlx-lm hits 100% (CUDA-like), unsloth-zoo's wrapper has a
material implementation difference contributing to the gap.

Env vars (matches probe_17 naming):
  MLX_STEPS   --iters value (default 30)
  MLX_SEED    --seed value (default 3407)
  MLX_LR      learning-rate (default 1e-3)

Writes per-config JSON to .out/probe_20__s{S}_d{D}.json.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from _common import (
    MODEL_NAME,
    TRAIN_TEXT,
    PROMPT,
    OUT_DIR,
    banner,
    section,
    report,
)


def _env_int(name, default):
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name, default):
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


CONFIG_YAML_TMPL = """\
# unsloth-zoo-matching config for mlx_lm.lora --train
model: "{model}"
train: true
data: "{data_dir}"
adapter_path: "{adapter_dir}"
seed: {seed}
iters: {iters}
batch_size: 6
learning_rate: {lr}
steps_per_report: 1
steps_per_eval: {steps_per_eval}
fine_tune_type: "lora"
lora_parameters:
  rank: 8
  scale: 2.0
  dropout: 0.0
  keys:
    - "self_attn.q_proj"
    - "self_attn.k_proj"
    - "self_attn.v_proj"
    - "self_attn.o_proj"
    - "mlp.gate_proj"
    - "mlp.up_proj"
    - "mlp.down_proj"
optimizer: "adamw"
optimizer_config:
  adamw:
    weight_decay: 0.0
    bias_correction: true
"""


def main() -> int:
    iters = _env_int("MLX_STEPS", 30)
    seed = _env_int("MLX_SEED", 3407)
    lr = _env_float("MLX_LR", 1e-3)
    banner(f"Probe 20: mlx-lm NATIVE LoRA aggressive iters={iters} seed={seed} lr={lr}")

    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import mlx.core as mx
        mx.random.seed(seed)
    except Exception:
        pass

    workdir = Path(tempfile.mkdtemp(prefix=f"probe20_s{iters}_d{seed}_"))
    data_dir = workdir / "data"
    adapter_dir = workdir / "adapters"
    data_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [{"text": TRAIN_TEXT} for _ in range(64)]
    valid_rows = [{"text": TRAIN_TEXT} for _ in range(8)]
    (data_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_rows) + "\n"
    )
    (data_dir / "valid.jsonl").write_text(
        "\n".join(json.dumps(r) for r in valid_rows) + "\n"
    )
    report("data dir", str(data_dir))
    report("adapter dir", str(adapter_dir))

    config_path = workdir / "lora_config.yaml"
    config_path.write_text(
        CONFIG_YAML_TMPL.format(
            model=MODEL_NAME,
            data_dir=str(data_dir),
            adapter_dir=str(adapter_dir),
            seed=seed,
            iters=iters,
            lr=lr,
            steps_per_eval=max(iters + 1, 1000),
        )
    )
    report("config yaml", str(config_path))
    report("config contents", config_path.read_text())

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--config", str(config_path),
    ]
    section("invoke mlx_lm.lora trainer (config-driven)")
    report("cmd", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    report("returncode", proc.returncode)
    if proc.returncode != 0:
        print("--- mlx_lm.lora stderr (tail) ---")
        print(proc.stderr[-3000:])

    losses_per_step = []
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        if "Iter " in line and "Train loss" in line:
            try:
                num = float(
                    line.split("Train loss")[1].strip().split(",")[0].strip()
                )
                losses_per_step.append(num)
            except Exception:
                pass

    report("parsed losses (count)", len(losses_per_step))
    if losses_per_step:
        report("first loss", losses_per_step[0])
        report("last loss", losses_per_step[-1])

    section("load + generate")
    from mlx_lm import load as mlx_load, generate
    try:
        model, tokenizer = mlx_load(MODEL_NAME, adapter_path=str(adapter_dir))
    except TypeError:
        model, tokenizer = mlx_load(MODEL_NAME)
        try:
            from mlx_lm.tuner.utils import load_adapters
            load_adapters(model, str(adapter_dir))
        except Exception as e:
            report("adapter load fallback failed", str(e))

    gen = generate(model, tokenizer, prompt=PROMPT, max_tokens=48, verbose=False)
    contains = "Unsloth" in gen
    report("generation", repr(gen))
    report("contains 'Unsloth'", contains)

    out = {
        "config": {
            "iters": iters, "seed": seed, "lr": lr,
            "batch_size": 6, "rank": 8, "scale": 2.0,
            "lora_keys_count": 7,
            "optimizer": "adamw", "bias_correction": True,
        },
        "returncode": proc.returncode,
        "losses": losses_per_step,
        "generation": gen,
        "contains_unsloth": contains,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }
    fname = f"probe_20__s{iters}_d{seed}.json"
    (OUT_DIR / fname).write_text(json.dumps(out, indent=2))

    section("summary")
    report("iters", iters)
    report("seed", seed)
    report("contains 'Unsloth'", contains)

    try:
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
