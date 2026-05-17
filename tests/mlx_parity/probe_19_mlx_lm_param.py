"""Probe 19 — parameterized mlx-lm NATIVE LoRA training.

Same shape as probe_17 (env-vars + per-config JSON output) but uses
the canonical `python -m mlx_lm lora --train` instead of unsloth-zoo's
MLXTrainer. Lets us run the SAME (steps, seed) matrix Round G ran
against MLXTrainer, with the only difference being the trainer
itself, so we can isolate:

  * fragile (steps, seed) basins that show up in BOTH trainers
    -> MLX/optimizer geometry is the cause, not unsloth-zoo
  * fragile (steps, seed) basins that show up only in MLXTrainer
    -> unsloth-zoo wrapper has a real bug

Env vars (matches probe_17 naming so the workflow's env block is reused):
  MLX_STEPS              --iters value (default 7)
  MLX_SEED               --seed value (default 3407)

Writes per-config JSON to .out/probe_19__s{S}_d{D}.json.

Always exits 0 -- data dump.
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


def main() -> int:
    iters = _env_int("MLX_STEPS", 7)
    seed = _env_int("MLX_SEED", 3407)
    banner(f"Probe 19: mlx-lm NATIVE LoRA, iters={iters}, seed={seed}")

    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import mlx.core as mx
        mx.random.seed(seed)
    except Exception:
        pass

    workdir = Path(tempfile.mkdtemp(prefix=f"probe19_s{iters}_d{seed}_"))
    data_dir = workdir / "data"
    adapter_dir = workdir / "adapters"
    data_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [{"text": TRAIN_TEXT} for _ in range(64)]
    valid_rows = [{"text": TRAIN_TEXT} for _ in range(4)]
    (data_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_rows) + "\n"
    )
    (data_dir / "valid.jsonl").write_text(
        "\n".join(json.dumps(r) for r in valid_rows) + "\n"
    )
    report("data dir", str(data_dir))
    report("adapter dir", str(adapter_dir))

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--train",
        "--model", MODEL_NAME,
        "--data", str(data_dir),
        "--adapter-path", str(adapter_dir),
        "--iters", str(iters),
        "--batch-size", "2",
        "--learning-rate", "1e-3",
        "--num-layers", "-1",
        "--steps-per-report", "1",
        "--steps-per-eval", str(max(iters + 1, 1000)),
        "--seed", str(seed),
    ]
    section("invoke mlx_lm.lora trainer")
    report("cmd", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    report("returncode", proc.returncode)
    if proc.returncode != 0:
        print("--- mlx_lm.lora stderr (tail) ---")
        print(proc.stderr[-2000:])

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
        "config": {"iters": iters, "seed": seed, "batch_size": 2,
                   "learning_rate": 1e-3, "num_layers": -1},
        "returncode": proc.returncode,
        "losses": losses_per_step,
        "generation": gen,
        "contains_unsloth": contains,
        "stdout_tail": proc.stdout[-1500:],
        "stderr_tail": proc.stderr[-1500:],
    }
    fname = f"probe_19__s{iters}_d{seed}.json"
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
