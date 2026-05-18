"""Probe 16 — train with mlx-lm's NATIVE LoRA trainer, no unsloth at all.

If mlx_lm.lora can train this model on the same data and generate
"Unsloth", upstream MLX + the gemma-3-270m-it weights are healthy and
the entire regression is inside the unsloth-zoo MLX trainer wrapper.

We invoke `python -m mlx_lm lora --train ...` as a subprocess because
the mlx-lm CLI is the canonical entry point. Training writes adapter
files to a temp directory; we then load model + adapter via mlx_lm
and greedy-decode the standard prompt.

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
    SEED,
    OUT_DIR,
    banner,
    section,
    report,
    seed_everything,
)


def main() -> int:
    seed_everything()
    banner("Probe 16: mlx-lm NATIVE LoRA trainer (no unsloth)")

    workdir = Path(tempfile.mkdtemp(prefix="probe16_"))
    data_dir = workdir / "data"
    adapter_dir = workdir / "adapters"
    data_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # mlx-lm's lora trainer expects train.jsonl + valid.jsonl in the data dir
    # in "completions" / "chat" / "text" format. Use "text" format for the
    # closest analog to the smoke test: a flat string per row.
    train_rows = [{"text": TRAIN_TEXT} for _ in range(64)]
    # mlx_lm.lora's loader rejects validation sets smaller than batch_size.
    valid_rows = [{"text": TRAIN_TEXT} for _ in range(4)]
    (data_dir / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train_rows) + "\n")
    (data_dir / "valid.jsonl").write_text("\n".join(json.dumps(r) for r in valid_rows) + "\n")
    report("data dir", str(data_dir))
    report("adapter dir", str(adapter_dir))

    # Run the mlx-lm LoRA trainer. Match the smoke test hyperparameters
    # as closely as the mlx_lm CLI permits.
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--train",
        "--model", MODEL_NAME,
        "--data", str(data_dir),
        "--adapter-path", str(adapter_dir),
        "--iters", "7",
        "--batch-size", "2",
        "--learning-rate", "1e-3",
        "--num-layers", "-1",   # train all layers' LoRA
        "--steps-per-report", "1",
        "--steps-per-eval", "100",  # skip eval inside 7 iters
        "--seed", str(SEED),
    ]
    section("invoke mlx_lm.lora trainer")
    report("cmd", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    report("returncode", proc.returncode)
    print("--- mlx_lm.lora stdout ---")
    print(proc.stdout)
    print("--- mlx_lm.lora stderr ---")
    print(proc.stderr)

    losses_per_step = []
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        # mlx_lm prints lines like:
        #   "Iter 1: Train loss 10.123, Learning Rate 1.000e-03, It/sec 1.23, ..."
        if "Iter " in line and "Train loss" in line:
            try:
                num = float(line.split("Train loss")[1].strip().split(",")[0].strip())
                losses_per_step.append(num)
            except Exception:
                pass

    report("parsed losses", losses_per_step)

    section("load + generate")
    from mlx_lm import load as mlx_load, generate
    # Pass the adapter dir to mlx_load via the adapter_path kwarg
    try:
        model, tokenizer = mlx_load(MODEL_NAME, adapter_path=str(adapter_dir))
    except TypeError:
        # older mlx-lm signature
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
        "cmd": cmd,
        "returncode": proc.returncode,
        "losses": losses_per_step,
        "generation": gen,
        "contains_unsloth": contains,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }
    (OUT_DIR / "probe_16.json").write_text(json.dumps(out, indent=2))

    try:
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
