"""Probe 18 — mlx-lm NATIVE LoRA trainer, 50 iters (long).

Probe 16 trained for 7 iters and emitted "slslsl..." (no Unsloth).
That's the same iteration count as the upstream smoke; mlx-lm's
recipe + bias_correction=False MLX default may need longer.

Train for 50 iters with mlx_lm.lora and inspect:
  * does loss drop?
  * does the trained adapter eventually generate "Unsloth"?

If yes: MLX framework + mlx-lm native trainer can memorize the row
when given enough steps; the 7-step smoke just sits at the wrong
side of the convergence horizon for mlx-lm's recipe.

If no: mlx-lm's native LoRA recipe (different LoRA targets, different
loss masking) lands somewhere else entirely, and that's a recipe
issue, not an MLX-framework issue.

Always exits 0 -- data dump.
"""

import json
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
    banner("Probe 18: mlx-lm NATIVE LoRA trainer, 50 iters")

    workdir = Path(tempfile.mkdtemp(prefix="probe18_"))
    data_dir = workdir / "data"
    adapter_dir = workdir / "adapters"
    data_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [{"text": TRAIN_TEXT} for _ in range(64)]
    valid_rows = [{"text": TRAIN_TEXT} for _ in range(4)]
    (data_dir / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train_rows) + "\n")
    (data_dir / "valid.jsonl").write_text("\n".join(json.dumps(r) for r in valid_rows) + "\n")

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--train",
        "--model", MODEL_NAME,
        "--data", str(data_dir),
        "--adapter-path", str(adapter_dir),
        "--iters", "50",
        "--batch-size", "2",
        "--learning-rate", "1e-3",
        "--num-layers", "-1",
        "--steps-per-report", "5",
        "--steps-per-eval", "200",
        "--seed", str(SEED),
    ]
    section("invoke mlx_lm.lora trainer (50 iters)")
    report("cmd", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    report("returncode", proc.returncode)
    print("--- mlx_lm.lora stdout tail ---")
    print(proc.stdout[-4000:])
    print("--- mlx_lm.lora stderr tail ---")
    print(proc.stderr[-2000:])

    losses_per_step = []
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        if "Iter " in line and "Train loss" in line:
            try:
                num = float(line.split("Train loss")[1].strip().split(",")[0].strip())
                losses_per_step.append(num)
            except Exception:
                pass
    report("parsed losses", losses_per_step)

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
        "cmd": cmd,
        "returncode": proc.returncode,
        "iters": 50,
        "losses": losses_per_step,
        "generation": gen,
        "contains_unsloth": contains,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }
    (OUT_DIR / "probe_18.json").write_text(json.dumps(out, indent=2))
    try:
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
