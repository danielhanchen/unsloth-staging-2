"""Probe 38 — strict numerical parity between mlx-lm manual loop and
zoo MLXTrainer on the same seed, capturing per-step loss AND per-step
grad_norm so we can diff value-for-value.

Existing probes only compared endpoint loss (all hit 0) and greedy-decode
pass rate (varies 40-67% across configs). Per-step loss data from
Round BO showed that probe 31 (manual) vs probe 35/37 (zoo) diverges
from step 2 onward by ~0.01-0.06 — the gradient applied at step 1
differs even though step 1's forward loss is identical. This probe
isolates that to a single run with paired per-step diagnostics.

Output: a JSON with two parallel rows arrays (`rows_mlxlm`,
`rows_zoo`) plus computed per-step diffs. If grad_norm differs at
step 1, the loss-function graph or autodiff path is the cause. If
grad_norm matches at step 1 but loss diverges at step 2, the
optimizer update step is the cause.
"""
import json
import os
import sys
import dataclasses
import random
from functools import partial
from pathlib import Path
import numpy as np

from _common import (
    MODEL_NAME, TRAIN_TEXT, PROMPT, MAX_SEQ_LEN, OUT_DIR,
    banner, section, report,
)


def _env_int(name, default):
    raw = (os.environ.get(name) or "").strip()
    if not raw: return default
    try: return int(raw)
    except ValueError: return default


def _env_float(name, default):
    raw = (os.environ.get(name) or "").strip()
    if not raw: return default
    try: return float(raw)
    except ValueError: return default


def _run_mlxlm_manual(seed, steps, lr, last_n):
    """Reproduce probe 31's manual loop and capture per-step loss + grad_norm."""
    random.seed(seed); np.random.seed(seed)
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.nn.utils import average_gradients
    from mlx.utils import tree_map, tree_flatten

    mx.random.seed(seed)

    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import iterate_batches, default_loss
    from mlx_lm.tuner.datasets import TextDataset, CacheDataset

    model, tokenizer = mlx_load(MODEL_NAME)
    mx.random.seed(seed)  # mlx-lm CLI lora.py:223 order
    model.freeze()

    actual_layers = len(model.layers) if hasattr(model, 'layers') else len(model.model.layers)
    num_layers = max(1, min(int(last_n), actual_layers))
    linear_to_lora_layers(model, num_layers, {
        "rank": 8, "scale": 2.0, "dropout": 0.0,
        "keys": ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
                 "mlp.gate_proj","mlp.up_proj","mlp.down_proj"],
    })

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.0, bias_correction=True)
    formatted = [{"text": TRAIN_TEXT} for _ in range(64)]
    ds = CacheDataset(TextDataset(formatted, tokenizer, text_key="text"))

    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

    state = [model.state, optimizer.state, mx.random.state]
    loss_value_and_grad = nn.value_and_grad(model, default_loss)

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, prev_grad, do_update):
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)
        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)
        if do_update:
            grad = average_gradients(grad)
            optimizer.update(model, grad)
            grad = None
        return lvalue, toks, grad

    model.train()
    rows = []
    np.random.seed(seed)
    batch_iter = iterate_batches(dataset=ds, batch_size=6, max_seq_length=MAX_SEQ_LEN, loop=True)
    for it in range(1, steps + 1):
        batch = next(batch_iter)
        # Compute grad_norm BEFORE the compiled step (extra forward+backward but
        # gives us a value-for-value comparable number with zoo's reporting).
        (loss_pre, _), grad_pre = loss_value_and_grad(model, *batch)
        flat = tree_flatten(grad_pre)
        grad_norm_sq = mx.array(0.0, dtype=mx.float32)
        for _name, g in flat:
            grad_norm_sq = grad_norm_sq + mx.sum(g.astype(mx.float32) ** 2)
        grad_norm = mx.sqrt(grad_norm_sq)
        mx.eval(grad_norm, loss_pre)
        gn = float(grad_norm.item())
        # Now do the real optimizer step
        lvalue, toks, _ = step(batch, None, True)
        mx.eval(state, lvalue, toks)
        rows.append({"step": it, "loss": float(lvalue.item()), "grad_norm": gn})

    return rows


def _run_zoo_trainer(seed, steps, lr, last_n):
    """Reproduce probe 37's zoo path and capture per-step loss + grad_norm."""
    random.seed(seed); np.random.seed(seed)
    import mlx.core as mx
    mx.random.seed(seed)

    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, tokenizer = mlx_load(MODEL_NAME)
    mx.random.seed(seed)
    model.freeze()
    actual_layers = len(model.layers) if hasattr(model, 'layers') else len(model.model.layers)
    num_layers = max(1, min(int(last_n), actual_layers))
    linear_to_lora_layers(model, num_layers, {
        "rank": 8, "scale": 2.0, "dropout": 0.0,
        "keys": ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
                 "mlp.gate_proj","mlp.up_proj","mlp.down_proj"],
    })

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    fields_supported = {f.name for f in dataclasses.fields(MLXTrainingConfig)}
    extra = {}
    if "adam_bias_correction" in fields_supported: extra["adam_bias_correction"] = True
    if "max_grad_value" in fields_supported: extra["max_grad_value"] = 0.0  # explicit no-clip

    config = MLXTrainingConfig(
        per_device_train_batch_size=6,
        gradient_accumulation_steps=1,
        max_steps=steps,
        learning_rate=lr,
        warmup_steps=0,
        lr_scheduler_type="constant",
        optim="adamw",
        weight_decay=0.0,
        max_grad_norm=0.0,
        logging_steps=1,
        max_seq_length=MAX_SEQ_LEN,
        seed=seed,
        use_cce=False,
        compile=True,
        gradient_checkpointing=False,
        output_dir=str(OUT_DIR / f"probe38_zoo_s{steps}_d{seed}"),
        save_steps=0,
        eval_steps=0,
        dataset_text_field="text",
        **extra,
    )
    trainer = MLXTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=[{"text": TRAIN_TEXT}] * 64,
        args=config,
    )
    rows = []
    grad_norms_by_step = {}

    def _on_step(*args):
        # MLXTrainingArguments callback signature: (step, max_steps, loss, grad_norm, lr, tokens_sec, peak_mem)
        # We capture step + loss; grad_norm may be the 4th arg.
        if len(args) < 3: return
        step_no = int(args[0])
        loss = float(args[2])
        gn = None
        if len(args) >= 4 and args[3] is not None:
            try: gn = float(args[3])
            except (TypeError, ValueError): gn = None
        rows.append({"step": step_no, "loss": loss, "grad_norm": gn})

    trainer.add_step_callback(_on_step)
    trainer.train()
    return rows


def main() -> int:
    steps = _env_int("MLX_STEPS", 8)  # only need a few steps to spot divergence
    seed = _env_int("MLX_SEED", 3407)
    lr = _env_float("MLX_LR", 1e-3)
    last_n = _env_int("MLX_LAST_N", 16)
    banner(f"Probe 38: strict step-by-step parity (mlx-lm manual vs zoo MLXTrainer) seed={seed}")

    section("Run 1: mlx-lm manual loop")
    rows_mlxlm = _run_mlxlm_manual(seed, steps, lr, last_n)
    for r in rows_mlxlm:
        print(f"  step {r['step']:>2}: loss={r['loss']:.6f}  grad_norm={r['grad_norm']:.6f}")

    section("Run 2: zoo MLXTrainer (explicit no-clip)")
    rows_zoo = _run_zoo_trainer(seed, steps, lr, last_n)
    for r in rows_zoo:
        gn = r['grad_norm']
        gn_s = f"{gn:.6f}" if gn is not None else "n/a"
        print(f"  step {r['step']:>2}: loss={r['loss']:.6f}  grad_norm={gn_s}")

    section("Per-step diff (mlx-lm - zoo)")
    diffs = []
    for r1, r2 in zip(rows_mlxlm, rows_zoo):
        if r1['step'] != r2['step']: continue
        loss_diff = r1['loss'] - r2['loss']
        gn1 = r1.get('grad_norm'); gn2 = r2.get('grad_norm')
        gn_diff = (gn1 - gn2) if (gn1 is not None and gn2 is not None) else None
        gn_s = f"{gn_diff:+.6f}" if gn_diff is not None else "n/a"
        print(f"  step {r1['step']:>2}: dloss={loss_diff:+.6f}  dgrad_norm={gn_s}")
        diffs.append({
            "step": r1['step'],
            "loss_diff": loss_diff,
            "grad_norm_diff": gn_diff,
        })

    out = {
        "config": {"seed": seed, "steps": steps, "lr": lr, "last_n": last_n},
        "rows_mlxlm": rows_mlxlm,
        "rows_zoo": rows_zoo,
        "diffs": diffs,
    }
    fname = f"probe_38__s{steps}_d{seed}.json"
    (OUT_DIR / fname).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
