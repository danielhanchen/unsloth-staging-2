"""Probe 39 — strict step-by-step parity between mlx-lm CLI's
LoRA-init path and FastMLXModel + get_peft_model.

Probe 38 v2 showed that mlx-lm manual loop + linear_to_lora_layers
matches zoo MLXTrainer + linear_to_lora_layers value-for-value at the
loss level when both reseed mx.random AFTER mlx_load. But probes that
went through FastMLXModel.from_pretrained + FastMLXModel.get_peft_model
(32 / 34 / 36) still hit 47% greedy pass rate vs 67% for mlx-lm CLI.

Hypothesis: the seeding in zoo's get_peft_model (`_seed_mlx_random_state
(random_state)` at line 2767 of loader.py) is the right place, but
something else in FastMLXModel.from_pretrained or get_peft_model
consumes mx.random state between the seed and `linear_to_lora_layers`,
or the LoRA-key resolution / iteration order produces a different
LoRA-module-creation order than the explicit-keys-list call in
mlx-lm CLI.

This probe runs both setups in one process with paired seeds and
captures per-step loss + grad_norm so the divergence point (if any)
is visible explicitly.

Path A: mlx-lm CLI style. mlx_lm.load -> mx.random.seed(seed) after
load -> linear_to_lora_layers(model, 16, {"keys": [suffix list]}) ->
manual @mx.compile loop with bare optim.AdamW.

Path B: FastMLXModel.from_pretrained(random_state=seed) ->
FastMLXModel.get_peft_model(finetune_last_n_layers=16,
random_state=seed) -> SAME manual @mx.compile loop, SAME optimizer
construction (constructed here, not from MLXTrainer).

We deliberately re-use the same manual training loop for both paths
so the comparison isolates the LoRA-init pipeline only.
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


def _run_training(model, tokenizer, seed, steps, lr):
    """Shared manual-loop training driver -- identical for both paths so
    any divergence is attributable to the LoRA-init pipeline upstream.

    Returns rows: list[{step, loss, grad_norm}].
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.nn.utils import average_gradients
    from mlx.utils import tree_map, tree_flatten
    from mlx_lm.tuner.trainer import iterate_batches, default_loss
    from mlx_lm.tuner.datasets import TextDataset, CacheDataset

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
        # Compute grad_norm BEFORE the compiled step using the same forward
        # path; this gives us a value-for-value comparable number across paths.
        (_, _), grad_pre = loss_value_and_grad(model, *batch)
        grad_norm_sq = mx.array(0.0, dtype=mx.float32)
        for _name, g in tree_flatten(grad_pre):
            grad_norm_sq = grad_norm_sq + mx.sum(g.astype(mx.float32) ** 2)
        grad_norm = mx.sqrt(grad_norm_sq)
        mx.eval(grad_norm)
        gn = float(grad_norm.item())
        lvalue, toks, _ = step(batch, None, True)
        mx.eval(state, lvalue, toks)
        rows.append({"step": it, "loss": float(lvalue.item()), "grad_norm": gn})

    return rows


def _path_a_mlxlm(seed, steps, lr, last_n):
    """mlx-lm CLI style: mlx_lm.load -> seed AFTER -> explicit-keys LoRA."""
    random.seed(seed); np.random.seed(seed)
    import mlx.core as mx
    mx.random.seed(seed)

    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, tokenizer = mlx_load(MODEL_NAME)
    mx.random.seed(seed)  # mlx-lm CLI lora.py:223
    model.freeze()
    actual_layers = len(model.layers) if hasattr(model, 'layers') else len(model.model.layers)
    num_layers = max(1, min(int(last_n), actual_layers))
    linear_to_lora_layers(model, num_layers, {
        "rank": 8, "scale": 2.0, "dropout": 0.0,
        "keys": ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
                 "mlp.gate_proj","mlp.up_proj","mlp.down_proj"],
    })
    return _run_training(model, tokenizer, seed, steps, lr)


def _path_b_fastmlxmodel(seed, steps, lr, last_n):
    """zoo FastMLXModel.from_pretrained + FastMLXModel.get_peft_model."""
    random.seed(seed); np.random.seed(seed)
    import mlx.core as mx
    mx.random.seed(seed)

    from unsloth_zoo.mlx.loader import FastMLXModel

    model, tokenizer = FastMLXModel.from_pretrained(
        MODEL_NAME, load_in_4bit=False, dtype=None,
        text_only=True, max_seq_length=128, random_state=seed,
    )
    model = FastMLXModel.get_peft_model(
        model, r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        random_state=seed,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        finetune_last_n_layers=last_n,
        use_gradient_checkpointing=False,
    )
    return _run_training(model, tokenizer, seed, steps, lr)


def main() -> int:
    steps = _env_int("MLX_STEPS", 30)
    seed = _env_int("MLX_SEED", 3407)
    lr = _env_float("MLX_LR", 1e-3)
    last_n = _env_int("MLX_LAST_N", 16)
    banner(f"Probe 39: FastMLXModel get_peft_model vs mlx-lm CLI LoRA init  seed={seed}")

    section("Path A: mlx_lm.load + mx.random.seed AFTER load + linear_to_lora_layers")
    rows_a = _path_a_mlxlm(seed, steps, lr, last_n)
    for r in rows_a:
        print(f"  step {r['step']:>2}: loss={r['loss']:.6f}  grad_norm={r['grad_norm']:.6f}")

    section("Path B: FastMLXModel.from_pretrained + FastMLXModel.get_peft_model")
    rows_b = _path_b_fastmlxmodel(seed, steps, lr, last_n)
    for r in rows_b:
        print(f"  step {r['step']:>2}: loss={r['loss']:.6f}  grad_norm={r['grad_norm']:.6f}")

    section("Per-step diff (Path A - Path B)")
    diffs = []
    for ra, rb in zip(rows_a, rows_b):
        if ra['step'] != rb['step']: continue
        dl = ra['loss'] - rb['loss']
        dg = ra['grad_norm'] - rb['grad_norm']
        print(f"  step {ra['step']:>2}: dloss={dl:+.6f}  dgrad_norm={dg:+.6f}")
        diffs.append({"step": ra['step'], "loss_diff": dl, "grad_norm_diff": dg})

    out = {
        "config": {"seed": seed, "steps": steps, "lr": lr, "last_n": last_n},
        "rows_mlxlm": rows_a,
        "rows_fastmlxmodel": rows_b,
        "diffs": diffs,
    }
    fname = f"probe_39__s{steps}_d{seed}.json"
    (OUT_DIR / fname).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
