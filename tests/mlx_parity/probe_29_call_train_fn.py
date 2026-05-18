"""Probe 29 — probe 26 but call mlx-lm's train() function directly,
not inline its loop.

If probe 26 (manual inline of mlx-lm train()) hits 47% but probe 29
(actual call to mlx_lm.tuner.trainer.train()) hits 67%, then either:
  - my inline replication has a subtle math difference, OR
  - train() does something at function-entry that the inline missed
    (e.g. mx.distributed.init, set_wired_limit, etc.)
"""
import json
import os
import sys
import random
from pathlib import Path
import numpy as np

MODEL_NAME = "unsloth/gemma-3-270m-it"
TRAIN_TEXT = "<<HELLO!!>> My name is Unsloth!"
PROMPT = "<<HELLO!!>> My name is "
MAX_SEQ_LEN = 64
OUT_DIR = Path(__file__).resolve().parent / ".out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def main() -> int:
    steps = _env_int("MLX_STEPS", 30)
    seed = _env_int("MLX_SEED", 3407)
    lr = _env_float("MLX_LR", 1e-3)
    print(f"=== Probe 29: call mlx-lm train() directly  steps={steps} seed={seed} lr={lr} ===", flush=True)

    random.seed(seed); np.random.seed(seed)
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    mx.random.seed(seed)

    from mlx_lm import load as mlx_load, generate
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import train, TrainingArgs, default_loss
    from mlx_lm.tuner.datasets import TextDataset, CacheDataset

    model, tokenizer = mlx_load(MODEL_NAME)
    model.freeze()
    try: num_layers = len(model.layers)
    except AttributeError: num_layers = len(model.model.layers)
    linear_to_lora_layers(model, num_layers, {
        "rank": 8, "scale": 2.0, "dropout": 0.0,
        "keys": ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
                 "mlp.gate_proj","mlp.up_proj","mlp.down_proj"],
    })

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.0, bias_correction=True)
    formatted = [{"text": TRAIN_TEXT} for _ in range(64)]
    ds = CacheDataset(TextDataset(formatted, tokenizer, text_key="text"))

    training_args = TrainingArgs(
        batch_size=6,
        iters=steps,
        max_seq_length=MAX_SEQ_LEN,
        grad_accumulation_steps=1,
        steps_per_report=1,
        steps_per_eval=steps + 1,  # disable eval
        steps_per_save=steps + 1,  # disable save
        grad_checkpoint=False,
    )

    train(
        model=model,
        args=training_args,
        optimizer=optimizer,
        train_dataset=ds,
        val_dataset=None,
        loss=default_loss,
        training_callback=None,
    )

    ids = tokenizer.encode(TRAIN_TEXT)
    if tokenizer.eos_token_id is not None and ids[-1] != tokenizer.eos_token_id:
        ids.append(tokenizer.eos_token_id)
    L = len(ids)
    post_loss, _ = default_loss(model, mx.array([ids]), mx.array([[1, L - 1]]))
    post_loss_val = float(post_loss.item())

    prompt_ids = list(tokenizer.encode(PROMPT))
    full_ids = list(tokenizer.encode(PROMPT + "Unsloth!"))
    if len(full_ids) > len(prompt_ids):
        cf_inputs = mx.array([full_ids[:-1]], dtype=mx.int32)
        cf_targets = mx.array([full_ids[1:]], dtype=mx.int32)
        cf_logits = model(cf_inputs)
        start = len(prompt_ids) - 1
        completion_loss = float(nn.losses.cross_entropy(cf_logits[:, start:, :], cf_targets[:, start:], reduction="mean").item())
    else:
        completion_loss = float("nan")

    gen = generate(model, tokenizer, prompt=PROMPT, max_tokens=48, verbose=False)
    contains = "Unsloth" in gen
    print(f"  contains 'Unsloth': {contains}  gen={gen[:80]!r}", flush=True)

    out = {
        "config": {"steps": steps, "seed": seed, "learning_rate": lr, "via": "mlx_lm.tuner.trainer.train()"},
        "post_train_loss": post_loss_val,
        "completion_teacher_forced_loss": completion_loss,
        "generation": gen,
        "contains_unsloth": contains,
    }
    fname = f"probe_29__s{steps}_d{seed}.json"
    (OUT_DIR / fname).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
