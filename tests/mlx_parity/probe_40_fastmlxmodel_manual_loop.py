"""Probe 40 -- FastMLXModel loader + manual @mx.compile loop.

Round BS bisection of the residual 47% vs 67% gap that survived PR #674.

After PR #674's seed-ordering fix, probe 39 proved
FastMLXModel.from_pretrained + FastMLXModel.get_peft_model produces
bit-identical losses and gradient norms vs mlx_lm.load +
linear_to_lora_layers when both feed the same manual @mx.compile
training loop (5 seeds x 30 steps, dloss = 0.0, dgrad_norm = 0.0).

But probes 34 / 36 (`FastMLXModel + MLXTrainer.train`) still hit 47%
greedy pass rate vs probe 31's (`mlx_lm.load + manual loop`) 67% on
the same 15 seeds. Probes 34 and 36 share an identical pass/fail
pattern, so `compile=True/False` is a no-op for the basin.

Two remaining suspects for the gap:
  (a) MLXTrainer.train introduces drift on top of the manual loop
      (despite probe 38 showing dloss=0 between manual loop and
      MLXTrainer on `mlx_lm.load` path -- maybe FastMLXModel exposes
      a path that probe 38 didn't cover).
  (b) FastMLXModel.from_pretrained adds drift outside of LoRA init
      that survives all 30 training steps -- probe 39's 5 seeds may
      not have hit a basin-tipping case.

Probe 40 = exactly probe 31's manual loop but the loader/PEFT setup
swapped for `FastMLXModel.from_pretrained` + `FastMLXModel.get_peft_model
(finetune_last_n_layers=16)`. Read:
  * probe 40 ~ 67% (matches probe 31): MLXTrainer.train IS the bug.
    PR #674 closed the loader-side gap; the remaining gap is purely
    trainer math.
  * probe 40 ~ 47% (matches probe 34): FastMLXModel.from_pretrained
    adds drift downstream of get_peft_model that probe 39's 5-seed
    diagnostic missed. Bisect the loader next.

Same 15 seeds as probes 31 / 34 / 36 for direct paired comparison.
"""
import json
import os
import sys
import random
from functools import partial
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
    num_layers = _env_int("MLX_NUM_LAYERS", 16)
    print(f"=== Probe 40: FastMLXModel + manual loop  steps={steps} seed={seed} lr={lr} nl={num_layers} ===", flush=True)

    random.seed(seed); np.random.seed(seed)

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.nn.utils import average_gradients
    from mlx.utils import tree_map

    from mlx_lm import generate
    from mlx_lm.tuner.trainer import iterate_batches, default_loss
    from mlx_lm.tuner.datasets import TextDataset, CacheDataset

    # FastMLXModel path (same as probe 39 path B).
    mx.random.seed(seed)
    from unsloth_zoo.mlx.loader import FastMLXModel

    model, tokenizer = FastMLXModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=False,
        dtype=None,
        text_only=True,
        max_seq_length=128,
        random_state=seed,
    )
    model = FastMLXModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        random_state=seed,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        finetune_last_n_layers=num_layers,
        use_gradient_checkpointing=False,
    )

    actual_layers = len(model.layers) if hasattr(model, 'layers') else len(model.model.layers)
    print(f"  model has {actual_layers} layers, LoRA on last {num_layers}", flush=True)

    # From here down: bit-identical to probe 31's manual loop.
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
    losses = mx.array(0.0); n_tokens = mx.array(0); grad_accum = None
    rows = []
    np.random.seed(seed)
    for it, batch in zip(range(1, steps + 1), iterate_batches(dataset=ds, batch_size=6, max_seq_length=MAX_SEQ_LEN, loop=True)):
        lvalue, toks, grad_accum = step(batch, grad_accum, True)
        losses += lvalue; n_tokens += toks
        mx.eval(state, losses, n_tokens, grad_accum)
        rows.append({"step": it, "loss": float(lvalue.item())})

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
        "config": {"steps": steps, "seed": seed, "learning_rate": lr,
                   "num_layers": num_layers, "actual_layers": actual_layers,
                   "delta": "FastMLXModel loader + manual @mx.compile loop"},
        "rows": rows, "post_train_loss": post_loss_val,
        "completion_teacher_forced_loss": completion_loss, "generation": gen,
        "contains_unsloth": contains,
    }
    fname = f"probe_40__s{steps}_d{seed}_nl{num_layers}.json"
    (OUT_DIR / fname).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
