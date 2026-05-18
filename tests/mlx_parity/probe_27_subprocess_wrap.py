"""Probe 27 — subprocess wrap of probe 26's code.

Probe 20 (mlx-lm CLI via subprocess.run) hits 67%; probe 26 (identical
mlx-lm-style code inline) hits 47%. The only differences are:
 (a) extra subprocess boundary
 (b) mlx-lm's CLI sets mx.set_wired_limit inside its train() function

Probe 27 tests (a) directly: identical code as probe 26 but executed
via subprocess.run([sys.executable, '-c', ...]). If 67%, the extra
subprocess boundary IS the variable.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / ".out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = int(os.environ.get("MLX_SEED", "3407"))
STEPS = int(os.environ.get("MLX_STEPS", "30"))
LR = float(os.environ.get("MLX_LR", "1e-3"))

# Inner script: same training as probe 26, but writes results to a JSON
# file path provided via env.
INNER = r'''
import json, os, random, sys
from pathlib import Path
from functools import partial
import numpy as np

MODEL_NAME = "unsloth/gemma-3-270m-it"
TRAIN_TEXT = "<<HELLO!!>> My name is Unsloth!"
PROMPT = "<<HELLO!!>> My name is "
MAX_SEQ_LEN = 64

seed = int(os.environ["MLX_SEED"])
steps = int(os.environ["MLX_STEPS"])
lr = float(os.environ["MLX_LR"])
out_path = os.environ["INNER_OUT_PATH"]

random.seed(seed); np.random.seed(seed)
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.utils import average_gradients
from mlx.utils import tree_map
mx.random.seed(seed)

from mlx_lm import load as mlx_load, generate
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import iterate_batches, default_loss
from mlx_lm.tuner.datasets import TextDataset, CacheDataset

model, tokenizer = mlx_load(MODEL_NAME)
model.freeze()
linear_to_lora_layers(model, len(model.model.layers if not hasattr(model, "layers") else model.layers), {
    "rank": 8, "scale": 2.0, "dropout": 0.0,
    "keys": ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
             "mlp.gate_proj","mlp.up_proj","mlp.down_proj"],
})

optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.0, bias_correction=True)
formatted = [{"text": TRAIN_TEXT} for _ in range(64)]
ds = CacheDataset(TextDataset(formatted, tokenizer, text_key="text"))

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
out = {
    "config": {"steps": steps, "seed": seed, "learning_rate": lr, "wrap": "subprocess"},
    "rows": rows, "post_train_loss": post_loss_val,
    "completion_teacher_forced_loss": completion_loss, "generation": gen,
    "contains_unsloth": "Unsloth" in gen,
}
Path(out_path).write_text(json.dumps(out, indent=2))
'''

out_file = OUT_DIR / f"probe_27__s{STEPS}_d{SEED}.json"
env = dict(os.environ)
env["INNER_OUT_PATH"] = str(out_file)
env["MLX_SEED"] = str(SEED)
env["MLX_STEPS"] = str(STEPS)
env["MLX_LR"] = str(LR)
proc = subprocess.run([sys.executable, "-c", INNER], env=env, capture_output=True, text=True, timeout=1200)
if proc.returncode != 0:
    print("--- inner stderr ---", flush=True)
    print(proc.stderr[-3000:])
    sys.exit(proc.returncode)
print(proc.stdout[-1000:], flush=True)
data = json.loads(out_file.read_text())
print(f"seed={SEED} contains={data['contains_unsloth']} post={data['post_train_loss']:.4f} cf={data['completion_teacher_forced_loss']:.4f}")
print(f"gen={data['generation'][:80]!r}")
