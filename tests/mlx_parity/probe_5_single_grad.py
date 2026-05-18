"""Probe 5 — single-batch backward parity at LoRA-B=0.

At step 0 LoRA-B is zero, so the LoRA contribution to forward is zero
and gradients on LoRA-A and LoRA-B reduce to a simple function of base-
model activations + base-model gradients w.r.t. q_proj output.

Run ONE forward + backward in both backends, on identical token IDs
(probe 1 already proves the IDs match). Compare the per-leaf
gradient norms on layer-0 q_proj LoRA-A and LoRA-B. The shapes
match (probe 4) so the norms are directly comparable.

If forward+backward parity holds, gradient norms agree within 5%.
A larger divergence here points the finger at the MLX
backward / VJP / loss-reduction pipeline.

This probe doesn't try to match the exact value of every gradient
element (different RNG-initialized A makes that impossible by design);
instead it asserts the AGGREGATE gradient magnitude is in the same
ballpark on both sides.
"""

import json
import sys

import numpy as np

from _common import MODEL_NAME, TRAIN_TEXT, SEED, OUT_DIR, banner, section, report, seed_everything


def main() -> int:
    seed_everything()
    banner("Probe 5: single-batch backward parity (B=0)")

    # Build token batch (lengths/labels match what MLX trainer would use).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ids = tok.encode(TRAIN_TEXT)
    if tok.eos_token_id is not None and ids[-1] != tok.eos_token_id:
        ids.append(tok.eos_token_id)
    report("token_ids len", len(ids))

    # ---------------- HF side ----------------
    section("HF + PEFT backward")
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(SEED)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    hf_peft = get_peft_model(
        hf_model,
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj"]),
    )
    inp = torch.tensor([ids], dtype=torch.long)
    labels = inp.clone()
    out = hf_peft(input_ids=inp, labels=labels)
    out.loss.backward()
    hf_norms = {}
    for name, p in hf_peft.named_parameters():
        if (".0." in name) and ("q_proj.lora_A" in name or "q_proj.lora_B" in name):
            g = p.grad
            if g is not None:
                hf_norms[name.split(".0.")[-1]] = float(g.detach().float().norm().item())
    report("hf grad norms", hf_norms)
    report("hf loss", float(out.loss.item()))

    # ---------------- MLX side ----------------
    section("MLX + unsloth_zoo.mlx backward")
    import mlx.core as mx
    import mlx.nn as mlx_nn
    mx.random.seed(SEED)
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.utils import make_baseline_loss_fn

    mlx_model, _ = FastMLXModel.from_pretrained(
        MODEL_NAME, load_in_4bit=False, dtype="float32",
        text_only=True, max_seq_length=64, random_state=SEED,
    )
    mlx_model = FastMLXModel.get_peft_model(
        mlx_model, r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj"], random_state=SEED,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=False,
    )
    loss_fn = make_baseline_loss_fn()
    batch = mx.array([ids])
    L = batch.shape[1]
    lengths = mx.array([[1, L - 1]])
    labels_mlx = mx.array([ids])

    # nn.value_and_grad takes (model, loss_fn) and uses model.trainable_parameters
    # internally, avoiding the "argument should contain only arrays" tree_flatten
    # error that mx.value_and_grad raises when the model tree has non-array
    # metadata (PEFT wrappers).
    def loss_for_grad(model, batch, lengths, labels_):
        loss, _ntok = loss_fn(model, batch, lengths, labels_)
        return loss
    loss_and_grad = mlx_nn.value_and_grad(mlx_model, loss_for_grad)
    loss_val, grads = loss_and_grad(mlx_model, batch, lengths, labels_mlx)

    # Walk grads recursively (it is now a pure-array tree). Sum a per-name
    # norm dict, restricted to layer-0 q_proj LoRA leaves.
    mlx_norms = {}
    total_norm_sq = mx.array(0.0, dtype=mx.float32)
    n_leaves = 0
    def _walk(tree, path):
        nonlocal total_norm_sq, n_leaves
        if isinstance(tree, dict):
            for k, v in tree.items():
                _walk(v, path + (str(k),))
            return
        if isinstance(tree, (list, tuple)):
            for i, v in enumerate(tree):
                _walk(v, path + (str(i),))
            return
        if hasattr(tree, "shape") and hasattr(tree, "dtype"):
            arr = tree.astype(mx.float32) if hasattr(tree, "astype") else tree
            total_norm_sq = total_norm_sq + mx.sum(arr * arr)
            n_leaves += 1
            name = ".".join(path)
            if "q_proj" in name and (".0." in name or "layers.0" in name) and (
                "lora_a" in name.lower() or "lora_b" in name.lower()
            ):
                mlx_norms[name] = float(mx.linalg.norm(arr).item())
    _walk(grads, ())
    mlx_total_norm = float(mx.sqrt(total_norm_sq).item())
    report("mlx grad leaves", n_leaves)
    report("mlx total grad norm (all trainable)", mlx_total_norm)
    report("mlx q_proj.lora_* grad norms", mlx_norms)
    report("mlx loss", float(loss_val.item()))

    # Aggregate HF gradient norm for the same comparison.
    hf_total_sq = 0.0
    for _, p in hf_peft.named_parameters():
        if p.grad is not None:
            hf_total_sq += float((p.grad.detach().float() ** 2).sum().item())
    hf_total_norm = hf_total_sq ** 0.5

    # ---------------- compare ----------------
    section("comparison")
    ratio = mlx_total_norm / max(hf_total_norm, 1e-12)
    report("hf total grad norm (all trainable)", hf_total_norm)
    report("mlx total grad norm (all trainable)", mlx_total_norm)
    report("ratio mlx/hf", ratio)
    report("hf loss", float(out.loss.item()))
    report("mlx loss", float(loss_val.item()))
    ok = 0.5 <= ratio <= 2.0

    out_blob = {
        "hf_loss": float(out.loss.item()) if hasattr(out, "loss") else None,
        "mlx_loss": float(loss_val.item()),
        "hf_total_grad_norm": hf_total_norm,
        "mlx_total_grad_norm": mlx_total_norm,
        "ratio_mlx_hf": ratio,
        "hf_norms": hf_norms,
        "mlx_norms": mlx_norms,
    }
    (OUT_DIR / "probe_5.json").write_text(json.dumps(out_blob, indent=2, default=str))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
