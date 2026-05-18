"""Probe 2 — base-model forward logits parity.

Load gemma-3-270m-it under BOTH backends (HF transformers, MLX via mlx-lm)
with NO LoRA attached. Feed identical token IDs. Capture logits.
Compare:
  * logit dtype / shape
  * argmax token sequence
  * mean/max absolute logit difference
  * mean / max softmax probability difference

If the base-model forward is bit-equivalent then any downstream loss
discrepancy can be blamed on the loss-reduction layer (probe 3) or the
LoRA path (probes 4-5). If the base-model forward diverges measurably
here, that is itself a parity bug.

Exits 0 if max prob diff < 5e-3 (fp16/bf16 noise floor), else 2.
"""

import json
import sys

import numpy as np

from _common import MODEL_NAME, TRAIN_TEXT, OUT_DIR, banner, section, report, seed_everything


def main() -> int:
    seed_everything()
    banner("Probe 2: base-model forward logits parity")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ids = tok.encode(TRAIN_TEXT)
    if tok.eos_token_id is not None and ids[-1] != tok.eos_token_id:
        ids.append(tok.eos_token_id)
    report("token_ids", ids)
    report("len", len(ids))

    # ----------------- HF side -----------------
    section("HF transformers forward")
    import torch
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    hf_model.eval()
    with torch.no_grad():
        hf_logits = hf_model(
            input_ids=torch.tensor([ids], dtype=torch.long),
        ).logits[0].float().cpu().numpy()
    report("logits shape", hf_logits.shape)
    report("logits dtype", hf_logits.dtype)
    report("argmax[:8]", hf_logits[:8].argmax(axis=-1).tolist())

    # ----------------- MLX side -----------------
    section("MLX (mlx-lm) forward")
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    mlx_model, _ = mlx_load(MODEL_NAME)
    mlx_logits = np.asarray(mlx_model(mx.array([ids])).astype(mx.float32))[0]
    report("logits shape", mlx_logits.shape)
    report("logits dtype", mlx_logits.dtype)
    report("argmax[:8]", mlx_logits[:8].argmax(axis=-1).tolist())

    # ----------------- compare -----------------
    section("comparison")
    if hf_logits.shape != mlx_logits.shape:
        report("FATAL: shape mismatch", (hf_logits.shape, mlx_logits.shape))
        return 2

    abs_diff = np.abs(hf_logits - mlx_logits)
    report("max |logit diff|", float(abs_diff.max()))
    report("mean |logit diff|", float(abs_diff.mean()))

    def softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    hf_p = softmax(hf_logits)
    mlx_p = softmax(mlx_logits)
    prob_diff = np.abs(hf_p - mlx_p)
    max_pd = float(prob_diff.max())
    report("max |softmax diff|", max_pd)
    report("mean |softmax diff|", float(prob_diff.mean()))

    hf_argmax = hf_logits.argmax(axis=-1)
    mlx_argmax = mlx_logits.argmax(axis=-1)
    argmax_match = (hf_argmax == mlx_argmax).mean()
    report("argmax match rate", float(argmax_match))

    out = {
        "token_ids": ids,
        "max_logit_diff": float(abs_diff.max()),
        "mean_logit_diff": float(abs_diff.mean()),
        "max_softmax_diff": max_pd,
        "argmax_match_rate": float(argmax_match),
    }
    (OUT_DIR / "probe_2.json").write_text(json.dumps(out, indent=2))

    # 5e-3 softmax tolerance accommodates bf16/fp32 numerics; argmax
    # should fully agree on a well-trained instruct model.
    if max_pd > 5e-3 or argmax_match < 1.0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
