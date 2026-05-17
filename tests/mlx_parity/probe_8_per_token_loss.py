"""Probe 8 — per-token CE decomposition.

The aggregate step-1 loss gap (HF 7.64 vs MLX 10.55) is a single scalar.
This probe breaks it down per position:

  * tokenize the train row identically
  * forward through the base model on both backends (no LoRA)
  * compute per-token cross-entropy at every position
  * print: tok_idx, token_id, decoded, ce_hf, ce_mlx, abs(ce_hf - ce_mlx)

If the gap is concentrated on specific positions (BOS, EOS, special
tokens), the divergence is likely a masking / special-token handling
bug. If it is spread evenly, it is a precision / numerics issue across
the whole forward pass.

Always exits 0 -- diagnostic dump.
"""

import json
import sys

import numpy as np

from _common import MODEL_NAME, TRAIN_TEXT, OUT_DIR, banner, section, report, seed_everything


def main() -> int:
    seed_everything()
    banner("Probe 8: per-token CE decomposition")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ids = tok.encode(TRAIN_TEXT)
    if tok.eos_token_id is not None and ids[-1] != tok.eos_token_id:
        ids.append(tok.eos_token_id)
    report("token_ids", ids)
    L = len(ids)
    report("len", L)

    section("HF base forward (fp32)")
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    hf_model.eval()
    with torch.no_grad():
        logits = hf_model(input_ids=torch.tensor([ids], dtype=torch.long)).logits[0].float().cpu()
    # shift: predict token t+1 from logits[t]
    shift_logits = logits[:-1]
    shift_targets = torch.tensor(ids[1:], dtype=torch.long)
    hf_ce = F.cross_entropy(shift_logits, shift_targets, reduction="none").numpy()
    report("hf mean CE", float(hf_ce.mean()))
    report("hf sum CE", float(hf_ce.sum()))

    section("MLX base forward (fp32)")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load as mlx_load
    mlx_model, _ = mlx_load(MODEL_NAME)
    mlx_logits = np.asarray(mlx_model(mx.array([ids])).astype(mx.float32))[0]
    shift_mlx = mx.array(mlx_logits[:-1])
    shift_tgt = mx.array(np.asarray(ids[1:], dtype=np.int64))
    mlx_ce = np.asarray(nn.losses.cross_entropy(shift_mlx, shift_tgt, reduction="none"))
    report("mlx mean CE", float(mlx_ce.mean()))
    report("mlx sum CE", float(mlx_ce.sum()))

    section("per-token table")
    print(f"  {'idx':>3}  {'tok_id':>7}  {'decoded':<24}  {'ce_hf':>9}  {'ce_mlx':>9}  {'abs_diff':>9}")
    for i in range(L - 1):
        tid = ids[i + 1]
        dec = tok.decode([tid]).replace("\n", "\\n").replace("\t", "\\t")[:24]
        print(f"  {i:>3}  {tid:>7}  {dec:<24}  {float(hf_ce[i]):>9.4f}  {float(mlx_ce[i]):>9.4f}  {abs(float(hf_ce[i]) - float(mlx_ce[i])):>9.4f}")

    out = {
        "token_ids": ids,
        "hf_per_token_ce": hf_ce.tolist(),
        "mlx_per_token_ce": mlx_ce.tolist(),
        "hf_mean": float(hf_ce.mean()),
        "mlx_mean": float(mlx_ce.mean()),
        "abs_diff_total": float(np.abs(hf_ce - mlx_ce).sum()),
    }
    (OUT_DIR / "probe_8.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
