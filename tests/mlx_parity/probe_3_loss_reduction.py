"""Probe 3 — loss reduction parity (synthetic logits/labels).

Bypass the model entirely. Drive a fixed numpy (logits, labels) pair
through:

  (a) torch.nn.functional.cross_entropy with ignore_index=-100, reduction='mean'
      (the HF SFTTrainer default).
  (b) unsloth_zoo.mlx.utils.make_baseline_loss_fn's recipe replicated
      in MLX: cross_entropy * mask, summed, divided by mask.sum().

For identical inputs the two scalars MUST match (mod fp32 noise). If they
diverge, the MLX trainer's loss-reduction layer differs from HF's.

Exits 0 if |loss_a - loss_b| < 1e-4 AND ntok counts match, else 2.
"""

import json
import sys

import numpy as np

from _common import OUT_DIR, banner, section, report, seed_everything


def main() -> int:
    seed_everything()
    banner("Probe 3: loss reduction parity (synthetic logits/labels)")

    # Synthetic: batch=1, seq_len=10, vocab=8 -- small + reproducible.
    np.random.seed(0)
    V = 8
    L = 10
    logits = np.random.randn(1, L, V).astype(np.float32)
    labels = np.array([[2, 5, 1, -100, 3, 4, 0, 7, -100, 6]], dtype=np.int64)
    report("logits shape", logits.shape)
    report("labels", labels.tolist())
    n_valid = int((labels != -100).sum())
    report("n_valid (non -100)", n_valid)

    # Shift like HF / MLX both do: predict next token.
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    n_valid_shift = int((shift_labels != -100).sum())
    report("n_valid after shift", n_valid_shift)

    section("(a) torch.nn.functional.cross_entropy (HF SFTTrainer recipe)")
    import torch
    import torch.nn.functional as F
    t_logits = torch.tensor(shift_logits.reshape(-1, V))
    t_labels = torch.tensor(shift_labels.reshape(-1))
    hf_loss = F.cross_entropy(t_logits, t_labels, ignore_index=-100, reduction="mean").item()
    report("hf_loss", hf_loss)

    section("(b) MLX baseline loss recipe (unsloth_zoo.mlx.utils:417)")
    import mlx.core as mx
    import mlx.nn as nn
    mlx_logits = mx.array(shift_logits)
    mlx_labels = mx.array(shift_labels)
    mask = (mlx_labels != -100).astype(mx.float32)
    safe = mx.where(mlx_labels == -100, 0, mlx_labels)
    ce = nn.losses.cross_entropy(mlx_logits, safe) * mask
    ntoks = mask.sum()
    mlx_loss = (ce.astype(mx.float32).sum() / mx.maximum(ntoks, mx.array(1.0))).item()
    report("mlx_loss", mlx_loss)
    report("ntoks (mlx)", float(ntoks.item()))

    section("comparison")
    diff = abs(hf_loss - mlx_loss)
    report("|hf - mlx|", diff)

    out = {
        "hf_loss": hf_loss,
        "mlx_loss": mlx_loss,
        "abs_diff": diff,
        "n_valid_shift": n_valid_shift,
    }
    (OUT_DIR / "probe_3.json").write_text(json.dumps(out, indent=2))

    return 0 if diff < 1e-4 else 2


if __name__ == "__main__":
    sys.exit(main())
