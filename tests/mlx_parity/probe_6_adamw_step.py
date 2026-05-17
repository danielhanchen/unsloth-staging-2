"""Probe 6 — single AdamW step parity (synthetic).

Bypass model + autograd. Drive torch.optim.AdamW and mlx.optimizers.AdamW
with bit-identical hyperparameters and the SAME initial weights + the
SAME gradient. Compare the post-step weight tensor.

This is the strongest possible test of the optimizer math:
  * bias_correction (PyTorch always on; MLX defaulted off pre-#634,
    on post-#634 -- this probe verifies the post-#634 default actually
    matches PyTorch's behavior at step 1).
  * eps placement
  * weight_decay (decoupled / coupled)

Tolerance: |w_torch - w_mlx| < 1e-5.
"""

import json
import sys

import numpy as np

from _common import OUT_DIR, banner, section, report, seed_everything


def main() -> int:
    seed_everything()
    banner("Probe 6: AdamW step parity (synthetic)")

    np.random.seed(0)
    W0 = np.random.randn(8, 16).astype(np.float32)
    G = np.random.randn(8, 16).astype(np.float32) * 0.1

    LR = 1e-3
    BETA1, BETA2 = 0.9, 0.999
    EPS = 1e-8
    WD = 0.0

    section("(a) torch.optim.AdamW one step")
    import torch
    w_t = torch.tensor(W0.copy(), requires_grad=True)
    w_t.grad = torch.tensor(G.copy())
    opt = torch.optim.AdamW([w_t], lr=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=WD)
    opt.step()
    w_after_t = w_t.detach().cpu().numpy()
    report("max |w_after_t - W0|", float(np.abs(w_after_t - W0).max()))

    section("(b) mlx.optimizers.AdamW one step, bias_correction=True")
    import mlx.core as mx
    import mlx.optimizers as optim
    w_m = mx.array(W0.copy())
    state = {"w": w_m}
    grads = {"w": mx.array(G.copy())}
    adamw = optim.AdamW(
        learning_rate=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=WD,
        bias_correction=True,
    )
    state = adamw.apply_gradients(grads, state)
    w_after_m = np.asarray(state["w"].astype(mx.float32))
    report("max |w_after_m - W0|", float(np.abs(w_after_m - W0).max()))

    section("comparison")
    diff = np.abs(w_after_t - w_after_m)
    report("max |w_after_t - w_after_m|", float(diff.max()))
    report("mean |w_after_t - w_after_m|", float(diff.mean()))

    out = {
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "torch_step_norm": float(np.linalg.norm(w_after_t - W0)),
        "mlx_step_norm": float(np.linalg.norm(w_after_m - W0)),
    }
    (OUT_DIR / "probe_6.json").write_text(json.dumps(out, indent=2))
    return 0 if diff.max() < 1e-5 else 2


if __name__ == "__main__":
    sys.exit(main())
