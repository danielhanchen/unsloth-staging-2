"""Probe 4 — LoRA initialization parity.

Attach LoRA r=8 alpha=16 on q_proj of layer 0 in both backends with
seed=SEED. Inspect the resulting LoRA-A and LoRA-B matrices.

Expected baseline (standard LoRA init):
  A ~ Kaiming uniform (non-zero, small magnitude)
  B ~ zero matrix

If both backends honor this, the LoRA contribution at step 0 is zero
and the base-model forward dominates (i.e. probe 2 + LoRA-attached
forward should produce the same logits up to fp noise).

This probe does not enforce A == A across backends (different RNGs),
but DOES enforce:
  * B is exactly zero in both
  * |A.std()| within 2x across backends
  * shapes match
"""

import json
import sys

import numpy as np

from _common import MODEL_NAME, SEED, OUT_DIR, banner, section, report, seed_everything


def main() -> int:
    seed_everything()
    banner("Probe 4: LoRA initialization parity")

    # ---------------- HF / torch / PEFT ----------------
    section("HF + PEFT LoRA")
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(SEED)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    hf_peft = get_peft_model(
        hf_model,
        LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
            target_modules=["q_proj"],
        ),
    )
    # Find layer-0 q_proj LoRA-A and LoRA-B
    hf_A = None
    hf_B = None
    for name, p in hf_peft.named_parameters():
        if "q_proj.lora_A.default.weight" in name and ".0." in name:
            hf_A = p.detach().float().cpu().numpy()
        if "q_proj.lora_B.default.weight" in name and ".0." in name:
            hf_B = p.detach().float().cpu().numpy()
        if hf_A is not None and hf_B is not None:
            break
    report("hf A shape / std", (None if hf_A is None else (hf_A.shape, float(hf_A.std()))))
    report("hf B shape / max|B|", (None if hf_B is None else (hf_B.shape, float(np.abs(hf_B).max()))))

    # ---------------- MLX / mlx-lm / unsloth_zoo.mlx ----------------
    section("MLX + unsloth_zoo.mlx LoRA")
    import mlx.core as mx
    mx.random.seed(SEED)
    from unsloth_zoo.mlx.loader import FastMLXModel
    mlx_model, _tok = FastMLXModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=False,
        dtype="float32",
        text_only=True,
        max_seq_length=64,
        random_state=SEED,
    )
    mlx_model = FastMLXModel.get_peft_model(
        mlx_model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj"],
        random_state=SEED,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=False,
    )
    mlx_A = None
    mlx_B = None
    # Walk module tree and grab layer-0 q_proj LoRA params.
    def walk(mod, prefix=""):
        for name, child in getattr(mod, "named_modules", lambda: [])():
            yield name, child
    try:
        for name, child in mlx_model.named_modules():
            if name.endswith(".q_proj") and (".layers.0." in name or ".0.q_proj" in name):
                for attr in ("lora_a", "lora_b", "lora_A", "lora_B"):
                    if hasattr(child, attr):
                        v = getattr(child, attr)
                        arr = np.asarray(mx.eval(v) if callable(getattr(v, "eval", None)) else v)
                        if attr.lower().endswith("a") and mlx_A is None:
                            mlx_A = arr
                        if attr.lower().endswith("b") and mlx_B is None:
                            mlx_B = arr
                break
    except Exception as e:
        report("introspection error", str(e))
    report("mlx A shape / std", (None if mlx_A is None else (mlx_A.shape, float(mlx_A.std()))))
    report("mlx B shape / max|B|", (None if mlx_B is None else (mlx_B.shape, float(np.abs(mlx_B).max()))))

    section("comparison")
    ok = True
    issues = []
    if hf_A is None or hf_B is None:
        issues.append("could not locate HF layer-0 q_proj LoRA params")
        ok = False
    if mlx_A is None or mlx_B is None:
        issues.append("could not locate MLX layer-0 q_proj LoRA params")
        ok = False
    if hf_B is not None and float(np.abs(hf_B).max()) != 0.0:
        issues.append(f"HF B is non-zero (max|B|={float(np.abs(hf_B).max())})")
        ok = False
    if mlx_B is not None and float(np.abs(mlx_B).max()) != 0.0:
        issues.append(f"MLX B is non-zero (max|B|={float(np.abs(mlx_B).max())})")
        ok = False
    if hf_A is not None and mlx_A is not None and hf_A.shape != mlx_A.shape:
        issues.append(f"shape mismatch A: hf={hf_A.shape} mlx={mlx_A.shape}")
        ok = False
    if hf_A is not None and mlx_A is not None and hf_A.shape == mlx_A.shape:
        ratio = float(mlx_A.std()) / max(float(hf_A.std()), 1e-12)
        report("std ratio mlx/hf", ratio)
        if not (0.5 <= ratio <= 2.0):
            issues.append(f"A std ratio out of [0.5, 2.0]: {ratio:.3f}")
            ok = False

    for i in issues:
        report("FAIL", i)
    if ok:
        report("OK", "B==0 in both and A stds within 2x")

    out = {
        "hf_A_shape": None if hf_A is None else list(hf_A.shape),
        "hf_A_std": None if hf_A is None else float(hf_A.std()),
        "hf_B_max_abs": None if hf_B is None else float(np.abs(hf_B).max()),
        "mlx_A_shape": None if mlx_A is None else list(mlx_A.shape),
        "mlx_A_std": None if mlx_A is None else float(mlx_A.std()),
        "mlx_B_max_abs": None if mlx_B is None else float(np.abs(mlx_B).max()),
        "issues": issues,
    }
    (OUT_DIR / "probe_4.json").write_text(json.dumps(out, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
