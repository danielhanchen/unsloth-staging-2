"""Tiny Qwen-Image-2.1 pipeline (tiny Qwen3-VL text encoder, tiny VAE) with a real-geometry denoiser: 32 x 128 heads
(dim 4096), mlp x3, real RoPE axes, only num_layers shrunk; random bf16 weights.
python build_pipe.py --src TINY_PIPE_DIR --out OUT [--heads 32 --layers 2]"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel

ap = argparse.ArgumentParser()
ap.add_argument("--src", required = True)
ap.add_argument("--out", required = True)
ap.add_argument("--heads", type = int, default = 32)
ap.add_argument("--layers", type = int, default = 2)
a = ap.parse_args()
src, out = Path(a.src), Path(a.out)
if out.exists():
    shutil.rmtree(out)
shutil.copytree(src, out, ignore = shutil.ignore_patterns(".cache", "transformer"))
cfg = json.loads((src / "transformer" / "config.json").read_text())
cfg.update(attention_head_dim = 128, num_attention_heads = a.heads, num_layers = a.layers, mlp_ratio = 3,
           axes_dims_rope = [16, 56, 56])
kw = {k: v for k, v in cfg.items() if not k.startswith("_")}
torch.manual_seed(0)
model = QwenImage21Transformer2DModel(**kw)
with torch.no_grad():
    for n, p in model.named_parameters():
        if "norm" in n and p.dim() == 1:
            p.copy_(0.1 * torch.randn_like(p) + (1.0 if "zero" not in n.lower() else 0.0))
model.to(torch.bfloat16).save_pretrained(out / "transformer", safe_serialization = True)
print("built", out, kw, sum(p.numel() for p in model.parameters()) / 1e6, "M params")
