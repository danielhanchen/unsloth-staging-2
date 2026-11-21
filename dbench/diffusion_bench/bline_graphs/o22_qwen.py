"""Qwen-Image-2.1 t2i: the external tester's graph (2026-10-22 bundle, bench.py bline_graph_qwen) verbatim.

Bline-Org workflow_templates image_qwen_image_2_1_t2i (API format, o22_templates/) with the resolution selector and
prompt-enhancer nodes removed (13, 459:471..475), prompt / size / seed / steps / cfg 1 set, int8_convrot DiT + TE,
QwenImage21Cache(device auto, dtype default) kept, TextEncodeQwenImage21 resolution kept at the template's 1024 (the
tester never changed it, also at 2048^2), SaveImageAdvanced png kept (renamed to "save" for the backend).
"""
import json
from pathlib import Path

from . import SAVE, GraphParams

TEMPLATE = "image_qwen_image_2_1_t2i.json"
DEFAULTS = {"sampler": "euler", "scheduler": "simple", "cfg": 1.0, "steps": 25}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae"}


def build(p: GraphParams) -> dict:
    g = json.load(open(Path(__file__).parent / "o22_templates" / TEMPLATE))
    for k in ["13", "459:471", "459:472", "459:473", "459:474", "459:475"]:
        g.pop(k, None)
    g["459:452"]["inputs"]["prompt"] = p.prompt
    g["459:456"]["inputs"].update(width = int(p.width), height = int(p.height), batch_size = 1)
    g["459:458"]["inputs"].update(seed = int(p.seed), steps = int(p.steps), cfg = 1)
    g["459:451"]["inputs"].update(unet_name = p.files["dit"], weight_dtype = p.weight_dtype or "default")
    g["459:453"]["inputs"]["clip_name"] = p.files["te"]
    g["459:454"]["inputs"]["vae_name"] = p.files["vae"]
    save = g.pop("461")
    save["inputs"]["filename_prefix"] = p.prefix
    g[SAVE] = save
    return g
