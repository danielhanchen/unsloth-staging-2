"""MiniMax-H3 t2v + audio: the external tester's graph (2026-10-22 bundle, bench.py bline_graph_h3) verbatim.

Bline-Org workflow_templates video_minimax_h3_t2v (API format, o22_templates/) with the resolution selector, the
prompt enhancer and the turbo switch removed (115, 140:132..139), the scheduler and guider wired to the UNETLoader
(turbo off), res_multistep / simple, BasicGuider (no CFG), CreateVideo 24 fps + SaveVideo kept (renamed "save").
"""
import json
from pathlib import Path

from . import SAVE, GraphParams

TEMPLATE = "video_minimax_h3_t2v.json"
DEFAULTS = {"sampler": "res_multistep", "scheduler": "simple", "steps": 20}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae", "audio_vae": "vae"}
VIDEO = True


def build(p: GraphParams) -> dict:
    g = json.load(open(Path(__file__).parent / "o22_templates" / TEMPLATE))
    for k in ["115", "140:132", "140:133", "140:134", "140:135", "140:136", "140:137", "140:138", "140:139"]:
        g.pop(k, None)
    g["140:124"]["inputs"].update(steps = int(p.steps), model = ["140:127", 0])
    g["140:126"]["inputs"]["model"] = ["140:127", 0]
    g["140:131"]["inputs"].update(prompt = p.prompt, width = int(p.width), height = int(p.height),
                                  length = int(p.frames or 124))
    g["140:129"]["inputs"]["noise_seed"] = int(p.seed)
    g["140:127"]["inputs"].update(unet_name = p.files["dit"], weight_dtype = p.weight_dtype or "default")
    te = p.files["te"] if isinstance(p.files["te"], str) else p.files["te"][0]
    g["140:128"]["inputs"]["clip_name"] = te
    g["140:119"]["inputs"]["vae_name"] = p.files["vae"]
    g["140:120"]["inputs"]["vae_name"] = p.files["audio_vae"]
    save = g.pop("92")
    save["inputs"]["filename_prefix"] = "video/" + p.prefix
    g[SAVE] = save
    return g
