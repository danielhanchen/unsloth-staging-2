"""HiDream-I1 (Full / Dev / Fast) text to image: Bline-Org's hidream_i1_{full,dev,fast} templates (no subgraph):
UNETLoader(hidream_i1_<v>_fp8) -> ModelSamplingSD3(shift: Full 3, Dev 6, Fast 3) -> KSampler(Full 50 / cfg 5 /
uni_pc / simple; Dev 28 / cfg 1 / lcm / normal; Fast 16 / cfg 1 / lcm / normal; denoise 1);
QuadrupleCLIPLoader(clip_l_hidream, clip_g_hidream, t5xxl_fp8_e4m3fn_scaled, llama_3.1_8b_instruct_fp8_scaled) ->
CLIPTextEncode positive / negative ("bad ugly jpeg artifacts"); EmptySD3LatentImage 1024x1024; VAEDecode(ae).
The variant's numbers come from the cell (the cards models/hidream-i1-*.json); DEFAULTS = Full.
Files: dit (diffusion_models), te [4 names in loader order] (text_encoders), vae (vae).
"""

from . import SAVE, GraphParams, pick, save_node, wrap_model

TEMPLATE = "hidream_i1_full.json"
DEFAULTS = {"sampler": "uni_pc", "scheduler": "simple", "cfg": 5.0, "shift": 3.0, "steps": 50}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae"}
NEGATIVE = "bad ugly jpeg artifacts"


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    te = p.files["te"]
    if isinstance(te, str) or len(te) != 4:
        raise ValueError("hidream-i1: files.te must be [clip_l, clip_g, t5xxl, llama_3.1_8b] in that order")
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}},
        "shift": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["unet", 0], "shift": float(pick(p, mod, "shift"))}},
        "clip": {"class_type": "QuadrupleCLIPLoader",
                 "inputs": {"clip_name1": te[0], "clip_name2": te[1], "clip_name3": te[2], "clip_name4": te[3]}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.prompt}},
        "neg": {"class_type": "CLIPTextEncode",
                "inputs": {"clip": ["clip", 0], "text": NEGATIVE if p.negative is None else p.negative}},
        "latent": {"class_type": "EmptySD3LatentImage",
                   "inputs": {"width": int(p.width), "height": int(p.height), "batch_size": 1}},
    }
    model = wrap_model(g, ["shift", 0], p)
    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": model, "seed": int(p.seed), "steps": int(p.steps),
                               "cfg": float(pick(p, mod, "cfg")), "sampler_name": pick(p, mod, "sampler"),
                               "scheduler": pick(p, mod, "scheduler"), "positive": ["pos", 0],
                               "negative": ["neg", 0], "latent_image": ["latent", 0], "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g[SAVE] = save_node(["decode", 0], p.prefix)
    return g
