"""Wan2.1-T2V (1.3B / 14B) text to video: BlineUI's text_to_video_wan template (blineui-workflow-templates 0.11.96).
UNETLoader(wan2.1_t2v_1.3B_fp16) -> ModelSamplingSD3(shift 8) -> KSampler(uni_pc / simple, CFG 6, 30 steps);
CLIPLoader(umt5_xxl_fp8_e4m3fn_scaled, type wan) -> CLIPTextEncode pos / neg (Wan's Chinese default negative);
EmptyHunyuanLatentVideo(832, 480, 33); VAEDecode(wan_2.1_vae) -> CreateVideo(16) -> SaveVideo (auto = H.264 MP4).
Files here are built from the OFFICIAL Wan-AI repos (vscv/convert_bline.py), not the Bline-Org repack.
Length must be 4k+1 frames; width / height multiples of 16.
"""

from . import GraphParams, pick, save_video_nodes, wrap_model
from .wan22_5b import DEFAULT_NEGATIVE

TEMPLATE = "text_to_video_wan.json"
DEFAULTS = {"sampler": "uni_pc", "scheduler": "simple", "cfg": 6.0, "shift": 8.0, "steps": 30}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae"}
VIDEO = True


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    frames = int(p.frames or 33)
    if (frames - 1) % 4:
        raise ValueError(f"wan2.1-t2v: frames must be 4k+1, got {frames}")
    te = p.files["te"] if isinstance(p.files["te"], str) else p.files["te"][0]
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}},
        "shift": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["unet", 0], "shift": float(pick(p, mod, "shift"))}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "wan", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0],
                                                          "text": DEFAULT_NEGATIVE if p.negative is None else p.negative}},
        "latent": {"class_type": "EmptyHunyuanLatentVideo",
                   "inputs": {"width": p.width, "height": p.height, "length": frames, "batch_size": 1}},
    }
    model = wrap_model(g, ["shift", 0], p)
    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": model, "seed": int(p.seed), "steps": int(p.steps), "cfg": float(pick(p, mod, "cfg")),
                               "sampler_name": pick(p, mod, "sampler"), "scheduler": pick(p, mod, "scheduler"),
                               "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
                               "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    save_video_nodes(g, ["decode", 0], p.fps or 16, p.prefix)
    return g
