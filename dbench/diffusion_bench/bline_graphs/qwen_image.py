"""Qwen-Image (original, 20B) text to image: Bline-Org's image_qwen_image template, non-Lightning branch
(blineui-workflow-templates 0.11.70): UNETLoader(qwen_image_fp8_e4m3fn) -> ModelSamplingAuraFlow(3.1) ->
KSampler(euler / simple, 20 steps, CFG 4) ; CLIPLoader(qwen_2.5_vl_7b_fp8_scaled, type qwen_image) -> CLIPTextEncode
pos / neg ("") ; EmptySD3LatentImage ; VAEDecode(qwen_image_vae). The template's default size is 1328x1328; the cell
passes the size used on both frameworks.
Files (Bline-Org/Qwen-Image_BlineUI split_files): dit qwen_image_fp8_e4m3fn.safetensors, te
qwen_2.5_vl_7b_fp8_scaled.safetensors, vae qwen_image_vae.safetensors.
"""

from . import SAVE, GraphParams, pick, save_node, wrap_model

TEMPLATE = "image_qwen_image.json"
DEFAULTS = {"sampler": "euler", "scheduler": "simple", "cfg": 4.0, "shift": 3.1, "steps": 20}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae"}


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": float(pick(p, mod, "shift"))}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": p.files["te"], "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.negative or ""}},
        "latent": {"class_type": "EmptySD3LatentImage", "inputs": {"width": p.width, "height": p.height, "batch_size": 1}},
    }
    model = wrap_model(g, ["shift", 0], p)
    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": model, "seed": int(p.seed), "steps": int(p.steps), "cfg": float(pick(p, mod, "cfg")),
                               "sampler_name": pick(p, mod, "sampler"), "scheduler": pick(p, mod, "scheduler"),
                               "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
                               "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g[SAVE] = save_node(["decode", 0], p.prefix)
    return g
