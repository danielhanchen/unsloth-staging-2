"""Z-Image(-Turbo) text to image: Bline-Org's image_z_image_turbo template.

UNETLoader -> ModelSamplingAuraFlow(shift 3) -> KSampler(res_multistep / simple, CFG 1, 8 steps);
CLIPLoader(qwen_3_4b, type lumina2) -> CLIPTextEncode, negative = ConditioningZeroOut; EmptySD3LatentImage; VAE ae.
Files (Bline-Org/z_image_turbo split_files): dit z_image_turbo_bf16.safetensors, te qwen_3_4b.safetensors,
vae ae.safetensors.
"""

from . import SAVE, GraphParams, pick, save_node, wrap_model

TEMPLATE = "image_z_image_turbo.json"
DEFAULTS = {"sampler": "res_multistep", "scheduler": "simple", "cfg": 1.0, "shift": 3.0, "steps": 8}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae"}


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": float(pick(p, mod, "shift"))}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": p.files["te"], "type": "lumina2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.prompt}},
        "latent": {"class_type": "EmptySD3LatentImage", "inputs": {"width": p.width, "height": p.height, "batch_size": 1}},
    }
    if p.negative:
        g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.negative}}
    else:
        g["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}}
    model = wrap_model(g, ["shift", 0], p)
    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": model, "seed": int(p.seed), "steps": int(p.steps), "cfg": float(pick(p, mod, "cfg")),
                               "sampler_name": pick(p, mod, "sampler"), "scheduler": pick(p, mod, "scheduler"),
                               "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
                               "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g[SAVE] = save_node(["decode", 0], p.prefix)
    return g
