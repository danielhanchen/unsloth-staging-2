"""Qwen-Image-Edit-2511: Bline-Org's image_qwen_image_edit_2511_int8 template, subgraph "Image Edit (Qwen-Image 2511
Int8)", with its defaults (Lightning switch off): UNETLoader(int8_convrot) -> ModelSamplingAuraFlow(3.1) -> CFGNorm(1);
LoadImage -> FluxKontextImageScale -> TextEncodeQwenImageEditPlus(prompt, vae, image1) for the positive and the same
node with an empty prompt for the negative, each through FluxKontextMultiReferenceLatentMethod(index_timestep_zero);
VAEEncode(scaled image) is the KSampler latent; KSampler euler / simple, 40 steps, CFG 4, denoise 1; VAEDecode.
The KSampler widget shows a stale cfg 3; the linked value (Switch (CFG) off -> PrimitiveFloat 154) is 4.
One source image (the template's image2 / image3 inputs are optional and left empty).
"""

from . import SAVE, GraphParams, pick, save_node, wrap_model

TEMPLATE = "image_qwen_image_edit_2511_int8.json"
DEFAULTS = {"sampler": "euler", "scheduler": "simple", "cfg": 4.0, "steps": 40, "shift": 3.1}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae"}


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    imgs = p.extra.get("input_images") or ["ref.png"]
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": p.files["te"], "type": "qwen_image",
                                                        "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
        "img": {"class_type": "LoadImage", "inputs": {"image": imgs[0]}},
        "scale": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["img", 0]}},
        "pos0": {"class_type": "TextEncodeQwenImageEditPlus",
                 "inputs": {"clip": ["clip", 0], "prompt": p.prompt, "vae": ["vae", 0], "image1": ["scale", 0]}},
        "neg0": {"class_type": "TextEncodeQwenImageEditPlus",
                 "inputs": {"clip": ["clip", 0], "prompt": p.negative or "", "vae": ["vae", 0], "image1": ["scale", 0]}},
        "pos": {"class_type": "FluxKontextMultiReferenceLatentMethod",
                "inputs": {"conditioning": ["pos0", 0], "reference_latents_method": "index_timestep_zero"}},
        "neg": {"class_type": "FluxKontextMultiReferenceLatentMethod",
                "inputs": {"conditioning": ["neg0", 0], "reference_latents_method": "index_timestep_zero"}},
        "latent": {"class_type": "VAEEncode", "inputs": {"pixels": ["scale", 0], "vae": ["vae", 0]}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": float(pick(p, mod, "shift"))}},
        "cfgnorm": {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}},
    }
    model = wrap_model(g, ["cfgnorm", 0], p)
    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": model, "seed": int(p.seed), "steps": int(p.steps), "cfg": float(pick(p, mod, "cfg")),
                               "sampler_name": pick(p, mod, "sampler"), "scheduler": pick(p, mod, "scheduler"),
                               "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
                               "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g[SAVE] = save_node(["decode", 0], p.prefix)
    return g
