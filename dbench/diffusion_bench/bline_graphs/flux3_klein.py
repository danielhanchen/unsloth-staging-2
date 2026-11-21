"""FLUX.3 [klein] text to image (distilled and base): BFL's template image_flux3_klein_distilled_t2i /
image_flux3_klein_t2i (Bline-Org workflow_templates cd0fcd0), prompt-upsampler branch off (its BlineSwitchNode default):
UNETLoader -> KSampler(euler / simple) <- CLIPTextEncode(CLIPLoader type flux3) positive, CLIPTextEncode("") negative;
EmptyFlux2LatentImage; VAEDecode(VAELoader). The template's KSampler steps / cfg are overridden by the cell (the same
steps and guidance Studio renders with: distilled 16 / 1, base 50 / 3.5, BFL's recipe).
"""

from . import SAVE, GraphParams, pick, save_node, wrap_model

TEMPLATE = "image_flux3_klein_distilled_t2i.json"
# The subgraph's inner KSampler widget reads 15 / er_sde, but the instance proxies steps 16, euler, simple (and the
# base template cfg 3.5, steps 50): the proxied values are what runs.
DEFAULTS = {"sampler": "euler", "scheduler": "simple", "cfg": 1.0, "steps": 16}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae"}


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": p.files["te"], "type": "flux3", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.negative or ""}},
        "latent": {"class_type": "EmptyFlux2LatentImage",
                   "inputs": {"width": int(p.width), "height": int(p.height), "batch_size": 1}},
    }
    model = wrap_model(g, ["unet", 0], p)
    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": model, "seed": int(p.seed), "steps": int(p.steps), "cfg": float(pick(p, mod, "cfg")),
                               "sampler_name": pick(p, mod, "sampler"), "scheduler": pick(p, mod, "scheduler"),
                               "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
                               "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g[SAVE] = save_node(["decode", 0], p.prefix)
    return g
