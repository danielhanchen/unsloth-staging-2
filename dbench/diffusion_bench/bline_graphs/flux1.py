"""FLUX.1 (schnell / dev) text to image: Bline-Org's flux_schnell template, with split files instead of the fp8
all-in-one checkpoint (flux_schnell_full_text_to_image): UNETLoader + DualCLIPLoader(clip_l, t5xxl, type flux) +
VAELoader(ae); CLIPTextEncode; EmptySD3LatentImage; KSampler(euler / simple, CFG 1). schnell uses 4 steps. For dev,
set options.guidance (FluxGuidance, 3.5 in the dev template). A single ``checkpoint`` file uses
CheckpointLoaderSimple instead (the template's flux1-schnell-fp8.safetensors).
Files: dit flux1-schnell.safetensors (BFL original works), te [clip_l.safetensors, t5xxl_fp16.safetensors], vae ae.
"""

from . import SAVE, GraphParams, pick, save_node, wrap_model

TEMPLATE = "flux_schnell.json"
DEFAULTS = {"sampler": "euler", "scheduler": "simple", "cfg": 1.0, "steps": 4}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae", "checkpoint": "checkpoints"}


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    g: dict = {}
    if p.files.get("checkpoint"):
        g["ckpt"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": p.files["checkpoint"]}}
        model, clip, vae = ["ckpt", 0], ["ckpt", 1], ["ckpt", 2]
    else:
        te = p.files["te"]
        if isinstance(te, str) or len(te) != 2:
            raise ValueError("flux.1 needs te = [clip_l file, t5xxl file]")
        clip_l = next((t for t in te if "clip_l" in t.lower()), te[0])
        t5 = next(t for t in te if t != clip_l)
        g["unet"] = {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}}
        g["clip"] = {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": clip_l, "clip_name2": t5, "type": "flux",
                                                                "device": "default"}}
        g["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}}
        model, clip, vae = ["unet", 0], ["clip", 0], ["vae", 0]
    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": p.prompt}}
    g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": p.negative or ""}}
    pos = ["pos", 0]
    if p.guidance is not None:
        g["fluxguidance"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": pos, "guidance": float(p.guidance)}}
        pos = ["fluxguidance", 0]
    g["latent"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": p.width, "height": p.height, "batch_size": 1}}
    model = wrap_model(g, model, p)
    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": model, "seed": int(p.seed), "steps": int(p.steps), "cfg": float(pick(p, mod, "cfg")),
                               "sampler_name": pick(p, mod, "sampler"), "scheduler": pick(p, mod, "scheduler"),
                               "positive": pos, "negative": ["neg", 0], "latent_image": ["latent", 0], "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": vae}}
    g[SAVE] = save_node(["decode", 0], p.prefix)
    return g
