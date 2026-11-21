"""SDXL-Turbo: Bline-Org's sdxlturbo_example template. CheckpointLoaderSimple (sd_xl_turbo_1.0_fp16.safetensors) or,
when the cell gives a diffusers directory instead, DiffusersLoader over it (deprecated in BlineUI but still
shipped; it is how a diffusers snapshot runs without converting it). CLIPTextEncode pos / neg ("text, watermark"
in the template); EmptyLatentImage 512; SDTurboScheduler(steps, denoise 1) + KSamplerSelect(euler_ancestral) ->
SamplerCustom(add_noise, CFG 1). SDTurboScheduler caps steps at 10.
"""

from . import SAVE, GraphParams, pick, save_node, wrap_model

TEMPLATE = "sdxlturbo_example.json"
DEFAULTS = {"sampler": "euler_ancestral", "cfg": 1.0, "steps": 1}
FILES = {"checkpoint": "checkpoints", "diffusers": "diffusers"}


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    if int(p.steps) > 10:
        raise ValueError("sdxl-turbo: SDTurboScheduler takes at most 10 steps")
    g: dict = {}
    if p.files.get("checkpoint"):
        g["ckpt"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": p.files["checkpoint"]}}
    elif p.files.get("diffusers"):
        g["ckpt"] = {"class_type": "DiffusersLoader", "inputs": {"model_path": p.files["diffusers"]}}
    else:
        raise ValueError("sdxl-turbo needs options.files.checkpoint or options.files.diffusers")
    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ckpt", 1], "text": p.prompt}}
    g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ckpt", 1], "text": p.negative or ""}}
    g["latent"] = {"class_type": "EmptyLatentImage", "inputs": {"width": p.width, "height": p.height, "batch_size": 1}}
    model = wrap_model(g, ["ckpt", 0], p)
    g["sigmas"] = {"class_type": "SDTurboScheduler", "inputs": {"model": model, "steps": int(p.steps), "denoise": 1.0}}
    g["ksel"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": pick(p, mod, "sampler")}}
    g["sampler"] = {"class_type": "SamplerCustom",
                    "inputs": {"model": model, "add_noise": True, "noise_seed": int(p.seed), "cfg": float(pick(p, mod, "cfg")),
                               "positive": ["pos", 0], "negative": ["neg", 0], "sampler": ["ksel", 0],
                               "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["ckpt", 2]}}
    g[SAVE] = save_node(["decode", 0], p.prefix)
    return g
