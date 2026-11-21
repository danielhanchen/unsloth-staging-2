"""FLUX.2 [klein] 4B distilled text to image: Bline-Org's image_flux2_klein_text_to_image template, the
"Text to Image (Flux.2 Klein 4B Distilled)" subgraph: UNETLoader(flux-2-klein-4b) + CLIPLoader(qwen_3_4b, type flux2)
+ VAELoader(flux2-vae); CLIPTextEncode -> CFGGuider(cfg 1) with ConditioningZeroOut(positive) as the negative;
Flux2Scheduler(steps 4, width, height); KSamplerSelect(euler); RandomNoise(seed); EmptyFlux2LatentImage;
SamplerCustomAdvanced; VAEDecode. The base (non-distilled) subgraph is 20 steps, CFG 5, a real negative prompt.
Files (Bline-Org/flux2-klein-4B split_files): dit flux-2-klein-4b.safetensors, te qwen_3_4b.safetensors,
vae flux2-vae.safetensors.
"""

from . import SAVE, GraphParams, pick, save_node, wrap_model

TEMPLATE = "image_flux2_klein_text_to_image.json"
DEFAULTS = {"sampler": "euler", "cfg": 1.0, "steps": 4}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae"}


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    te = p.files["te"] if isinstance(p.files["te"], str) else p.files["te"][0]
    g: dict = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "flux2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
    }
    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.prompt}}
    cfg = float(pick(p, mod, "cfg"))
    if p.negative is None and cfg == 1.0:
        g["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}}
    else:
        g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.negative or ""}}
    model = wrap_model(g, ["unet", 0], p)
    g["guider"] = {"class_type": "CFGGuider",
                   "inputs": {"model": model, "positive": ["pos", 0], "negative": ["neg", 0], "cfg": cfg}}
    g["ksampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": pick(p, mod, "sampler")}}
    g["sigmas"] = {"class_type": "Flux2Scheduler",
                   "inputs": {"steps": int(p.steps), "width": int(p.width), "height": int(p.height)}}
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(p.seed)}}
    g["latent"] = {"class_type": "EmptyFlux2LatentImage",
                   "inputs": {"width": int(p.width), "height": int(p.height), "batch_size": 1}}
    g["sampler"] = {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["ksampler", 0],
                               "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g[SAVE] = save_node(["decode", 0], p.prefix)
    return g
