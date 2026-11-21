"""Qwen-Image-2.1-Turbo text to image: Bline-Org's image_qwen_image_2_1_turbo_t2i template (workflow_templates 447747b).

UNETLoader -> QwenImage21Cache(auto, default) -> CFGGuider(cfg 1) <- TextEncodeQwenImage21(CLIPLoader type qwen_image,
resolution 1024); KSamplerSelect(euler); ManualSigmas (the template's 8-step list); RandomNoise; SamplerCustomAdvanced;
EmptyLatentImage; VAEDecode(VAELoader). The template's prompt-enhancer branch (TextGenerate, BlineSwitchNode false) is
off by default and left out. The step count is the sigma list (8); ``p.steps`` is ignored.
Files here are built from the official Qwen/Qwen-Image-2.1-Turbo repo (bf16), not the template's Bline-Org int8 files.
"""

from . import SAVE, GraphParams, save_node, wrap_model

TEMPLATE = "image_qwen_image_2_1_turbo_t2i.json"
SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"
DEFAULTS = {"sampler": "euler", "scheduler": None, "cfg": 1.0, "steps": 8}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae"}


def _sigma_list(text: str) -> list:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def build(p: GraphParams) -> dict:
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": p.files["te"], "type": "qwen_image",
                                                        "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
        "cond": {"class_type": "TextEncodeQwenImage21",
                 "inputs": {"clip": ["clip", 0], "prompt": p.prompt, "negative_prompt": p.negative or "",
                            "resolution": int(p.extra.get("te_resolution") or 1024)}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": p.width, "height": p.height, "batch_size": 1}},
        "prefix_cache": {"class_type": "QwenImage21Cache", "inputs": {"model": ["unet", 0], "device": "auto",
                                                                     "dtype": "default"}},
    }
    model = wrap_model(g, ["prefix_cache", 0], p)
    g["guider"] = {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["cond", 0], "negative": ["cond", 1],
                                                         "cfg": float(p.cfg if p.cfg is not None else 1.0)}}
    g["ksampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": p.sampler or "euler"}}
    sig = p.extra.get("sigmas") or SIGMAS
    if not p.extra.get("sigmas") and p.steps and int(p.steps) != len(_sigma_list(SIGMAS)) - 1:
        # a shorter run (short_steps, smoke): the first N distilled sigmas, then 0, so the step count is honoured
        v = _sigma_list(SIGMAS)[: max(1, min(int(p.steps), len(_sigma_list(SIGMAS)) - 1))] + ["0.0"]
        sig = ", ".join(v)
    g["sigmas"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": sig}}
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(p.seed)}}
    g["sampler"] = {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["ksampler", 0],
                               "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g[SAVE] = save_node(["decode", 0], p.prefix)
    return g
