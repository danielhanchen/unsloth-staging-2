"""MOSS-Image (SFT / Base) text to image: Bline-Org's image_moss_image_t2i / image_moss_image_base_t2i templates, the
"Text to Image (MOSS-Image)" subgraph with the prompt upsampler switched off (raw prompt to CLIPTextEncode):
UNETLoader(moss_image_{sft,base}_int8_convrot) [-> SkipLayerGuidanceDiTSimple(double "", single "22", 0, 0.5) when the
template's switch_1 is on: Base yes, SFT no] -> KSampler(er_sde / simple, 25 steps, CFG 2.5, denoise 1);
CLIPLoader(qwen3.5_9b_w4a8_gptq, type moss_image) -> CLIPTextEncode positive and negative ("nsfw, worst quality, bad
anatomy, watermark, jpeg artifact"); EmptyFlux2LatentImage 1024x1024; VAELoader(flux2-vae_small); VAEDecode.
The model's own sampling settings carry shift 5 (no ModelSampling node in the template); ``shift`` adds
ModelSamplingAuraFlow only when a cell asks for it. ``graph_extra`` {"slg": true|false} picks the SLG branch.
Files (Bline-Org/MOSS-Image): diffusion_models/moss_image_{sft,base}_{int8_convrot,bf16}.safetensors,
text_encoders/qwen3.5_9b_{w4a8_gptq,int8_convrot,bf16}.safetensors, vae/flux2-vae_small.safetensors.
"""

from . import SAVE, GraphParams, pick, save_node, wrap_model

TEMPLATE = "image_moss_image_t2i.json"
DEFAULTS = {"sampler": "er_sde", "scheduler": "simple", "cfg": 2.5, "steps": 25}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae"}
NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    te = p.files["te"] if isinstance(p.files["te"], str) else p.files["te"][0]
    g: dict = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "moss_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.prompt}},
        "latent": {"class_type": "EmptyFlux2LatentImage",
                   "inputs": {"width": int(p.width), "height": int(p.height), "batch_size": 1}},
    }
    negative = NEGATIVE if p.negative is None else p.negative
    g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": negative}}
    model = ["unet", 0]
    if p.shift is not None:
        g["shift"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": float(p.shift)}}
        model = ["shift", 0]
    if (p.extra or {}).get("slg"):
        g["slg"] = {"class_type": "SkipLayerGuidanceDiTSimple",
                    "inputs": {"model": model, "double_layers": "", "single_layers": "22",
                               "start_percent": 0.0, "end_percent": 0.5}}
        model = ["slg", 0]
    model = wrap_model(g, model, p)
    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": model, "seed": int(p.seed), "steps": int(p.steps),
                               "cfg": float(pick(p, mod, "cfg")), "sampler_name": pick(p, mod, "sampler"),
                               "scheduler": pick(p, mod, "scheduler"), "positive": ["pos", 0],
                               "negative": ["neg", 0], "latent_image": ["latent", 0], "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g[SAVE] = save_node(["decode", 0], p.prefix)
    return g
