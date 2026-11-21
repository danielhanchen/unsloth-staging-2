"""Krea 2 Turbo text to image: Bline-Org's image_krea2_turbo_t2i_int8 / image_krea2_turbo_t2i templates (workflow_templates
v0.11.111 @ 85984c2; byte-identical to a8bc07a), with the subgraph's prompt-enhancer (TextGenerate) and style-LoRA
branches off: UNETLoader -> KSampler(euler / simple, CFG 1, 8 steps, denoise 1) <- CLIPTextEncode(CLIPLoader type
krea2) positive, ConditioningZeroOut negative; EmptyLatentImage (1024 x 1024 from the template's ResolutionSelector
1:1 / 1 MP / multiple 8; BlineUI widens it to the 16-channel Wan21 latent, bline/sample.py fix_empty_latent_channels);
VAEDecode(VAELoader qwen_image_vae). No ModelSampling node: the shift is the model's own sampling setting
(bline/supported_models.py Krea2.sampling_settings shift 1.15, ModelSamplingFlux), so the card sets
shift_in_graph false.

The template's KSampler is not proxied by the subgraph instance (only prompt, enhancer, size, seed, LoRA and file
names are), so its inner 8 / 1.0 / euler / simple widget is what runs. The instance's prompt_enhance input is true
in both templates: benchmarks switch it off (card trap enhancer_off) and send the raw prompt to both stacks.
"""

from . import SAVE, GraphParams, pick, save_node, wrap_model

TEMPLATE = "image_krea2_turbo_t2i_int8.json"
DEFAULTS = {"sampler": "euler", "scheduler": "simple", "cfg": 1.0, "steps": 8}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae"}


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": p.files["te"], "type": "krea2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.prompt}},
        "latent": {"class_type": "EmptyLatentImage",
                   "inputs": {"width": int(p.width), "height": int(p.height), "batch_size": 1}},
    }
    if p.negative:
        g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.negative}}
    else:  # the template's negative: zeroed positive conditioning (unused at CFG 1)
        g["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}}
    model = wrap_model(g, ["unet", 0], p)
    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": model, "seed": int(p.seed), "steps": int(p.steps), "cfg": float(pick(p, mod, "cfg")),
                               "sampler_name": pick(p, mod, "sampler"), "scheduler": pick(p, mod, "scheduler"),
                               "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
                               "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g[SAVE] = save_node(["decode", 0], p.prefix)
    return g
