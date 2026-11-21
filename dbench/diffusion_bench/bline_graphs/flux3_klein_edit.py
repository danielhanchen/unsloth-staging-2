"""FLUX.3 [klein] image edit (base and distilled): Bline-Org's image_flux3_klein_image_edit /
image_flux3_klein_distilled_image_edit templates with the prompt upsampler off (the raw prompt goes to CLIPTextEncode;
the layout-prompt / bounding-box helper nodes only feed the upsampler):
UNETLoader -> KSampler(euler / simple, denoise 1) <- ReferenceMultiLatent(positive = CLIPTextEncode(prompt),
negative = CLIPTextEncode(""), vae, resolution "resize" 1 MP, images.image0 = LoadImage) outputs (positive, negative,
empty_latent); VAEDecode(VAELoader flux2-vae) -> SaveImage. The output size is the first reference at 1 MP.

What runs vs the template note: the BASE edit template's KSampler runs steps 50, cfg 1 (50 evaluations), while its
note (and BFL's recipe, and Studio) say CFG 3.5 (100 evaluations). DEFAULTS keep what the template RUNS; a like-for-like
speed pair sets cfg 3.5 on both stacks or counts the 2x (models/flux.3-klein-base-edit.json, trap
template_note_vs_sampler). Distilled edit: 16 steps, cfg 1.
Files: dit flux3-image-klein{,-base}-int8-convrot.safetensors, te qwen3vl_4b_int8_convrot.safetensors, vae
flux2-vae.safetensors. graph_extra: input_images [names] (image0 first), megapixels (1.0).
"""

from . import SAVE, GraphParams, pick, save_node, wrap_model

TEMPLATE = "image_flux3_klein_image_edit.json"
DEFAULTS = {"sampler": "euler", "scheduler": "simple", "cfg": 1.0, "steps": 50}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae"}


def build(p: GraphParams) -> dict:
    import sys

    mod = sys.modules[__name__]
    x = p.extra or {}
    imgs = list(x.get("input_images") or ["ref.png"])
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": p.files["te"], "type": "flux3", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": p.negative or ""}},
        "refs": {"class_type": "ReferenceMultiLatent",
                 "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0], "resolution": "resize",
                            "resolution.megapixels": float(x.get("megapixels", 1.0))}},
    }
    for i, name in enumerate(imgs):
        g[f"ref{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g["refs"]["inputs"][f"images.image{i}"] = [f"ref{i}", 0]
    model = wrap_model(g, ["unet", 0], p)
    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": model, "seed": int(p.seed), "steps": int(p.steps),
                               "cfg": float(pick(p, mod, "cfg")), "sampler_name": pick(p, mod, "sampler"),
                               "scheduler": pick(p, mod, "scheduler"), "positive": ["refs", 0],
                               "negative": ["refs", 1], "latent_image": ["refs", 2], "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g[SAVE] = save_node(["decode", 0], p.prefix)
    return g
