"""Qwen-Image-2.1 image edit, template-faithful: Bline-Org's image_qwen_image_2_1_image_edit, subgraph "Image Edit
(Qwen Image 2.1)" with the prompt enhancer off (its switch default):
UNETLoader -> QwenImage21Cache(auto, default) -> KSampler(euler / simple, 25 steps, cfg 1, denoise 1);
CLIPLoader(qwen3vl_8b_int8_convrot, qwen_image) -> TextEncodeQwenImage21(prompt, negative "", vae,
images.image_1.., resolution 0) -> positive / negative, and its output 2 = the reference-sized latent (the template's
"custom size" switch is off, so EmptyLatentImage is unused); VAEDecode -> save.
resolution 0 = each reference keeps its own size (rounded to 32) and the OUTPUT is the first reference's size: an
edit cell must give Studio the same output size (the reference's), or the pair is not like-for-like.
The external tester's variant (resolution 1024 + a 1024^2 empty latent, Studio-request-matched) stays registered as
"o22-qwen-edit". graph_extra: input_images [names] (image_1 first), resolution (0), prefix_cache ("auto"; "off" drops
the node).
"""

from dataclasses import replace

from . import qwen_image_21

TEMPLATE = "image_qwen_image_2_1_image_edit.json"
DEFAULTS = dict(qwen_image_21.DEFAULTS)
FILES = dict(qwen_image_21.FILES)


def build(p):
    x = dict(p.extra or {})
    cache = x.get("prefix_cache", "auto")
    x["prefix_cache"] = None if cache in (None, False, "off") else cache
    g = qwen_image_21.build(replace(p, extra = x))
    imgs = list(x.get("input_images") or ["ref.png"])
    g["cond"]["inputs"].update(vae = ["vae", 0], resolution = int(x.get("resolution", 0)))
    for i, name in enumerate(imgs, start = 1):
        g[f"ref{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g["cond"]["inputs"][f"images.image_{i}"] = [f"ref{i}", 0]
    del g["latent"]
    g["sampler"]["inputs"]["latent_image"] = ["cond", 2]
    return g
