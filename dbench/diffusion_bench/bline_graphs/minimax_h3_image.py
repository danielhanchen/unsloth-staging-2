"""MiniMax-H3-Image text to image: Bline-Org's image_minimax_h3_image_t2i template, the "Text to Image (MiniMax H3
Image)" subgraph with the prompt enhancer switched off (its BlineSwitchNode default is off for t2i):
UNETLoader(minimax_h3_image_pruned_int8_convrot) -> KSampler(euler / simple, 6 steps, cfg 1, denoise 1);
CLIPLoader(qwen3vl_32b_minimax_h3_int8_convrot, type minimax) -> MiniMaxH3ImageConditioning(prompt, width, height)
-> positive, ConditioningZeroOut -> negative (skipped at cfg 1, so 6 model evaluations), latent = its output 1;
VAELoader(minimax_h3_image_vae_fp16) -> VAEDecode -> SaveImage.

Size: the template's ResolutionSelector is 1:1 at 4 MP, multiple 32 = 2048x2048 (the template note: "doesn't work well
in low resolution"). The node itself treats width / height 0 as image_1's VAE size or 2048^2. The external tester's
bench.py (bline_graph_h3image) is this graph with the size pinned on both stacks; this module follows it.
Files (Bline-Org/MiniMax-H3-Image): diffusion_models/minimax_h3_image_{pruned_int8_convrot,...}.safetensors,
text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors (the template's; the H3 video templates use the
nvfp4_awq TE), vae/minimax_h3_image_vae_fp16.safetensors. Width / height multiples of 32.

``graph_extra``: input_images [names in BlineUI's input dir] wires LoadImage -> images.image_N (no resize; the edit
template's resize chain lives in minimax_h3_image_edit).
"""

from . import SAVE, GraphParams, pick, save_node, wrap_model

TEMPLATE = "image_minimax_h3_image_t2i.json"
DEFAULTS = {"sampler": "euler", "scheduler": "simple", "cfg": 1.0, "steps": 6}
FILES = {"dit": "diffusion_models", "te": "text_encoders", "vae": "vae"}


def base_graph(p: GraphParams, mod) -> dict:
    if int(p.width) % 32 or int(p.height) % 32:
        raise ValueError(f"minimax-h3-image: width / height must be multiples of 32, got {p.width}x{p.height}")
    te = p.files["te"] if isinstance(p.files["te"], str) else p.files["te"][0]
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.files["dit"], "weight_dtype": p.weight_dtype}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": p.files["vae"]}},
        "cond": {"class_type": "MiniMaxH3ImageConditioning",
                 "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": p.prompt, "width": int(p.width),
                            "height": int(p.height)}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["cond", 0]}},
    }
    model = wrap_model(g, ["unet", 0], p)
    g["sampler"] = {"class_type": "KSampler",
                    "inputs": {"model": model, "seed": int(p.seed), "steps": int(p.steps),
                               "cfg": float(pick(p, mod, "cfg")), "sampler_name": pick(p, mod, "sampler"),
                               "scheduler": pick(p, mod, "scheduler"), "positive": ["cond", 0],
                               "negative": ["neg", 0], "latent_image": ["cond", 1], "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g[SAVE] = save_node(["decode", 0], p.prefix)
    return g


def build(p: GraphParams) -> dict:
    import sys

    g = base_graph(p, sys.modules[__name__])
    for i, name in enumerate((p.extra or {}).get("input_images") or [], start = 1):
        g[f"ref{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g["cond"]["inputs"][f"images.image_{i}"] = [f"ref{i}", 0]
    return g
