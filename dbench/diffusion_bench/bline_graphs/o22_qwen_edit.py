"""Qwen-Image-2.1 edit (one reference): the external tester's graph (2026-10-22 bundle, bench.py bline_graph_qwen_edit):
their t2i graph (o22_qwen) plus LoadImage(ref) wired into TextEncodeQwenImage21 (vae, images.image_1, resolution 1024).
"""
from . import o22_qwen

TEMPLATE = o22_qwen.TEMPLATE
DEFAULTS = o22_qwen.DEFAULTS
FILES = o22_qwen.FILES


def build(p):
    g = o22_qwen.build(p)
    imgs = p.extra.get("input_images") or ["ref.png"]
    g["ref"] = {"class_type": "LoadImage", "inputs": {"image": imgs[0]}}
    g["459:452"]["inputs"].update({"vae": ["459:454", 0], "images.image_1": ["ref", 0], "resolution": 1024})
    return g
