"""MiniMax-H3-Image edit: Bline-Org's image_minimax_h3_image_edit template with the prompt enhancer switched OFF
(the template ships it ON: BlineSwitchNode / PrimitiveBoolean true, a Qwen3.5-9B TextGenerateV2 rewrite; a benchmark
feeds the raw prompt to both stacks). Graph = the t2i graph (minimax_h3_image) plus the template's Picture 1 chain:
LoadImage -> ResizeImageMaskNode(scale total pixels, 4 MP, area) -> ResizeImageMaskNode(scale to multiple, 32, area)
-> images.image_1, and GetImageSize of that image -> width / height, so the output is the 4 MP reference size.
Extra references (image_2..) are wired as given, without the resize chain (the template has one picture).
KSampler 6 steps, cfg 1, euler / simple, ConditioningZeroOut negative: 6 model evaluations.

``graph_extra``:
  input_images  [names in BlineUI's input dir], image_1 first (required; defaults to ["ref.png"])
  edit_size     "template" (default: GetImageSize of the resized reference) | "cell" (the cell's width / height,
                as Studio's request pins it; the tester pinned both stacks this way)
Traps: a NEW reference image per timed render (a reused one hits Studio's reference-latent cache) and the same PNG
bytes on both stacks (Studio saved at compress level 1, BlineUI's SaveImage at 4): see models/minimax-h3-image-edit.json.
"""

from . import GraphParams
from .minimax_h3_image import DEFAULTS, FILES, base_graph

TEMPLATE = "image_minimax_h3_image_edit.json"
DEFAULTS = dict(DEFAULTS)
FILES = dict(FILES)


def build(p: GraphParams) -> dict:
    import sys

    x = p.extra or {}
    g = base_graph(p, sys.modules[__name__])
    imgs = list(x.get("input_images") or ["ref.png"])
    g["ref1"] = {"class_type": "LoadImage", "inputs": {"image": imgs[0]}}
    g["resize_mp"] = {"class_type": "ResizeImageMaskNode",
                      "inputs": {"input": ["ref1", 0], "resize_type": "scale total pixels",
                                 "resize_type.megapixels": 4, "scale_method": "area"}}
    g["resize_32"] = {"class_type": "ResizeImageMaskNode",
                      "inputs": {"input": ["resize_mp", 0], "resize_type": "scale to multiple",
                                 "resize_type.multiple": 32, "scale_method": "area"}}
    g["cond"]["inputs"]["images.image_1"] = ["resize_32", 0]
    if x.get("edit_size", "template") == "template":
        g["size"] = {"class_type": "GetImageSize", "inputs": {"image": ["resize_32", 0]}}
        g["cond"]["inputs"].update(width = ["size", 0], height = ["size", 1])
    elif x.get("edit_size") != "cell":
        raise ValueError(f"minimax-h3-image-edit: edit_size must be template or cell, got {x.get('edit_size')!r}")
    for i, name in enumerate(imgs[1:], start = 2):
        g[f"ref{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g["cond"]["inputs"][f"images.image_{i}"] = [f"ref{i}", 0]
    return g
