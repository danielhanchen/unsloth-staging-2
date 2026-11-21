"""BlineUI API-format graphs per model family, for the ``blineui`` backend.

Each family module mirrors the official Bline-Org workflow template for that model (the node list and widget
values of blineui_workflow_templates_json/templates/<name>.json; the pinned release is models/BLINE_PIN.json),
rebuilt in API format with the cell's prompt, seed, size and steps. A module defines:

    TEMPLATE  the template file it follows
    DEFAULTS  sampler / scheduler / shift / cfg the template uses (a cell's options override them)
    FILES     {role: bline folder} the graph needs, e.g. {"dit": "diffusion_models", "te": "text_encoders"}
    build(p) -> graph dict; the node that saves the output is always id "save"
    VIDEO     (optional) True for video graphs: CreateVideo -> SaveVideo, like the templates' export

The per-family settings a benchmark should run (template defaults, passes, files, traps) live in the model cards,
``diffusion_bench/models/<family>.json``; a card's ``bline.graph`` is a key of FAMILIES below.

``p`` is a GraphParams. Shared wrappers (TorchCompileModel, EasyCache, the cache-busting nonce) are applied here,
so every family gets them the same way.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Optional

FAMILIES = {
    "qwen-image-2.1": "qwen_image_21",
    "z-image": "z_image",
    "flux.1": "flux1",
    "sdxl-turbo": "sdxl_turbo",
    "wan2.2-5b": "wan22_5b",
    "minimax-h3": "minimax_h3",
    "o22-qwen": "o22_qwen",
    "o22-h3": "o22_h3",
    "o22-qwen-edit": "o22_qwen_edit",
    "qwen-image-edit-2511": "qwen_image_edit_2511",
    "sdxl": "sdxl",
    "flux2-klein": "flux2_klein",
    "qwen-image-2.1-turbo": "qwen_image_21_turbo",
    "flux3-klein": "flux3_klein",
    "ltx-2.5": "ltx25",
    "ltx-2.3": "ltx23",
    "hunyuanvideo-1.5": "hunyuanvideo15",
    "wan2.2-a14b": "wan22_a14b",
    "wan2.1-t2v": "wan21_t2v",
    "moss-image": "moss_image",
    "minimax-h3-image": "minimax_h3_image",
    "minimax-h3-image-edit": "minimax_h3_image_edit",
    "flux3-klein-edit": "flux3_klein_edit",
    "qwen-image": "qwen_image",
    "ltx-2.3-dev": "ltx23_dev",
    "hidream-i1": "hidream_i1",
    "ideogram-4": "ideogram4",
    "qwen-image-2.1-edit": "qwen_image_21_edit",
    "krea-2": "krea2",
}
ALIASES = {
    "qwen-image-21": "qwen-image-2.1", "qwen_image_2.1": "qwen-image-2.1", "qwen21": "qwen-image-2.1",
    "z-image-turbo": "z-image", "zimage": "z-image", "z_image": "z-image",
    "flux.1-schnell": "flux.1", "flux.1-dev": "flux.1", "flux-schnell": "flux.1", "flux1": "flux.1", "flux": "flux.1",
    "sdxl_turbo": "sdxl-turbo", "sdxlturbo": "sdxl-turbo",
    "minimax_h3": "minimax-h3", "h3": "minimax-h3",
    "ltx2.5": "ltx-2.5", "ltx25": "ltx-2.5",
    "wan2.2-ti2v-5b": "wan2.2-5b", "wan22-5b": "wan2.2-5b", "wan2.2": "wan2.2-5b",
    "flux.2-klein": "flux2-klein", "flux2_klein": "flux2-klein", "flux.3-klein": "flux3-klein",
    "flux3_klein": "flux3-klein", "qwen_image_edit_2511": "qwen-image-edit-2511",
    "ltx2.3": "ltx-2.3", "ltx23": "ltx-2.3", "ltx-2": "ltx-2.3-dev",
    "hv15": "hunyuanvideo-1.5", "hunyuanvideo15": "hunyuanvideo-1.5", "hunyuanvideo1.5": "hunyuanvideo-1.5",
    "wan2.2-t2v-a14b": "wan2.2-a14b", "wan-a14b": "wan2.2-a14b", "wan22_a14b": "wan2.2-a14b",
    "wan2.1-t2v-1.3b": "wan2.1-t2v", "wan21-1.3b": "wan2.1-t2v", "wan2.1": "wan2.1-t2v",
    "moss_image": "moss-image", "mossimage": "moss-image", "moss-image-sft": "moss-image", "moss-image-base": "moss-image",
    "minimax_h3_image": "minimax-h3-image", "h3-image": "minimax-h3-image", "h3image": "minimax-h3-image",
    "minimax_h3_image_edit": "minimax-h3-image-edit", "h3-image-edit": "minimax-h3-image-edit",
    "flux.3-klein-edit": "flux3-klein-edit", "flux3_klein_edit": "flux3-klein-edit",
    "qwen-image-2512": "qwen-image", "qwen_image": "qwen-image",
    "ltx23_dev": "ltx-2.3-dev", "hidream": "hidream-i1", "hidream_i1": "hidream-i1", "ideogram4": "ideogram-4",
    "qwen21-edit": "qwen-image-2.1-edit", "qwen_image_21_edit": "qwen-image-2.1-edit",
    "krea2": "krea-2", "krea-2-turbo": "krea-2", "krea2-turbo": "krea-2",
}
SAVE = "save"


@dataclass
class GraphParams:
    prompt: str
    seed: int
    steps: int
    width: int = 1024
    height: int = 1024
    negative: Optional[str] = None
    cfg: Optional[float] = None
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    shift: Optional[float] = None
    guidance: Optional[float] = None  # FLUX-dev distilled guidance (FluxGuidance)
    frames: Optional[int] = None
    fps: Optional[float] = None
    files: dict = field(default_factory = dict)  # role -> basename (as BlineUI lists it)
    weight_dtype: str = "default"
    compile: Any = None  # None / False, True, or a TorchCompileModel backend name
    easycache: Optional[float] = None
    easycache_start: float = 0.15
    easycache_end: float = 0.95
    prefix: str = "bench"
    nonce: Optional[int] = None  # an ignored input that changes the cache signature (see apply_nonce)
    nonce_nodes: str = "all"  # all | sampler | none
    extra: dict = field(default_factory = dict)


def canonical(family: str) -> str:
    fam = (family or "").strip().lower()
    fam = ALIASES.get(fam, fam)
    if fam not in FAMILIES:
        raise KeyError(f"no BlineUI graph for family {family!r}; known: {sorted(FAMILIES)} (aliases {sorted(ALIASES)})")
    return fam


def module(family: str):
    return importlib.import_module(f"{__name__}.{FAMILIES[canonical(family)]}")


def pick(p: GraphParams, mod, key: str):
    val = getattr(p, key)
    return mod.DEFAULTS.get(key) if val is None else val


def wrap_model(g: dict, model_link: list, p: GraphParams, suffix: str = "") -> list:
    """TorchCompileModel then EasyCache on the MODEL link, in the order the EasyCache template chains them."""
    if p.compile:
        backend = p.compile if isinstance(p.compile, str) else "inductor"
        g["compile" + suffix] = {"class_type": "TorchCompileModel", "inputs": {"model": model_link, "backend": backend}}
        model_link = ["compile" + suffix, 0]
    if p.easycache is not None:
        g["easycache" + suffix] = {"class_type": "EasyCache",
                          "inputs": {"model": model_link, "reuse_threshold": float(p.easycache),
                                     "start_percent": p.easycache_start, "end_percent": p.easycache_end,
                                     "verbose": False}}
        model_link = ["easycache" + suffix, 0]
    return model_link


_LOADERS = ("UNETLoader", "SaveVideo", "CreateVideo", "CLIPLoader", "DualCLIPLoader", "VAELoader", "CheckpointLoaderSimple", "DiffusersLoader",
            "TorchCompileModel", "EasyCache", "ModelSamplingAuraFlow", "ModelSamplingSD3", "MiniMaxH3SigmaShift",
            "QwenImage21Cache", "UnetLoaderGGUF", "LatentUpscaleModelLoader", "LTXAVTextEncoderLoader",
            "LTXVAudioVAELoader", "LoraLoaderModelOnly", "LoraLoader", "QuadrupleCLIPLoader", "CFGOverride",
            "CFGNorm")  # MODEL -> MODEL patches (like ModelSampling*): re-patching per render is not work Studio does
_SAMPLERS = ("KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced")


def apply_nonce(g: dict, p: GraphParams) -> dict:
    """BlineUI caches every node output keyed on its inputs, so re-running a graph it has seen returns the cached
    image in ~0.1 s without sampling. run_cell renders the same prompt and seed twice (the timed pass and the
    long/short pairs), so without this the second one is a cache hit. An input the node does not declare is part
    of the cache signature (bline_execution/caching.py hashes every raw input) but is dropped before the node runs
    (execution.get_input_data keeps declared inputs only), so a per-render ``_bench_nonce`` forces re-execution
    without changing any result. ``all`` re-runs text encoding, sampling and decoding (the models stay loaded):
    the same work Studio does per render. ``sampler`` re-runs from the sampler on, ``none`` leaves BlineUI's cache
    alone."""
    if p.nonce is None or p.nonce_nodes == "none":
        return g
    for node in g.values():
        ct = node["class_type"]
        if ct in _LOADERS:
            continue
        if p.nonce_nodes == "sampler" and ct not in _SAMPLERS:
            continue
        node["inputs"]["_bench_nonce"] = int(p.nonce)
    return g


def save_node(images_link: list, prefix: str) -> dict:
    return {"class_type": "SaveImage", "inputs": {"images": images_link, "filename_prefix": prefix}}


def save_video_nodes(g: dict, images_link: list, fps: float, prefix: str) -> None:
    """The video templates' export: CreateVideo(fps) -> SaveVideo(auto / auto = H.264 MP4). Node ids "video" and SAVE."""
    g["video"] = {"class_type": "CreateVideo", "inputs": {"images": images_link, "fps": float(fps)}}
    g[SAVE] = {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": prefix,
                                                     "format": "auto", "format.codec": "auto"}}


def gguf_loaders(g: dict) -> dict:
    """A .gguf denoiser goes through BlineUI-GGUF's UnetLoaderGGUF (city96/BlineUI-GGUF, the loader BlineUI users
    install for GGUF): same MODEL output, no weight_dtype input."""
    for node in g.values():
        if node["class_type"] == "UNETLoader" and str(node["inputs"].get("unet_name", "")).endswith(".gguf"):
            node["class_type"] = "UnetLoaderGGUF"
            node["inputs"] = {"unet_name": node["inputs"]["unet_name"]}
    return g


def frames_save(g: dict, p: GraphParams) -> dict:
    """``graph_extra.video_save = "frames"``: replace CreateVideo -> SaveVideo with SaveImage on the same frames, for a
    backend that reads PNG frames only. The default (the templates' MP4 export) is what a BlineUI user runs."""
    if (p.extra or {}).get("video_save") != "frames" or g.get(SAVE, {}).get("class_type") != "SaveVideo":
        return g
    video = g.pop(g[SAVE]["inputs"]["video"][0])
    g[SAVE] = save_node(video["inputs"]["images"], p.prefix)
    return g


# Families BlineUI-GGUF's UnetLoaderGGUF refuses: a .gguf denoiser for these cannot be a BlineUI arm, so the graph
# refuses it instead of failing inside the server. city96/BlineUI-GGUF @ 6ea2651e (upstream HEAD, 2026-01-12)
# loader.py:12 IMG_ARCH_LIST = flux, sd1, sdxl, sd3, aura, hidream, cosmos, ltxv, hyvid, wan, lumina2, qwen_image, and
# tools/convert.py detect_arch (files without general.architecture) knows only those templates: no MiniMax-H3 (video or
# image) and no MOSS-Image. H3 video was also refused on a real server (outputs/video_vs_bline_nov5/README.md:411).
GGUF_REFUSED = ("minimax-h3", "o22-h3", "minimax-h3-image", "minimax-h3-image-edit", "moss-image")


def build(family: str, p: GraphParams) -> dict:
    fam = canonical(family)
    mod = module(fam)
    if fam in GGUF_REFUSED and str((p.files or {}).get("dit", "")).endswith(".gguf"):
        raise ValueError(f"{fam}: BlineUI-GGUF's UnetLoaderGGUF has no architecture entry for this family (trap "
                         "gguf_refused); compare its GGUF on sd.cpp / Studio only")
    g = frames_save(gguf_loaders(mod.build(p)), p)
    if SAVE not in g:
        raise RuntimeError(f"bline_graphs.{mod.__name__} built no '{SAVE}' node")
    return apply_nonce(g, p)
