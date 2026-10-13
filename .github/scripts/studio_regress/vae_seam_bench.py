"""`vae_seam_bench`: does a Studio tree's tiled VAE decode leave seams, per family and low-VRAM tier?

    <python> studio_regress/vae_seam_bench.py --src <unsloth tree> --out <dir> [--families a,b] [--tiers floor,4g,8g]
    <python> studio_regress/vae_seam_bench.py --compare <before out> <after out>

For every image and video family in the tree's registries (plus LTX-2.3, whose VAE Studio assembles from a single
file), loads the family's real VAE (bf16, pinned revision), encodes a natural photo untiled into a real latent, and
decodes that latent twice: untiled (the reference) and tiled the way the tree's Studio would decode it on a
low-VRAM tier (`enable_tiling()` plus every tile override module the tree ships, see `tile_overrides`). A tier is
the free VRAM the override's planner sees at decode time: `floor` reports 0 bytes free (the smallest tiles the tree
can pick), `4g` / `8g` report 4 / 8 GiB. Trees without a VRAM-aware override decode identically on every tier.

Metrics, per (family, canvas, photo, tier), on the decoded frames in 8-bit levels:
- psnr: PSNR of the tiled decode against the untiled decode of the same latent.
- seam (reported, not gated): the mean |tiled - untiled| on each tile boundary row / column over its median across
  the image, max over boundaries. ~1 when the error is spread over the tiles, high when it piles up on a seam.
- window: the worst 64 px window mean of |tiled - untiled| (PR #12696's line metric).
- The tile grid is read from the decode itself (every slice of the latent's last two dims), not from attributes.

A check FAILs when `window` or `psnr` is past the VAE's limits (CLASS_LIMITS, else DEFAULT_LIMITS). A canvas that
fits one tile is untiled and passes when the tiled path matches the untiled decode. `known` marks a family whose seams are a
recorded, still-open bug on main (KNOWN_SEAMS) so it does not force a base run by itself.

Writes <out>/results.json ({"results": [{"surface", "check", "status", "known", ...}]}, the diffusion_bench shape
external.py compares per check) and <out>/summary.md. Exit 0 no new FAIL, 1 a FAIL, 2 a family that could not be
loaded or decoded (infra: VOID). `--compare` prints the per-check table of two finished runs and exits 1 when a
check that passes before fails after.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

# Run as a script, this directory is sys.path[0] and its profile.py would shadow the stdlib one torch imports.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE]

# Gates: PSNR of tiled vs untiled and the worst 64 px window mean |tiled - untiled| (PR #12696's line metric). A
# boundary-step score does not separate lines from photo content (the image seam audit: Qwen-Image-2.1's lines
# 1.4-3.5 against 1.4-3.3 on a content control), so `seam` (that score on |tiled - untiled|) is reported only.
# Content sets the floor of both gates per VAE, so the default only catches a gross grid, and VAEs with a measured
# fixed geometry get their own, tighter limits: (window max, PSNR min).
#   default: HunyuanImage-2.1 stock (12/3-latent tiles, a visible grid) window 30-61, PSNR 20.9-30.5; every other
#     VAE on main, stock or wide, stays under window 28.2 / over PSNR 27.3 (FLUX.2's 27.3 dB is a per-tile colour
#     shift, not a line).
#   Qwen-Image-2.1: stock window 3.4-14.1, PSNR 32.7-46.2; PR #12696 window 0.27-2.4, PSNR 46.7-64.5.
#   LTX-2 / 2.3: stock window 2.3-7.7, PSNR 39.7-49.3 (the video audit: 43.4-43.7 dB); PR #12698 window 0.47-1.88,
#     PSNR >= 53.2.
DEFAULT_LIMITS = (30.0, 25.0)
CLASS_LIMITS = {
    "AutoencoderKLQwenImage21": (3.0, 45.0),
    "AutoencoderKLLTX2Video": (2.5, 45.0),
}
TIERS = {"floor": 0, "4g": 4 << 30, "8g": 8 << 30}
VIDEO_FRAMES = 9

# The natural photo every latent is encoded from (a real photo, flat background + fur: seams show on both).
IMAGE_REPO = "huggingface/documentation-images"
IMAGE_FILES = {"cat": "diffusers/sdxl_reference_input_cat.jpg", "astronaut": "diffusers/astronaut.jpg"}
VIDEO_IMAGE = "astronaut"
IMAGE_REVISION = "9a5332498b493c7a37c9dfc167b710bfe8ebcf9b"

# Revisions the numbers were measured at (2026-10-05). A family not listed loads its repo's default branch and
# records the revision it got.
PINNED = {
    "unsloth/FLUX.1-schnell": "9df3faa7ae3b6ddf0b2b69bb78616372897cc65c",
    "black-forest-labs/FLUX.2-klein-4B": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
    "unsloth/FLUX.2-dev": "351bbb0ebab35d4dfec0445f67bcec34b383ddf9",
    "unsloth/FLUX.1-Kontext-dev": "ed585f929575da3142d1cdd7a49ef25db6f12161",
    "Qwen/Qwen-Image-Edit-2511": "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
    "Qwen/Qwen-Image": "75e0b4be04f60ec59a75f475837eced720f823b6",
    "Qwen/Qwen-Image-2.1": "d26bb61231c349cf6b7896fa83353113880e1ba3",
    "Tongyi-MAI/Z-Image-Turbo": "f332072aa78be7aecdf3ee76d5c247082da564a6",
    "unsloth/Krea-2-Turbo": "2489d0f4cb200616c0b5410f96f64d8fdc611b06",
    "Alpha-VLLM/Lumina-Image-2.0": "53504abd8178b30685b6c4c7a4cd181ff78b73e9",
    "hunyuanvideo-community/HunyuanImage-2.1-Diffusers": "7e7b7a177de58591aeaffca0929f4765003d7ced",
    "HiDream-ai/HiDream-I1-Full": "8ccbbfb270ccdae26d6bb0081df67dc81e4033bf",
    "unsloth/ideogram-4-fp8": "518e05e745736960acb56d8726138ffbc174f0a9",
    "stabilityai/stable-diffusion-xl-base-1.0": "462165984030d82259a11f4367a4eed129e94a7b",
    "MiniMaxAI/MiniMax-H3": "42ed227ee7df40d41602854ae760620d6eb651fe",
    "Lightricks/LTX-2": "dfcc2108383fe1aaa0584bdf55d368a4bdadd90c",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": "b8fff7315c768468a5333511427288870b2e9635",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "5be7df9619b54f4e2667b2755bc6a756675b5cd7",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v": "286be7ce72277246578a3e3cc2487e95ddae5bcf",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v": "f4dbc4a1efa4ac8ea56680cdf79d9f455105e814",
}
LTX23_FAMILY = "ltx-2.3"
LTX23_VAE = ("unsloth/LTX-2.3-GGUF", "vae/ltx-2.3-22b-distilled_video_vae.safetensors")
# The switchboard runs suites offline against its own HF cache (engine.SUITE_HF_HOME); --prefetch fills it.
SUITE_HUB = Path(os.environ.get("WORKSPACE") or _HERE.parent) / "temp" / "studio_regress" / "hf_home" / "hub"


def prefetch(cache_dir):
    """Every pinned VAE folder, LTX-2.3's single-file VAE and the photos, into ``cache_dir`` (online)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    for repo, rev in PINNED.items():
        snapshot_download(repo, revision=rev, allow_patterns=["vae/*"], cache_dir=cache_dir)
        print(f"prefetched {repo}@{rev[:10]} vae/", flush=True)
    hf_hub_download(LTX23_VAE[0], LTX23_VAE[1], cache_dir=cache_dir)
    for name in IMAGE_FILES.values():
        hf_hub_download(IMAGE_REPO, name, repo_type="dataset", revision=IMAGE_REVISION, cache_dir=cache_dir)
    print(f"prefetched {LTX23_VAE[0]}/{LTX23_VAE[1]} and {len(IMAGE_FILES)} photos into {cache_dir}", flush=True)

# VAEs whose untiled decode is not a reference, so tiled vs untiled measures nothing about seams.
NO_UNTILED_REFERENCE = {
    "AutoencoderKLMiniMaxH3": "transformer decoder working at its 256 px tile: its untiled decode of a photo latent is "
    "20 dB from the photo, the tiled one 30 dB (2026-10-05); Studio always decodes it tiled",
}

GUARD_TEST = "studio/backend/tests/test_vae_tile_geometry.py"


def known_seams(src):
    """Families the tree's own seam guard lists in KNOWN_SEAMS (measured, still-open seam bugs): one list for the
    CPU guard and this bench. Read with ast (the test imports pytest / diffusers); {} when the tree has no guard."""
    import ast

    path = Path(src) / GUARD_TEST
    if not path.is_file():
        return {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        target = getattr(node, "target", None) or (node.targets[0] if isinstance(node, ast.Assign) else None)
        if not (isinstance(target, ast.Name) and target.id == "KNOWN_SEAMS" and isinstance(node.value, ast.Dict)):
            continue
        names = []
        for key, value in zip(node.value.keys, node.value.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                names.append(key.value)
            elif key is None and isinstance(value, ast.DictComp) and isinstance(value.generators[0].iter, ast.Tuple):
                names += [e.value for e in value.generators[0].iter.elts if isinstance(e, ast.Constant)]
        return {n: "KNOWN_SEAMS in " + GUARD_TEST for n in names}
    return {}


KNOWN_SEAMS: dict[str, str] = {}

# Image canvases: Studio's default square plus a size whose stock tile grid ends in a sliver for 8x VAEs.
IMAGE_SIZES = ((1024, 1024), (1600, 1600))


# ---------------------------------------------------------------------------------------------------- tile grid
def _recorder_cls():
    import torch
    from torch.overrides import TorchFunctionMode

    class TileRecorder(TorchFunctionMode):
        """Records every (start, stop) slice taken of the last two dims of a tensor shaped like the latent."""

        def __init__(self, hw):
            super().__init__()
            self.hw = tuple(int(v) for v in hw)
            self.axes = (set(), set())

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.Tensor.__getitem__ and len(args) > 1 and torch.is_tensor(args[0]):
                self._record(args[0], args[1])
            return func(*args, **kwargs)

        def _record(self, t, idx):
            if t.dim() < 4 or tuple(t.shape[-2:]) != self.hw:
                return
            idx = idx if isinstance(idx, tuple) else (idx,)
            if any(i is Ellipsis for i in idx):
                k = next(n for n, i in enumerate(idx) if i is Ellipsis)
                idx = idx[:k] + (slice(None),) * (t.dim() - len(idx) + 1) + idx[k + 1 :]
            if len(idx) != t.dim() or not all(isinstance(s, slice) for s in idx[-2:]):
                return
            for axis, (s, n) in enumerate(zip(idx[-2:], self.hw)):
                lo, hi, _ = s.indices(n)
                if (lo, hi) != (0, n) and hi > lo:
                    self.axes[axis].add((lo, hi))

    return TileRecorder


def boundaries(intervals, length, ratio):
    """Interior tile edges, in pixels, of one axis's recorded tiles."""
    edges = set()
    for lo, hi in intervals:
        if lo > 0:
            edges.add(lo * ratio)
        if hi < length:
            edges.add(hi * ratio)
    return sorted(edges)


def seam_score(tiled, ref, tiles_h, tiles_w, lh, lw, ratio):
    """Boundary score on the error image: for each boundary the decode used, the mean |tiled - untiled| on the row
    (resp. column) at it (+-1 px), over the median of that row mean across the image; max over boundaries, with where.
    ~1 when the error is spread over the tile, high when it piles up on the seam. Reported, not gated."""
    err = (tiled - ref).abs()
    best, where = 1.0, None
    for axis, tiles, length in ((0, tiles_h, lh), (1, tiles_w, lw)):
        edges = boundaries(tiles, length, ratio)
        if not edges:
            continue
        dim = 2 + axis
        other = [d for d in range(err.dim()) if d != dim]
        prof = err.permute(dim, *other).reshape(err.shape[dim], -1).mean(1)
        med = float(prof.median().clamp_min(1e-3))
        for e in edges:
            k0, k1 = max(0, e - 1), min(prof.numel(), e + 2)
            if k1 <= k0:
                continue
            k = k0 + int(prof[k0:k1].argmax())
            if float(prof[k]) / med > best:
                best, where = float(prof[k]) / med, f"{'row' if axis == 0 else 'col'} {k}"
    return best, where


def window_error(tiled, ref, size=64):
    """Worst size x size window mean |tiled - untiled|, in levels (PR #12696's line metric)."""
    import torch

    e = (tiled - ref).abs().mean(dim=(0, 1))[None, None]
    size = min(size, e.shape[-2], e.shape[-1])
    return float(torch.nn.functional.avg_pool2d(e, size, stride=max(1, size // 4)).max())


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse)


# ---------------------------------------------------------------------------------------------------- the tree
def tile_overrides(src):
    """The tree's tile override modules: every core/inference module that rebinds a VAE's ``tiled_decode`` with
    ``types.MethodType`` and exposes ``install(vae, logger=None)``. Discovered, so a new override is benched as soon
    as it lands; a module patching ``tiled_decode`` another way is listed in ``other`` (reported, not applied)."""
    inf = Path(src) / "studio" / "backend" / "core" / "inference"
    found, other = [], []
    for path in sorted(inf.glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"\.tiled_decode\s*=", text):
            continue
        if re.search(r"tiled_decode\s*=\s*types\.MethodType", text) and re.search(r"^def install\(\s*vae\b", text, re.M):
            found.append(path.stem)
        else:
            other.append(path.stem)
    return found, other


def families(src, wanted=None):
    """[(name, kind, spec)] from the tree's registries; spec = {"repo", "revision"} or {"ltx23": True}."""
    sys.path.insert(0, str(Path(src) / "studio" / "backend"))
    from core.inference import diffusion_families as DF
    from core.inference import video_families as VF

    out = []
    for kind, fams in (("image", DF._FAMILIES), ("video", VF._FAMILIES)):
        for f in fams:
            repo = DF.mirror_repo(f.base_repo) if DF.upstream_is_gated(f.base_repo) else f.base_repo
            repo = repo or f.base_repo
            out.append((f.name, kind, {"repo": repo, "revision": PINNED.get(repo), "family": f}))
            if f.name == "ltx-2":
                out.append((LTX23_FAMILY, kind, {"ltx23": True, "family": f}))
    if wanted:
        out = [x for x in out if x[0] in wanted]
    return out


def load_vae(spec, dtype, device):
    import diffusers

    if spec.get("ltx23"):
        from core.inference.video_ltx2 import load_ltx23_vae

        vae = load_ltx23_vae({}, variant="distilled", torch_dtype=dtype, hf_token=None)
        return vae.to(device).eval(), "unsloth/LTX-2.3-GGUF (single file)"
    from huggingface_hub import hf_hub_download

    cfg = json.loads(Path(hf_hub_download(spec["repo"], "vae/config.json", revision=spec["revision"])).read_text())
    name = cfg["_class_name"]
    cls = getattr(diffusers, name, None)
    if cls is None:
        raise RuntimeError(f"diffusers {diffusers.__version__} has no {name}")
    vae = cls.from_pretrained(spec["repo"], subfolder="vae", revision=spec["revision"], torch_dtype=dtype)
    return vae.to(device).eval(), f"{spec['repo']}@{spec['revision'] or 'main'}"


def is_video_vae(vae):
    """True when the VAE takes (B, C, T, H, W): its ``_encode`` reads a frame count (Wan-style image VAEs do too)."""
    import inspect

    try:
        src = inspect.getsource(type(vae)._encode)
    except (OSError, TypeError, AttributeError):
        src = ""
    return re.search(r"num_frame|frames|shape\[2\]|(?:\w+\s*,\s*){4}\w+\s*=\s*\w+\.shape", src) is not None


def photo(name, width, height, frames):
    """(1, 3, T, H, W) in [-1, 1]: the photo resized to cover the canvas, a slow pan across frames."""
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image

    img = Image.open(hf_hub_download(IMAGE_REPO, IMAGE_FILES[name], repo_type="dataset", revision=IMAGE_REVISION)).convert("RGB")
    pan = 4 * (frames - 1)
    scale = max((width + pan) / img.width, height / img.height)
    img = img.resize((math.ceil(img.width * scale), math.ceil(img.height * scale)), Image.LANCZOS)
    top = (img.height - height) // 2
    arr = np.asarray(img, dtype=np.float32)
    clip = [arr[top : top + height, 4 * t : 4 * t + width] for t in range(frames)]
    x = torch.from_numpy(np.stack(clip)).permute(3, 0, 1, 2)[None]
    return x / 127.5 - 1.0


def to_levels(x):
    """Decoder output -> (C, T, H, W) float in 0..255."""
    if x.dim() == 4:
        x = x[:, :, None]
    return ((x[0, :3].float().clamp(-1, 1) + 1) * 127.5).cpu()


@contextlib.contextmanager
def free_vram(budget):
    """Make every free-VRAM read (mem_get_info, reserved - allocated) report ``budget`` bytes free."""
    import torch

    saved = torch.cuda.mem_get_info, torch.cuda.memory_reserved
    torch.cuda.empty_cache()
    torch.cuda.mem_get_info = lambda device=None: (int(budget), saved[0](device)[1])
    torch.cuda.memory_reserved = lambda device=None: torch.cuda.memory_allocated(device)
    try:
        yield
    finally:
        torch.cuda.mem_get_info, torch.cuda.memory_reserved = saved


def decode(vae, z):
    out = vae.decode(z, return_dict=False)
    return out[0] if isinstance(out, (tuple, list)) else out.sample


def encode(vae, x):
    out = vae.encode(x)
    dist = getattr(out, "latent_dist", None)
    if dist is None:
        dist = out[0] if isinstance(out, (tuple, list)) else out
    return dist.mode() if hasattr(dist, "mode") else dist


def canvases(name, kind, spec):
    if kind == "image":
        return IMAGE_SIZES
    presets = tuple(getattr(spec["family"], "resolution_presets", ()) or ((768, 512),))
    return tuple(dict.fromkeys(presets))


def bench_family(name, kind, spec, tiers, overrides, device, log, photos=None):
    import torch

    rows = []
    dtype = torch.bfloat16
    t0 = time.time()
    vae, source = load_vae(spec, dtype, device)
    video = is_video_vae(vae)
    ratio = int(getattr(vae, "spatial_compression_ratio", 0) or 2 ** (len(vae.config.block_out_channels) - 1))
    frames = VIDEO_FRAMES if kind == "video" else 1
    clip = getattr(vae.config, "clip_length", None)
    if kind == "video" and clip:  # MiniMax-H3 decodes whole clips only: two of them, so a chunk boundary is in
        frames = max(frames, 2 * int(clip) + 1)
    applied = []
    for mod in overrides:
        m = importlib.import_module(f"core.inference.{mod}")
        if m.install(vae):
            applied.append(mod)
    if type(vae).__name__ in NO_UNTILED_REFERENCE:
        log(f"{name}: SKIP {NO_UNTILED_REFERENCE[type(vae).__name__]}")
        return [{"surface": name, "check": "untiled_reference", "status": "SKIP", "known": False,
                 "error": NO_UNTILED_REFERENCE[type(vae).__name__], "vae": type(vae).__name__, "source": source}]
    Recorder = _recorder_cls()
    images = photos or ([VIDEO_IMAGE] if kind == "video" else list(IMAGE_FILES))
    for width, height in canvases(name, kind, spec):
        if width % ratio or height % ratio:
            continue
        for image in images:
            x = photo(image, width, height, frames).to(device, dtype)
            if getattr(vae.config, "in_channels", 3) == 4:  # RGBA VAEs (Qwen-Image-2.1): opaque alpha
                x = torch.cat([x, torch.ones_like(x[:, :1])], 1)
            if not video:
                x = x[:, :, 0]
            with torch.no_grad():
                vae.use_tiling = False
                try:
                    z = encode(vae, x)
                except RuntimeError as exc:  # e.g. MiniMax-H3's padded clip overflows 32-bit indexing untiled
                    if "32-bit" not in str(exc) and "out of memory" not in str(exc).lower():
                        raise
                    torch.cuda.empty_cache()
                    vae.use_tiling = True  # the VAE's own tiled encode: still a real latent of the photo
                    z = encode(vae, x)
                    vae.use_tiling = False
                ref = to_levels(decode(vae, z))
                vae.use_tiling = True
                for tier in tiers:
                    rec = Recorder(z.shape[-2:])
                    with free_vram(TIERS[tier]), rec:
                        tiled = to_levels(decode(vae, z))
                    lh, lw = int(z.shape[-2]), int(z.shape[-1])
                    tiles_h, tiles_w = sorted(rec.axes[0]), sorted(rec.axes[1])
                    n = min(ref.shape[1], tiled.shape[1])
                    a, b = tiled[:, :n], ref[:, :n]
                    seam, where = seam_score(a, b, tiles_h, tiles_w, lh, lw, ratio)
                    win = window_error(a, b)
                    p = psnr(a, b)
                    win_max, psnr_min = CLASS_LIMITS.get(type(vae).__name__, DEFAULT_LIMITS)
                    fail = []
                    if win > win_max:
                        fail.append(f"64 px window off by {win:.2f} levels > {win_max}")
                    if p < psnr_min:
                        fail.append(f"psnr {p:.1f} dB < {psnr_min}")
                    rows.append({
                        "surface": name, "check": f"{width}x{height}/{image}/{tier}",
                        "status": "FAIL" if fail else "PASS", "known": bool(fail) and name in KNOWN_SEAMS,
                        "error": "; ".join(fail) or None, "seam": round(seam, 3), "seam_at": where,
                        "window": round(win, 3), "psnr": None if math.isinf(p) else round(p, 2),
                        "vae": type(vae).__name__, "ratio": ratio, "latent": [lh, lw], "tiles_h": tiles_h,
                        "tiles_w": tiles_w, "overrides": applied, "source": source,
                    })
                    log(f"{name:22s} {width}x{height} {image:9s} {tier:5s} seam {seam:6.2f} window {win:6.2f} "
                        f"psnr {p:6.2f}  tiles {len(tiles_h) or 1}x{len(tiles_w) or 1}  {rows[-1]['status']}")
                    del tiled
            del x, z, ref
    del vae
    torch.cuda.empty_cache()
    log(f"{name}: {time.time() - t0:.1f}s")
    return rows


def summary_md(rows, meta):
    lines = [f"# vae_seam_bench: {meta['src']}", "",
             f"limits (64 px window levels, PSNR dB): default {DEFAULT_LIMITS}, per VAE {CLASS_LIMITS}; seam reported only; "
             f"overrides {meta['overrides']}; torch {meta['torch']}, diffusers {meta['diffusers']}", "",
             "| family | VAE | canvas/photo/tier | tiles h x w | seam | window | PSNR dB | status |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] in ("ERROR", "SKIP"):
            lines.append(f"| {r['surface']} | {r.get('vae', '')} | | | | | | {r['status']}: {r['error']} |")
            continue
        status = r["status"] + (" (known)" if r.get("known") else "")
        lines.append(f"| {r['surface']} | {r['vae']} | {r['check']} | {len(r['tiles_h']) or 1} x {len(r['tiles_w']) or 1} "
                     f"| {r['seam']:.2f} | {r['window']:.2f} | {r['psnr'] if r['psnr'] is not None else 'inf'} | {status} |")
    return "\n".join(lines) + "\n"


def compare(before, after):
    def load(d):
        rows = json.loads((Path(d) / "results.json").read_text())["results"]
        return {(r["surface"], r["check"]): r for r in rows}

    b, a = load(before), load(after)
    print("| family | canvas/photo/tier | before seam / window / PSNR | after seam / window / PSNR | verdict |")
    print("|---|---|---|---|---|")
    worse = 0
    for key in sorted(set(b) | set(a)):
        rb, ra = b.get(key), a.get(key)
        fmt = lambda r: "-" if r is None else (r["status"] if r["status"] in ("ERROR", "SKIP") else f"{r['seam']:.2f} / {r['window']:.2f} / {r['psnr']}")
        verdict = "SAME"
        if rb and ra:
            if rb["status"] == "PASS" and ra["status"] != "PASS":
                verdict, worse = "REGRESSION", worse + 1
            elif rb["status"] != "PASS" and ra["status"] == "PASS":
                verdict = "FIXED"
            elif rb["status"] == ra["status"] == "FAIL":
                verdict = "FAIL_BOTH"
        else:
            verdict = "MISSING_" + ("BEFORE" if rb is None else "AFTER")
        print(f"| {key[0]} | {key[1]} | {fmt(rb)} | {fmt(ra)} | {verdict} |")
    return 1 if worse else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src")
    ap.add_argument("--out")
    ap.add_argument("--families", default=os.environ.get("VAE_SEAM_BENCH_FAMILIES", ""),
                    help="comma list (default: every family; env VAE_SEAM_BENCH_FAMILIES scopes a switchboard run)")
    ap.add_argument("--tiers", default="floor,4g,8g")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--photos", default="", help=f"comma list of {sorted(IMAGE_FILES)}; default both for image "
                    f"families, {VIDEO_IMAGE} for video ones")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--prefetch", nargs="?", const=str(SUITE_HUB), metavar="HUB_CACHE",
                    help=f"download every VAE and photo the bench reads (default cache {SUITE_HUB}) and exit")
    args = ap.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if args.prefetch:
        prefetch(args.prefetch)
        return 0
    if not args.src or not args.out:
        ap.error("--src and --out are required")
    import diffusers
    import torch

    args.src = str(Path(args.src).resolve())
    out = Path(args.out).resolve()  # before the chdir below
    out.mkdir(parents=True, exist_ok=True)
    tiers = [t for t in args.tiers.split(",") if t]
    bad = [t for t in tiers if t not in TIERS]
    if bad:
        ap.error(f"unknown tiers {bad}; known {sorted(TIERS)}")
    os.chdir(Path(args.src) / "studio" / "backend")
    overrides, other = tile_overrides(args.src)
    KNOWN_SEAMS.update(known_seams(args.src))
    wanted = {f for f in args.families.split(",") if f}
    fams = families(args.src, wanted or None)
    if wanted - {f[0] for f in fams}:
        print(f"unknown families: {sorted(wanted - {f[0] for f in fams})}", file=sys.stderr)
        return 2
    logf = (out / "bench.log").open("a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"src {args.src}; overrides {overrides}; other tiled_decode patches {other}; tiers {tiers}; "
        f"known {sorted(KNOWN_SEAMS)}")
    rows, infra = [], False
    for name, kind, spec in fams:
        try:
            rows += bench_family(name, kind, spec, tiers, overrides, args.device, log,
                                 [p for p in args.photos.split(",") if p] or None)
        except Exception as exc:  # noqa: BLE001 - one family's load / decode failure must not hide the rest
            infra = True
            rows.append({"surface": name, "check": "load", "status": "ERROR", "known": False,
                         "error": f"{type(exc).__name__}: {exc}"[:500]})
            log(f"{name}: ERROR {traceback.format_exc()}")
            torch.cuda.empty_cache()
    meta = {"src": str(args.src), "overrides": overrides, "known": sorted(KNOWN_SEAMS), "other_tile_patches": other, "tiers": tiers,
            "torch": torch.__version__, "diffusers": diffusers.__version__,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "limits": {"default": DEFAULT_LIMITS, **CLASS_LIMITS},
            "photos": {k: f"{IMAGE_REPO}/{v}@{IMAGE_REVISION}" for k, v in IMAGE_FILES.items()}}
    (out / "results.json").write_text(json.dumps({"meta": meta, "results": rows}, indent=1))
    (out / "summary.md").write_text(summary_md(rows, meta))
    if infra:
        return 2
    return 1 if any(r["status"] == "FAIL" and not r.get("known") for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
