"""Shared helpers for the diffusion journeys (10 diffusion_gen, 11 diffusion_train).

Fixture: `hf-internal-testing/tiny-stable-diffusion-xl-pipe` (51 MB, StableDiffusionXLPipeline).
Studio restricts non-GGUF diffusion loads to unsloth/* repos OR a local path, so the fixture is
a LOCAL snapshot (`ensure_tiny_sdxl`) and loaded with model_kind="pipeline"; family detection
reads its model_index.json (`sdxl`). Measured on B200: load < 5 s, 256x256 1-step generate 1.9 s,
~1.2 GB GPU in the inference subprocess; 2-step LoRA train on 4 images at 64 px: 44 s, 0.16 GB peak.
Video (12 diffusion_video) uses hf-internal-testing/tiny-wan-pipe; see `tiny_wan_http`.
Tiny snapshots live in <tiny_root>/<repo name> ($WORKSPACE/hf_tiny), the layout diffusion_bench's
`run_edge.py --tiny-root` reads, so both suites share one download.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

from studio_regress.contract import StepFailed

TINY_SDXL_REPO = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
TINY_WAN_REPO = "hf-internal-testing/tiny-wan-pipe"
PROMPT = "a red square on a white background"


def _workspace():
    return Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[3])


def tiny_root() -> Path:
    """Shared root for hf-internal-testing tiny pipelines, one dir per repo name. Same layout
    diffusion_bench (`run_edge.py --tiny-root`, default $WORKSPACE/hf_tiny) expects."""
    return Path(os.environ.get("STUDIO_REGRESS_TINY_ROOT") or _workspace() / "hf_tiny")


def ensure_tiny(repo: str, dest=None) -> str:
    """Local snapshot of a tiny diffusers pipeline at <tiny_root>/<repo name> (downloads once).
    Studio refuses non-GGUF remote loads outside unsloth/*, so tiny fixtures load from a path."""
    dest = Path(dest or tiny_root() / repo.rsplit("/", 1)[-1])
    if not (dest / "model_index.json").is_file():
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_dir=str(dest),
                          allow_patterns=["*.json", "*.txt", "*.model", "*.bin", "*.safetensors"],
                          ignore_patterns=["*.onnx", "*.msgpack", "openvino*", "vae_decoder/*", "vae_encoder/*"])
    return str(dest)


def ensure_tiny_sdxl(dest=None) -> str:
    """Local snapshot of the tiny SDXL pipeline (~51 MB); $STUDIO_REGRESS_TINY_SDXL overrides."""
    return ensure_tiny(TINY_SDXL_REPO, dest or os.environ.get("STUDIO_REGRESS_TINY_SDXL"))


def ensure_tiny_wan(dest=None) -> str:
    """Local snapshot of the tiny Wan pipeline (~0.9 MB)."""
    return ensure_tiny(TINY_WAN_REPO, dest)


# Explicit speed_mode "off": an UNSET speed_mode on a dense image load defers torch.compile
# ("default") to the 3rd image, and on a video load resolves to "default" outright, so pixels for a
# fixed seed drift across repeats (the old "2 of 3 identical"). memory_mode "fast" = fully resident,
# no offload path chosen by a VRAM measurement that differs between hosts.
DETERMINISTIC_LOAD = {"speed_mode": "off", "memory_mode": "fast"}


def image_load_body(ctx) -> dict:
    return {"model_path": model_path(ctx), "model_kind": "pipeline", **DETERMINISTIC_LOAD}


def model_path(ctx) -> str:
    return (ctx.models or {}).get("tiny_sdxl") or ensure_tiny_sdxl()


# -- video --------------------------------------------------------------------------------------
# Studio's HTTP video route only takes the family's resolution presets (Wan2.2-TI2V-5B: 1280x704 /
# 704x1280; 256x256 is a 422), and tiny-wan-pipe ships rope_max_seq_len 32, i.e. at most 32 latent
# patches per axis (512 px at VAE 8x / patch 2): 1280x704 fails inside WanRotaryPosEmbed. The rope
# tables are non-persistent buffers computed from the config, so a copy whose config says 128 has
# the SAME weights and renders the presets (1280/8/2 = 80 <= 128).
VIDEO_FAMILY = "wan2.2-ti2v-5b"
TINY_WAN_ROPE = 128


def tiny_wan_http(src=None) -> str:
    """<tiny_root>/tiny-wan-pipe-rope128: config-only derivative of the tiny Wan snapshot `src`
    (a local dir or an HF-cache snapshot; symlinks are copied as files). Idempotent."""
    import json
    import shutil
    src = Path(src or ensure_tiny_wan())
    dest = tiny_root() / f"{TINY_WAN_REPO.rsplit('/', 1)[-1]}-rope{TINY_WAN_ROPE}"
    cfg = dest / "transformer" / "config.json"

    def done():
        return cfg.is_file() and json.loads(cfg.read_text()).get("rope_max_seq_len") == TINY_WAN_ROPE
    if done():
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Per-process tmp: two runs building the copy at once must not delete each other's half-copy.
    tmp = dest.with_name(f"{dest.name}.tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        shutil.copytree(src, tmp, ignore=shutil.ignore_patterns(".cache"))
        c = json.loads((tmp / "transformer" / "config.json").read_text())
        c["rope_max_seq_len"] = TINY_WAN_ROPE
        (tmp / "transformer" / "config.json").write_text(json.dumps(c, indent=2))
        if not done():   # another process may have finished its identical copy (and be using it)
            shutil.rmtree(dest, ignore_errors=True)
            try:
                tmp.rename(dest)
            except OSError:
                if not done():
                    raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return str(dest)


def video_model_path(ctx) -> str:
    return tiny_wan_http((ctx.models or {}).get("tiny_wan"))


def video_load_body(ctx) -> dict:
    return {"model_path": video_model_path(ctx), "model_kind": "pipeline",
            "family_override": VIDEO_FAMILY, **DETERMINISTIC_LOAD}


def _boxes(buf, start, end):
    i = start
    while i + 8 <= end:
        size = int.from_bytes(buf[i:i + 4], "big")
        kind = bytes(buf[i + 4:i + 8]).decode("latin-1")
        hdr = 8
        if size == 1:
            size, hdr = int.from_bytes(buf[i + 8:i + 16], "big"), 16
        elif size == 0:
            size = end - i
        if size < hdr:
            return
        yield kind, i + hdr, min(i + size, end)
        i += size


def mp4_video_info(data: bytes) -> dict:
    """Frame count / fps / size of the first video track, read from the MP4 boxes (stdlib only):
    stsz sample_count, mdhd timescale + duration, tkhd width/height. {} if not an MP4."""
    buf = memoryview(data)
    moov = next(((a, b) for k, a, b in _boxes(buf, 0, len(buf)) if k == "moov"), None)
    if not moov:
        return {}
    for k, a, b in _boxes(buf, *moov):
        if k != "trak":
            continue
        info, box = {}, {kk: (aa, bb) for kk, aa, bb in _boxes(buf, a, b)}
        if "tkhd" in box:
            ta, tb = box["tkhd"]
            info["width"] = int.from_bytes(buf[tb - 8:tb - 6], "big")
            info["height"] = int.from_bytes(buf[tb - 4:tb - 2], "big")
        mdia = {kk: (aa, bb) for kk, aa, bb in _boxes(buf, *box.get("mdia", (0, 0)))}
        if "hdlr" not in mdia or bytes(buf[mdia["hdlr"][0] + 8:mdia["hdlr"][0] + 12]) != b"vide":
            continue
        ma = mdia["mdhd"][0]
        if buf[ma] == 1:
            timescale = int.from_bytes(buf[ma + 20:ma + 24], "big")
            duration = int.from_bytes(buf[ma + 24:ma + 32], "big")
        else:
            timescale = int.from_bytes(buf[ma + 12:ma + 16], "big")
            duration = int.from_bytes(buf[ma + 16:ma + 20], "big")
        minf = {kk: (aa, bb) for kk, aa, bb in _boxes(buf, *mdia["minf"])}
        stbl = {kk: (aa, bb) for kk, aa, bb in _boxes(buf, *minf["stbl"])}
        sa = stbl["stsz"][0]
        frames = int.from_bytes(buf[sa + 8:sa + 12], "big")
        info["frames"] = frames
        if timescale and duration:
            info["fps"] = round(frames / (duration / timescale), 2)
            info["duration_s"] = round(duration / timescale, 3)
        return info
    return {}


def _deadline(ctx, timeout_s):
    """Monotonic deadline for an inner wait: `timeout_s` stretched by the host load factor, and never
    past the running step's budget (timeouts.inner), so the wait ends with its own diagnostic."""
    import time
    from studio_regress import timeouts
    return time.monotonic() + timeouts.inner(timeout_s, getattr(ctx, "state", None))


def _left(deadline):
    import time
    return deadline - time.monotonic()


def _backoff(interval, cap):
    """Poll interval: start short (a fast Studio answers in well under a second), grow to `cap`."""
    return min(cap, interval * 1.5)


async def wait_video_loaded(ctx, timeout_s=300):
    p, dl, iv = {}, _deadline(ctx, timeout_s), 0.25
    while True:
        s = await ctx.api.get("/api/inference/video/status")
        if s.get("loaded"):
            return s
        p = await ctx.api.get("/api/inference/video/load-progress")
        if p.get("phase") == "error" or p.get("error"):
            raise StepFailed(f"video load failed: {p}")
        if _left(dl) <= 0:
            raise StepFailed(f"video model did not load in time: {p}")
        await asyncio.sleep(iv)
        iv = _backoff(iv, 1.0)


# generate-progress phases while a clip is still being made.
VIDEO_RUNNING = ("queued", "denoise", "decode", "export", "running")


async def wait_video_done(ctx, timeout_s=300):
    """generate-progress until it leaves the running phases; returns the terminal record."""
    g, dl, iv = {}, _deadline(ctx, timeout_s), 0.2
    while True:
        g = await ctx.api.get("/api/inference/video/generate-progress")
        if not g.get("active") and g.get("phase") not in VIDEO_RUNNING:
            return g
        if _left(dl) <= 0:
            raise StepFailed(f"video generation still active: {g}")
        await asyncio.sleep(iv)
        iv = _backoff(iv, 0.5)


async def wait_images_loaded(ctx, timeout_s=300):
    p, dl, iv = {}, _deadline(ctx, timeout_s), 0.25
    while True:
        s = await ctx.api.get("/api/inference/images/status")
        if s.get("loaded"):
            return s
        p = await ctx.api.get("/api/inference/images/load-progress")
        if p.get("error") or p.get("status") == "error":
            raise StepFailed(f"image load failed: {p}")
        if _left(dl) <= 0:
            raise StepFailed(f"image model did not load in time: {p}")
        await asyncio.sleep(iv)
        iv = _backoff(iv, 2.0)


async def wait_generation_idle(ctx, timeout_s=180):
    p, dl, iv = {}, _deadline(ctx, timeout_s), 0.2
    while True:
        p = await ctx.api.get("/api/inference/images/generate-progress")
        if not p.get("active"):
            return p
        if _left(dl) <= 0:
            raise StepFailed(f"generation still active: {p}")
        await asyncio.sleep(iv)
        iv = _backoff(iv, 1.0)


async def wait_generation_started(ctx, gallery0, timeout_s=10):
    """After a Generate click: until generate-progress reports active, or the gallery already grew
    (a 1-step tiny image can finish between two polls). Replaces a blind 1 s sleep, which let
    wait_generation_idle read "idle" before a slow host had even started the job. Returns True
    when a start was seen; False after the (short) deadline, and the caller's gallery check then
    fails a Generate that never ran."""
    dl, iv = _deadline(ctx, timeout_s), 0.1
    while True:
        if (await ctx.api.get("/api/inference/images/generate-progress")).get("active"):
            return True
        if len(await gallery(ctx)) > gallery0:
            return True
        if _left(dl) <= 0:
            return False
        await asyncio.sleep(iv)
        iv = _backoff(iv, 0.5)


async def wait_result_shown(page, timeout_ms=10_000):
    """Until the page shows a generated image (gallery URL / blob / data URL) that finished
    decoding. Replaces fixed 1-1.5 s waits before a capture; bounded, never raises (the step's
    API facts carry the verdict, the shot is the page as it stands)."""
    from studio_regress import timeouts
    try:
        await page.wait_for_function(
            """() => [...document.querySelectorAll('img')].some(im => {
                 const s = im.currentSrc || im.src || '';
                 return (s.includes('/images/gallery') || s.startsWith('blob:') || s.startsWith('data:image'))
                        && im.complete && im.naturalWidth > 0; })""",
            timeout=timeouts.scaled_ms(timeout_ms))
        return True
    except Exception:
        return False


async def gallery(ctx):
    return (await ctx.api.get("/api/inference/images/gallery")).get("images", [])


async def image_sha(ctx, image):
    r = await ctx.api.raw("GET", image["url"])
    if r.status_code != 200:
        raise StepFailed(f"image file {image['url']} -> {r.status_code}")
    try:  # decoded pixels: the PNG container carries per-save metadata
        import io
        from PIL import Image
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        return hashlib.sha256(im.tobytes()).hexdigest()[:16], im.size
    except Exception:  # noqa: BLE001
        return hashlib.sha256(r.content).hexdigest()[:16], len(r.content)


async def mark_volatile(page, patterns):
    """Tag leaf elements whose text matches any regex with data-volatile (masked by the fixture's
    global `[data-volatile]` selector): run-specific paths, throughput, VRAM readings."""
    await page.evaluate(
        """pats => { const res = pats.map(p => new RegExp(p));
             for (const el of document.querySelectorAll('body *')) {
               if (el.children.length) continue;
               const t = (el.textContent || '').trim();
               if (t && res.some(r => r.test(t))) el.setAttribute('data-volatile', '1');
             } }""", list(patterns))


async def mask_results(page):
    """Result images / thumbnails are masked in screenshots (the random-weight pipeline paints noise and
    thumbnails scale it differently per layout); pixel identity is carried by the `image_sha` fact,
    which is bit-stable for a fixed seed once the load pins speed_mode off (DETERMINISTIC_LOAD). Before
    that, an unset speed_mode engaged torch.compile on the 3rd image, which is the old "2 of 3"."""
    await page.evaluate(
        """() => { for (const im of document.querySelectorAll('img')) {
             const s = im.currentSrc || im.src || '';
             if (s.includes('/images/gallery') || s.startsWith('blob:') || s.startsWith('data:image'))
               im.setAttribute('data-volatile', '1'); } }""")


async def dismiss_toasts(page):
    for btn in await page.get_by_role("button", name="Close toast").all():
        try:
            await btn.click(timeout=1000)
        except Exception:
            pass
