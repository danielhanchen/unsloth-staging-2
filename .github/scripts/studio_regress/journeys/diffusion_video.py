"""Journey 12: text-to-video on the tiny Wan pipeline, modelled on diffusion_bench (unslothai/scripts
#330) edge checks video.sanity and video.cancel.

Fixture: hf-internal-testing/tiny-wan-pipe (0.9 MB, random weights), loaded as a LOCAL pipeline with
family_override wan2.2-ti2v-5b, speed_mode off, memory_mode fast. Over HTTP the Wan family only takes
its presets (1280x704 / 704x1280; 256x256 answers 422) and tiny Wan's 32-entry rope table cannot
cover 1280x704, so the journey loads the config-only rope-128 copy (`_diffusion.tiny_wan_http`).

Steps: load (API) -> /video page with the model -> 9-frame 2-step seed-0 clip (API; the Duration
select starts at 25 frames) whose MP4 must hold 9 frames at 24 fps -> the clip in the page is a
playable <video> -> Generate from the UI (100 steps, 121 frames) and press Cancel mid-denoise; the
page must return to idle with the model still loaded and no clip saved -> unload.
Measured on B200 (shared GPU): 30-71 s for all 6 steps (load 2-4 s, 9-frame clip 3-15 s, cancel 14-40 s).
Peak for the Studio process tree: 8.5 GB when memory_mode fast stays resident, 1.5-3.1 GB when the planner
(which prices the real 5B against free VRAM) picks model offload.
"""

from __future__ import annotations

import asyncio
import re

from studio_regress import timeouts
from studio_regress.contract import Journey, Step, StepFailed, StepUnreachable
from studio_regress.journeys import _diffusion as d

MASKS = ("[data-sonner-toaster]",)
PROMPT = "a monarch butterfly opening and closing its wings on a purple flower"
CANCEL_RE = re.compile(r"^\s*Cancel\s*$")
CLIP = {"width": 1280, "height": 704, "num_frames": 9, "steps": 2, "guidance": 1.0, "seed": 0}


async def load(ctx):
    await ctx.api.post("/api/inference/images/unload")
    await ctx.api.post("/api/inference/video/unload")
    r = await ctx.api.raw("POST", "/api/inference/video/load", json=d.video_load_body(ctx))
    if r.status_code != 200:
        raise StepFailed(f"video load {r.status_code}: {r.text[:300]}")
    s = await d.wait_video_loaded(ctx)
    dfl = s.get("defaults") or {}
    return {"family": s.get("family"), "model_kind": s.get("model_kind"), "speed_mode": s.get("speed_mode"),
            "memory_mode": s.get("memory_mode"),
            # advisory: "fast" is priced from the family size table (the real 5B, ~24 GB + clip headroom)
            # against FREE VRAM, so a shared GPU flips it between none and model offload
            "_offload_policy": s.get("offload_policy"),
            "speed_optims": s.get("speed_optims"), "resolution_presets": dfl.get("resolution_presets"),
            "frame_step": dfl.get("frame_step"), "default_fps": dfl.get("fps")}


async def _mask_media(page):
    """Clip pixels (random weights) and the header's Offload value (host free-VRAM dependent)."""
    await page.evaluate(
        """() => { for (const v of document.querySelectorAll('video')) v.setAttribute('data-volatile', '1');
             for (const el of document.querySelectorAll('body *')) {
               if (el.children.length || (el.textContent || '').trim() !== 'Offload') continue;
               const val = el.nextElementSibling;
               if (val) val.setAttribute('data-volatile', '1'); } }""")
    await d.mask_results(page)


async def video_page(ctx):
    p = ctx.page
    await p.goto(ctx.base_url + "/video", wait_until="domcontentloaded")
    try:
        await p.get_by_role("button", name="Generate", exact=True).first.wait_for(
            timeout=timeouts.scaled_ms(60_000))
    except Exception as e:
        raise StepUnreachable(f"/video has no Generate button: {e}") from None
    await d.dismiss_toasts(p)
    combos = [(await c.inner_text()).strip() for c in await p.get_by_role("combobox").all()]
    body = await p.locator("body").inner_text()
    await _mask_media(p)
    return {"url": p.url.replace(ctx.base_url, ""), "model_shown": d.video_model_path(ctx).rsplit("/", 1)[-1] in body,
            "selects": combos[:2]}


async def clip(ctx):
    """#330 video.sanity at the only size HTTP accepts: the MP4 itself must hold 9 frames at 24 fps."""
    off = await ctx.api.raw("POST", "/api/inference/video/generate", json={"prompt": PROMPT, **CLIP, "width": 256,
                                                                          "height": 256})
    r = await ctx.api.raw("POST", "/api/inference/video/generate", json={"prompt": PROMPT, **CLIP})
    if r.status_code != 200:
        raise StepFailed(f"video generate {r.status_code}: {r.text[:300]}")
    g = await d.wait_video_done(ctx, timeout_s=300)
    vid = g.get("video")
    if g.get("phase") != "completed" or not vid:
        raise StepFailed(f"clip ended {g.get('phase')}: {g.get('error')}")
    f = await ctx.api.raw("GET", vid["url"])
    if f.status_code != 200:
        raise StepFailed(f"clip file {vid['url']} -> {f.status_code}")
    mp4 = d.mp4_video_info(f.content)
    if mp4.get("frames") != CLIP["num_frames"]:
        raise StepFailed(f"MP4 holds {mp4.get('frames')} frames, asked {CLIP['num_frames']}: {mp4}")
    ctx.state["clip_id"] = vid["id"]
    return {"off_preset_256_status": off.status_code, "width": vid.get("width"), "height": vid.get("height"),
            "num_frames": vid.get("num_frames"), "fps": vid.get("fps"), "duration_s": vid.get("duration_s"),
            "steps": vid.get("steps"), "seed": vid.get("seed"), "mp4_frames": mp4.get("frames"),
            "mp4_fps": mp4.get("fps"), "mp4_size": [mp4.get("width"), mp4.get("height")]}


async def clip_in_ui(ctx):
    p = ctx.page
    await p.reload(wait_until="domcontentloaded")
    vid = ctx.state.get("clip_id") or ""
    info = None
    probe = """id => { const v = [...document.querySelectorAll('video')]
                   .find(e => (e.currentSrc || e.src || '').includes(id));
                 return v ? {ready: v.readyState, duration: v.duration, w: v.videoWidth, h: v.videoHeight,
                             error: v.error ? v.error.code : null} : null; }"""
    # In the page, until the clip's <video> has data (readyState >= 2) or reports a media error:
    # no 0.5 s polling from Python. Bounded like the old 60 x 0.5 s loop, stretched by host load.
    try:
        await p.wait_for_function(
            """id => [...document.querySelectorAll('video')].some(e => (e.currentSrc || e.src || '').includes(id)
                 && (e.readyState >= 2 || e.error))""", arg=vid, timeout=timeouts.inner(30, ctx.state) * 1000)
    except Exception:
        pass
    info = await p.evaluate(probe, vid)
    if not info:
        raise StepFailed("the new clip has no <video> element on /video")
    await d.dismiss_toasts(p)
    await _mask_media(p)
    return {"playable": info["ready"] >= 2 and info["error"] is None,
            "duration_s": round(info["duration"] or 0, 3), "video_size": [info["w"], info["h"]]}


async def _fill(page, selector, value):
    loc = page.locator(selector)
    if await loc.count() == 0:
        raise StepUnreachable(f"input {selector} not found")
    await loc.first.fill(str(value))
    await loc.first.press("Tab")


async def cancel(ctx):
    """#330 video.cancel through the page: Generate a long clip, press Cancel (the button that
    replaces Generate while a clip runs) mid-denoise."""
    p = ctx.page
    videos0 = len((await ctx.api.get("/api/inference/video/gallery")).get("videos", []))
    await p.locator("textarea").first.fill(PROMPT)
    await _fill(p, 'input[aria-label="Steps"]', 100)
    await _fill(p, 'input[placeholder="Random if empty"]', 0)
    await p.get_by_role("button", name="Generate", exact=True).first.click()
    prog, dl = {}, d._deadline(ctx, 60)
    while True:   # until the first denoise step (the old 240 x 0.25 s bound, load-scaled)
        prog = await ctx.api.get("/api/inference/video/generate-progress")
        if prog.get("active") and int(prog.get("step") or 0) >= 1:
            break
        if prog.get("phase") == "error" or d._left(dl) <= 0:
            break   # never started / failed at once: recorded below (was_active False)
        await asyncio.sleep(0.2)
    was_active = bool(prog.get("active"))
    # visible text "Cancel"; its accessible name is not exactly that, so match the text
    cancel_btn = p.get_by_role("button").filter(has_text=CANCEL_RE)
    try:  # the page swaps Generate for Cancel on its own progress poll, a beat after the API
        await cancel_btn.first.wait_for(timeout=timeouts.scaled_ms(10_000))
        has_cancel = True
    except Exception:
        has_cancel = False
    if has_cancel:
        await cancel_btn.first.click()
    else:
        await ctx.api.post("/api/inference/video/generate/cancel")
    after = await d.wait_video_done(ctx, timeout_s=120)
    gen = p.get_by_role("button", name="Generate", exact=True).first
    try:
        await gen.wait_for(timeout=timeouts.scaled_ms(30_000))
        # Until Generate is enabled again, in the page (the old 60 x 0.5 s loop, load-scaled).
        try:
            await p.wait_for_function(
                """() => [...document.querySelectorAll('button')].some(b => b.innerText.trim() === 'Generate'
                     && !b.disabled)""", timeout=timeouts.inner(30, ctx.state) * 1000)
        except Exception:
            pass
        idle = await gen.is_enabled()
    except Exception:
        idle = False
    # The cancel outcome toast arrives on the page's own progress poll, a beat after Generate is
    # back; this wait is what lets both sides show (and dismiss) it the same way. No state to wait on.
    await p.wait_for_timeout(1000)
    await d.dismiss_toasts(p)
    await _mask_media(p)
    s = await ctx.api.get("/api/inference/video/status")
    videos = (await ctx.api.get("/api/inference/video/gallery")).get("videos", [])
    return {"was_active": was_active, "cancel_control": has_cancel, "after_active": bool(after.get("active")),
            "after_phase": after.get("phase"), "cancel_reported": "cancel" in str(after.get("error") or "").lower(),
            "generate_idle": idle,
            "cancel_gone": await p.get_by_role("button").filter(has_text=CANCEL_RE).count() == 0,
            "still_loaded": bool(s.get("loaded")), "clips_saved": len(videos) - videos0}


async def unload(ctx):
    s = await ctx.api.post("/api/inference/video/unload")
    return {"loaded": s.get("loaded")}


JOURNEY = Journey(
    name="diffusion_video", tier="gpu", needs=("tiny_wan",), routes=("/video",),
    steps=(
        Step("s01_load", load, shot=False, timeout_s=300),
        Step("s02_video_page", video_page, masks=MASKS),
        Step("s03_clip", clip, shot=False, timeout_s=330),
        Step("s04_clip_in_ui", clip_in_ui, masks=MASKS, timeout_s=60),
        Step("s05_cancel", cancel, masks=MASKS, timeout_s=240),
        Step("s06_unload", unload, shot=False),
    ),
)
