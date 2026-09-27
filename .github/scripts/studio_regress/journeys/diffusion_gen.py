"""Journey 10: image generation on the smallest pipeline Studio accepts (tiny SDXL, local).

Steps: load (API) -> Images page -> fixed params (prompt, 256x256, 1 step, cfg 0, seed 0) ->
Generate -> result + gallery facts (image sha for the same seed) -> image menu -> Recipe ->
long generation + Stop -> unload. Loads with speed_mode off (see _diffusion.DETERMINISTIC_LOAD).
Video is journeys/diffusion_video.py.
"""

from __future__ import annotations

import asyncio
import re

from studio_regress import timeouts
from studio_regress.contract import Journey, Step, StepFailed, StepUnreachable
from studio_regress.journeys import _diffusion as d

MASKS = ("[data-sonner-toaster]",)


async def load(ctx):
    await ctx.api.post("/api/inference/images/unload")
    await ctx.api.post("/api/inference/images/load", json=d.image_load_body(ctx))
    s = await d.wait_images_loaded(ctx)
    ctx.state["gallery0"] = len(await d.gallery(ctx))
    return {"family": s.get("family"), "model_kind": s.get("model_kind"), "device": s.get("device"),
            "dtype": s.get("dtype"), "workflows": s.get("workflows"), "supports_lora": s.get("supports_lora")}


async def open_images(ctx):
    p = ctx.page
    await p.goto(ctx.base_url + "/images", wait_until="domcontentloaded")
    try:
        await p.get_by_role("button", name="Generate", exact=True).first.wait_for(
            timeout=timeouts.scaled_ms(30_000))
    except Exception as e:
        raise StepUnreachable(f"/images has no Generate button: {e}") from None
    await d.dismiss_toasts(p)
    return {"url": p.url.replace(ctx.base_url, "")}


async def _fill(page, label, value):
    loc = page.locator(f'input[aria-label="{label}"], input[placeholder="{label}"]')
    if await loc.count() == 0:
        raise StepUnreachable(f"input {label!r} not found")
    await loc.first.fill(str(value))
    await loc.first.press("Tab")


async def set_params(ctx):
    p = ctx.page
    box = p.locator("textarea").first
    await box.fill(d.PROMPT)
    want = (("Width", 256), ("Height", 256), ("Steps", 1), ("Guidance", 0), ("Random if empty", 0))
    for label, value in want:
        await _fill(p, label, value)
    # Until the inputs hold the typed values (React re-renders on blur), not a fixed 300 ms; a field
    # the page clamps or rejects never matches, and the read below records what it shows instead.
    try:
        await p.wait_for_function(
            """want => want.every(([l, v]) => { const e = document.querySelector(
                 `input[aria-label="${l}"], input[placeholder="${l}"]`); return e && e.value === String(v); })""",
            arg=[list(w) for w in want[:4]], timeout=timeouts.scaled_ms(2_000))
    except Exception:
        pass
    vals = {}
    for label in ("Width", "Height", "Steps", "Guidance"):
        vals[label.lower()] = await p.locator(f'input[aria-label="{label}"], input[placeholder="{label}"]').first.input_value()
    return vals


async def generate(ctx):
    p = ctx.page
    await p.get_by_role("button", name="Generate", exact=True).first.click()
    # Until the job is seen (active, or already in the gallery), not a blind 1 s: a Generate that
    # never starts falls through after a short bound and fails the gallery check below.
    await d.wait_generation_started(ctx, ctx.state.get("gallery0", 0))
    await d.wait_generation_idle(ctx)
    imgs = await d.gallery(ctx)
    if len(imgs) <= ctx.state.get("gallery0", 0):
        raise StepFailed("Generate produced no gallery image")
    img = imgs[0]
    # Compared fact: with speed_mode off the decoded pixels for seed 0 are bit-identical (measured 5/5
    # in one load, again after a reload, and across four Studio processes on fresh homes).
    sha, size = await d.image_sha(ctx, img)
    await d.wait_result_shown(p)   # the result image decoded in the page, not a fixed 1.5 s
    await d.dismiss_toasts(p)
    await d.mask_results(p)
    return {"width": img.get("width"), "height": img.get("height"), "steps": img.get("steps"),
            "seed": img.get("seed"), "image_sha": sha, "decoded_size": size, "gallery_count": len(imgs)}


async def _hover_result(p):
    """The Recipe / Download / menu toolbar only renders while the result image is hovered."""
    imgs = p.locator("img")
    best, area = None, 0
    for i in range(await imgs.count()):
        b = await imgs.nth(i).bounding_box()
        if b and b["width"] * b["height"] > area:
            best, area = imgs.nth(i), b["width"] * b["height"]
    if best is not None:
        await best.hover()
        # The hover toolbar is up once its menu button is visible (was a fixed 300 ms).
        try:
            await p.get_by_role("button", name="More actions for this image").first.wait_for(
                state="visible", timeout=3_000)
        except Exception:
            pass


async def image_menu(ctx):
    p = ctx.page
    await _hover_result(p)
    btn = p.get_by_role("button", name="More actions for this image")
    if await btn.count() == 0:
        raise StepUnreachable("no image action menu")
    await btn.first.click()
    try:   # the menu has rendered once its first item is visible (was a fixed 500 ms)
        await p.get_by_role("menuitem").first.wait_for(state="visible", timeout=5_000)
    except Exception:
        pass
    items = [t.strip() for t in await p.get_by_role("menuitem").all_inner_texts()]
    await d.mask_results(p)
    return {"menu_items": items}


async def recipe(ctx):
    p = ctx.page
    await p.keyboard.press("Escape")
    await _hover_result(p)
    btn = p.get_by_role("button").filter(has_text=re.compile(r"^\s*Recipe\s*$"))
    if await btn.count() == 0:
        raise StepUnreachable("no Recipe button")
    await btn.first.click()
    dlg = p.locator("[role=dialog]")
    try:   # the Recipe dialog is open and has its text (was a fixed 700 ms)
        await dlg.first.wait_for(state="visible", timeout=5_000)
        await p.wait_for_function(
            "p => [...document.querySelectorAll('[role=dialog]')].some(d => d.innerText.includes(p))",
            arg=d.PROMPT, timeout=timeouts.scaled_ms(2_000))
    except Exception:
        pass
    text = (await dlg.first.inner_text()) if await dlg.count() else ""
    await d.mask_results(p)
    return {"recipe_has_prompt": d.PROMPT in text, "recipe_has_seed": "0" in text}


async def stop(ctx):
    p = ctx.page
    await p.keyboard.press("Escape")
    await _fill(p, "Steps", 100)
    await _fill(p, "Width", 1024)
    await _fill(p, "Height", 1024)
    await p.get_by_role("button", name="Generate", exact=True).first.click()
    active = False
    for _ in range(20):
        if (await ctx.api.get("/api/inference/images/generate-progress")).get("active"):
            active = True
            break
        await asyncio.sleep(0.25)
    stop_btn = p.get_by_role("button").filter(has_text="Stop")
    has_stop = await stop_btn.count() > 0
    if has_stop:
        await stop_btn.last.click()
    else:
        await ctx.api.post("/api/inference/images/generate/cancel")
    after = await d.wait_generation_idle(ctx, timeout_s=90)
    # The page is back to idle once Generate is enabled again (was a fixed 1 s).
    try:
        await p.wait_for_function(
            """() => [...document.querySelectorAll('button')].some(b => b.innerText.trim() === 'Generate'
                 && !b.disabled)""", timeout=timeouts.scaled_ms(10_000))
    except Exception:
        pass
    await d.dismiss_toasts(p)
    await d.mask_results(p)
    return {"was_active": active, "stop_control": has_stop, "after_active": bool(after.get("active"))}


async def unload(ctx):
    s = await ctx.api.post("/api/inference/images/unload")
    return {"loaded": s.get("loaded")}


JOURNEY = Journey(
    name="diffusion_gen", tier="gpu", needs=("tiny_sdxl",), routes=("/images",),
    steps=(
        Step("s01_load", load, shot=False, timeout_s=300),
        Step("s02_images_page", open_images, masks=MASKS),
        Step("s03_params", set_params, masks=MASKS),
        Step("s04_generate", generate, masks=MASKS, timeout_s=180),
        Step("s05_image_menu", image_menu, masks=MASKS),
        Step("s06_recipe", recipe, masks=MASKS),
        Step("s07_stop", stop, masks=MASKS, timeout_s=150),
        Step("s08_unload", unload, shot=False),
    ),
)
