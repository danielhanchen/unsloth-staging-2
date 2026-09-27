"""Journey 11: diffusion LoRA training on tiny data + the tiny SDXL pipeline, then use the LoRA.

Steps: upload a 4-image captioned dataset (API, multipart) -> Images > Train tab (form, dataset
thumbnails) -> start a 2-step LoRA run on the local tiny SDXL (API: the UI base-model picker
only offers hub bases, which are GB-scale) -> watch the Train panel reach "complete" (loss /
grad-norm charts; throughput, VRAM and run paths masked) -> adapter listed in
/api/models/diffusion-loras -> Deploy to Create -> generate with the LoRA applied.
Measured: 44 s for the train step on B200, 0.16 GB peak.
"""

from __future__ import annotations

import asyncio
import io
import math

from studio_regress import timeouts
from studio_regress.contract import Journey, Step, StepFailed, StepUnreachable
from studio_regress.journeys import _diffusion as d

DATASET = "sr_tiny_imgs"
ADAPTER = "sr_tiny_lora"
VOLATILE = (r"^Saved: ", r"img/s$", r"^\d+(\.\d+)? GB$", r"/outputs/", r"/loras/")
MASKS = ("[data-sonner-toaster]",)


def _images():
    from PIL import Image, ImageDraw
    files = []
    for i, col in enumerate(("red", "green", "blue", "yellow")):
        im = Image.new("RGB", (64, 64), "white")
        ImageDraw.Draw(im).rectangle([16, 16, 48, 48], fill=col)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        files.append(("files", (f"img{i}.png", buf.getvalue(), "image/png")))
        files.append(("files", (f"img{i}.txt", f"a {col} square, sks style".encode(), "text/plain")))
    return files


async def upload(ctx):
    r = await ctx.api.raw("POST", "/api/train/diffusion/dataset", data={"name": DATASET}, files=_images())
    if r.status_code != 200:
        raise StepFailed(f"dataset upload {r.status_code}: {r.text[:200]}")
    j = r.json()
    return {"image_count": j.get("image_count"), "caption_count": j.get("caption_count")}


async def train_tab(ctx):
    p = ctx.page
    await p.goto(ctx.base_url + "/images", wait_until="domcontentloaded")
    tab = p.get_by_role("tab", name="Train")
    try:
        await tab.wait_for(timeout=timeouts.scaled_ms(30_000))
    except Exception as e:
        raise StepUnreachable(f"Images > Train tab missing: {e}") from None
    await tab.click()
    await p.get_by_role("button", name="Start training").wait_for(timeout=timeouts.scaled_ms(20_000))
    await d.dismiss_toasts(p)
    fam = p.get_by_role("combobox").first
    return {"family_picker": (await fam.inner_text()).strip() if await fam.count() else None}


async def start(ctx):
    await ctx.api.post("/api/inference/images/unload")
    body = {"base_model": d.model_path(ctx), "data_dir": DATASET, "output_dir": ADAPTER, "model_family": "sdxl",
            "resolution": 64, "train_steps": 2, "lora_rank": 4, "seed": 0, "cache_latents": False, "save_steps": 2}
    r = await ctx.api.raw("POST", "/api/train/diffusion/start", json=body)
    if r.status_code != 200:
        raise StepFailed(f"diffusion start {r.status_code}: {r.text[:300]}")
    ctx.state["job_id"] = r.json().get("job_id")
    return {"status": r.json().get("status")}


async def complete(ctx):
    s, dl, iv = {}, d._deadline(ctx, 600), 0.5
    while True:   # terminal status ends the wait at once (error / failed / stopped fail below)
        s = await ctx.api.get("/api/train/diffusion/status")
        if s.get("status") in ("completed", "error", "failed", "stopped") or d._left(dl) <= 0:
            break
        await asyncio.sleep(iv)
        iv = d._backoff(iv, 2.0)
    if s.get("status") != "completed":
        raise StepFailed(f"diffusion training ended {s.get('status')}: {s.get('message')}")
    # "completed" is reported while the trainer subprocess is still exiting (`active` = process alive),
    # and until it exits every image load answers 409, so a Deploy click in that window never loads.
    dl = d._deadline(ctx, 120)
    while s.get("active") and d._left(dl) > 0:
        await asyncio.sleep(0.25)
        s = await ctx.api.get("/api/train/diffusion/status")
    if s.get("active"):
        raise StepFailed("trainer process still alive 120 s (load-scaled) after completion")
    loss = s.get("loss")
    if not (isinstance(loss, (int, float)) and math.isfinite(loss)):
        raise StepFailed(f"non-finite loss {loss}")
    p = ctx.page
    await p.reload(wait_until="domcontentloaded")
    await p.get_by_role("tab", name="Train").click()
    try:
        await p.get_by_text("Training Complete", exact=False).first.wait_for(timeout=timeouts.scaled_ms(30_000))
    except Exception:
        raise StepFailed("Train panel never showed completion") from None
    await d.mark_volatile(p, VOLATILE)
    return {"step": s.get("step"), "total_steps": s.get("total_steps"), "num_images": s.get("num_images"),
            "family": s.get("family"), "loss_finite": True, "grad_norm_finite": math.isfinite(s.get("grad_norm") or float("nan")),
            "lora_saved": bool(s.get("lora_path"))}


async def lora_listed(ctx):
    loras = (await ctx.api.get("/api/models/diffusion-loras", params={"family": "sdxl"})).get("loras", [])
    mine = [l for l in loras if l.get("id") == ADAPTER]
    if not mine:
        raise StepFailed("trained adapter missing from the LoRA picker listing")
    return {"families": mine[0].get("families"), "format": mine[0].get("format")}


async def deploy(ctx):
    p = ctx.page
    btn = p.get_by_role("button", name="Deploy to Create")
    if await btn.count() == 0:
        raise StepUnreachable("no Deploy to Create button")
    await btn.first.click()
    # Deploy loads the trained base in the background: settle on "loaded" so both sides are
    # photographed in the same state, not mid-load.
    s = await d.wait_images_loaded(ctx, timeout_s=180)
    try:
        await p.get_by_role("button", name="Cancel load").wait_for(state="hidden",
                                                                   timeout=timeouts.scaled_ms(30_000))
    except Exception:
        pass
    # The "<model> loaded" toast lands on the page's own status poll after the API says loaded;
    # this wait lets it arrive so dismiss_toasts clears it on both sides. No state to wait on.
    await p.wait_for_timeout(1000)
    await d.dismiss_toasts(p)
    await d.mask_results(p)
    body = (await p.locator("body").inner_text())
    values = await p.evaluate("() => [...document.querySelectorAll('input')].map(i => i.value)")
    return {"lora_chip_shown": any(v.startswith(ADAPTER) for v in values), "on_create": "Create images" in body,
            "deployed_family": s.get("family"), "deployed_loras": s.get("loras") or s.get("active_loras")}


async def generate_with_lora(ctx):
    # Deploy to Create starts its own load of the trained base (a second load meanwhile 409s). That
    # UI load leaves speed_mode unset, so once it settles, reload with the deterministic body unless
    # it already is one: the LoRA is applied per request, so nothing the deploy set is lost.
    s = await ctx.api.get("/api/inference/images/status")
    if not s.get("loaded"):
        r = await ctx.api.raw("POST", "/api/inference/images/load", json=d.image_load_body(ctx))
        if r.status_code not in (200, 409):
            raise StepFailed(f"image load {r.status_code}: {r.text[:200]}")
    s = await d.wait_images_loaded(ctx)
    base_after_deploy, deploy_speed = s.get("family"), s.get("speed_mode")
    if (s.get("speed_mode"), s.get("memory_mode")) != ("off", "fast"):
        await ctx.api.post("/api/inference/images/unload")
        r = await ctx.api.raw("POST", "/api/inference/images/load", json=d.image_load_body(ctx))
        if r.status_code != 200:
            raise StepFailed(f"image reload {r.status_code}: {r.text[:200]}")
        s = await d.wait_images_loaded(ctx)
    r = await ctx.api.raw("POST", "/api/inference/images/generate", timeout=300, json={
        "prompt": "a red square, sks style", "width": 256, "height": 256, "steps": 1, "guidance": 0.0, "seed": 0,
        "loras": [{"id": ADAPTER, "weight": 1.0}]})
    if r.status_code != 200:
        raise StepFailed(f"generate with LoRA {r.status_code}: {r.text[:300]}")
    img = r.json()["images"][0]
    sha, _ = await d.image_sha(ctx, img)
    await ctx.api.post("/api/inference/images/unload")
    # _image_sha stays advisory: the LoRA weights come from a GPU training run. Observed identical in
    # 3 of 3 separate Studio processes; promote once more runs (and another GPU) agree.
    return {"base_after_deploy": base_after_deploy, "_deploy_speed_mode": deploy_speed, "speed_mode": s.get("speed_mode"),
            "_image_sha": sha, "loras": [l.get("id") if isinstance(l, dict) else l for l in img.get("loras") or []]}


JOURNEY = Journey(
    name="diffusion_train", tier="gpu", needs=("tiny_sdxl",), routes=("/images",),
    steps=(
        Step("s01_upload", upload, shot=False),
        Step("s02_train_tab", train_tab, masks=MASKS),
        Step("s03_start", start, shot=False),
        Step("s04_complete", complete, masks=MASKS, timeout_s=660),
        Step("s05_lora_listed", lora_listed, shot=False),
        Step("s06_deploy", deploy, masks=MASKS, timeout_s=240),
        Step("s07_generate_with_lora", generate_with_lora, shot=False, timeout_s=360),
    ),
)
