"""Scene: an oversized image generation that the memory guard used to refuse (PR 11798).

Reported from Unsloth Desktop: a 2x Upscale to 2048x2048 was refused as too big for the GPU while
ComfyUI ran it on the same card, and the only way past was an environment variable a desktop
install cannot set. The PR tiles the VAE for such a call instead of refusing, and gives the
remaining refusals an in-app override.

Three readings per side, all against the same Studio:

1. API: a real 2x Upscale (1024 -> 2048) via /images/generate with init_image + upscale, the
   workflow that was reported. Status and detail go into the facts.
2. UI: Create at 2048x2048 on the Images page, which hits the same guard. BEFORE shows the refusal
   toast naming the env var; AFTER renders the image.
3. UI, frontend only: the SAME tagged 400 (the refusal text the head backend produces for a request
   that cannot run even tiled) is served to both pages through Playwright routing, so the toast
   difference is the frontend's alone: AFTER offers "Generate anyway". Plus the Advanced panel,
   which gains the "Allow oversized generations" switch.

GPU 4 is held down with gpu_ballast.py so the card reads like a ~14 GiB consumer card on both
sides; free_mib is recorded per side.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
import urllib.error
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session, api_get, api_post  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

DEFAULT_REPO = "unsloth/Z-Image-Turbo-GGUF"
DEFAULT_FILE = "z-image-turbo-Q4_K_M.gguf"
SRC = WORKSPACE / "outputs" / "upscale_gate" / "branch_12g_src.png"
_TOAST = "[data-sonner-toast]"

# What the head backend says for a request that cannot run even tiled (Qwen-Image-2.1, ten 2048
# condition images, 16 GB card): produced by image_activation_verdict, pasted verbatim.
MOCK_DETAIL = os.environ.get("UPSCALE_GATE_MOCK_DETAIL", "")


def _free_mib(gpu_index: int) -> int:
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
         "-i", str(gpu_index)],
        text=True, capture_output=True, check=True).stdout
    return int(out.strip().splitlines()[0])


def _load_and_wait(session: Session, body: dict, timeout_s: int = 1200) -> dict:
    api_post(session, "/api/inference/images/unload", {}, timeout=300)
    api_post(session, "/api/inference/images/load", body, timeout=timeout_s)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(5)
        status = api_get(session, "/api/inference/images/status")
        if status.get("loaded"):
            return status
        progress = api_get(session, "/api/inference/images/load-progress")
        if progress.get("error"):
            raise RuntimeError(f"load failed: {str(progress['error'])[:400]}")
    raise RuntimeError(f"model did not load within {timeout_s}s")


def _api_generate(session: Session, body: dict) -> dict:
    try:
        out = api_post(session, "/api/inference/images/generate", body, timeout=1800)
        img = (out.get("images") or [{}])[0]
        return {"status": 200, "width": img.get("width"), "height": img.get("height"),
                "workflow": img.get("workflow")}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "refusal_header": exc.headers.get("X-Unsloth-Refusal"),
                "detail": exc.read().decode()[:900]}


async def _wait_generation(page, session: Session, timeout_s: int = 900) -> str:
    """Until a toast shows or a started run finishes. Returns what ended the wait."""
    toast = page.locator(_TOAST)
    seen_active = False
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if await toast.count() > 0 and await toast.last.is_visible():
            return "toast"
        try:
            active = bool(api_get(session, "/api/inference/images/generate-progress").get("active"))
        except Exception:  # noqa: BLE001
            active = False
        if active:
            seen_active = True
        elif seen_active:
            return "finished"
        await page.wait_for_timeout(2_000)
    return "timeout"


async def drive(session: Session, out_dir: Path, label: str,
                repo: str = DEFAULT_REPO, filename: str = DEFAULT_FILE,
                gpu_index: int = 4, record_video: bool = False,
                **_: object) -> tuple[list[Path], dict]:
    status = _load_and_wait(session, {"model_path": repo, "gguf_filename": filename,
                                      "model_kind": "gguf"})
    facts: dict = {"loaded": status.get("loaded"), "offload_policy": status.get("offload_policy"),
                   "free_mib": _free_mib(gpu_index)}

    # 1. The reported workflow, through the API.
    src_b64 = base64.b64encode(SRC.read_bytes()).decode()
    facts["api_upscale_2x"] = _api_generate(session, {
        "prompt": "a red fox in a snowy forest, detailed fur", "steps": 8, "seed": 7,
        "init_image": src_b64, "upscale": 2.0})

    shots: list[Path] = []
    init = seed_init_script(
        type("A", (), {"access_token": session.access_token,
                       "refresh_token": session.refresh_token})(), []
    )
    async with open_chat(session.base_url, init_scripts=[init],
                         viewport=(1600, 1000), headless=True,
                         video_dir=(out_dir / "video") if record_video else None,
                         video_name=f"{label.lower()}_session") as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/images", wait_until="domcontentloaded")
        await page.wait_for_timeout(8_000)
        prompt = page.get_by_placeholder("Describe", exact=False).first
        if await prompt.count():
            await prompt.fill("a red fox in a snowy forest, detailed fur")

        # 2. Create at 2048x2048 on the real backend.
        for name, value in (("Width", 2048), ("Height", 2048)):
            box = page.get_by_label(name, exact=True).first
            await box.fill(str(value))
            await box.press("Tab")
        await page.wait_for_timeout(1_000)
        await page.get_by_role("button", name="Generate", exact=True).first.click()
        facts["ui_2048_ended_by"] = await _wait_generation(page, session)
        await page.wait_for_timeout(2_500)
        toast = page.locator(_TOAST)
        facts["ui_2048_toast"] = (
            (await toast.last.inner_text()).replace("\n", " ")[:700] if await toast.count() else None)
        shot = out_dir / f"{label.lower()}_1_create_2048.png"
        await sp.screenshot(shot, full_page=False)
        shots.append(shot)
        # Dismiss whatever toast is up before the next reading.
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(1_500)
        for t in await toast.all():
            try:
                await t.hover()
                close = t.locator("[data-close-button]")
                if await close.count():
                    await close.first.click()
            except Exception:  # noqa: BLE001
                pass
        await page.wait_for_timeout(1_500)

        # 3. Same tagged 400 on both sides: the toast difference is the frontend's alone.
        if MOCK_DETAIL:
            async def _refuse(route):
                await route.fulfill(
                    status=400,
                    headers={"Content-Type": "application/json",
                             "X-Unsloth-Refusal": "memory-estimate"},
                    body=json.dumps({"detail": MOCK_DETAIL}))

            await page.route("**/api/inference/images/generate", _refuse)
            await page.get_by_role("button", name="Generate", exact=True).first.click()
            await page.locator(_TOAST).last.wait_for(state="visible", timeout=60_000)
            await page.wait_for_timeout(1_500)
            t = page.locator(_TOAST).last
            facts["mock_refusal_toast"] = (await t.inner_text()).replace("\n", " ")[:900]
            facts["mock_refusal_has_generate_anyway"] = (
                await t.get_by_text("Generate anyway").count()) > 0
            shot = out_dir / f"{label.lower()}_2_refusal_toast.png"
            await t.screenshot(path=str(shot))
            shots.append(shot)
            await page.unroute("**/api/inference/images/generate")

        # 4. The Advanced panel.
        adv = page.get_by_role("button", name="Advanced", exact=True).first
        if (await adv.get_attribute("aria-expanded")) != "true":
            await adv.click()
            await page.wait_for_timeout(1_000)
        panel = adv.locator("xpath=..")
        facts["advanced_has_allow_oversized"] = (
            await panel.get_by_text("Allow oversized generations").count()) > 0
        switch_row = panel.get_by_text("CPU offload", exact=False).first
        await switch_row.scroll_into_view_if_needed()
        await page.wait_for_timeout(800)
        shot = out_dir / f"{label.lower()}_3_advanced.png"
        await panel.screenshot(path=str(shot))
        shots.append(shot)

    print(f"[{label}] {json.dumps(facts)}", flush=True)
    api_post(session, "/api/inference/images/unload", {}, timeout=300)
    return shots, facts
