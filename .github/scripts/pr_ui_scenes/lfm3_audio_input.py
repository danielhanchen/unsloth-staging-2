"""Metadata-derived presentation evidence; no model is loaded and no generation runs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from pr_ui_scenes._common import Session


def _detect(checkout: Path, fixture: Path, python: str) -> dict:
    backend = checkout.resolve() / "studio" / "backend"
    program = """
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
from utils.models import model_config
from utils.audio_tokens import is_audio_input_type
assert pathlib.Path(model_config.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]))
audio_type, definitive = model_config.detect_audio_type_checked(sys.argv[2], local_files_only=True)
print('SCENE_RESULT ' + json.dumps({'audio_type': audio_type, 'definitive': definitive,
                                  'has_audio_input': is_audio_input_type(audio_type)}))
"""
    result = subprocess.run(
        [python, "-c", program, str(backend), str(fixture.resolve())],
        cwd = checkout,
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"},
        capture_output = True,
        text = True,
        timeout = 60,
    )
    result.check_returncode()
    rows = [
        line.removeprefix("SCENE_RESULT ")
        for line in result.stdout.splitlines()
        if line.startswith("SCENE_RESULT ")
    ]
    if len(rows) != 1:
        raise RuntimeError(f"detector did not return one result: {result.stdout[-2000:]}")
    detected = json.loads(rows[0])
    if not detected["definitive"]:
        raise RuntimeError(f"metadata probe was inconclusive: {detected}")
    return detected


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    before_checkout: str,
    after_checkout: str,
    fixture: str,
    detector_python: str = "python",
    repo: str = "LiquidAI/LFM3-Omni-2B",
    **_: object,
) -> tuple[list[Path], dict]:
    from playwright.async_api import async_playwright
    from studio_test_kit.auth import seed_init_script

    checkout = Path(before_checkout if label.upper() == "BEFORE" else after_checkout).resolve()
    out_dir.mkdir(parents = True, exist_ok = True)
    metadata = out_dir / "metadata_fixture"
    metadata.mkdir(exist_ok = True)
    hashes = {}
    for name in ("config.json", "tokenizer_config.json", "processor_config.json"):
        source = Path(fixture).resolve() / name
        shutil.copyfile(source, metadata / name)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    detected = await asyncio.to_thread(_detect, checkout, metadata, detector_python)
    status = {
        "active_model": repo,
        "model_identifier": repo,
        "is_vision": bool(
            json.loads((metadata / "processor_config.json").read_text(encoding = "utf-8")).get(
                "image_processor"
            )
        ),
        "is_gguf": False,
        "is_audio": bool(detected["audio_type"]),
        "audio_type": detected["audio_type"],
        "has_audio_input": detected["has_audio_input"],
        "loading": [],
        "loaded": [repo],
        "context_length": 2048,
        "supports_reasoning": False,
        "supports_tools": False,
    }
    (out_dir / "metadata_status_fixture.json").write_text(
        json.dumps(status, indent = 2), encoding = "utf-8"
    )
    facts = {
        "evidence_scope": "metadata-derived presentation; status response intercepted; no model load or generation",
        "metadata_sha256": hashes,
        "detector_result": detected,
        "_detector_checkout": str(checkout),
        "_status_fixture": str(out_dir / "metadata_status_fixture.json"),
    }
    init = seed_init_script(session, [])
    status_reads = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless = True)
        context = await browser.new_context(viewport = {"width": 1440, "height": 1000})
        try:
            await context.add_init_script(init)

            async def supply_status(route):
                status_reads.append(route.request.url)
                await route.fulfill(status = 200, json = status)

            await context.route("**/api/inference/status*", supply_status)
            page = await context.new_page()
            await page.goto(f"{session.base_url}/chat", wait_until = "domcontentloaded")
            composer = page.locator("form:has(textarea)").first
            await composer.locator("textarea").wait_for(state = "visible", timeout = 30_000)
            selected = page.get_by_role("button").filter(has_text = repo.rsplit("/", 1)[-1]).first
            await selected.wait_for(state = "visible", timeout = 30_000)
            facts["selected_model_text"] = await selected.inner_text()
            await page.goto(
                f"{session.base_url}/chat?compare=5cb1eed7-e09e-4e17-a855-54b5638fc9fa",
                wait_until = "domcontentloaded",
            )
            await page.locator("textarea:visible").first.wait_for(state = "visible", timeout = 30_000)
            await (
                page.get_by_role("button", name = "Tools and attachments", exact = True)
                .filter(visible = True)
                .click()
            )
            menu = page.get_by_role("menu").last
            await menu.wait_for(state = "visible", timeout = 10_000)
            await menu.get_by_role("menuitem", name = "Browser use", exact = True).wait_for(
                state = "visible", timeout = 30_000
            )
            upload = menu.get_by_role("menuitem", name = "Upload audio", exact = True)
            visible = await upload.count() > 0 and await upload.is_visible()
            if visible != detected["has_audio_input"]:
                raise RuntimeError(
                    f"frontend audio control disagrees with detector: {detected}, visible={visible}"
                )
            if not status_reads:
                raise RuntimeError("frontend never consumed the metadata-derived status")
            await menu.evaluate(
                "el => Promise.all(el.getAnimations({subtree:true}).map(a => a.finished.catch(() => {})))"
            )
            facts["upload_audio_visible"] = visible
            facts["surface"] = "Compare chat tools and attachments menu"
            facts["menu_text"] = await menu.inner_text()
            facts["_menu_box"] = await menu.bounding_box()
            facts["_intercepted_status_requests"] = len(status_reads)
            shot = out_dir / f"{label.lower()}_audio_input_menu.png"
            await page.screenshot(path = str(shot), full_page = False)
        finally:
            await context.close()
            await browser.close()
    return [shot], facts
