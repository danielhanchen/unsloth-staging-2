"""Scene: the Video tab's "Loaded build" summary after an AUTO-precision load (PR 13618).

PR 13618 makes LTX-2.5's auto precision seed the hosted INT8-ConvRot DiT even where the bf16
DiT fits resident, as ComfyUI's LTX-2.5 templates do. Nothing in the frontend changes; what
moves is the backend's resolved ``transformer_quant``, which the Video page's "Loaded build"
summary reads from ``/api/inference/video/status`` (Transformer row). On a card where bf16
fits, BEFORE reports the dense checkpoint and AFTER reports INT8.

The load is real (``needs_model``): weights come from the shared HF cache, offline.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or os.environ.get("UNSLOTH_WORKSPACE") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session, api_get, api_post  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

DEFAULT_REPO = "Lightricks/LTX-2.5-Diffusers"


def _load_and_wait(session: Session, body: dict, timeout_s: int = 2400) -> dict:
    """POST the load, then poll status: /video/load returns before the load finishes."""
    api_post(session, "/api/inference/video/unload", {}, timeout=300)
    api_post(session, "/api/inference/video/load", body, timeout=timeout_s)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(10)
        status = api_get(session, "/api/inference/video/status")
        if status.get("loaded"):
            return status
        progress = api_get(session, "/api/inference/video/load-progress")
        if progress.get("error"):
            raise RuntimeError(f"load failed: {str(progress['error'])[:400]}")
    raise RuntimeError(f"model did not load within {timeout_s}s")


async def drive(session: Session, out_dir: Path, label: str,
                repo: str = DEFAULT_REPO, **_: object) -> tuple[list[Path], dict]:
    status = _load_and_wait(session, {"model_path": repo, "model_kind": "pipeline"})
    facts = {k: status.get(k) for k in
             ("loaded", "repo_id", "family", "model_kind", "transformer_quant",
              "text_encoder_quant", "dtype", "device", "offload_policy")}
    shots: list[Path] = []
    init = seed_init_script(
        type("A", (), {"access_token": session.access_token,
                       "refresh_token": session.refresh_token})(), [],
        # The summary sits in the right-docked Advanced panel, closed by default.
        extra_local_storage={"unsloth_video_advanced_open": "true"},
    )
    async with open_chat(session.base_url, init_scripts=[init],
                         viewport=(1600, 1000), headless=True) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/video", wait_until="domcontentloaded")
        heading = page.get_by_text("Loaded build").first
        await heading.wait_for(state="visible", timeout=120_000)
        await page.wait_for_timeout(2_000)
        box = heading.locator("xpath=..")
        text = await box.inner_text()
        facts["loaded_build_text"] = " | ".join(t.strip() for t in text.splitlines() if t.strip())
        if "Transformer" not in facts["loaded_build_text"]:
            raise RuntimeError(f"Loaded build panel not found: {facts['loaded_build_text']!r}")
        shot = out_dir / f"{label.lower()}_loaded_build.png"
        await box.screenshot(path=str(shot))
        shots.append(shot)
        full = out_dir / f"{label.lower()}_page.png"
        await sp.screenshot(full, full_page=False)
        shots.append(full)
    api_post(session, "/api/inference/video/unload", {}, timeout=300)
    return shots, facts
