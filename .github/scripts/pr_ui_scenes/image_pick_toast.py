"""Scene: what the Images page tells the user in the seconds after picking an uncached model, for PR 11626.

Before the PR, a Hub pick of a model that is not on disk stages its download plan and raises no toast
until ``handleLoad`` runs, which is after the download finishes: seconds to minutes of silence. The PR
raises a pick toast before the plan request ("Preparing download..."), shows the pick's own download
progress, and hands the same toast to the load.

The evidence is a time series, not one frame: the scene polls the Sonner toaster every 250 ms from the
click and records when the first toast appeared and what it said. The shot is taken at a fixed moment
after the pick, so both halves are photographed at the same elapsed time.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import time
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

DEFAULT_REPO = "unsloth/Z-Image-Turbo-GGUF"
DEFAULT_QUANT = "Q4_K_M"
# Every repo the pick can stage: the GGUF plus the companion base it borrows VAE / encoder from.
DEFAULT_PURGE = ("unsloth/Z-Image-Turbo-GGUF", "Tongyi-MAI/Z-Image-Turbo", "unsloth/Z-Image-Turbo")


def _purge(cache_hub: Path, repos) -> list[str]:
    """Empty the shared cache of this pick, so neither side finds it on disk (the second side would
    otherwise load at once and raise the load toast for a reason that is not the fix)."""
    removed = []
    for repo in repos:
        d = cache_hub / ("models--" + repo.replace("/", "--"))
        if d.exists():
            shutil.rmtree(d, ignore_errors = True)
            removed.append(repo)
    # The Xet chunk cache sits beside the hub cache and survives it: a warm one lets the second
    # side finish in seconds what the first side fetched cold.
    xet = cache_hub.parent / "xet"
    if xet.exists():
        shutil.rmtree(xet, ignore_errors = True)
        removed.append("xet chunk cache")
    return removed


async def _toasts(page) -> list[str]:
    try:
        return [t.strip() for t in await page.locator("[data-sonner-toast]").all_inner_texts() if t.strip()]
    except Exception:  # noqa: BLE001 -- toaster not mounted yet
        return []


async def drive(session: Session, out_dir: Path, label: str, repo: str = DEFAULT_REPO,
                quant: str = DEFAULT_QUANT, cache_hub: str = "", purge = DEFAULT_PURGE,
                shot_at_s: float = 8.0, watch_s: float = 20.0, **_: object) -> tuple[list[Path], dict]:
    facts: dict = {}
    if cache_hub:
        facts["purged"] = _purge(Path(cache_hub), purge)
    shots: list[Path] = []
    init = seed_init_script(
        type("A", (), {"access_token": session.access_token,
                       "refresh_token": session.refresh_token})(), []
    )
    async with open_chat(session.base_url, init_scripts = [init],
                         viewport = (1600, 1000), headless = True) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/images", wait_until = "domcontentloaded")
        await page.wait_for_timeout(6_000)
        facts["toasts_before_pick"] = await _toasts(page)
        await page.get_by_role("button", name = "Select image model").first.click()
        await page.wait_for_timeout(3_000)
        # The row reads "<name> (Slow)" for a GGUF repo, so match the start of the label.
        await page.get_by_text(re.compile("^" + re.escape(repo.split("/")[-1]) + r"(\s|$)")).first.click()
        await page.wait_for_timeout(3_000)
        quant_row = page.get_by_text(quant, exact = True).first
        await quant_row.wait_for(timeout = 30_000)

        t0 = time.monotonic()
        await quant_row.click()
        timeline: list[tuple[float, str]] = []
        shot_taken = False
        while (elapsed := time.monotonic() - t0) < watch_s:
            texts = await _toasts(page)
            joined = " || ".join(texts)
            if texts and (not timeline or timeline[-1][1] != joined):
                timeline.append((round(elapsed, 2), joined[:300]))
            if not shot_taken and elapsed >= shot_at_s:
                shot = out_dir / f"{label.lower()}_page_{int(shot_at_s)}s.png"
                await sp.screenshot(shot, full_page = False)
                shots.append(shot)
                facts["toasts_at_shot"] = texts
                shot_taken = True
            await page.wait_for_timeout(250)
        facts["first_toast_s"] = timeline[0][0] if timeline else None
        facts["first_toast_text"] = timeline[0][1] if timeline else None
        facts["toast_timeline"] = timeline
        facts["shown_progress"] = any("of" in t and "B" in t for _, t in timeline)
        try:
            facts["download_panel"] = " | ".join(
                t.strip() for t in (await page.locator(".hub-download-panel").first.inner_text(timeout = 3_000)).splitlines()
                if t.strip()
            )[:300]
        except Exception:  # noqa: BLE001 -- panel absent is a fact too
            facts["download_panel"] = None

        # Stop the transfer: a multi-GB checkpoint nobody loads is this scene's only side effect.
        cancels = page.get_by_role("button", name = "Cancel download")
        for i in range(await cancels.count()):
            try:
                await cancels.nth(i).click(timeout = 5_000)
            except Exception:  # noqa: BLE001 -- already finished or unmounted
                pass
        await page.wait_for_timeout(3_000)
        facts["toasts_after_cancel"] = await _toasts(page)
    return shots, facts
