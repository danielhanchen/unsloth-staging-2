"""Scene: how the Hub shows a repo whose scoped download is still running.

An image GGUF fetches its "Required assets" (text encoder, VAE) from a base repo through a SCOPED
job (variant "@scope"). While that job writes, the base repo holds an `.incomplete` blob, so the
On Device list drew the amber "Partial download, open it to finish" dot, and the repo's own card
(keyed on the snapshot job, which has none) showed a "Partial" tag and a "Resume" button. Users
read that as the download being paused while the panel and toast said it was running.

Both sides start the same scoped HTTP download through the API before the page loads, so the
frontend adopts it through `/api/hub/active-downloads` exactly as it does after a reload, then:
  0. hover the On Device row: its status dot and tooltip
  1. open the row: the detail card's status tag and action button
The download is checked to be STILL RUNNING after the shots (fact `running_after_shots`), since
a transfer that finished mid-scene photographs a different state on each side. It is then
cancelled.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
import urllib.parse
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session, api_get, api_post  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

DEFAULT_REPO = "unsloth/Qwen-Image-2.1-FP8"
DEFAULT_FILES = ("Qwen-Image-2.1-text_encoder-FP8.safetensors",)
SCOPE = "uidiff-required-assets"


def _status(session: Session, repo: str) -> dict:
    q = urllib.parse.urlencode({"repo_id": repo, "gguf_variant": f"@{SCOPE}"})
    return api_get(session, f"/api/hub/download-status?{q}")


def _progress(session: Session, repo: str, expected: int) -> dict:
    q = urllib.parse.urlencode(
        {"repo_id": repo, "variant": f"@{SCOPE}", "expected_bytes": expected}
    )
    return api_get(session, f"/api/hub/gguf-download-progress?{q}")


async def drive(session: Session, out_dir: Path, label: str, repo: str = DEFAULT_REPO,
                files: list[str] | tuple[str, ...] = DEFAULT_FILES, expected_bytes: int = 0,
                min_bytes: int = 200_000_000, cache_hub: str = "",
                record_video: bool = False,
                **_: object) -> tuple[list[Path], dict]:
    facts: dict = {}
    # Sides run one after the other on one cache: without a wipe AFTER would resume BEFORE's
    # partial and could finish before the page is up.
    repo_dir = Path(cache_hub) / ("models--" + repo.replace("/", "--")) if cache_hub else None
    if repo_dir is not None:
        shutil.rmtree(repo_dir, ignore_errors=True)
    start = api_post(session, "/api/hub/download", {
        "repo_id": repo,
        "gguf_variant": None,
        "scope_id": SCOPE,
        "files": list(files),
        "use_xet": False,
        "transport_mode": "http",
    })
    facts["start"] = {k: start.get(k) for k in ("state", "accepted", "attached")}
    t0 = time.time()
    while time.time() - t0 < 180:
        prog = _progress(session, repo, expected_bytes)
        if prog.get("downloaded_bytes", 0) >= min_bytes:
            break
        await asyncio.sleep(1)
    facts["bytes_before_page"] = prog.get("downloaded_bytes")
    facts["status_before_page"] = _status(session, repo).get("state")

    shots: list[Path] = []
    init = seed_init_script(
        type("A", (), {"access_token": session.access_token,
                       "refresh_token": session.refresh_token})(), []
    )
    short = repo.split("/")[-1]
    try:
        async with open_chat(session.base_url, init_scripts=[init],
                             viewport=(1500, 1000), headless=True,
                             video_dir=(out_dir / "video") if record_video else None,
                             video_name=f"{label.lower()}_session") as sp:
            page = sp.page
            await page.goto(f"{session.base_url}/hub", wait_until="domcontentloaded")
            on_device = page.get_by_role("radio", name="On Device").first
            await on_device.wait_for(state="visible", timeout=60_000)
            # The segmented control animates while the first inventory walk lands, and
            # Playwright's stability wait can starve on it; retry, then force.
            for attempt in range(4):
                try:
                    await page.wait_for_timeout(2_000)
                    await on_device.click(timeout=10_000, force=attempt == 3)
                    break
                except Exception:  # noqa: BLE001
                    if attempt == 3:
                        raise
            await page.wait_for_function(
                "() => !!document.querySelector('input[placeholder=\"Search on-device models\"]')",
                timeout=30_000,
            )
            search = page.get_by_placeholder("Search on-device models").first
            await search.wait_for(state="visible", timeout=60_000)
            await search.fill(short)
            name = page.get_by_role("button", name=short, exact=True).first
            await name.wait_for(state="visible", timeout=180_000)
            # Adoption of the running job (active-downloads) and the inventory walk both land
            # after first paint; settle so the sides differ by code, not by timing.
            await page.wait_for_timeout(6_000)

            row = name.locator("xpath=..")
            facts["row_status_dots"] = sorted({
                (await el.get_attribute("aria-label")) or ""
                for el in await row.locator('[role="img"]').all()
            } - {""})
            await name.hover()
            await page.wait_for_timeout(1_500)
            tips = await page.locator('[role="tooltip"]').all_inner_texts()
            facts["row_tooltip"] = [" ".join(t.split()) for t in tips if t.strip()]
            list_shot = out_dir / f"{label.lower()}_0_on_device_row.png"
            await page.screenshot(path=str(list_shot),
                                  clip={"x": 0, "y": 180, "width": 1500, "height": 320})
            shots.append(list_shot)

            await name.click()
            heading = page.get_by_role("heading", name=short).first
            await heading.wait_for(state="visible", timeout=60_000)
            await page.wait_for_timeout(4_000)
            main_text = " ".join((await page.locator("main").first.inner_text()).split())
            facts["detail_shows_partial_tag"] = " Partial " in f" {main_text} "
            facts["detail_shows_downloading_tag"] = "Downloading" in main_text
            facts["detail_action_buttons"] = sorted({
                " ".join(t.split())
                for t in await page.get_by_role("button").all_inner_texts()
                if t.strip() and any(w in t for w in ("Continue", "Download", "Resume", "%"))
            })
            aria = await page.locator(
                'button[aria-label="Pause download"], button[aria-label="Cancel download"]'
            ).count()
            facts["detail_has_stop_control"] = aria > 0
            detail_shot = out_dir / f"{label.lower()}_1_detail.png"
            await page.screenshot(path=str(detail_shot),
                                  clip={"x": 440, "y": 180, "width": 1060, "height": 360})
            shots.append(detail_shot)
        facts["running_after_shots"] = _status(session, repo).get("state") == "running"
        facts["bytes_after_shots"] = _progress(session, repo, expected_bytes).get(
            "downloaded_bytes"
        )
    finally:
        try:
            api_post(session, "/api/hub/download/cancel",
                     {"repo_id": repo, "gguf_variant": f"@{SCOPE}"})
        except Exception as exc:  # noqa: BLE001 -- recorded, the shots already exist
            facts["cancel_error"] = f"{type(exc).__name__}: {exc}"
        for _ in range(60):
            if _status(session, repo).get("state") not in ("running", "cancelling"):
                break
            await asyncio.sleep(1)
        if repo_dir is not None:
            shutil.rmtree(repo_dir, ignore_errors=True)
    return shots, facts


if __name__ == "__main__":
    import argparse

    from pr_ui_scenes._common import studio_session

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--home", type=Path, required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="AFTER")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s = studio_session(a.url, a.home, a.password)
    print(asyncio.run(drive(s, a.out, a.label)))
