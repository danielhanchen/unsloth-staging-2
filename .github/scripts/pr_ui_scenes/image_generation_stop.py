"""Scene: the Images composer while a generation is running (PR 8219, issue 8187).

Video generation has always had a cancel; images had none, so a 50 step run at batch 4
could only be waited out. This PR adds `POST /images/generate/cancel` and a Stop control
that replaces Generate for the duration of the run.

Two pairs, because "a Stop button exists" is only half the claim:

  0  a few seconds into the run: BEFORE has no way to stop it, AFTER shows Stop
  1  a few seconds after Stop is clicked: AFTER is back to Generate with the run
     cancelled, BEFORE is still going

The run has to be made long enough to photograph twice. Steps and batch size are pushed
to their maxima first: at the defaults a turbo model on this box finishes before the
second shot, and a run that ended on its own is indistinguishable in a screenshot from a
run that was stopped.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import urllib.error
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or os.environ.get("UNSLOTH_WORKSPACE") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session, api_get, api_post  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

DEFAULT_REPO = "unsloth/Z-Image-Turbo-GGUF"
DEFAULT_FILE = "z-image-turbo-Q4_K_M.gguf"

# Generate, Stop, or whatever the button says mid-run. Matching all three is what lets
# ONE locator find the action row on both sides, so the two shots frame the same thing.
_ACTION = re.compile(r"^(Generate|Stop|Generating)", re.I)


def _load_and_wait(session: Session, body: dict, timeout_s: int = 900) -> dict:
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


async def _crank(page, index: int, presses: int = 60) -> None:
    """Push a slider to its maximum with the keyboard, which is exact where a drag is not."""
    slider = page.get_by_role("slider").nth(index)
    await slider.click()
    for _ in range(presses):
        await page.keyboard.press("ArrowRight")


async def drive(session: Session, out_dir: Path, label: str,
                repo: str = DEFAULT_REPO, filename: str = DEFAULT_FILE,
                **_: object) -> tuple[list[Path], dict]:
    status = _load_and_wait(session, {"model_path": repo, "gguf_filename": filename,
                                      "model_kind": "gguf"})
    facts: dict = {"loaded": status.get("loaded"), "gguf_variant": status.get("gguf_variant")}
    # The route itself, before any clicking: it is the half of the claim a picture cannot
    # carry. 404 on the base, an answer on the head.
    try:
        api_post(session, "/api/inference/images/generate/cancel", {}, timeout=60)
        facts["cancel_route"] = "200"
    except urllib.error.HTTPError as exc:
        facts["cancel_route"] = str(exc.code)

    shots: list[Path] = []
    init = seed_init_script(
        type("A", (), {"access_token": session.access_token,
                       "refresh_token": session.refresh_token})(), []
    )
    async with open_chat(session.base_url, init_scripts=[init],
                         viewport=(1600, 1000), headless=True) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/images", wait_until="domcontentloaded")
        await page.wait_for_timeout(8_000)
        await _crank(page, 0)   # Steps
        await _crank(page, 2)   # Batch size
        await page.get_by_role("button", name="Generate", exact=True).first.click()
        await page.wait_for_timeout(6_000)

        progress = api_get(session, "/api/inference/images/generate-progress")
        facts["running"] = {"active": progress.get("active"),
                            "total_steps": progress.get("total_steps")}
        if not progress.get("active"):
            raise RuntimeError(
                "no generation is in flight six seconds after Generate; there is nothing "
                f"to photograph. progress={json.dumps(progress)}"
            )

        action = page.get_by_role("button").filter(has_text=_ACTION).last
        row = action.locator("xpath=ancestor::div[1]")
        shot = out_dir / f"{label.lower()}_running.png"
        await row.screenshot(path=str(shot))
        shots.append(shot)
        facts["action_text"] = (await action.inner_text()).strip()

        # Filtered on TEXT, not on the accessible name: `get_by_role("button", name="Stop",
        # exact=True)` matched nothing here even though the button's inner_text is exactly
        # "Stop" -- the icon inside it contributes to the accessible name. That silently
        # skipped the click and left the second pair showing two still-running sides.
        stop = page.get_by_role("button").filter(has_text=re.compile(r"^\s*Stop\s*$", re.I))
        facts["stop_control"] = bool(await stop.count())
        if facts["stop_control"]:
            await stop.last.click()

        # Polled to inactive, not a fixed sleep. Measured on this box: a batch of 4 keeps
        # going for about 20 s after the click (the cancel lands at a step boundary and a
        # batched step is slow), while a batch of 1 stops within 4 s. A fixed 8 s wait
        # photographed AFTER still showing Stop, which reads as "the button does nothing".
        waited = 0
        after = api_get(session, "/api/inference/images/generate-progress")
        while after.get("active") and waited < 60:
            await page.wait_for_timeout(4_000)
            waited += 4
            after = api_get(session, "/api/inference/images/generate-progress")
        facts["after_stop"] = {"active": after.get("active"), "step": after.get("step"),
                               "seconds_waited": waited}
        shot2 = out_dir / f"{label.lower()}_after_stop.png"
        await page.get_by_role("button").filter(has_text=_ACTION).last.locator(
            "xpath=ancestor::div[1]").screenshot(path=str(shot2))
        shots.append(shot2)

    # The BEFORE run cannot be stopped from the UI -- that is the finding -- so drop it
    # here rather than leave a batch of 4 at 50 steps grinding on a shared GPU.
    api_post(session, "/api/inference/images/unload", {}, timeout=300)
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
