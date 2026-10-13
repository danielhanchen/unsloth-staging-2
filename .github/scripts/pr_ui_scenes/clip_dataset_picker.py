"""Scene: the Images -> Train tab's "Training images" dataset picker.

The clip-dataset PR widens Studio's diffusion dataset layer from images to video
clips. Everything about it is backend: ``_diffusion_dataset_summary`` counts clips,
``GET /api/train/diffusion/info`` admits a folder on "an image OR a clip" instead of
"an image", and the upload allowlist accepts a container beside its ``.txt`` sidecar.
Zero of that is visible anywhere except in one Select: the "Training images" picker,
which is built entirely from ``info.datasets``.

So the scene seeds TWO folders into the Studio home's datasets root -- an image set
and a clip set -- and photographs the open picker on both sides. The image set is the
control: it must appear identically on BOTH halves. If it does not, the shot is
telling you the panel failed to load, not that the PR did something.

The clips are real MP4s produced by ffmpeg, not stub bytes. Nothing in this path
decodes them, but a scene that seeds a file the product would reject is exactly the
kind of evidence that looks fine and proves nothing.

Listings only: no weights, no GPU, no model download.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

WORKSPACE = Path(os.environ.get("WORKSPACE") or os.environ.get("UNSLOTH_WORKSPACE") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session, api_get, open_menu  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

IMAGE_DATASET = "photo-style"
CLIP_DATASET = "clip-style"

# The row that must be present on BOTH sides. Opening the menu on a value only the
# AFTER side contains would time out on BEFORE, and the run would read as a broken
# selector rather than as the finding.
ANCHOR_OPTION = re.compile(re.escape(IMAGE_DATASET))

# The left settings column around the dataset picker. A full 1500 px viewport renders
# at ~440 px per half in a GitHub comment, where the row text is unreadable.
COLUMN_CLIP = {"x": 0, "y": 46, "width": 640, "height": 620}


def _seed_datasets(home: Path, n_images: int = 3, n_clips: int = 3) -> dict:
    """Write an image folder and a clip folder into this home's datasets root.

    Both get a ``<stem>.txt`` sidecar per item, which is the caption rule the two kinds
    share and the reason the clip half needed no new caption code at all.
    """
    from PIL import Image

    root = home / "assets" / "datasets"
    facts: dict = {"datasets_root": str(root)}

    images = root / IMAGE_DATASET
    images.mkdir(parents=True, exist_ok=True)
    for i in range(n_images):
        Image.new("RGB", (256, 256), (40 + 60 * i, 90, 200 - 40 * i)).save(
            images / f"still_{i}.png", format="PNG")
        (images / f"still_{i}.txt").write_text(f"a flat colour field, take {i}", "utf-8")

    clips = root / CLIP_DATASET
    clips.mkdir(parents=True, exist_ok=True)
    made = 0
    for i in range(n_clips):
        dest = clips / f"clip_{i}.mp4"
        proc = subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi",
             "-i", f"testsrc=size=256x256:rate=24:duration=1,hue=h={i * 60}",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
             "-shortest", str(dest)],
            capture_output=True, text=True)
        if dest.is_file() and dest.stat().st_size > 0:
            made += 1
        else:
            facts.setdefault("ffmpeg_error", proc.stderr.strip()[-300:])
        (clips / f"clip_{i}.txt").write_text(f"a test pattern turning hue, take {i}", "utf-8")

    facts["seeded_images"] = n_images
    facts["seeded_clips"] = made
    facts["clip_bytes"] = [p.stat().st_size for p in sorted(clips.glob("*.mp4"))]
    return facts


async def drive(session: Session, out_dir: Path, label: str,
                clip: Optional[dict] = None,
                **_: object) -> tuple[list[Path], dict]:
    """Seed both folders, then photograph the open "Training images" list."""
    facts: dict = _seed_datasets(session.home)

    # The numeric half, read from the same server that gets photographed. The claim is
    # a listing membership, so read the listing rather than counting rows by eye.
    try:
        info = api_get(session, "/api/train/diffusion/info")
        rows = info.get("datasets", [])
        facts["api_dataset_names"] = [d.get("name") for d in rows]
        facts["api_dataset_count"] = len(rows)
        facts["api_rows"] = [
            {k: d.get(k) for k in ("name", "image_count", "clip_count", "caption_count")}
            for d in rows
        ]
        facts["clip_dataset_in_api"] = any(
            d.get("name") == CLIP_DATASET for d in rows)
    except Exception as exc:  # noqa: BLE001 -- the screenshot is still worth taking
        facts["info_error"] = f"{type(exc).__name__}: {exc}"[:300]

    shots: list[Path] = []
    init = seed_init_script(
        type("A", (), {"access_token": session.access_token,
                       "refresh_token": session.refresh_token})(), []
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    async with open_chat(session.base_url, init_scripts=[init],
                         viewport=(1500, 1000), headless=True) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/images", wait_until="domcontentloaded")
        # PillTabs keeps real tab roles, so this is unambiguous -- unlike a text match on
        # "Train", which also hits the panel heading and the Start training button.
        await page.get_by_role("tab", name="Train").first.click(timeout=60_000)
        trigger = page.get_by_role("combobox", name="Training images").first
        await trigger.wait_for(state="visible", timeout=60_000)
        # /info is fetched on mount and the Select is built from it; a shot taken before
        # it lands shows an empty picker on BOTH sides.
        await page.wait_for_timeout(6_000)

        listbox = page.locator("[role=listbox]").first
        anchor = page.get_by_role("option", name=ANCHOR_OPTION).first
        # Radix menus lose races with the panel's background /info refresh: retry the
        # trigger rather than hunting for a better selector.
        await open_menu(page, trigger, anchor)
        await page.wait_for_timeout(1_500)

        options = page.get_by_role("option")
        labels = [(await options.nth(i).inner_text()).strip()
                  for i in range(await options.count())]
        facts["menu_options"] = labels
        facts["menu_option_count"] = len(labels)
        facts["clip_dataset_in_menu"] = any(CLIP_DATASET in t for t in labels)
        facts["image_dataset_in_menu"] = any(IMAGE_DATASET in t for t in labels)

        shot = out_dir / f"{label.lower()}_dataset_menu.png"
        await listbox.screenshot(path=str(shot))
        shots.append(shot)

        # The same driving script on both sides, different outcome: on the base there is
        # no clip-style row to click, so the picker keeps whatever it had. That IS the
        # result; do not branch earlier to "fix" it.
        picked = page.get_by_role("option", name=re.compile(re.escape(CLIP_DATASET)))
        if await picked.count():
            await picked.first.click()
        else:
            await page.keyboard.press("Escape")
        await page.wait_for_timeout(3_000)
        facts["dataset_after_pick"] = (await trigger.inner_text()).strip()

        panel = out_dir / f"{label.lower()}_panel.png"
        await page.screenshot(path=str(panel), clip=clip or COLUMN_CLIP)
        shots.append(panel)

        full = out_dir / f"{label.lower()}_page.png"
        await sp.screenshot(full, full_page=False)
        shots.append(full)

    print(f"[{label}] {json.dumps(facts)[:900]}", flush=True)
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
