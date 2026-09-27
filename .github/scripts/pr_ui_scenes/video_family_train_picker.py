"""Scene: the Images -> Train tab's "Model family" picker (PR 8196).

PR 8196 adds LTX-2, the first VIDEO family, to the flow-matching DiT LoRA trainer.
The trainer, the routing and the preflight are backend, but one listing decides
whether any of it is reachable by hand: ``family_train_infos()``, which is what
``GET /api/train/diffusion/info`` returns and what the Train panel's "Model family"
Select is built from.

Before this PR that listing walked ``trainable_family_names()``, the IMAGE registry
alone, so a video family could not appear in it however complete its trainer was.
The PR replaces that with ``_all_trainable_family_names()`` -- the image registry's
trainable families UNION ``TRAINABLE_VIDEO_FAMILIES`` -- and drops any family whose
pipeline class the installed diffusers lacks (``family_pipeline_available``).

That last clause is why this scene reads the API as well as photographing the
dropdown: on a diffusers older than 0.39 there is no ``LTX2Pipeline`` and the head
would honestly show no new row. The facts record the diffusers version so a pair
can never be mistaken for the wrong reason.

Listings only: no weights, no GPU, no disk.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

WORKSPACE = Path(os.environ.get("WORKSPACE") or os.environ.get("UNSLOTH_WORKSPACE") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session, api_get, open_menu  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

# The option to open the menu ON: a value the list CONTAINS on BOTH sides, never a
# category word. "SDXL (U-Net)" is the one family that predates every DiT and is in
# the frontend's own preset fallback, so it is present even if /info answers nothing.
ANCHOR_OPTION = re.compile(r"SDXL")

# The row the PR adds. Backend label is exactly "LTX-2" (_FAMILY_LABELS["ltx-2"]).
TARGET_LABEL = "LTX-2"

# The left settings column, top: family select, its chips, and the base repo under it.
# A fixed clip identical on both sides -- a full 1500 px viewport renders at ~440 px
# per half in a GitHub comment, where 11 px chip text is unreadable.
COLUMN_CLIP = {"x": 0, "y": 46, "width": 640, "height": 560}


def _diffusers_version(session: Session, pipeline_attr: str = "LTX2Pipeline") -> str:
    """The version of diffusers THIS install carries, read off the same home.

    ``family_train_infos`` hides a family whose pipeline class diffusers lacks, so an
    AFTER side on an old diffusers would show no new row for a reason that has
    nothing to do with the PR. Record it rather than assume it.
    """
    import subprocess

    py = session.home / "unsloth_studio" / "bin" / "python"
    if not py.exists():
        return "unknown (no venv python)"
    out = subprocess.run(
        [str(py), "-c",
         "import diffusers,json;print(json.dumps([diffusers.__version__,"
         f"hasattr(diffusers,{pipeline_attr!r})]))"],
        text=True, capture_output=True)
    for line in reversed(out.stdout.strip().splitlines()):
        try:
            ver, has = json.loads(line)
            return f"{ver} ({pipeline_attr}: {has})"
        except Exception:  # noqa: BLE001 -- diffusers prints warnings on stdout
            continue
    return f"unknown ({out.stderr.strip()[-160:]})"


async def drive(session: Session, out_dir: Path, label: str,
                target: str = TARGET_LABEL,
                family_key: str = "ltx-2",
                pipeline_attr: str = "LTX2Pipeline",
                clip: Optional[dict] = None,
                seed_clips: int = 0,
                **_: object) -> tuple[list[Path], dict]:
    """Photograph the open Model family list, then the panel after picking `target`.

    `clip` overrides the panel crop. The default keeps 8196's framing (the top of the
    settings column). A taller clip that reaches the foot of the column also catches the
    floating Start control, which is worth having in the same frame when the question is
    not only "is the family listed" but "can it be started".
    """
    facts: dict = {"diffusers": _diffusers_version(session, pipeline_attr),
                   "target": target, "family_key": family_key}

    if seed_clips:
        # A family that trains from CLIPS is withheld from this picker while every listed
        # dataset is stills, so with no clip folder both sides show the same list for a
        # reason that has nothing to do with the family registry. Seeded identically on
        # BOTH sides through clip_dataset_picker's own helper, rather than a second copy
        # of it here: the point of the pair is the family list, so the datasets behind it
        # have to match or the comparison is not one.
        from pr_ui_scenes.clip_dataset_picker import _seed_datasets
        facts["seeded"] = _seed_datasets(session.home, n_images=2, n_clips=seed_clips)

    # The numeric half, from the same server we photograph. The claim is a listing
    # membership, so read the listing.
    try:
        info = api_get(session, "/api/train/diffusion/info")
        fams = info.get("families", [])
        facts["family_count"] = len(fams)
        facts["family_names"] = [f.get("name") for f in fams]
        hit = next((f for f in fams if str(f.get("name", "")).lower() == family_key), None)
        facts["target_in_api"] = hit is not None
        if hit is not None:
            facts["target_info"] = {
                "label": hit.get("label"), "base_repos": hit.get("base_repos"),
                "defaults": hit.get("defaults"), "vram_note": hit.get("vram_note"),
                "params": hit.get("params"), "qlora_vram_gb": hit.get("qlora_vram_gb"),
                "note": hit.get("note"), "deploy_base": hit.get("deploy_base"),
                "precision_modes": hit.get("precision_modes"),
                "recommended_precision": hit.get("recommended_precision"),
            }
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
        # PillTabs keeps real tab roles, so this is unambiguous -- unlike a text match
        # on "Train", which also hits the panel heading and the Start training button.
        await page.get_by_role("tab", name="Train").first.click(timeout=60_000)
        # /info is fetched by the panel on mount and the Select is built from it; a shot
        # taken before it lands shows the frontend's 4-row preset fallback on BOTH sides.
        await page.get_by_role("combobox", name="Model family").first.wait_for(
            state="visible", timeout=60_000)
        await page.wait_for_timeout(6_000)

        trigger = page.get_by_role("combobox", name="Model family").first
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
        facts["target_in_menu"] = any(target.lower() in t.lower() for t in labels)

        shot = out_dir / f"{label.lower()}_family_menu.png"
        await listbox.screenshot(path=str(shot))
        shots.append(shot)

        # Same driving script on both sides, different outcome: on the base there is no
        # LTX-2 row to click, so the panel keeps its default family. That IS the result;
        # do not "fix" it by branching earlier.
        picked = page.get_by_role("option", name=re.compile(re.escape(target)))
        if await picked.count():
            await picked.first.click()
        else:
            await page.keyboard.press("Escape")
        await page.wait_for_timeout(3_000)
        facts["family_after_pick"] = (await trigger.inner_text()).strip()

        # The floating Start control, read on BOTH sides. A family can be listed and
        # selectable and still be unstartable: the panel disables Start whenever the
        # backend advertises an empty precision_modes for a non-sdxl family. Recording
        # its text and disabled state is what turns "the row is there" into "the row
        # works", and it is not visible in a clip that stops above the button.
        try:
            start_btn = page.locator("button").filter(has_text=re.compile(
                "Start training|Not supported on this GPU|Training in progress|Starting"
            )).first
            facts["start_text"] = (await start_btn.inner_text()).strip()
            facts["start_disabled"] = await start_btn.is_disabled()
        except Exception as exc:  # noqa: BLE001 -- the screenshot is still worth taking
            facts["start_error"] = f"{type(exc).__name__}: {exc}"[:200]

        panel = out_dir / f"{label.lower()}_panel.png"
        await page.screenshot(path=str(panel), clip=clip or COLUMN_CLIP)
        shots.append(panel)

        full = out_dir / f"{label.lower()}_page.png"
        await sp.screenshot(full, full_page=False)
        shots.append(full)

    print(f"[{label}] {json.dumps(facts)[:700]}", flush=True)
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
