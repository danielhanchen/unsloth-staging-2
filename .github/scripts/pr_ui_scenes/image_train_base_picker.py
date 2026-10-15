"""Scene: the Images -> Train tab's "Base model" picker and its facts chips (PR 8267).

PR 8267 changes what the FLUX.2 Klein family offers to train against. Before it,
``train_base_repos`` held one entry, ``black-forest-labs/FLUX.2-klein-4B``. After it
there are two, ``FLUX.2-klein-base-4B`` and ``FLUX.2-klein-base-9B``, because the
undistilled bases are what the upstream fine-tuning guidance points at, and each is
paired to its own inference base through the new ``deploy_base_repos``.

The second half of the change is the chips. ``_FAMILY_TRAIN_SPECS["flux.2-klein"]``
advertises ``params: "4B"`` and ``qlora_vram_gb: 10`` for the whole family, which
understates the 9B option badly: the auto policy measures 16.4 GB for its bf16 text
encoder alone. ``_BASE_TRAIN_SPECS`` now overlays ``9B`` / ``18`` on that one base,
and ``FamilyFacts`` takes ``baseModel``, so the chips move when the base is picked.

So the pair has to show both: the list gaining a row, and the chips changing when the
9B row is selected. Photographing only the open list would miss the half that matters
to a user deciding whether their card fits.

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

# Open the FAMILY menu on a value the list contains on both sides. SDXL predates every
# DiT and is in the frontend's own preset fallback, so it survives even if /info is slow.
FAMILY_ANCHOR = re.compile(r"SDXL")

# Open the BASE menu on the one entry the panel always appends, on both sides. Anchoring
# on a klein repo id would anchor on the very thing the PR changes, and anchoring on a
# category word would open the wrong dropdown (trap 4).
BASE_ANCHOR = re.compile(r"Custom repo or local path")

FAMILY_LABEL = "FLUX.2 Klein"
TARGET_BASE = "FLUX.2-klein-base-9B"

# Family select, its chips, and the base select under it. A fixed clip identical on both
# sides: a full 1500 px viewport renders at ~440 px per half in a GitHub comment, where
# the 11 px chip text is unreadable.
COLUMN_CLIP = {"x": 0, "y": 46, "width": 640, "height": 560}


async def drive(session: Session, out_dir: Path, label: str,
                family_label: str = FAMILY_LABEL,
                family_key: str = "flux.2-klein",
                target_base: str = TARGET_BASE,
                clip: Optional[dict] = None,
                **_: object) -> tuple[list[Path], dict]:
    """Photograph the open Base model list, then the panel after picking `target_base`."""
    facts: dict = {"family_key": family_key, "target_base": target_base}

    # The numeric half, from the same server we photograph. The claim is a listing
    # membership plus two chip values, so read both off /info.
    try:
        info = api_get(session, "/api/train/diffusion/info")
        fams = info.get("families", [])
        hit = next((f for f in fams if str(f.get("name", "")).lower() == family_key), None)
        facts["family_in_api"] = hit is not None
        if hit is not None:
            facts["base_repos"] = hit.get("base_repos")
            facts["base_repo_count"] = len(hit.get("base_repos") or [])
            facts["default_base"] = hit.get("default_base")
            facts["params"] = hit.get("params")
            facts["qlora_vram_gb"] = hit.get("qlora_vram_gb")
            facts["base_specs"] = hit.get("base_specs")
            facts["deploy_bases"] = hit.get("deploy_bases")
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
        await page.get_by_role("tab", name="Train").first.click(timeout=60_000)
        # /info is fetched on mount and both Selects are built from it; a shot taken
        # before it lands shows the frontend's preset fallback on BOTH sides.
        family_trigger = page.get_by_role("combobox", name="Model family").first
        await family_trigger.wait_for(state="visible", timeout=60_000)
        await page.wait_for_timeout(6_000)

        # Pick the family first: the base list is a function of it.
        await open_menu(page, family_trigger,
                        page.get_by_role("option", name=FAMILY_ANCHOR).first)
        await page.wait_for_timeout(1_000)
        klein = page.get_by_role("option", name=re.compile(re.escape(family_label)))
        facts["family_in_menu"] = bool(await klein.count())
        if not await klein.count():
            # Nothing to photograph on either side; say so rather than shoot the default.
            await page.keyboard.press("Escape")
            facts["aborted"] = f"{family_label} absent from the family menu"
            print(f"[{label}] {json.dumps(facts)[:700]}", flush=True)
            return shots, facts
        await klein.first.click()
        await page.wait_for_timeout(2_500)
        # The click can miss and leave the panel on its previous family, which would
        # photograph an unrelated family's bases on one side only (trap 3). The family
        # is not a heading, so assert on the trigger's own value rather than using
        # assert_showing, which looks for a heading and would pass vacuously here.
        shown_family = (await family_trigger.inner_text()).strip()
        facts["family_after_pick"] = shown_family
        if family_label.lower() not in shown_family.lower():
            raise AssertionError(
                f"family select shows {shown_family!r}, expected {family_label!r}"
            )

        base_trigger = page.get_by_role("combobox", name="Base model").first
        await open_menu(page, base_trigger,
                        page.get_by_role("option", name=BASE_ANCHOR).first)
        await page.wait_for_timeout(1_500)

        options = page.get_by_role("option")
        labels = [(await options.nth(i).inner_text()).strip()
                  for i in range(await options.count())]
        facts["base_menu_options"] = labels
        facts["base_menu_option_count"] = len(labels)
        facts["target_in_menu"] = any(target_base.lower() in t.lower() for t in labels)

        shot = out_dir / f"{label.lower()}_base_menu.png"
        await page.locator("[role=listbox]").first.screenshot(path=str(shot))
        shots.append(shot)

        # Same driving script on both sides, different outcome: on the base build there
        # is no 9B row to click, so the panel keeps its only base. That IS the result.
        picked = page.get_by_role("option", name=re.compile(re.escape(target_base)))
        if await picked.count():
            await picked.first.click()
        else:
            await page.keyboard.press("Escape")
        await page.wait_for_timeout(2_500)
        facts["base_after_pick"] = (await base_trigger.inner_text()).strip()

        # The chips, read as text as well as photographed: "9B" and "18 GB" are the
        # claim, and a reviewer cannot diff a picture.
        try:
            chips = page.locator("[data-slot=badge], .badge, span").filter(
                has_text=re.compile(r"^\s*(\d+B|\d+ GB|~\d+ GB)\s*$"))
            facts["chip_texts"] = sorted({
                (await chips.nth(i).inner_text()).strip()
                for i in range(min(await chips.count(), 40))
            })
        except Exception as exc:  # noqa: BLE001 -- the screenshot is still worth taking
            facts["chip_error"] = f"{type(exc).__name__}: {exc}"[:200]

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
