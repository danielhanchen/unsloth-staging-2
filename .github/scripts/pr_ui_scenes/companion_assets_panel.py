"""Scene: what the Hub's On Device tab says before it deletes an image model.

Serves PR 8223 (issue 8116). An image GGUF caches only the denoiser; the text encoders,
VAE and tokenizer come from a separate companion base repo that every quant of the family
shares, and that base is usually the larger half. Three surfaces change, all on the same
tab, so one flow photographs all three:

  1. the On Device toolbar          gains a "Free up space" control
  2. delete the GGUF checkpoint     the dialog gains what it reclaims (4.4 GB) and what it
                                    strands (8.2 GB of shared assets)
  3. delete the shared base repo    refused while a quant is installed, with Delete disabled,
                                    where before it succeeded and left both quants unloadable

The cache is seeded by `scripts/seed_8223_cache.py` into an isolated HF_HOME; both Studios
are launched against that same cache, and this scene never confirms a delete, so the two
sides read an identical, unmutated disk.

No weights are loaded and no GPU is touched: every number here comes from a cache scan.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or os.environ.get("UNSLOTH_WORKSPACE") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import (  # noqa: E402
    Session,
    api_get,
    api_post,
    assert_showing,
    open_menu,
)
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

DEFAULT_GGUF_REPO = "unsloth/FLUX.2-klein-4B-GGUF"
DEFAULT_BASE_REPO = "black-forest-labs/FLUX.2-klein-4B"


def _pad_to(path: Path, width: int, height: int) -> None:
    """Centre `path` on a fixed white canvas, in place.

    The two dialogs are different heights (that IS the change), and `hstack_images`
    equalises heights by SCALING, which would blow the shorter one up and make the pair
    look retouched. Padding to a common canvas keeps both at 1:1. Raising rather than
    cropping when it does not fit is deliberate: a silently cropped dialog loses exactly
    the sentence the shot exists to show.
    """
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.width > width or img.height > height:
        raise RuntimeError(
            f"{path.name} is {img.width}x{img.height}, larger than the {width}x{height} "
            "canvas; widen it rather than cropping the text this shot exists to show"
        )
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
    canvas.save(path)


async def _open_delete_dialog(page, repo_id: str):
    """Open the row menu for `repo_id` and click Delete, proving the dialog is ITS dialog.

    Every row's dialog is titled "Delete cached model?", so the title alone cannot tell a
    hit from a miss -- only the body names the repo. A menu opened on the neighbouring row
    photographs perfectly and proves nothing, which is exactly the failure this scene is
    most exposed to: the two repos here sit next to each other in the list.
    """
    trigger = page.locator(f'button[aria-label="More options for {repo_id}"]').first
    await trigger.wait_for(state="visible", timeout=120_000)
    delete_item = page.get_by_role("menuitem", name="Delete").first
    # Retried by open_menu: the inventory refreshes on a timer and a refresh landing
    # between the click and the menu paint remounts the row, taking the dropdown with it.
    await open_menu(page, trigger, delete_item)
    await delete_item.click(timeout=20_000)
    await assert_showing(page, "Delete cached model?", timeout_ms=20_000)
    dialog = page.get_by_role("alertdialog").filter(has_text=repo_id).first
    await dialog.wait_for(state="visible", timeout=20_000)
    return dialog


async def _close_dialog(page) -> None:
    await page.keyboard.press("Escape")
    await page.get_by_role("alertdialog").first.wait_for(state="hidden", timeout=20_000)


async def drive(session: Session, out_dir: Path, label: str,
                gguf_repo: str = DEFAULT_GGUF_REPO,
                base_repo: str = DEFAULT_BASE_REPO,
                **_: object) -> tuple[list[Path], dict]:
    """Three shots: the On Device toolbar, then each repo's delete dialog."""
    facts: dict = {}
    # The numeric half, read from the same server that is photographed. On the merge base
    # both endpoints are absent, and that 404 IS the before-state, so it is recorded rather
    # than raised.
    for name, repo in (("gguf", gguf_repo), ("base", base_repo)):
        try:
            facts[f"delete_impact_{name}"] = api_post(
                session, "/api/hub/delete-impact", {"repo_id": repo}
            )
        except Exception as exc:  # noqa: BLE001
            facts[f"delete_impact_{name}"] = f"{type(exc).__name__}: {exc}"
    try:
        facts["orphan_companions"] = api_get(session, "/api/hub/orphan-companions")
    except Exception as exc:  # noqa: BLE001
        facts["orphan_companions"] = f"{type(exc).__name__}: {exc}"

    shots: list[Path] = []
    init = seed_init_script(
        type("A", (), {"access_token": session.access_token,
                       "refresh_token": session.refresh_token})(), []
    )
    async with open_chat(session.base_url, init_scripts=[init],
                         viewport=(1500, 1000), headless=True) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/hub", wait_until="domcontentloaded")
        on_device = page.get_by_role("radio", name="On Device").first
        await on_device.wait_for(state="visible", timeout=60_000)
        await on_device.click()
        search = page.get_by_placeholder("Search on-device models").first
        await search.wait_for(state="visible", timeout=60_000)
        await search.fill("FLUX.2-klein-4B")
        # The inventory is a cache walk; both rows must be present before anything is
        # clicked, or a shot of a half-loaded list differs by timing rather than by the fix.
        for repo in (gguf_repo, base_repo):
            await page.locator(
                f'button[aria-label="More options for {repo}"]'
            ).first.wait_for(state="visible", timeout=180_000)
        await page.wait_for_timeout(3_000)

        # Cropped, not the whole viewport: a 1500px-wide pair renders at ~440px per half in
        # a PR comment, where a dialog's small print is unreadable. The claim has to survive
        # the size GitHub shows it at.
        toolbar_shot = out_dir / f"{label.lower()}_0_on_device_toolbar.png"
        await page.screenshot(path=str(toolbar_shot),
                              clip={"x": 340, "y": 96, "width": 1110, "height": 128})
        shots.append(toolbar_shot)
        facts["free_up_space_button"] = (
            await page.get_by_test_id("free-up-space-trigger").count() > 0
        )

        for index, repo in ((1, gguf_repo), (2, base_repo)):
            dialog = await _open_delete_dialog(page, repo)
            # The dialog fetches its preview after opening; settle so the two sides differ
            # by content and not by how far each had rendered.
            await page.wait_for_timeout(4_000)
            key = "gguf" if repo == gguf_repo else "base"
            facts[f"dialog_text_{key}"] = " ".join((await dialog.inner_text()).split())
            confirm = dialog.get_by_role("button", name="Delete").first
            facts[f"delete_button_disabled_{key}"] = await confirm.is_disabled()
            shot = out_dir / f"{label.lower()}_{index}_delete_{key}.png"
            await dialog.screenshot(path=str(shot))
            _pad_to(shot, 480, 480)
            shots.append(shot)
            await _close_dialog(page)

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
