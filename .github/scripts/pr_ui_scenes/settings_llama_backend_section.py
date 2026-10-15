"""Scene: Settings > System (resources tab) > "GGUF inference engine" section.

Opens the settings dialog via the /settings deep link, switches to the System tab,
scrolls the "GGUF inference engine" section into view and clips the shot to it.
Facts record whether the "Exact concurrency" row (PR 10358) is rendered and what
/api/settings/exact-concurrency answers, so a missing row cannot pass silently.
No model is needed.
"""

from __future__ import annotations

import os
import sys
import urllib.error
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or os.environ.get("UNSLOTH_WORKSPACE") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(WORKSPACE))

from pr_ui_scenes._common import Session, api_get  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

VIEWPORT = (1440, 1400)


async def drive(session: Session, out_dir: Path, label: str,
                section_title: str = "GGUF inference engine",
                tab_label: str = "System", open_select: bool = True,
                clip_height: int = 420, **_: object):
    facts: dict = {}
    try:
        facts["api_exact_concurrency"] = api_get(session, "/api/settings/exact-concurrency")
    except urllib.error.HTTPError as exc:
        facts["api_exact_concurrency"] = f"HTTP {exc.code}"
    init = seed_init_script(
        type("A", (), {"access_token": session.access_token,
                       "refresh_token": session.refresh_token})(), []
    )
    shots: list[Path] = []
    async with open_chat(session.base_url, init_scripts=[init],
                         viewport=VIEWPORT, headless=True) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/settings", wait_until="domcontentloaded")
        dialog = page.get_by_role("dialog")
        await dialog.first.wait_for(state="visible", timeout=60_000)
        tab = dialog.get_by_role("button", name=tab_label, exact=True)
        if await tab.count() == 0:
            tab = dialog.get_by_role("tab", name=tab_label, exact=True)
        await tab.first.click()
        heading = dialog.get_by_text(section_title, exact=True).first
        await heading.wait_for(state="visible", timeout=60_000)
        await heading.scroll_into_view_if_needed()
        # Let the section's own GET calls (backend status, path, exact concurrency) settle:
        # the rows render "Loading..." and empty selects until they answer.
        sel = page.locator("[data-testid=exact-concurrency-select]")
        for _ in range(20):
            loading = await dialog.get_by_text("Loading...", exact=False).count()
            empty_sel = await sel.count() and not (await sel.first.inner_text()).strip()
            if not loading and not empty_sel:
                break
            await page.wait_for_timeout(1000)
        facts["section_still_loading"] = bool(
            await dialog.get_by_text("Loading...", exact=False).count())
        facts["exact_concurrency_select_count"] = await sel.count()
        facts["exact_concurrency_label_visible"] = await dialog.get_by_text(
            "Exact concurrency", exact=True).count() > 0
        if await sel.count():
            facts["exact_concurrency_select_value"] = (await sel.first.inner_text()).strip()
        facts["llama_backend_select_count"] = await page.locator(
            "[data-testid=llama-backend-select]").count()
        # Clip to the section: from its title down to the end of its rows.
        section = heading.locator("xpath=ancestor::*[.//*[@data-testid='llama-backend-select']][1]")
        if await section.count():
            # Top of the scroll pane, so the fixed-height clip stays inside the viewport.
            await section.evaluate("el => el.scrollIntoView({block: 'start'})")
            await page.wait_for_timeout(300)
        box = await section.bounding_box() if await section.count() else None
        # Measured BEFORE opening the listbox: an open Radix Select hides the dialog from
        # the accessibility tree and the section locator then finds nothing.
        # Open the new select so its Auto / On / Off options are in the shot.
        if await sel.count() and open_select:
            await sel.first.click()
            opts = page.get_by_role("option")
            await opts.first.wait_for(state="visible", timeout=15_000)
            facts["exact_concurrency_options"] = [
                (await opts.nth(i).inner_text()).strip() for i in range(await opts.count())]
        out = out_dir / f"{label}_settings_llama_backend.png"
        if box:
            pad = 16
            # Same fixed height on both sides (room for the open listbox), so the
            # composite does not rescale one half against the other.
            await page.screenshot(path=str(out), clip={
                "x": max(box["x"] - pad, 0), "y": max(box["y"] - pad, 0),
                "width": box["width"] + 2 * pad, "height": clip_height})
        else:
            await page.screenshot(path=str(out))
        facts["clipped_to_section"] = bool(box)
        shots.append(out)
    return shots, facts
