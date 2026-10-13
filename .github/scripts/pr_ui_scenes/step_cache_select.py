"""Scene: the "Step cache" select in the Images or Video advanced panel, opened, recorded.

Serves the static step skip PRs (11737 images, 11748 video: a "Static skip" option joins
Auto / Off / First-Block-Cache) and 11791 (the hint text changes: auto FBCache only on the
max speed tier). Listings only: the option list and the hint are static frontend, so an
identical pair is a missed click, not an unreportable host.

kwargs: path ("/images" or "/video"), select_label (default "Step cache"), hint_needles
(strings whose presence in the page text is recorded as facts), record (webm per side).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

WORKSPACE = Path(
    os.environ.get("WORKSPACE")
    or os.environ.get("UNSLOTH_WORKSPACE")
    or Path(__file__).resolve().parents[2]
)
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402


async def _open_advanced(page, select_label: str, timeout_ms: int = 30_000) -> None:
    trigger = page.get_by_text("Advanced", exact=False).first
    for attempt in range(6):
        try:
            if await page.get_by_text(select_label, exact=True).first.is_visible():
                return
        except Exception:  # noqa: BLE001
            pass
        try:
            await trigger.scroll_into_view_if_needed()
            await trigger.click(timeout=timeout_ms)
            await page.get_by_text(select_label, exact=True).first.wait_for(
                state="visible", timeout=8_000
            )
            return
        except Exception:  # noqa: BLE001
            if attempt == 5:
                raise
            await page.wait_for_timeout(1_000)


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    path: str = "/images",
    select_label: str = "Step cache",
    hint_needles: Optional[list] = None,
    record: bool = True,
    hover_hint: bool = False,
    open_list: bool = True,
    **_: object,
) -> tuple[list[Path], dict]:
    facts: dict = {"path": path, "select_label": select_label}
    shots: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    init = seed_init_script(
        type("A", (), {"access_token": session.access_token,
                       "refresh_token": session.refresh_token})(),
        [],
    )
    video_dir = out_dir / "video" if record else None
    async with open_chat(
        session.base_url, init_scripts=[init], viewport=(1500, 1100), headless=True,
        video_dir=video_dir, video_name=f"{label}_{path.strip('/')}_step_cache",
    ) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}{path}", wait_until="domcontentloaded")
        await page.wait_for_timeout(9_000)
        await _open_advanced(page, select_label)
        lbl = page.get_by_text(select_label, exact=True).first
        await lbl.scroll_into_view_if_needed()
        await page.wait_for_timeout(600)
        body = await page.inner_text("body")
        for needle in hint_needles or []:
            facts[f"text_has[{needle}]"] = needle in body
        # the combobox that follows the label
        container = page.locator("div").filter(has=lbl).filter(
            has=page.get_by_role("combobox")).last
        trigger = container.get_by_role("combobox").first
        if hover_hint:
            info = container.locator('[data-slot="info-hint"]').first
            tip = page.get_by_role("tooltip").first
            for how in ("focus", "hover"):
                try:
                    if how == "focus":
                        await info.focus()
                    else:
                        await info.hover(force=True)
                    await tip.wait_for(state="visible", timeout=6_000)
                    break
                except Exception:  # noqa: BLE001 - a label span covers the icon; try the next way in
                    if how == "hover":
                        raise
            await page.wait_for_timeout(500)
            tip_text = (await tip.inner_text()).strip()
            facts["hint_text"] = tip_text
            for needle in hint_needles or []:
                facts[f"hint_has[{needle}]"] = needle in tip_text
            hshot = out_dir / f"{label}_{path.strip('/')}_step_cache_hint.png"
            await page.screenshot(path=str(hshot))
            shots.append(hshot)
            await page.mouse.move(5, 5)
            await page.wait_for_timeout(400)
        facts["trigger_text"] = (await trigger.inner_text()).strip()
        for attempt in range(6 if open_list else 0):
            await trigger.scroll_into_view_if_needed()
            await trigger.click()
            try:
                await page.get_by_role("option").first.wait_for(state="visible", timeout=8_000)
                break
            except Exception:  # noqa: BLE001 - Radix race (trap 7)
                if attempt == 5:
                    raise
                await page.keyboard.press("Escape")
                await page.wait_for_timeout(800)
        if not open_list:
            await page.wait_for_timeout(300)
        await page.wait_for_timeout(700)
        options = page.get_by_role("option")
        texts = [(await options.nth(i).inner_text()).strip() for i in range(await options.count())]
        facts["options"] = texts
        facts["has_static_option"] = any("static" in t.lower() for t in texts)
        if not open_list:
            texts = None
        if open_list:
            shot = out_dir / f"{label}_{path.strip('/')}_step_cache_open.png"
            await page.screenshot(path=str(shot))
            shots.append(shot)
        if not open_list:
            pass
        elif facts["has_static_option"]:
            await page.get_by_role("option").filter(has_text="Static").first.click()
            await page.wait_for_timeout(900)
            facts["trigger_text_after_pick"] = (await trigger.inner_text()).strip()
            shot2 = out_dir / f"{label}_{path.strip('/')}_step_cache_picked.png"
            await page.screenshot(path=str(shot2))
            shots.append(shot2)
        else:
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(600)
            facts["trigger_text_after_pick"] = (await trigger.inner_text()).strip()
            shot2 = out_dir / f"{label}_{path.strip('/')}_step_cache_picked.png"
            await page.screenshot(path=str(shot2))
            shots.append(shot2)
    if record and getattr(sp, "video_webm", None):
        facts["video_webm"] = str(sp.video_webm)
    return shots, facts
