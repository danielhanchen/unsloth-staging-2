"""Scene: the composer model picker's org filters (PR 18108).

Before: an org with 3+ models on device (Qwen here) gets a TAB in the "Model sources"
tablist beside Favorites / Discover / On Device, listing its on-device models.
After: org chips sit on a second row (role=group "Filter by organization", aria-pressed
buttons), shown on Discover and On Device only, narrowing whichever is open; Settings >
Chat > "Choose organizations" (localStorage `unsloth_chat_picker_orgs`) picks them outright.

Needs a seeded, isolated HF cache (--studio-env HF_HOME/HF_HUB_CACHE/HF_XET_CACHE/
XDG_CACHE_HOME) holding 3+ `Qwen/` or `unsloth/Qwen*` repos plus non-Qwen controls.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

WORKSPACE = Path(
    os.environ.get("WORKSPACE")
    or os.environ.get("UNSLOTH_WORKSPACE")
    or Path(__file__).resolve().parents[1]
)
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402


def _init(session: Session, extra: dict | None = None) -> str:
    auth = type(
        "A", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
    )()
    return seed_init_script(auth, [], extra_local_storage = extra)


async def _labels(loc) -> list[dict]:
    out = []
    for i in range(await loc.count()):
        b = loc.nth(i)
        out.append(
            {
                "label": await b.get_attribute("aria-label"),
                "role": await b.get_attribute("role"),
                "selected": await b.get_attribute("aria-selected"),
                "pressed": await b.get_attribute("aria-pressed"),
            }
        )
    return out


async def _state(page) -> dict:
    tabs = page.get_by_role("tablist", name = "Model sources").first.locator("button")
    group = page.get_by_role("group", name = "Filter by organization")
    chips = group.first.locator("button") if await group.count() else None
    listbox = page.get_by_role("listbox").first
    await page.wait_for_timeout(1500)
    rows = []
    opts = listbox.get_by_role("option")
    for i in range(await opts.count()):
        txt = (await opts.nth(i).inner_text()).strip().split("\n")[0]
        rows.append(txt)
    return {
        "source_row": await _labels(tabs),
        "org_row": (await _labels(chips)) if chips is not None else None,
        "listbox": await listbox.get_attribute("aria-label"),
        "rows": rows,
        "row_count": len(rows),
    }


async def _shot(page, out: Path) -> Path:
    """Clip to the picker: from the tab strip to the bottom of the list."""
    out.parent.mkdir(parents = True, exist_ok = True)
    top = await page.get_by_role("tablist", name = "Model sources").first.bounding_box()
    lb = await page.get_by_role("listbox").first.bounding_box()
    x = max(0, min(top["x"], lb["x"]) - 16)
    y = max(0, top["y"] - 16)
    w = max(top["x"] + top["width"], lb["x"] + lb["width"]) - x + 16
    h = min(lb["y"] + lb["height"], y + 900) - y + 16
    await page.screenshot(path = str(out), clip = {"x": x, "y": y, "width": w, "height": h})
    return out


async def _open_picker(page, base_url: str) -> None:
    await page.goto(f"{base_url}/chat", wait_until = "domcontentloaded")
    trigger = page.locator('button[aria-label^="Model:"]').first
    await trigger.wait_for(state = "visible", timeout = 60_000)
    await page.wait_for_timeout(3000)
    await trigger.click()
    await page.get_by_role("tablist", name = "Model sources").first.wait_for(
        state = "visible", timeout = 30_000
    )


async def _tab(page, name: str) -> None:
    await (
        page.get_by_role("tablist", name = "Model sources")
        .first.locator(f'button[aria-label="{name}"]')
        .first.click()
    )
    await page.wait_for_timeout(2500)


async def _press_org(page, org: str) -> str:
    """Head: the chip in the org row. Base: the org tab in the source row."""
    group = page.get_by_role("group", name = "Filter by organization")
    if await group.count():
        await group.first.locator(f'button[aria-label="{org}"]').first.click()
        where = "org_row"
    else:
        await _tab(page, org)
        where = "source_row"
    await page.wait_for_timeout(2500)
    return where


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    org: str = "Qwen",
    picked: list | None = None,
    **_: object,
) -> tuple[list[Path], dict]:
    picked = picked or ["google", "qwen"]
    lab = label.lower()
    shots: list[Path] = []
    facts: dict = {}
    async with open_chat(
        session.base_url, init_scripts = [_init(session)], viewport = (1500, 1100), headless = True
    ) as sp:
        page = sp.page
        await _open_picker(page, session.base_url)
        await _tab(page, "On Device")
        facts["device"] = await _state(page)
        shots.append(await _shot(page, out_dir / f"{lab}_1_on_device.png"))

        if any(t["label"] == org for t in facts["device"]["source_row"]) or (
            facts["device"]["org_row"]
            and any(t["label"] == org for t in facts["device"]["org_row"])
        ):
            facts["device_pressed_via"] = await _press_org(page, org)
            facts["device_pressed"] = await _state(page)
            shots.append(await _shot(page, out_dir / f"{lab}_2_on_device_{org.lower()}.png"))
        else:
            facts["device_pressed"] = f"no {org} tab or chip"

        await _press_org(page, org) if facts.get("device_pressed_via") == "org_row" else None
        await _tab(page, "Discover models")
        facts["discover"] = await _state(page)
        shots.append(await _shot(page, out_dir / f"{lab}_3_discover.png"))
        if facts["discover"]["org_row"] and any(
            t["label"] == org for t in facts["discover"]["org_row"]
        ):
            await _press_org(page, org)
            facts["discover_pressed"] = await _state(page)
            shots.append(await _shot(page, out_dir / f"{lab}_4_discover_{org.lower()}.png"))
        else:
            # Same shot count both sides: Discover as it stands, no chip to press.
            facts["discover_pressed"] = f"no {org} chip on Discover"
            shots.append(await _shot(page, out_dir / f"{lab}_4_discover_{org.lower()}.png"))

    async with open_chat(
        session.base_url,
        init_scripts = [_init(session, {"unsloth_chat_picker_orgs": picked})],
        viewport = (1500, 1100),
        headless = True,
    ) as sp:
        page = sp.page
        await _open_picker(page, session.base_url)
        await _tab(page, "On Device")
        facts["picked_device"] = await _state(page)
        shots.append(await _shot(page, out_dir / f"{lab}_5_picked_{'_'.join(picked)}.png"))

        await page.keyboard.press("Escape")
        await page.wait_for_timeout(800)
        dialog = page.get_by_role("dialog", name = "Settings").first
        for key in ("Control+Comma", "Meta+Comma"):
            await page.locator("body").press(key)
            try:
                await dialog.wait_for(state = "visible", timeout = 8_000)
                break
            except Exception:  # noqa: BLE001
                continue
        else:
            await page.get_by_role("button", name = re.compile(r"^\s*Settings\s*$")).first.click()
            await dialog.wait_for(state = "visible", timeout = 30_000)
        await (
            dialog.get_by_role("tab", name = re.compile(r"^\s*Chat\s*$"))
            .or_(dialog.get_by_role("button", name = re.compile(r"^\s*Chat\s*$")))
            .first.click()
        )
        await page.wait_for_timeout(2000)
        sw = dialog.get_by_role("switch", name = "Choose organizations")
        facts["settings_switch"] = await sw.count() > 0
        if facts["settings_switch"]:
            await sw.first.scroll_into_view_if_needed()
            facts["settings_switch_checked"] = await sw.first.get_attribute("aria-checked")
            group = dialog.get_by_role("group", name = "Choose organizations")
            items = group.first.locator("button") if await group.count() else None
            on = []
            if items is not None:
                for i in range(await items.count()):
                    if (await items.nth(i).get_attribute("data-state")) == "on":
                        on.append((await items.nth(i).inner_text()).strip())
                facts["settings_org_count"] = await items.count()
            facts["settings_orgs_on"] = on
            target = sw.first
        else:
            target = dialog.get_by_text(re.compile("Model selection", re.I)).first
            await target.scroll_into_view_if_needed()
        await page.wait_for_timeout(800)
        box = await dialog.bounding_box()
        tb = await target.bounding_box()
        y = max(box["y"], tb["y"] - 260)
        shot = out_dir / f"{lab}_6_settings_chat.png"
        await page.screenshot(
            path = str(shot),
            clip = {
                "x": box["x"],
                "y": y,
                "width": box["width"],
                "height": min(560, box["y"] + box["height"] - y),
            },
        )
        shots.append(shot)
    import json

    (out_dir / f"facts_{lab}.json").write_text(json.dumps(facts, indent = 1))
    return shots, facts
