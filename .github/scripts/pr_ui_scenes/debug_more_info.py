"""Debug scene (not for posting): is the chat page's run-settings "More info" control clickable?

The switchboard crawl recorded `button:More info -> error:Error` on 14348's head only. This
replays the crawl's steps with long timeouts and records every match and the click error text.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402


async def _matches(page):
    loc = page.get_by_role("button", name = re.compile("^More info"))
    out = []
    for i in range(await loc.count()):
        el = loc.nth(i)
        out.append(
            {
                "visible": await el.is_visible(),
                "box": await el.bounding_box(),
                "html": (
                    await el.evaluate(
                        "e => e.closest('section,div[data-slot],aside,form')?.outerHTML.slice(0,300) || ''"
                    )
                ),
            }
        )
    return loc, out


async def drive(session: Session, out_dir: Path, label: str, **_: object):
    facts: dict = {}
    init = seed_init_script(
        type(
            "A", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
        )(),
        [],
    )
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = (1440, 900), headless = True
    ) as sp:
        page = sp.page
        page.set_default_navigation_timeout(240_000)
        await page.goto(f"{session.base_url}/chat", wait_until = "domcontentloaded")
        await page.get_by_role("button", name = "Open run settings").first.click(timeout = 60_000)
        await page.wait_for_timeout(1500)
        loc, facts["open_panel"] = await _matches(page)
        await page.get_by_role("button", name = "Close run settings").first.click(timeout = 60_000)
        await page.wait_for_timeout(1500)
        loc, facts["after_close"] = await _matches(page)
        try:
            await loc.first.click(timeout = 10_000)
            facts["click"] = "ok"
        except Exception as e:  # noqa: BLE001
            facts["click"] = f"{type(e).__name__}: {str(e)[:600]}"
        shot = out_dir / f"{label.lower()}_more_info.png"
        await page.screenshot(path = str(shot))
    (out_dir / f"{label.lower()}_facts.json").write_text(json.dumps(facts, indent = 1, default = str))
    return [shot], facts
