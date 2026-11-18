"""Scene: the chat composer model picker's On Device rows, with the
"Model size and parameters" setting (localStorage `unsloth_chat_show_model_sizes`)
seeded on or off before the SPA boots.

Serves PR 18067. Setting on: head rows carry a param chip and their bytes on disk
in `[data-row-size]` columns; base has no such columns. Setting off (`show_sizes`
False): both sides should render the same rows (run that pass with --parity).

The rows depend on the HF cache, so launch both Studios on an isolated one
(--studio-env HF_HOME/HF_HUB_CACHE/HF_XET_CACHE/XDG_CACHE_HOME, trap 5).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WORKSPACE = Path(
    os.environ.get("WORKSPACE")
    or os.environ.get("UNSLOTH_WORKSPACE")
    or Path(__file__).resolve().parents[1]
)
sys.path.insert(0, str(WORKSPACE))

from pr_ui_scenes._common import Session  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

KEY = "unsloth_chat_show_model_sizes"


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    show_sizes: bool = True,
    tab: str = "On Device",
    **_: object,
) -> tuple[list[Path], dict]:
    auth = type(
        "A", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
    )()
    init = seed_init_script(auth, [], extra_local_storage = {KEY: "true" if show_sizes else "false"})
    out_dir.mkdir(parents = True, exist_ok = True)
    tag = "on" if show_sizes else "off"
    facts: dict = {"setting": tag, "tab": tab}
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = (1400, 1000), headless = True
    ) as sp:
        page = sp.page
        facts["localStorage"] = await page.evaluate(f"localStorage.getItem('{KEY}')")
        trigger = page.locator('[data-tour="chat-model-selector"]').first
        await trigger.wait_for(state = "visible", timeout = 60_000)
        await trigger.click()
        picker = page.locator("[data-composer-model-picker]").first
        await picker.wait_for(state = "visible", timeout = 30_000)
        tab_btn = picker.get_by_role("tab", name = tab, exact = True).first
        await tab_btn.click()
        if await tab_btn.get_attribute("aria-selected") != "true":
            raise RuntimeError(f"tab {tab!r} did not become selected")
        rows = picker.locator("[data-model-picker-list] [data-model-picker-option]")
        await rows.first.wait_for(state = "visible", timeout = 60_000)
        await page.wait_for_timeout(4_000)  # inventory + quant listings settle
        n = await rows.count()
        facts["rows"] = [" | ".join((await rows.nth(i).inner_text()).split("\n")) for i in range(n)]
        facts["row_size_cells"] = await picker.locator("[data-row-size]").count()
        facts["row_size_text"] = [
            " | ".join((await c.inner_text()).split("\n"))
            for c in await picker.locator("[data-row-size]").all()
        ]
        shot = out_dir / f"{label.lower()}_picker_{tag}.png"
        await picker.screenshot(path = str(shot))
    return [shot], facts
