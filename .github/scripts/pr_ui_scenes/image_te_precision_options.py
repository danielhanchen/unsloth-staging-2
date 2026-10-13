"""Scene: the Images advanced panel's "Text encoder precision" select, opened.

Serves PR 11539, which makes an unset request resolve to the family's own encoder
scheme (Qwen-Image-2.1 takes its hosted pre-cast fp8 encoder). That turns "Default"
from a synonym for bf16 into a real choice, so the released bf16 encoder needs a
spelling of its own: the option list gains "Dense (bf16)", and the request schema
gains the none/off it was rejecting.

Listings only: no weights, no model load, no GPU capability needed. The option list is
static frontend, which is the point -- both halves render it unconditionally, so an
identical pair here is a missed click rather than an unreportable host.

The numeric half is the option list itself, in `facts`, because six 12 px rows in a
440 px comment half are not readable and two lists are not diffable by eye.
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

# The control's own label. Matched on the label rather than on any option text: an
# option-text match would be the trap-4 shape, since "FP8" also appears in the
# transformer precision select directly above and would open an identical-looking menu
# on both sides.
SELECT_LABEL = "Text encoder precision"


async def _open_advanced(page, timeout_ms: int = 30_000) -> None:
    """Expand the Advanced disclosure that holds the precision selects.

    Retried: the panel is a Radix collapsible and loses races with the model-inventory
    refresh the page fires on mount, which surfaces as the select never appearing and
    reads like a bad selector.
    """
    trigger = page.get_by_text("Advanced", exact=False).first
    for attempt in range(6):
        try:
            if await page.get_by_text(SELECT_LABEL, exact=False).first.is_visible():
                return
        except Exception:  # noqa: BLE001 -- not present yet is not an answer
            pass
        try:
            await trigger.scroll_into_view_if_needed()
            await trigger.click(timeout=timeout_ms)
            await page.get_by_text(SELECT_LABEL, exact=False).first.wait_for(
                state="visible", timeout=8_000
            )
            return
        except Exception:  # noqa: BLE001
            if attempt == 5:
                raise
            await page.wait_for_timeout(1_000)
    raise AssertionError("unreachable")


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    clip: Optional[dict] = None,
    **_: object,
) -> tuple[list[Path], dict]:
    """Open the select and photograph its options, returning the option list as facts."""
    facts: dict = {"select_label": SELECT_LABEL}
    shots: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    init = seed_init_script(
        type(
            "A",
            (),
            {
                "access_token": session.access_token,
                "refresh_token": session.refresh_token,
            },
        )(),
        [],
    )
    async with open_chat(
        session.base_url, init_scripts=[init], viewport=(1500, 1100), headless=True
    ) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/images", wait_until="domcontentloaded")
        # The composer and its advanced panel mount after the model inventory lands.
        await page.wait_for_timeout(9_000)
        await _open_advanced(page)

        # The select trigger is the control that FOLLOWS the label, not the label.
        container = page.locator("div").filter(has_text=SELECT_LABEL).last
        trigger = container.get_by_role("combobox").first
        facts["trigger_text_before_open"] = (await trigger.inner_text()).strip()

        for attempt in range(6):
            await trigger.scroll_into_view_if_needed()
            await trigger.click()
            try:
                await page.get_by_role("option").first.wait_for(
                    state="visible", timeout=8_000
                )
                break
            except Exception:  # noqa: BLE001 -- Radix race, trap 7
                if attempt == 5:
                    raise
                await page.keyboard.press("Escape")
                await page.wait_for_timeout(800)

        options = page.get_by_role("option")
        texts = [(await options.nth(i).inner_text()).strip()
                 for i in range(await options.count())]
        facts["options"] = texts
        facts["option_count"] = len(texts)
        # The claim, as a boolean the reader does not have to squint for.
        facts["has_dense_option"] = any(
            "dense" in t.lower() or "bf16" in t.lower() for t in texts
        )

        shot = out_dir / f"{label}_te_precision_options.png"
        await page.screenshot(path=str(shot), clip=clip) if clip else await page.screenshot(
            path=str(shot)
        )
        shots.append(shot)

    return shots, facts
