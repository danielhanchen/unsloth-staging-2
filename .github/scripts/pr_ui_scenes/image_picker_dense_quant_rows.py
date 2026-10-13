"""Scene: the Images model picker rows for the official dense-quant pipelines.

Serves PR 10888 (the auto transformer-precision ladder walks int8 first on every GPU
tier, so the row suffix every dense-quant host reads flips from "Fast FP8" to
"Fast INT8") and PR 10883 (the official image rows carry the HOSTED checkpoint's size
instead of the released bf16 size, so the figure beside the row and the fit verdict
behind it both change, and the MiniMax H3 row gets its speed suffix back).

Both changes are BACKEND-DRIVEN. The suffix comes from `/api/system`'s
`dense_quant_schemes`, which is why this needs a CUDA host to say anything at all: on a
host that reports no schemes the rows are identical on both sides and the pair proves
nothing. It does NOT need a model loaded -- the rows are catalog plus host class.

Listings only: no weights, no disk. The numeric half is `/api/system` plus the row text
itself, because a reviewer cannot read a 12 px suffix in a 440 px half.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

WORKSPACE = Path(os.environ.get("WORKSPACE") or os.environ.get("UNSLOTH_WORKSPACE") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session, api_get  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

# The rows to photograph, in catalog order. Anchored when matched, so "Qwen-Image" never
# takes the "Qwen-Image 2512" or "Qwen-Image-Edit 2511" row, which is the whole trap-3
# family: a near-match photographs a real row that is not the one the claim is about.
DEFAULT_ROWS = ("Z-Image-Turbo", "Qwen-Image", "MiniMax H3")

# A row's first line is the display name plus, when the host reports a dense-quant
# ladder, the speed suffix in brackets.
_NAME_LINE = r"^\s*{name}\s*(\((?:Fast|Dense|Slow)[^)]*\))?\s*$"


def _row_pattern(name: str) -> re.Pattern:
    return re.compile(_NAME_LINE.format(name=re.escape(name)))


async def _open_picker(page, timeout_ms: int = 60_000):
    """Click the Images model-selector trigger and wait for real option rows.

    The trigger carries the placeholder when nothing is loaded, and the CURRENT model's
    name once something is. Both shapes are matched, since a home that has loaded before
    would otherwise leave the picker shut and the shot would be of the page behind it.
    """
    trigger = page.locator("button").filter(
        has_text=re.compile(r"Select image model|Z-Image|Qwen-Image|FLUX")
    ).first
    options = page.locator("[data-model-picker-option]")
    for attempt in range(6):
        await trigger.scroll_into_view_if_needed()
        await trigger.click(timeout=timeout_ms)
        try:
            await options.first.wait_for(state="visible", timeout=8_000)
            return options
        except Exception:  # noqa: BLE001 -- Radix loses races with the background refresh
            if attempt == 5:
                raise
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(800)
    raise AssertionError("unreachable")


async def drive(session: Session, out_dir: Path, label: str,
                rows: tuple[str, ...] = DEFAULT_ROWS,
                clip: Optional[dict] = None,
                search: Optional[str] = None,
                **_: object) -> tuple[list[Path], dict]:
    """Photograph the open picker, then each named row on its own."""
    facts: dict = {"rows_requested": list(rows)}

    # Read the host capability from the very server being photographed: with no reported
    # scheme there is no suffix to move, and identical halves would be a HOST result, not
    # a PR result.
    #
    # POLL for it. `/api/system` is polled all through startup and answers
    # dense_quant_supported=false / schemes=[] until the post-warm worker has torch and
    # torchao up, which on this box takes 15 to 30 seconds. Shooting before that photographs
    # the no-capability fallback on BOTH sides, which is the trap this whole file exists for:
    # a clean run, two identical halves, and nothing proved.
    deadline = time.time() + 420
    while time.time() < deadline:
        try:
            if api_get(session, "/api/system").get("dense_quant_schemes"):
                break
        except Exception:  # noqa: BLE001 -- a startup 5xx is not the answer
            pass
        time.sleep(5)
    facts["dense_quant_wait_s"] = round(420 - (deadline - time.time()), 1)
    try:
        system = api_get(session, "/api/system")
        facts["dense_quant_schemes"] = system.get("dense_quant_schemes")
        facts["dense_quant_supported"] = system.get("dense_quant_supported")
        facts["device_backend"] = system.get("device_backend")
        gpu = system.get("gpu")
        if isinstance(gpu, list):
            gpu = gpu[0] if gpu else None
        if isinstance(gpu, dict):
            facts["gpu"] = gpu.get("name") or gpu.get("gpu_name")
            facts["gpu_total_gb"] = gpu.get("memory_total_gb") or gpu.get("total_memory_gb")
        else:
            facts["gpu"] = gpu
    except Exception as exc:  # noqa: BLE001 -- the screenshot is still worth taking
        facts["system_error"] = f"{type(exc).__name__}: {exc}"[:300]

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
        # /api/system and the model inventory are fetched on mount and the rows are built
        # from both; a shot taken before they land shows the no-host-knowledge fallback on
        # BOTH sides.
        await page.wait_for_timeout(9_000)
        options = await _open_picker(page)
        await page.wait_for_timeout(4_000)
        if search:
            # Narrows the list so the rows render on screen: the list is virtualised, and a row
            # far down is recycled DOM by the time it is photographed by index.
            box = page.get_by_placeholder(re.compile("Search", re.I)).first
            await box.fill(search)
            await page.wait_for_timeout(2_500)
            facts["search"] = search

        texts = []
        for i in range(await options.count()):
            texts.append(re.sub(r"\s*\n\s*", " | ", (await options.nth(i).inner_text()).strip()))
        facts["option_count"] = len(texts)
        facts["first_options"] = texts[:6]

        menu = page.locator(".unsloth-model-selector-menu").first
        shot = out_dir / f"{label.lower()}_picker.png"
        if clip:
            await page.screenshot(path=str(shot), clip=clip)
        elif await menu.count():
            await menu.screenshot(path=str(shot))
        else:
            await sp.screenshot(shot, full_page=False)
        shots.append(shot)

        # One element shot per named row. A row is ~40 px tall, so a full-menu shot
        # renders the suffix and the size unreadably small in a comment.
        for name in rows:
            pattern = _row_pattern(name)
            hit = None
            for i, text in enumerate(texts):
                if pattern.match(text.split(" | ")[0]):
                    hit = i
                    break
            facts[f"row:{name}"] = texts[hit] if hit is not None else None
            row_shot = out_dir / f"{label.lower()}_row_{re.sub(r'[^A-Za-z0-9]+', '_', name).lower()}.png"
            if hit is None:
                # Absent on this side is a RESULT, not a crash; shoot the menu so the pair
                # still has a comparable image at this index.
                if await menu.count():
                    await menu.screenshot(path=str(row_shot))
                else:
                    await sp.screenshot(row_shot, full_page=False)
            else:
                target = options.nth(hit)
                await target.scroll_into_view_if_needed()
                await page.wait_for_timeout(600)
                # Re-read after the scroll: a virtualised list reuses the element at that index.
                shown = re.sub(r"\s*\n\s*", " | ", (await target.inner_text()).strip())
                if not pattern.match(shown.split(" | ")[0]):
                    target = options.filter(has_text=re.compile(rf"^\s*{re.escape(name)}\b")).first
                facts[f"shot:{name}"] = re.sub(r"\s*\n\s*", " | ", (await target.inner_text()).strip())
                await target.screenshot(path=str(row_shot))
            shots.append(row_shot)

    print(f"[{label}] {json.dumps(facts)}", flush=True)
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
