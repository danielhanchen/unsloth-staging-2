"""Scene: a cached image GGUF whose companions are NOT cached, for PR 12557.

On the Hub model inspector of an image GGUF repo, a quant whose .gguf is in the
cache used to read green "On device" even though Run still has to fetch the text
encoder / VAE from the base repo. PR 12557 shows a warning "Partial" tag instead,
with the tooltip "Model on device. X GB of required assets download on Run.", on
both the selected-quant trigger and the quant's row in the menu.

The cache must hold ONLY the GGUF (companions absent), so the driver points both
Studios at a dedicated cache via --studio-env. The scene asserts the cache state
before shooting, since a cache holding the companions would make AFTER read
"On device" too and the pair would prove nothing.

Facts carry tag VALUES (inner_text), not headings (trap 13), plus the tooltip.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

WORKSPACE = Path(
    os.environ.get("WORKSPACE")
    or os.environ.get("UNSLOTH_WORKSPACE")
    or Path(__file__).resolve().parents[2]
)
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from pr_ui_scenes._common import Session, api_post  # noqa: E402
from pr_ui_scenes.gguf_picker_rows import _select_repo  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402
from studio_test_kit.ui import open_chat  # noqa: E402

DEFAULT_REPO = "unsloth/FLUX.2-klein-4B-GGUF"
DEFAULT_BASE = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_FILE = "flux-2-klein-4b-Q2_K.gguf"
DEFAULT_QUANT = "Q2_K"

TAG_RE = re.compile(r"^(On device|Partial)$")


def _cache_state(cache_hub: Path, repo: str, base_repo: str, filename: str) -> dict:
    def d(r: str) -> Path:
        return cache_hub / ("models--" + r.replace("/", "--"))

    return {
        "gguf_cached": any(d(repo).glob(f"snapshots/*/{filename}")),
        "base_repo_in_cache": d(base_repo).exists(),
    }


async def _tag_text(scope) -> list[str]:
    tags = scope.get_by_text(TAG_RE)
    return [(await tags.nth(i).inner_text()).strip() for i in range(await tags.count())]


async def _hover_tooltip(page, tag) -> str:
    await tag.hover()
    tip = page.get_by_role("tooltip")
    try:
        await tip.first.wait_for(state = "visible", timeout = 4_000)
        return (await tip.first.inner_text()).strip()
    except Exception:  # noqa: BLE001 -- "On device" has no tooltip; absence is the fact
        return ""


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    repo: str = DEFAULT_REPO,
    base_repo: str = DEFAULT_BASE,
    filename: str = DEFAULT_FILE,
    quant: str = DEFAULT_QUANT,
    cache_hub: str = "",
    **_: object,
) -> tuple[list[Path], dict]:
    facts: dict = {}
    if cache_hub:
        state = _cache_state(Path(cache_hub), repo, base_repo, filename)
        facts["cache"] = state
        if not state["gguf_cached"] or state["base_repo_in_cache"]:
            raise RuntimeError(
                f"[{label}] cache not in the scene's precondition "
                f"(GGUF cached, base absent): {state}"
            )

    # What Run would still fetch, from the server being photographed.
    try:
        plan = api_post(
            session,
            "/api/inference/images/download-plan",
            {"model_path": repo, "gguf_filename": filename, "model_kind": "gguf"},
            timeout = 300,
        )
        facts["plan_entries"] = [
            {
                "repo": e["repo_id"],
                "gb": round(e["bytes"] / 1e9, 2),
                "checkpoint": e.get("checkpoint"),
            }
            for e in plan.get("entries", [])
        ]
    except Exception as exc:  # noqa: BLE001
        facts["plan_error"] = f"{type(exc).__name__}: {exc}"

    shots: list[Path] = []
    out_dir.mkdir(parents = True, exist_ok = True)
    init = seed_init_script(
        type(
            "A", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
        )(),
        [],
    )
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = (1500, 1000), headless = True
    ) as sp:
        page = sp.page
        await page.goto(f"{session.base_url}/hub", wait_until = "domcontentloaded")
        await _select_repo(page, repo)

        trigger = (
            page.locator("button.hub-menu-trigger")
            .filter(has_text = re.compile(r"Q\d|BF16|F16|Select quantization"))
            .first
        )
        await trigger.wait_for(state = "visible", timeout = 60_000)

        # Select the cached quant from the menu (whatever the default pick was).
        await trigger.click()
        rows = page.locator("[role='button'][aria-pressed]")
        await rows.first.wait_for(state = "visible", timeout = 30_000)
        row = rows.filter(has = page.get_by_text(quant, exact = True)).first
        await row.click()
        await page.wait_for_timeout(1_000)
        if await rows.first.is_visible():
            await page.keyboard.press("Escape")

        trig_text = ""
        # The companion plan resolves asynchronously after selection: wait for the
        # status tag to settle (AFTER flips On device -> Partial once it lands).
        for _ in range(30):
            await page.wait_for_timeout(1_000)
            trig_text = (await trigger.inner_text()).replace("\n", " ")
            if quant in trig_text and "Partial" in trig_text:
                break
        await page.wait_for_timeout(3_000)
        trig_text = (await trigger.inner_text()).replace("\n", " ")
        if quant not in trig_text:
            raise RuntimeError(f"[{label}] trigger does not show {quant}: {trig_text!r}")
        facts["trigger_text"] = trig_text
        trig_tags = await _tag_text(trigger)
        facts["trigger_tags"] = trig_tags

        trig_box = await trigger.bounding_box()
        trig_tag = trigger.get_by_text(TAG_RE).first
        facts["trigger_tooltip"] = (await _hover_tooltip(page, trig_tag)) if trig_tags else ""
        shot = out_dir / f"{label.lower()}_trigger.png"
        if trig_box:
            await page.screenshot(
                path = str(shot),
                clip = {
                    "x": max(0, trig_box["x"] - 200),
                    "y": max(0, trig_box["y"] - 110),
                    "width": trig_box["width"] + 400,
                    "height": trig_box["height"] + 140,
                },
            )
        else:
            await sp.screenshot(shot, full_page = False)
        shots.append(shot)
        await page.mouse.move(5, 5)
        await page.wait_for_timeout(800)

        # The quant menu row.
        await trigger.click()
        await rows.first.wait_for(state = "visible", timeout = 30_000)
        row = rows.filter(has = page.get_by_text(quant, exact = True)).first
        await row.wait_for(state = "visible", timeout = 10_000)
        facts["row_text"] = (await row.inner_text()).replace("\n", " ")
        row_tags = await _tag_text(row)
        facts["row_tags"] = row_tags
        row_tag = row.get_by_text(TAG_RE).first
        facts["row_tooltip"] = (await _hover_tooltip(page, row_tag)) if row_tags else ""
        await page.wait_for_timeout(500)
        rbox = await row.bounding_box()
        shot = out_dir / f"{label.lower()}_menu_row.png"
        if rbox:
            await page.screenshot(
                path = str(shot),
                clip = {
                    "x": max(0, rbox["x"] - 200),
                    "y": max(0, rbox["y"] - 110),
                    "width": rbox["width"] + 260,
                    "height": rbox["height"] + 140,
                },
            )
        else:
            await sp.screenshot(shot, full_page = False)
        shots.append(shot)

        full = out_dir / f"{label.lower()}_page.png"
        await sp.screenshot(full, full_page = False)
        shots.append(full)
        await page.keyboard.press("Escape")
    return shots, facts


if __name__ == "__main__":
    import argparse

    from pr_ui_scenes._common import studio_session

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required = True)
    ap.add_argument("--home", type = Path, required = True)
    ap.add_argument("--password", required = True)
    ap.add_argument("--out", type = Path, required = True)
    ap.add_argument("--label", default = "AFTER")
    ap.add_argument("--cache-hub", default = "")
    a = ap.parse_args()
    s = studio_session(a.url, a.home, a.password)
    print(asyncio.run(drive(s, a.out, a.label, cache_hub = a.cache_hub)))
