"""Scene: the Video model picker's MiniMax H3 rows on a 24 GiB card.

Serves PR 12409 (`/api/system.diffusers_offload_tiers` + frontend `offload-fit-tiers.ts`): the
backend publishes an extra fit tier for the MiniMax-H3 Diffusers (INT8 streamed) row, which the
picker unions with the catalog's own tiers. Before, the catalog's lowest tier is 30 GiB VRAM /
80 GiB available RAM (plus quantised streaming), so on a 24 GiB card the Diffusers row reads
"Does not fit" and only the GGUF row is offered. After, the 11.5 GiB tier admits it.

HOW THE 24 GiB CARD IS MADE. The picker's verdict (`offloadTierMet`) compares the tier against
`DeviceBudget.gpuGb`, which `artifactBudget` takes from `GpuInfo.memoryTotalGb`, the SUM of
`/api/system.gpu.devices[].memory_total_gb` -- TOTAL VRAM, read from nvidia-smi. Free VRAM is not
consulted at all, so `gpu_ballast.py` on a 180 GB B200 leaves the verdict untouched (both halves
would show the B200 verdict). There is no backend override for an NVIDIA device total (the only
budget-sized total in hardware.py is the Vulkan inventory). So the scene rewrites exactly that
field in the browser: `/api/system` is fetched from the REAL backend and only
`gpu.devices` / `inference_gpu.devices` are narrowed to one device with `memory_total_gb` = 24.
Everything the PR changes on the backend (`diffusers_offload_tiers`, `quantised_streaming`) and
the host RAM figures pass through untouched, identically on both sides.

`quantised_streaming` is probed during warm-up and reads false until then; every H3 Diffusers
tier requires it, so shooting early photographs "Does not fit" on BOTH sides. Poll for it.
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

DEFAULT_GPU_GB = 24.0
DEFAULT_QUERY = "MiniMax-H3"


def _narrow(gpu: Optional[dict], gpu_gb: float) -> Optional[dict]:
    """One idle device with `gpu_gb` total and all of it free.

    Free/used are pinned rather than derived from the real card: the B200 underneath is shared,
    and another job's allocation moving between the two sides would otherwise put a difference
    in the pair that is not the PR's (seen live: 0.0 vs 20.5 GiB free on the first run)."""
    if not isinstance(gpu, dict) or not gpu.get("devices"):
        return gpu
    dev = dict(gpu["devices"][0])
    dev["memory_total_gb"] = gpu_gb
    dev["vram_used_gb"] = 0.0
    dev["vram_free_gb"] = gpu_gb
    dev["vram_utilization_pct"] = 0.0
    out = dict(gpu)
    out["devices"] = [dev]
    return out


async def _open_picker(page, timeout_ms: int = 60_000):
    trigger = page.locator("[data-tour='video-model']").first
    if not await trigger.count():
        trigger = page.locator("button").filter(
            has_text=re.compile(r"Select video model|MiniMax|LTX|Wan")).first
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
                gpu_gb: float = DEFAULT_GPU_GB, query: str = DEFAULT_QUERY,
                **_: object) -> tuple[list[Path], dict]:
    facts: dict = {"gpu_gb_override": gpu_gb}

    deadline = time.time() + 420
    while time.time() < deadline:
        try:
            if api_get(session, "/api/system").get("quantised_streaming") is True:
                break
        except Exception:  # noqa: BLE001 -- a startup 5xx is not the answer
            pass
        time.sleep(5)
    facts["quantised_streaming_wait_s"] = round(420 - (deadline - time.time()), 1)
    system = api_get(session, "/api/system")
    facts["quantised_streaming"] = system.get("quantised_streaming")
    facts["diffusers_offload_tiers"] = system.get("diffusers_offload_tiers", "<absent>")
    facts["ram_available_gb"] = (system.get("memory") or {}).get("available_gb")
    real = ((system.get("gpu") or {}).get("devices") or [{}])[0]
    facts["real_gpu"] = {k: real.get(k) for k in ("name", "memory_total_gb", "vram_free_gb")}
    if system.get("quantised_streaming") is not True:
        raise RuntimeError("quantised_streaming never became true: every H3 Diffusers tier "
                           "requires it, so both halves would read 'Does not fit'")

    seen: dict = {}

    async def _rewrite(route):
        resp = await route.fetch()
        try:
            body = await resp.json()
        except Exception:  # noqa: BLE001
            await route.fulfill(response=resp)
            return
        if isinstance(body, dict) and "gpu" in body:
            body["gpu"] = _narrow(body.get("gpu"), gpu_gb)
            if body.get("inference_gpu"):
                body["inference_gpu"] = _narrow(body.get("inference_gpu"), gpu_gb)
            dev = ((body.get("gpu") or {}).get("devices") or [{}])[0]
            seen.update({
                "picker_gpu_total_gb": dev.get("memory_total_gb"),
                "picker_gpu_free_gb": dev.get("vram_free_gb"),
                "picker_ram_available_gb": (body.get("memory") or {}).get("available_gb"),
                "picker_quantised_streaming": body.get("quantised_streaming"),
                "picker_diffusers_offload_tiers": body.get("diffusers_offload_tiers", "<absent>"),
            })
        await route.fulfill(response=resp, json=body)

    shots: list[Path] = []
    init = seed_init_script(
        type("A", (), {"access_token": session.access_token,
                       "refresh_token": session.refresh_token})(), []
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    async with open_chat(session.base_url, init_scripts=[init],
                         viewport=(1500, 1000), headless=True) as sp:
        page = sp.page
        await page.route(re.compile(r".*/api/system(\?.*)?$"), _rewrite)
        await page.goto(f"{session.base_url}/video", wait_until="domcontentloaded")
        await page.wait_for_timeout(9_000)
        options = await _open_picker(page)
        search = page.locator(".unsloth-model-selector-menu input").first
        if await search.count():
            await search.fill(query)
        await page.wait_for_timeout(6_000)

        await page.mouse.move(5, 995)  # hover undims a row: park the pointer off the menu
        await page.wait_for_timeout(800)
        texts = []
        rows_detail = []
        for i in range(await options.count()):
            opt = options.nth(i)
            text = re.sub(r"\s*\n\s*", " | ", (await opt.inner_text()).strip())
            texts.append(text)
            detail = await opt.evaluate("""el => {
                const inner = el.querySelector(':scope > span');
                const badges = [...el.querySelectorAll('[aria-label]')].map(b => b.getAttribute('aria-label'));
                return {opacity: inner ? getComputedStyle(inner).opacity : null, badges};
            }""")
            rows_detail.append({"row": text, **detail})
        facts["option_count"] = len(texts)
        h3 = [i for i, t in enumerate(texts) if "minimax-h3" in t.lower() or "minimax h3" in t.lower()]
        facts["h3_rows"] = [rows_detail[i] for i in h3]
        facts.update(seen)
        if not h3:
            raise RuntimeError(f"no MiniMax H3 row in the open picker: {texts[:8]}")
        diff_i = next((i for i in h3 if texts[i].startswith("MiniMax H3 ")), None)
        if diff_i is None:
            raise RuntimeError(f"no curated 'MiniMax H3' (Diffusers) row: {texts}")
        d = rows_detail[diff_i]
        facts["diffusers_row"] = texts[diff_i]
        facts["diffusers_row_does_not_fit_badge"] = "Does not fit" in d["badges"]
        facts["diffusers_row_dimmed"] = d["opacity"] not in (None, "1")

        menu = page.locator(".unsloth-model-selector-menu").first
        shot = out_dir / f"{label.lower()}_h3_rows.png"
        if await menu.count():
            await menu.screenshot(path=str(shot))
        else:
            await sp.screenshot(shot, full_page=False)
        shots.append(shot)

        # Second shot: hover the Diffusers row so its fit mark paints, and the mark itself so the
        # tooltip names the verdict in words. AFTER has no mark, so the row hover is the shot.
        row = options.nth(diff_i)
        await row.hover()
        badge = row.locator('[aria-label="Does not fit"]')
        if await badge.count():
            await badge.first.hover()
        await page.wait_for_timeout(1_200)
        box = await (menu if await menu.count() else row).bounding_box()
        hover_shot = out_dir / f"{label.lower()}_h3_diffusers_hover.png"
        await page.screenshot(path=str(hover_shot), clip={
            "x": max(0, box["x"] - 10), "y": max(0, box["y"] - 10),
            "width": box["width"] + 20, "height": box["height"] + 70})
        tip = page.locator("[role='tooltip']")
        facts["hover_tooltip"] = (await tip.first.inner_text()).strip() if await tip.count() else None
        shots.append(hover_shot)

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
