#!/usr/bin/env python3
"""Screenshot + GIF of the Images tab after a CI generation, so the platform evidence is visual.

Runs after probe.py against the same Studio, seeds the JWT into localStorage the way the QA
probes do, and records the page while scrolling the gallery. Never fails the job on its own: a
missing screenshot must not mask a passing generation.

Usage: python shot.py --base http://127.0.0.1:8899 --out artifacts/ubuntu-latest
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx

USER = "unsloth"


async def tokens(base: str, password: str) -> tuple[str, str]:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{base}/api/auth/login", json={"username": USER, "password": password})
        r.raise_for_status()
        d = r.json()
        return d["access_token"], d.get("refresh_token", "")


def frames_to_gif(frames: list[Path], dest: Path, duration_ms: int = 700) -> bool:
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001 -- no Pillow: the PNGs are still uploaded
        return False
    images = []
    for f in frames:
        with Image.open(f) as im:
            images.append(im.convert("P", palette=Image.Palette.ADAPTIVE).copy())
    if not images:
        return False
    images[0].save(
        dest, format="GIF", save_all=True, append_images=images[1:], duration=duration_ms, loop=0
    )
    return True


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("STUDIO_BASE", "http://127.0.0.1:8899"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--password", default=os.environ.get("STUDIO_PASS", ""))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Written whatever happens, so the gate step downstream can tell "the UI was served and
    # screenshotted" from "something went wrong and the PNGs show a 404".
    verdict: dict = {"base": args.base, "pages": [], "served": False}

    def flush() -> None:
        (out / "ui_results.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    try:
        from playwright.async_api import async_playwright
    except Exception as exc:  # noqa: BLE001
        (out / "shot_error.txt").write_text(f"playwright unavailable: {exc}")
        verdict["error"] = f"playwright unavailable: {exc}"
        flush()
        return 0

    try:
        access, refresh = await tokens(args.base, args.password)
        frames: list[Path] = []
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(args=["--no-sandbox"])
            ctx = await browser.new_context(viewport={"width": 1440, "height": 900})
            await ctx.add_init_script(
                f"localStorage.setItem('unsloth_auth_token', {json.dumps(access)});"
                f"localStorage.setItem('unsloth_auth_refresh_token', {json.dumps(refresh)});"
                "localStorage.setItem('unsloth_onboarding_done', 'true');"
            )
            page = await ctx.new_page()
            for i, (url, wait) in enumerate(
                [(f"{args.base}/images", 6000), (f"{args.base}/images", 2500), (f"{args.base}/video", 4000)]
            ):
                resp = await page.goto(url, wait_until="domcontentloaded")
                await page.wait_for_timeout(wait)
                if i == 1:
                    await page.mouse.wheel(0, 600)
                    await page.wait_for_timeout(1200)
                shot = out / f"{i:02d}_{url.rsplit('/', 1)[-1]}.png"
                await page.screenshot(path=str(shot), full_page=False)
                frames.append(shot)
                # A screenshot on its own proves nothing: an api-only launch answers /images with
                # {"detail":"Not Found"} and the capture succeeds, so the artifact LOOKS like UI
                # evidence while showing a 404 body. Record what the page actually was.
                served = bool(
                    resp is not None
                    and resp.status == 200
                    and await page.locator("#root, [data-tour='navbar']").count()
                )
                verdict["pages"].append({
                    "url": url,
                    "status": None if resp is None else resp.status,
                    "served": served,
                })
            await ctx.close()
            await browser.close()
        frames_to_gif(frames, out / "flow.gif")
        verdict["served"] = bool(verdict["pages"]) and all(p["served"] for p in verdict["pages"])
    except Exception as exc:  # noqa: BLE001 -- evidence capture never fails the job
        (out / "shot_error.txt").write_text(str(exc)[:800])
        verdict["error"] = str(exc)[:800]
    flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
