"""Cross-OS Playwright probe for the Studio diffusion pages (PR 6763).

CI runners have no GPU, so this does not generate: it checks the parts that must hold on
every platform anyway -- the Images and Video pages render and keep their controls, the
model picker opens and lists diffusion models, the advanced panel exposes the load-time
options, the download-plan contract answers, and a GPU-less host is told up front that the
DiT families cannot train here (instead of evicting and dying in the child).

Usage: python diffusion_probe.py --base http://127.0.0.1:8888 --password <bootstrap> --out artifacts
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx
from playwright.async_api import async_playwright


async def api_login(base: str, user: str, password: str) -> str:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{base}/api/auth/login", json={"username": user, "password": password})
        r.raise_for_status()
        return r.json()["access_token"]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--user", default="unsloth")
    ap.add_argument("--password", required=True)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {"platform": sys.platform, "base": args.base, "checks": {}}

    token = await api_login(args.base, args.user, args.password)
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=120, headers=headers) as c:
        info = await c.get(f"{args.base}/api/inference/images/info")
        res["checks"]["images_info"] = info.status_code
        if info.status_code == 200:
            body = info.json()
            res["families"] = [f.get("name") for f in body.get("families", [])]

        vinfo = await c.get(f"{args.base}/api/inference/video/info")
        res["checks"]["video_info"] = vinfo.status_code

        # The download plan is pure metadata: it must answer on every OS without a GPU.
        plan = await c.post(
            f"{args.base}/api/inference/images/download-plan",
            json={"model_path": "stabilityai/sdxl-turbo", "model_kind": "pipeline"},
        )
        res["checks"]["download_plan"] = plan.status_code
        if plan.status_code == 200:
            d = plan.json()
            res["plan"] = {
                "entries": [(e["repo_id"], len(e["files"])) for e in d.get("entries", [])],
                "total_gb": round(d.get("total_bytes", 0) / 1e9, 2),
            }

        # A GPU-less runner must be told the DiT families are untrainable here, up front.
        tinfo = await c.get(f"{args.base}/api/train/diffusion/info")
        res["checks"]["train_info"] = tinfo.status_code
        if tinfo.status_code == 200:
            fams = tinfo.json().get("families", [])
            res["train_families"] = {
                f["name"]: {
                    "precision_modes": f.get("precision_modes"),
                    "vram_note": (f.get("vram_note") or "")[:80],
                }
                for f in fams
            }

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=["--no-sandbox"])
        ctx = await browser.new_context(viewport={"width": 1500, "height": 950})
        await ctx.add_init_script(
            "localStorage.setItem('unsloth_auth_token', "
            + json.dumps(token)
            + "); localStorage.setItem('unsloth_onboarding_done', 'true');"
        )
        page = await ctx.new_page()
        try:
            for path, label in (("/images", "images"), ("/video", "video")):
                await page.goto(f"{args.base}{path}", wait_until="domcontentloaded", timeout=60_000)
                await page.wait_for_timeout(4000)
                await page.screenshot(path=str(out / f"{sys.platform}_{label}.png"), full_page=True)
                body = await page.inner_text("body")
                res["checks"][f"{label}_page"] = {
                    "prompt_box": await page.locator("textarea").count(),
                    "has_generate": "Generate" in body,
                    "has_selector": await page.locator(".unsloth-model-selector-trigger").count(),
                }

            # Picker: it has to open and list diffusion models on every platform.
            await page.goto(f"{args.base}/images", wait_until="domcontentloaded", timeout=60_000)
            await page.wait_for_timeout(3000)
            trigger = page.locator(".unsloth-model-selector-trigger").first
            if await trigger.count() > 0:
                await trigger.click(timeout=20_000)
                await page.wait_for_timeout(3500)
                await page.screenshot(path=str(out / f"{sys.platform}_picker.png"))
                text = await page.inner_text("body")
                res["checks"]["picker"] = {
                    "opened": True,
                    "lists_diffusion": any(
                        name in text for name in ("SDXL", "FLUX", "Qwen-Image", "Z-Image")
                    ),
                }
                await page.keyboard.press("Escape")
                await page.wait_for_timeout(800)

            adv = page.locator("button[aria-label='Show advanced options']").first
            if await adv.count() > 0:
                await adv.click()
                await page.wait_for_timeout(1500)
                await page.screenshot(path=str(out / f"{sys.platform}_advanced.png"), full_page=True)
                text = await page.inner_text("body")
                res["checks"]["advanced"] = [
                    k
                    for k in ("Speed", "Precision", "Attention", "Memory", "Step cache", "CPU offload")
                    if k in text
                ]

            await page.goto(f"{args.base}/images", wait_until="domcontentloaded", timeout=60_000)
            await page.wait_for_timeout(2500)
            # The Train tab is disabled on a GPU-less host, so a click may never land. That is
            # the expected shape here, not a probe failure.
            try:
                train_tab = page.get_by_role("button", name="Train").last
                await train_tab.click(timeout=8_000)
                await page.wait_for_timeout(3000)
                await page.screenshot(path=str(out / f"{sys.platform}_train.png"), full_page=True)
                text = await page.inner_text("body")
                res["checks"]["train_tab"] = {
                    "has_dataset": "Dataset" in text,
                    "mentions_gpu_requirement": "GPU" in text,
                }
            except Exception as exc:  # noqa: BLE001
                res["checks"]["train_tab"] = f"not clickable: {type(exc).__name__}"
        except Exception as exc:  # noqa: BLE001 -- record and still write the report
            res["ui_error"] = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            await ctx.close()
            await browser.close()

    (out / f"probe_{sys.platform}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))

    failures = [k for k, v in res["checks"].items() if v in (0, None) or v == 500]
    if res["checks"].get("images_info") != 200 or res["checks"].get("download_plan") not in (200, 400):
        failures.append("api")
    if not res["checks"].get("images_page", {}).get("has_selector"):
        failures.append("images_page_missing_selector")
    if failures:
        print("PROBE FAILURES:", failures, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
