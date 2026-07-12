"""Drive the real Unsloth Studio folder-browser UI with Playwright.

Goes to the throwaway /fbtest route (renders <FolderBrowser open/>), then:
  1. screenshots the browser open on HOME with the spoofed C:/D:/E: drive chips
  2. hops C: -> D: -> E: via the suggestion chips (the #6368 feature)
  3. confirms the spoofed C:/Windows + C:/Program Files are NOT listed (denied)
  4. navigates into a model dir on C:
records a video the caller turns into a GIF. Exits non-zero on failure and
dumps diagnostics (page HTML + visible button labels) so CI iteration is cheap.

Usage: python folder_browser_playwright.py --url http://127.0.0.1:8888 \
           --password <bootstrap> --out out
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx
from playwright.async_api import async_playwright

USERNAME = "unsloth"


async def login(url: str, password: str) -> dict:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(f"{url}/api/auth/login", json={"username": USERNAME, "password": password})
        r.raise_for_status()
        b = r.json()
        return {"access": b["access_token"], "refresh": b.get("refresh_token", "")}


def seed_js(tok: dict) -> str:
    # Exact keys the SPA reads (features/auth/session.ts). Seed onboarding-done
    # so the index guard never bounces us off /fbtest.
    return (
        "(() => { const s = {"
        f"'unsloth_auth_token': {tok['access']!r},"
        f"'unsloth_auth_refresh_token': {tok['refresh']!r},"
        "'unsloth_onboarding_done': 'true'"
        "}; for (const k of Object.keys(s)) { try { window.localStorage.setItem(k, s[k]); } catch(e){} } })();"
    )


async def dump(page, out: Path, tag: str):
    try:
        (out / f"DIAG_{tag}.html").write_text(await page.content(), encoding="utf-8")
        labels = await page.eval_on_selector_all(
            "button, [role=button]", "els => els.map(e => (e.title||e.textContent||'').trim()).filter(Boolean)"
        )
        (out / f"DIAG_{tag}_buttons.txt").write_text("\n".join(labels), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print("diag failed:", e)


async def run(url: str, password: str, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    vids = out / "video"
    vids.mkdir(exist_ok=True)
    tok = await login(url, password)
    print("logged in; access token len", len(tok["access"]))

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 860},
            record_video_dir=str(vids),
            record_video_size={"width": 1280, "height": 860},
        )
        await ctx.add_init_script(seed_js(tok))
        page = await ctx.new_page()

        shots = []

        async def shot(name: str):
            p = out / name
            await page.screenshot(path=str(p))
            shots.append(p)
            print("shot", name)

        try:
            await page.goto(f"{url}/fbtest", wait_until="domcontentloaded")
            dialog = page.locator('[data-testid="folder-browser-dialog"]')
            await dialog.wait_for(state="visible", timeout=30000)
            # entries/chips load async; give the first browse call a beat.
            await page.wait_for_timeout(2500)
            await shot("01_open_home.png")

            # The spoofed drives appear as suggestion chips (title = full path).
            for drive in ("C:", "D:", "E:"):
                chip = page.locator(f'button[title$="/{drive}"]').first
                assert await chip.count() > 0, f"no suggestion chip for drive {drive}"
            print("drive chips present: C: D: E:")

            # Hop to C: and confirm denied system dirs are hidden, model dir shown.
            await page.locator('button[title$="/C:"]').first.click()
            await page.wait_for_timeout(1800)
            await shot("02_drive_C.png")
            names = await page.eval_on_selector_all(
                '[data-testid="folder-browser-dialog"] span.truncate.font-mono',
                "els => els.map(e => e.textContent.trim())",
            )
            print("C: entries:", names)
            assert "Models" in names, f"expected Models on C:, got {names}"
            assert "Windows" not in names, "C:/Windows should be hidden (denied)"
            assert "Program Files" not in names, "C:/Program Files should be hidden (denied)"

            # Into C:/Models -> the gguf dir with a models badge.
            await page.locator('[data-testid="folder-browser-dialog"] span.truncate.font-mono', has_text="Models").first.click()
            await page.wait_for_timeout(1500)
            await shot("03_C_Models.png")

            # Hop to D: then E: -- the cross-drive navigation #6368 asked for.
            await page.locator('button[title$="/D:"]').first.click()
            await page.wait_for_timeout(1500)
            await shot("04_drive_D.png")
            await page.locator('button[title$="/E:"]').first.click()
            await page.wait_for_timeout(1500)
            await shot("05_drive_E.png")

            print("SUCCESS: drove folder browser across C:/D:/E: with denied dirs hidden")
            rc = 0
        except Exception as e:  # noqa: BLE001
            print("DRIVER ERROR:", repr(e))
            await dump(page, out, "fail")
            try:
                await shot("99_error.png")
            except Exception:
                pass
            rc = 1
        finally:
            await ctx.close()
            await browser.close()

        # Rename the finalized video deterministically.
        webms = sorted(vids.glob("*.webm"))
        if webms:
            final = out / "folder_browser.webm"
            webms[-1].rename(final)
            print("video:", final)
        return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    sys.exit(asyncio.run(run(a.url, a.password, Path(a.out))))


if __name__ == "__main__":
    main()
