#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0
"""Capture the Unsloth JupyterLab branding (login footer, top-bar logo, Help
About dialog) and a WebM of the spinning-logo splash during lab bootstrap.

Raw Playwright (no studio_test_kit) against the JupyterLab port. Every capture
step is independent and best-effort: a missing selector logs and continues, and
the script always exits 0 (screenshots are evidence, not gates).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def capture(base_url: str, password: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    shots: list[str] = []

    def shoot(page, name: str, full: bool = False) -> None:
        try:
            page.screenshot(path=str(out_dir / name), full_page=full)
            shots.append(name)
            print(f"[branding] shot {name}")
        except Exception as e:  # noqa: BLE001
            print(f"[branding] shot {name} failed: {e!r}"[:160])

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(video_dir),
            record_video_size={"width": 1440, "height": 900},
        )
        page = context.new_page()

        # 1. Login page -- the AGPLv3 / "Built by Unsloth" footer.
        page.goto(f"{base_url}/login", wait_until="domcontentloaded")
        page.wait_for_timeout(1500)
        shoot(page, "01_login_footer.png", full=True)

        # 2. Log in (recording captures the spinning splash during lab bootstrap).
        try:
            box = page.locator("input[name='password'], input#password_input").first
            box.fill(password)
            page.locator("button:has-text('Log in'), button[type='submit']").first.click()
        except Exception as e:  # noqa: BLE001
            print(f"[branding] login submit problem: {e!r}"[:160])

        # 3. Wait for the Lab shell; the splash overlay shows in the recorded video.
        try:
            page.locator("#jp-top-panel, .jp-MainLogo, #jp-MainLogo, .lm-MenuBar").first.wait_for(
                state="visible", timeout=60_000
            )
        except Exception as e:  # noqa: BLE001
            print(f"[branding] lab shell wait problem: {e!r}"[:160])
        page.wait_for_timeout(3000)
        shoot(page, "02_lab_loaded.png", full=False)

        # 4. Top-bar Unsloth logo close-up (best-effort crop via element shot).
        try:
            logo = page.locator("#jp-top-panel .jp-MainLogo, #jp-MainLogo, .jp-Toolbar img, #jp-top-panel img").first
            logo.wait_for(state="visible", timeout=8000)
            logo.screenshot(path=str(out_dir / "03_topbar_logo.png"))
            shots.append("03_topbar_logo.png")
            print("[branding] shot 03_topbar_logo.png")
        except Exception as e:  # noqa: BLE001
            print(f"[branding] topbar logo shot skipped: {e!r}"[:160])

        # 5. Help > About Unsloth Docker Studio dialog.
        try:
            page.locator(".lm-MenuBar-item:has-text('Help'), [role='menuitem']:has-text('Help')").first.click()
            page.wait_for_timeout(600)
            page.locator(
                ".lm-Menu-itemLabel:has-text('About Unsloth'), [role='menuitem']:has-text('About Unsloth')"
            ).first.click()
            page.locator(".jp-Dialog, .lm-Widget.jp-Dialog").first.wait_for(state="visible", timeout=8000)
            page.wait_for_timeout(800)
            shoot(page, "04_about_dialog.png", full=False)
            # dismiss the dialog before the splash reload
            page.locator(".jp-Dialog button:has-text('Close'), .jp-Dialog .jp-mod-accept").first.click()
            page.wait_for_timeout(400)
        except Exception as e:  # noqa: BLE001
            print(f"[branding] About dialog skipped: {e!r}"[:160])

        # 6. Spinning-logo splash: a reload re-triggers JupyterLab's ISplashScreen
        # (our Unsloth splash). Burst-screenshot to catch a clean spin frame, and
        # the recorded video captures the full splash -> lab transition for a GIF.
        try:
            page.reload(wait_until="commit")
            for k in range(14):
                page.wait_for_timeout(250)
                try:
                    page.screenshot(path=str(out_dir / f"05_splash_{k:02d}.png"))
                except Exception:  # noqa: BLE001
                    pass
            shots.append("05_splash_burst")
            print("[branding] splash reload burst captured")
        except Exception as e:  # noqa: BLE001
            print(f"[branding] splash reload skipped: {e!r}"[:160])

        context.close()
        browser.close()

    # Rename the recorded webm deterministically.
    webm_final = None
    webms = sorted(video_dir.glob("*.webm"))
    if webms:
        webm_final = video_dir / "jupyter_splash.webm"
        if webm_final.exists():
            webm_final.unlink()
        webms[-1].rename(webm_final)

    (out_dir / "branding_manifest.json").write_text(
        json.dumps({"screenshots": shots, "video_webm": str(webm_final) if webm_final else None}, indent=2)
    )
    print(f"[branding] screenshots={shots} webm={webm_final}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:18888")
    ap.add_argument("--password", help="JUPYTER_PASSWORD")
    ap.add_argument("--password-file")
    ap.add_argument("--out-dir", default="branding_capture")
    args = ap.parse_args(argv)
    pw = args.password
    if args.password_file:
        pw = Path(args.password_file).read_text(encoding="utf-8").strip()
    try:
        capture(args.base_url, pw or "", Path(args.out_dir))
    except Exception as e:  # noqa: BLE001 -- evidence capture must never fail the job
        print(f"[branding] FAILED (non-fatal): {e!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
