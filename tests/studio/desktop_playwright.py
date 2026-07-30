# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Drive the packaged desktop app's UI with Playwright, like a person.

Two layers, because no single one works everywhere:

  native            Windows only. WebView2 opens a CDP endpoint when the app is launched
                    with WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=N,
                    so `connect_over_cdp` attaches to the ACTUAL packaged renderer --
                    the real window, the real Tauri IPC bridge.
  backend-attached  Everywhere. A Chromium pointed at the backend the desktop app spawned
                    (http://127.0.0.1:<port>), with the app's own JWT seeded into
                    localStorage. Same backend and same React bundle as the window, but
                    a different renderer -- so it catches backend/frontend contract
                    regressions, not WebView-specific ones.

macOS gets only the second layer: there is no desktop WKWebView WebDriver, and Tauri's
embedded-WebDriver plugin cannot be retrofitted into an already-released bundle.

    python tests/studio/desktop_playwright.py --layer auto --out artifacts/ui

Requires `playwright` + a chromium build for the backend-attached layer; the native layer
needs neither (it attaches to a browser that is already running).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

# desktop_drive owns backend discovery and the desktop-secret exchange; reuse it rather
# than reimplementing the port/auth dance in a second place.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from desktop_drive import (  # noqa: E402
    Backend,
    Report,
    desktop_login,
    discover_backend,
)

CDP_URL = "http://127.0.0.1:9222"

# The composer is the one element that means "the app is actually usable". Scoping to
# the form avoids the sidebar's own buttons, the pitfall studio_test_kit's README calls
# out for `button:has-text(...)`.
COMPOSER = "form:has(textarea) textarea, textarea"


def _init_script(token: str, refresh: str = "") -> str:
    """Seed the JWT the way the frontend stores it, BEFORE the first navigation.

    The token lives in localStorage, not a cookie, so it has to be planted via
    add_init_script or the first render bounces to the login screen.
    """
    payload = json.dumps({"access": token, "refresh": refresh})
    return f"""
    (() => {{
      const t = {payload};
      try {{
        localStorage.setItem('unsloth_access_token', t.access);
        localStorage.setItem('unsloth_refresh_token', t.refresh);
        localStorage.setItem('unsloth_token', t.access);
      }} catch (e) {{}}
    }})();
    """


async def _shoot(page, out_dir: Path, name: str) -> Path:
    path = out_dir / f"{name}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    await page.screenshot(path=str(path), full_page=False)
    return path


async def _walk_ui(page, report: Report, out_dir: Path, label: str) -> None:
    """The clicking-around part: prove the shell renders, the composer works, and the
    main navigation destinations do not blow up."""
    await _shoot(page, out_dir, f"{label}-01-landing")

    try:
        await page.wait_for_selector(COMPOSER, timeout=90_000)
        report.ok(f"[{label}] the chat composer rendered")
    except Exception as error:
        report.fail(f"[{label}] no chat composer after 90s: {type(error).__name__}")
        await _shoot(page, out_dir, f"{label}-01-no-composer")
        return

    # A blank WebKitGTK window still reports a composer if the DOM loaded but nothing
    # painted, so assert the page has real rendered text too.
    body_text = (await page.inner_text("body"))[:4000]
    if len(body_text.strip()) < 20:
        report.fail(f"[{label}] the window rendered almost no text ({len(body_text)} chars)")
    else:
        report.ok(f"[{label}] window has rendered content ({len(body_text)} chars)")

    # Console errors are how a frontend/backend contract break announces itself.
    errors: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

    for name in ("Chat", "Train", "Models", "Settings"):
        try:
            target = page.get_by_role("link", name=re.compile(rf"^{name}$", re.I)).first
            if await target.count() == 0:
                target = page.get_by_role("button", name=re.compile(rf"^{name}$", re.I)).first
            if await target.count() == 0:
                report.warn(f"[{label}] no navigation entry for {name!r}")
                continue
            await target.click(timeout=15_000)
            await page.wait_for_timeout(2500)
            await _shoot(page, out_dir, f"{label}-nav-{name.lower()}")
            report.ok(f"[{label}] navigated to {name}")
        except Exception as error:
            report.fail(f"[{label}] navigating to {name} failed: {type(error).__name__}: {error}")

    if errors:
        report.warn(f"[{label}] {len(errors)} console errors, first: {errors[0][:200]}")
        (out_dir / f"{label}-console-errors.txt").write_text("\n".join(errors))
    else:
        report.ok(f"[{label}] no console errors during the walk")


async def drive_native(report: Report, out_dir: Path) -> None:
    """Attach to the real WebView2 renderer over CDP (Windows only)."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        try:
            browser = await pw.chromium.connect_over_cdp(CDP_URL, timeout=60_000)
        except Exception as error:
            report.fail(
                f"could not attach to the packaged renderer at {CDP_URL}: "
                f"{type(error).__name__}: {error}. Was the app launched with "
                f"WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9222?"
            )
            return
        report.ok(f"attached to the packaged WebView2 renderer at {CDP_URL}")
        contexts = browser.contexts
        pages = [p for c in contexts for p in c.pages]
        if not pages:
            report.fail("the packaged renderer exposed no pages over CDP")
            await browser.close()
            return
        report.note("native_pages", [p.url for p in pages])
        page = pages[0]
        report.ok(f"packaged renderer page URL: {page.url}")
        await _walk_ui(page, report, out_dir, "native")
        # Detach without killing the app: the desktop process must survive for the rest
        # of the job.
        await browser.close()


async def drive_backend_attached(backend: Backend, report: Report, out_dir: Path) -> None:
    """A Chromium against the same backend the desktop window is using.

    Only usable against a Studio that serves the frontend. The PACKAGED app spawns
    `unsloth studio --api-only` (process.rs:537-541) and that mode does not mount the
    React bundle -- `/` returns 404 -- because the window renders the frontend embedded
    in the Tauri binary instead. So this layer covers the control arm, and the packaged
    app's UI has to be driven through its native renderer.
    """
    from playwright.async_api import async_playwright

    code, _ = backend.http("GET", "/", auth=False, raw=True, timeout=30)
    if code != 200:
        report.skip(
            f"this backend serves no frontend at / (HTTP {code}); it is running "
            f"--api-only, so the UI lives in the packaged renderer, not here"
        )
        return

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        await context.add_init_script(_init_script(backend.token or ""))
        page = await context.new_page()
        try:
            # networkidle never settles on an SSE/WebSocket SPA; wait for the DOM and
            # then for the composer instead.
            await page.goto(backend.base, wait_until="domcontentloaded", timeout=120_000)
            report.ok(f"loaded the frontend the backend serves at {backend.base}")
        except Exception as error:
            report.fail(
                f"the desktop backend at {backend.base} did not serve a frontend: "
                f"{type(error).__name__}: {error}"
            )
            await browser.close()
            return
        await _walk_ui(page, report, out_dir, "attached")
        await browser.close()


async def amain(args) -> int:
    report = Report()
    out_dir = Path(args.out)

    report.begin("discovery")
    backend = discover_backend(report, timeout=args.discover_timeout, pinned_port=args.port)
    if not backend:
        authenticated = False
    elif args.password:
        from desktop_drive import password_login

        authenticated = password_login(backend, report, args.password)
    else:
        authenticated = desktop_login(backend, report)
    report.end()

    layer = args.layer
    if layer == "auto":
        layer = "native" if sys.platform == "win32" else "attached"

    if layer in ("native", "both"):
        report.begin("ui_native")
        if not authenticated:
            # The window sits on the startup screen until its backend is reachable, and
            # never opens the WebView2 debug port, so a CDP refusal here is the backend
            # defect again rather than a UI one. Reporting it separately double-counts.
            report.skip("no reachable backend, so the window never left the startup screen")
        else:
            await drive_native(report, out_dir)
        report.end()

    if layer in ("attached", "both"):
        report.begin("ui_attached")
        if authenticated and backend:
            await drive_backend_attached(backend, report, out_dir)
        else:
            report.skip("no authenticated backend to attach to")
        report.end()

    print(report.summary(), flush=True)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report.scenarios, indent=2, default=str))
    failed = report.failed_scenarios
    if failed:
        print(f"::error::failing UI scenarios: {', '.join(failed)}", flush=True)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layer",
        choices=["auto", "native", "attached", "both"],
        default="auto",
        help="auto = native on Windows, backend-attached elsewhere",
    )
    parser.add_argument("--out", default="artifacts/ui", help="screenshot directory")
    parser.add_argument("--report", help="write the JSON report here")
    parser.add_argument("--discover-timeout", type=float, default=300.0)
    parser.add_argument("--port", type=int, help="pin the backend port")
    parser.add_argument("--password", help="password login for the non-Tauri control arm")
    args = parser.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
