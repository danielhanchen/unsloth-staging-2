"""A Chromium page for scenes on a heavily loaded host: open_chat's first goto uses a fixed 30 s timeout."""

from __future__ import annotations

from contextlib import asynccontextmanager

from playwright.async_api import async_playwright


@asynccontextmanager
async def slow_page(
    base_url: str,
    init_js: str,
    viewport = (1400, 1000),
    timeout_ms: int = 240_000,
):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless = True)
        try:
            ctx = await browser.new_context(viewport = {"width": viewport[0], "height": viewport[1]})
            ctx.set_default_timeout(timeout_ms)
            ctx.set_default_navigation_timeout(timeout_ms)
            await ctx.add_init_script(init_js)
            page = await ctx.new_page()
            await page.goto(f"{base_url}/chat", wait_until = "domcontentloaded")
            yield page
        finally:
            await browser.close()
