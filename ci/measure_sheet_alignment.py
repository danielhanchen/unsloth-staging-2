#!/usr/bin/env python3
"""Measure sheet-header close-button alignment for unslothai/unsloth PR #7957.

Drives the throwaway Vite harness (see gen_sheet_harness.py) on two dev
servers -- BEFORE (main) and AFTER (main + PR) -- and reports, per sheet and
per UI font size, the signed vertical delta between the close button's centre
and the SheetTitle's centre.

    signedDelta = closeCentreY - titleCentreY     # + means the button sits low

Also records header height, close-button geometry, the rendered `type`
attribute, close-button count, and (for the document preview) whether the
title text rects intersect the close button.
"""

import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

MEASURE_JS = r"""
() => {
  const content = document.querySelector('[data-slot="sheet-content"]');
  if (!content) return { error: 'no sheet-content' };
  const header = content.querySelector('[data-slot="sheet-header"]');
  const title = content.querySelector('[data-slot="sheet-title"]');
  const closes = content.querySelectorAll('[data-slot="sheet-close"]');
  if (!title || closes.length === 0) {
    return { error: 'missing title or close', closeCount: closes.length };
  }
  const close = closes[0];
  const t = title.getBoundingClientRect();
  const c = close.getBoundingClientRect();
  const h = header ? header.getBoundingClientRect() : null;
  const cs = content.getBoundingClientRect();

  // Title text pieces (for the overlap check on the document preview).
  const pieces = [...title.children]
    .filter((el) => el.tagName !== 'svg')
    .map((el) => {
      const r = el.getBoundingClientRect();
      return { text: (el.textContent || '').slice(0, 40), left: r.left,
               right: r.right, top: r.top, bottom: r.bottom };
    });
  const overlaps = pieces.some(
    (p) => p.right > c.left + 0.01 && p.left < c.right - 0.01 &&
           p.bottom > c.top + 0.01 && p.top < c.bottom - 0.01);

  return {
    titleTop: t.top, titleBottom: t.bottom, titleHeight: t.height,
    titleCentre: t.top + t.height / 2,
    closeTop: c.top, closeBottom: c.bottom,
    closeHeight: c.height, closeWidth: c.width,
    closeCentre: c.top + c.height / 2,
    closeRightGap: cs.right - c.right,
    signedDelta: (c.top + c.height / 2) - (t.top + t.height / 2),
    headerHeight: h ? h.height : null,
    headerTop: h ? h.top : null,
    closeCount: closes.length,
    closeType: close.getAttribute('type'),
    closeLabel: (close.textContent || '').trim(),
    uiFontScale: getComputedStyle(document.documentElement)
      .getPropertyValue('--ui-font-scale').trim(),
    rootFontSize: getComputedStyle(document.documentElement).fontSize,
    titleFontSize: getComputedStyle(title).fontSize,
    titleLineHeight: getComputedStyle(title).lineHeight,
    titlePaddingRight: getComputedStyle(title).paddingRight,
    overlaps,
    pieces,
  };
}
"""


async def settle(page):
    """Wait for the sheet to finish opening before measuring."""
    await page.wait_for_selector('[data-slot="sheet-content"][data-state="open"]',
                                 state="visible", timeout=20_000)
    await page.evaluate("() => document.fonts.ready")
    # Sheet transition is 200ms; give it room, then require two identical reads.
    await page.wait_for_timeout(450)
    prev = None
    for _ in range(20):
        await page.evaluate(
            "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
        cur = await page.evaluate(MEASURE_JS)
        if prev is not None and cur.get("signedDelta") == prev.get("signedDelta") \
                and cur.get("titleTop") == prev.get("titleTop"):
            return cur
        prev = cur
        await page.wait_for_timeout(60)
    return prev


async def run(args):
    variants = {"AFTER": args.pr_url}
    sheets = ["details", "preview"]
    sizes = [int(s) for s in args.font_sizes.split(",")]
    shots = Path(args.screenshot_dir)
    shots.mkdir(parents=True, exist_ok=True)

    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=1, color_scheme="light")
        page = await ctx.new_page()
        for label, base in variants.items():
            for sheet in sheets:
                for fs in sizes:
                    url = (f"{base}/harness.html?sheet={sheet}&fs={fs}"
                           f"&name={args.name}&page={args.page}&w={args.preview_width}")
                    await page.goto(url, wait_until="domcontentloaded")
                    m = await settle(page)
                    m = dict(m or {})
                    m.update(variant=label, sheet=sheet, fontSize=fs)
                    results.append(m)
                    print(f"{label:6s} {sheet:8s} fs={fs:2d}  "
                          f"delta={m.get('signedDelta')}  "
                          f"titleH={m.get('titleHeight')}  "
                          f"hdrH={m.get('headerHeight')}  "
                          f"closes={m.get('closeCount')}  "
                          f"overlap={m.get('overlaps')}", flush=True)
                    if args.screenshots:
                        el = await page.query_selector('[data-slot="sheet-header"]')
                        if el:
                            await el.screenshot(
                                path=str(shots / f"{sheet}_fs{fs}_{label.lower()}.png"))
        await ctx.close()
        await browser.close()

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:5191")
    ap.add_argument("--pr-url", default="http://127.0.0.1:5192")
    ap.add_argument("--font-sizes", default="12,15,16,20")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--preview-width", type=int, default=704)
    ap.add_argument("--name", default="quarterly-report.pdf")
    ap.add_argument("--page", default="12")
    ap.add_argument("--screenshots", action="store_true")
    ap.add_argument("--screenshot-dir", default="outputs/pr7957/shots")
    ap.add_argument("--out", default="outputs/pr7957/geometry.json")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
