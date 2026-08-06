#!/usr/bin/env python3
"""Desktop titlebar occlusion check for unslothai/unsloth#7957, run on real
Windows and macOS GitHub runners.

`isTauri` is a module-scope const derived from `window.__TAURI__`, so that is
seeded before the bundle evaluates. Everything else is real on the runner: the
platform reported by navigator, the OS font metrics, and the browser engine
(msedge on Windows is the same Chromium family as WebView2; webkit on macOS is
the closest available proxy for WKWebView).

Reports, per variant, the fraction of the sheet close button that is actually
hit-testable, i.e. that document.elementFromPoint attributes to the button
rather than to the app's own window controls.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from playwright.async_api import async_playwright

SEED_TAURI = "(() => { window.__TAURI__ = {}; " \
             "window.__TAURI_INTERNALS__ = { invoke: () => Promise.resolve(null) }; })();"

PROBE_JS = r"""
() => {
  const content = document.querySelector('[data-slot="sheet-content"]');
  if (!content) return { error: 'no sheet-content' };
  const close = content.querySelector('[data-slot="sheet-close"]');
  const title = content.querySelector('[data-slot="sheet-title"]');
  if (!close || !title) return { error: 'no close/title' };
  const c = close.getBoundingClientRect();
  const t = title.getBoundingClientRect();
  const controls = document.querySelector('[role="toolbar"][aria-label="Window controls"]');
  const cb = controls ? controls.getBoundingClientRect() : null;

  const N = 9;
  let reachable = 0, total = 0;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const x = c.left + (c.width  * (i + 0.5)) / N;
      const y = c.top  + (c.height * (j + 0.5)) / N;
      const el = document.elementFromPoint(x, y);
      total++;
      if (el && (el === close || close.contains(el))) reachable++;
    }
  }
  const centreEl = document.elementFromPoint(c.left + c.width / 2, c.top + c.height / 2);
  const centreHit = centreEl && (centreEl === close || close.contains(centreEl))
    ? 'CLOSE'
    : (() => {
        if (!centreEl) return 'none';
        const owner = centreEl.closest('[role="toolbar"],header,[data-slot]');
        return owner
          ? (owner.getAttribute('aria-label') || owner.getAttribute('data-slot') || owner.tagName)
          : centreEl.tagName;
      })();

  return {
    closeTop: c.top,
    signedDelta: (c.top + c.height / 2) - (t.top + t.height / 2),
    titleHeight: t.height,
    controlsPresent: !!controls,
    yOverlapWithControls: cb
      ? Math.max(0, Math.min(c.bottom, cb.bottom) - Math.max(c.top, cb.top)) : 0,
    reachablePct: Math.round((reachable / total) * 1000) / 10,
    centreHit,
    uaPlatform: (navigator.userAgentData && navigator.userAgentData.platform)
      || navigator.platform || '',
    titleFont: getComputedStyle(title).fontFamily,
    titleLineHeight: getComputedStyle(title).lineHeight,
  };
}
"""


async def settle(page):
    await page.wait_for_selector('[data-slot="sheet-content"][data-state="open"]',
                                 state="visible", timeout=30_000)
    await page.evaluate("() => document.fonts.ready")
    await page.wait_for_timeout(700)


async def run(args):
    results = []
    async with async_playwright() as p:
        for spec in args.engines.split(","):
            name, _, channel = spec.partition(":")
            launch = {"headless": True}
            if channel:
                launch["channel"] = channel
            browser = await getattr(p, name).launch(**launch)
            label = spec
            ctx = await browser.new_context(viewport={"width": 1280, "height": 800},
                                            device_scale_factor=1, color_scheme="light")
            await ctx.add_init_script(SEED_TAURI)
            page = await ctx.new_page()
            for variant in ["before", "after", "fixed"]:
                for sheet in ["details", "preview"]:
                    for fs in [int(s) for s in args.font_sizes.split(",")]:
                        await page.goto(
                            f"{args.url}/titlebar-harness.html"
                            f"?variant={variant}&sheet={sheet}&fs={fs}",
                            wait_until="domcontentloaded")
                        await settle(page)
                        m = dict(await page.evaluate(PROBE_JS))
                        m.update(engine=label, variant=variant, sheet=sheet, fontSize=fs)
                        results.append(m)
                        print(f"{label:16s} {variant:6s} {sheet:8s} fs={fs:2d} "
                              f"top={m.get('closeTop')!s:>7} "
                              f"delta={m.get('signedDelta')!s:>7} "
                              f"ctrlOverlap={m.get('yOverlapWithControls')!s:>6} "
                              f"reachable={m.get('reachablePct')!s:>5}% "
                              f"centre={m.get('centreHit')}", flush=True)
            await ctx.close()
            await browser.close()

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    plat = results[0].get("uaPlatform", "") if results else ""
    print(f"\nnavigator platform reported by the runner: {plat!r}")
    print(f"title font: {results[0].get('titleFont') if results else ''}")

    custom_titlebar = any(r.get("controlsPresent") for r in results)
    print(f"custom titlebar rendered on this platform: {custom_titlebar}")

    def worst(variant):
        rows = [r for r in results if r["variant"] == variant]
        return min((r["reachablePct"] for r in rows), default=None), \
               sorted({r["centreHit"] for r in rows})

    for v in ["before", "after", "fixed"]:
        w, centres = worst(v)
        print(f"  {v:6s} worst reachable={w}%  centre hits={centres}")

    # The PR must not leave the close button unreachable at its centre, and the
    # candidate fix must restore it. On macOS there is no custom titlebar, so
    # every variant should already be clear.
    fixed_rows = [r for r in results if r["variant"] == "fixed"]
    bad = [r for r in fixed_rows if r["centreHit"] != "CLOSE"]
    if bad:
        print(f"\nFAIL: {len(bad)} 'fixed' rows still have an occluded centre")
        for r in bad[:10]:
            print(f"  {r['engine']} {r['sheet']} fs={r['fontSize']} centre={r['centreHit']}")
        sys.exit(1)
    print("\nOK: every 'fixed' row has a hit-testable close button centre")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5192")
    ap.add_argument("--engines", default="chromium")
    ap.add_argument("--font-sizes", default="12,15,20")
    ap.add_argument("--out", default="titlebar.json")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
