#!/usr/bin/env python3
"""Geometry check for unslothai/unsloth#7954 rebased onto the merged #7957.

For every SheetContent call site, in every chrome branch, reports where the
panel starts and ends and whether its close button is actually hit-testable
rather than covered by the desktop window controls.

The invariant: on the Windows/Linux custom-titlebar path a fixed sheet must
start at exactly 34px and still end at the viewport bottom. In browser mode and
on macOS it must start at 0. The in-container sheet (site 6) must never move.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from playwright.async_api import async_playwright

SITES = {
    1: "response details",
    2: "chat settings (mobile)",
    3: "deep research",
    4: "sidebar (mobile)",
    5: "document preview",
    6: "recipe block (in-container)",
    7: "chart settings",
}

PROBE_JS = r"""
() => {
  const content = document.querySelector('[data-slot="sheet-content"]');
  if (!content) return { error: 'no sheet-content' };
  const c = content.getBoundingClientRect();
  const close = content.querySelector('[data-slot="sheet-close"]');

  let reachablePct = null, centreHit = null;
  // The mobile sidebar hides the default close button with [&>button]:hidden,
  // so it has no box to hit-test. Report it as not applicable, not occluded.
  const closeHidden = close
    ? getComputedStyle(close).display === 'none'
      || close.getBoundingClientRect().width === 0
    : true;
  if (close && !closeHidden) {
    const b = close.getBoundingClientRect();
    const N = 9;
    let reachable = 0, total = 0;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const x = b.left + (b.width  * (i + 0.5)) / N;
        const y = b.top  + (b.height * (j + 0.5)) / N;
        const el = document.elementFromPoint(x, y);
        total++;
        if (el && (el === close || close.contains(el))) reachable++;
      }
    }
    reachablePct = Math.round((reachable / total) * 1000) / 10;
    const centreEl = document.elementFromPoint(b.left + b.width / 2,
                                               b.top + b.height / 2);
    centreHit = centreEl && (centreEl === close || close.contains(centreEl))
      ? 'CLOSE'
      : (() => {
          if (!centreEl) return 'none';
          const owner = centreEl.closest('[role="toolbar"],header,[data-slot]');
          return owner
            ? (owner.getAttribute('aria-label') || owner.getAttribute('data-slot')
               || owner.tagName)
            : centreEl.tagName;
        })();
  }

  // Site 6 is positioned against its container, not the viewport.
  const parent = content.offsetParent || document.documentElement;
  const p = parent.getBoundingClientRect();

  return {
    top: Math.round(c.top * 100) / 100,
    bottom: Math.round(c.bottom * 100) / 100,
    height: Math.round(c.height * 100) / 100,
    viewportHeight: window.innerHeight,
    parentTop: Math.round(p.top * 100) / 100,
    parentBottom: Math.round(p.bottom * 100) / 100,
    computedTop: getComputedStyle(content).top,
    computedMarginTop: getComputedStyle(content).marginTop,
    reachablePct,
    centreHit,
    closeHidden,
  };
}
"""

SEED_TAURI = ("(() => { window.__TAURI__ = {}; window.__TAURI_INTERNALS__ = "
              "{ invoke: () => Promise.resolve(null) }; })();")


async def settle(page):
    """The sheet slides in over ~200ms; measuring mid-transition reads a box
    that is not where it lands, so wait for the rect to stop moving."""
    await page.wait_for_selector('[data-slot="sheet-content"][data-state="open"]',
                                 state="visible", timeout=30_000)
    await page.evaluate("() => document.fonts.ready")
    await page.wait_for_timeout(450)
    await page.wait_for_function(
        """() => {
             const el = document.querySelector('[data-slot="sheet-content"]');
             if (!el) return false;
             const now = el.getBoundingClientRect().top;
             const stable = window.__lastTop === now;
             window.__lastTop = now;
             return stable && !el.getAnimations().some(a => a.playState === 'running');
           }""",
        timeout=15_000,
    )


async def measure(page, url, attempts=3):
    last = None
    for attempt in range(attempts):
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            await settle(page)
            return await page.evaluate(PROBE_JS)
        except Exception as exc:  # noqa: BLE001 - transient
            last = exc
            await page.wait_for_timeout(1000)
    return {"error": f"{type(last).__name__}: {last}"}


async def run_engine(p, spec, args, results):
    name, _, channel = spec.partition(":")
    launch = {"headless": True}
    if channel:
        launch["channel"] = channel
    browser = await getattr(p, name).launch(**launch, timeout=120_000)
    ctx = await browser.new_context(
        viewport={"width": args.width, "height": 800},
        device_scale_factor=args.dpr, color_scheme="light")
    await ctx.add_init_script(SEED_TAURI)
    page = await ctx.new_page()
    for chrome in args.chromes.split(","):
        for variant in args.variants.split(","):
            for site in [int(s) for s in args.sites.split(",")]:
                for fs in [int(x) for x in args.font_sizes.split(",")]:
                    m = dict(await measure(
                        page,
                        f"{args.url}/sheet-matrix-harness.html"
                        f"?site={site}&variant={variant}&chrome={chrome}&fs={fs}"))
                    m.update(engine=spec, chrome=chrome, variant=variant,
                             site=site, fontSize=fs, width=args.width,
                             dpr=args.dpr)
                    results.append(m)
                    print(f"{spec:14s} {chrome:8s} {variant:5s} site{site} "
                          f"({SITES[site][:22]:22s}) fs={fs:2d} "
                          f"top={m.get('top')!s:>7} bottom={m.get('bottom')!s:>7} "
                          f"vh={m.get('viewportHeight')!s:>4} "
                          f"reach={m.get('reachablePct')!s:>5} "
                          f"centre={m.get('centreHit')}", flush=True)
    await ctx.close()
    await browser.close()


def verdict(results):
    """Sites 1-5,7 are viewport-fixed; site 6 tracks its container."""
    bad = []
    for r in results:
        if r.get("error"):
            bad.append((r, f"error: {r['error']}"))
            continue
        site, chrome = r["site"], r["chrome"]
        expected_top = 34.0 if chrome == "custom" else 0.0
        if site == 6:
            # Must stay glued to its container regardless of chrome.
            if abs(r["top"] - r["parentTop"]) > 0.02:
                bad.append((r, f"in-container sheet moved: top={r['top']} "
                               f"parentTop={r['parentTop']}"))
            continue
        if abs(r["top"] - expected_top) > 0.02:
            bad.append((r, f"top={r['top']}, expected {expected_top}"))
        if abs(r["bottom"] - r["viewportHeight"]) > 0.02:
            bad.append((r, f"bottom={r['bottom']} != viewport "
                           f"{r['viewportHeight']} (overflow/short by "
                           f"{round(r['bottom'] - r['viewportHeight'], 2)}px)"))
        if r.get("centreHit") is not None and r["centreHit"] != "CLOSE":
            bad.append((r, f"close button centre occluded by {r['centreHit']}"))
    return bad


async def run(args):
    results, skipped = [], []
    async with async_playwright() as p:
        for spec in args.engines.split(","):
            try:
                await asyncio.wait_for(run_engine(p, spec, args, results),
                                       timeout=args.engine_timeout)
            except asyncio.TimeoutError:
                print(f"\n{spec}: exceeded {args.engine_timeout}s, skipping",
                      flush=True)
                skipped.append(f"{spec} (timeout)")
            except Exception as exc:  # noqa: BLE001
                print(f"\n{spec}: {type(exc).__name__}: {exc}", flush=True)
                skipped.append(f"{spec} ({type(exc).__name__})")

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    if skipped:
        print(f"\nengines skipped: {', '.join(skipped)}")

    print(f"\n{len(results)} measurement(s) -> {args.out}")
    if args.check_variant:
        checked = [r for r in results if r["variant"] == args.check_variant]
        bad = verdict(checked)
        print(f"checking {len(checked)} '{args.check_variant}' row(s)")
        if bad:
            print(f"\nFAIL: {len(bad)} row(s) violate the invariant")
            for r, why in bad[:20]:
                print(f"  {r['engine']} {r['chrome']} site{r['site']} "
                      f"fs={r['fontSize']}: {why}")
            sys.exit(1)
        print("OK: every row starts and ends where it should, close button clear")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5193")
    ap.add_argument("--engines", default="chromium")
    ap.add_argument("--chromes", default="custom,mac,browser")
    ap.add_argument("--variants", default="fixed")
    ap.add_argument("--sites", default="1,2,3,4,5,6,7")
    ap.add_argument("--font-sizes", default="15")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--dpr", type=float, default=1.0)
    ap.add_argument("--out", default="sheet_matrix.json")
    ap.add_argument("--check-variant", default=None)
    ap.add_argument("--engine-timeout", type=float, default=900.0)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
