"""Scene: API monitor and hardware monitor drag / native resize geometry (PR 12204).

The claim is geometric, so the evidence is numbers from getBoundingClientRect, with
screenshots only as illustration. Cells, per browser engine:

  resize_no_drag   native grip pulled (+80,+60) on a freshly opened API monitor
  resize_after_drag  panel dragged up-left first, then the same grip pull
  poll_after_resize  the same box after two poll ticks (panel must not creep)
  drag_clamp       dragged far past the top-left: where it lands, and no jump on release
  reopen           close then reopen: back at its opening box
  obstacle         hardware monitor open first: API monitor opening overlap area
  hw_drag / hw_resize  the hardware monitor alone (shared hook parity: must match base)
  short_viewport   900x600: footer controls on screen and hit-testable

A grip pull that changes neither width nor height is recorded as `resize_ok: false`:
a missed grip must never read as "the panel stayed put".
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(
    os.environ.get("WORKSPACE")
    or os.environ.get("UNSLOTH_WORKSPACE")
    or Path(__file__).resolve().parents[2]
)
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from playwright.async_api import async_playwright  # noqa: E402

from pr_ui_scenes._common import Session  # noqa: E402
from studio_test_kit.auth import seed_init_script  # noqa: E402

API_CLOSE = '[aria-label="Close API monitor"]'
HW_PANEL = '[data-testid="floating-monitor"]'
HW_HANDLE = '[data-testid="floating-monitor-drag-handle"]'
ROUTE = "/data-recipes"

_API_PANEL_JS = f"""() => {{
  const b = document.querySelector('{API_CLOSE}');
  return b ? b.closest('.menu-soft-surface') : null;
}}"""


def _init(session: Session, hw_open: bool) -> str:
    auth = type(
        "A", (), {"access_token": session.access_token, "refresh_token": session.refresh_token}
    )()
    extra = {
        "unsloth_monitor_overlay": {
            "state": {"isOpen": hw_open, "isMinimized": False},
            "version": 0,
        },
        # Traffic-driven auto open would race the shortcut.
        "unsloth_api_monitor_overlay": {"state": {"autoOpen": False}, "version": 0},
    }
    return seed_init_script(auth, [], extra_local_storage = extra)


async def _rect(page, which: str):
    if which == "api":
        return await page.evaluate(f"""() => {{
          const p = ({_API_PANEL_JS})();
          if (!p) return null;
          const r = p.getBoundingClientRect();
          return {{left: r.left, top: r.top, width: r.width, height: r.height}};
        }}""")
    return await page.evaluate(f"""() => {{
      const p = document.querySelector('{HW_PANEL}');
      if (!p) return null;
      const r = p.getBoundingClientRect();
      return {{left: r.left, top: r.top, width: r.width, height: r.height}};
    }}""")


def _r(rect):
    return None if rect is None else {k: round(v, 1) for k, v in rect.items()}


def _delta(a, b):
    if a is None or b is None:
        return None
    return {k: round(b[k] - a[k], 1) for k in ("left", "top", "width", "height")}


async def _drag(
    page,
    x,
    y,
    dx,
    dy,
    steps = 12,
    measure = None,
):
    await page.mouse.move(x, y)
    await page.mouse.down()
    await page.mouse.move(x + dx, y + dy, steps = steps)
    await page.wait_for_timeout(120)
    during = await _rect(page, measure) if measure else None
    await page.mouse.up()
    await page.wait_for_timeout(400)
    return during


async def _grip_pull(page, which, dx, dy):
    r = await _rect(page, which)
    await _drag(page, r["left"] + r["width"] - 3, r["top"] + r["height"] - 3, dx, dy)
    return r, await _rect(page, which)


async def _handle_center(page, which):
    if which == "api":
        box = await page.evaluate(f"""() => {{
          const b = document.querySelector('{API_CLOSE}');
          const h = b && b.previousElementSibling;
          if (!h) return null;
          const r = h.getBoundingClientRect();
          return {{x: r.left + r.width / 2, y: r.top + r.height / 2}};
        }}""")
    else:
        box = await page.locator(HW_HANDLE).bounding_box()
        box = box and {"x": box["x"] + box["width"] / 2, "y": box["y"] + box["height"] / 2}
    return box


async def _chord(page) -> str:
    # Playwright's WebKit sends a Mac Safari user agent, so Studio binds the macOS chord there.
    mac = await page.evaluate("() => /Mac/.test(navigator.userAgent)")
    return "Control+Shift+KeyU" if mac else "Control+Alt+Shift+KeyM"


async def _toggle_api(page):
    await page.mouse.click(5, 300)
    await page.keyboard.press(await _chord(page))
    await page.wait_for_timeout(1200)


async def _open_page(browser, session, hw_open, viewport):
    ctx = await browser.new_context(viewport = {"width": viewport[0], "height": viewport[1]})
    await ctx.add_init_script(_init(session, hw_open))
    page = await ctx.new_page()
    await page.goto(f"{session.base_url}{ROUTE}", wait_until = "domcontentloaded")
    await page.wait_for_timeout(5000)
    return ctx, page


def _resize_ok(before, after):
    d = _delta(before, after)
    return bool(d) and (abs(d["width"]) >= 40 or abs(d["height"]) >= 30)


async def _engine(
    pw,
    engine: str,
    session: Session,
    out_dir: Path,
    label: str,
    shots: list[Path],
    screenshots: bool,
) -> dict:
    browser = await getattr(pw, engine).launch(headless = True)
    f: dict = {}
    try:
        vp = (1600, 1000)
        # resize before any drag, then drag, then resize again, then poll ticks
        ctx, page = await _open_page(browser, session, False, vp)
        await _toggle_api(page)
        opened = await _rect(page, "api")
        f["api_open_rect"] = _r(opened)
        if opened is None:
            diag = await page.evaluate("""() => ({url: location.href,
              text: document.body.innerText.slice(0, 200),
              ua: navigator.userAgent, platform: navigator.platform})""")
            await page.screenshot(path = str(out_dir / f"{label.lower()}_{engine}_fail.png"))
            raise RuntimeError(f"{engine}: API monitor did not open on the shortcut {diag}")
        if screenshots:
            p = out_dir / f"{label.lower()}_{engine}_0_open.png"
            await page.screenshot(path = str(p))
            shots.append(p)
        b, a = await _grip_pull(page, "api", 80, 60)
        f["resize_no_drag"] = {"delta": _delta(b, a), "resize_ok": _resize_ok(b, a)}
        b, a = await _grip_pull(page, "api", -80, -60)
        f["shrink_no_drag"] = {"delta": _delta(b, a), "resize_ok": _resize_ok(b, a)}
        h = await _handle_center(page, "api")
        before_drag = await _rect(page, "api")
        await _drag(page, h["x"], h["y"], -500, -300)
        after_drag = await _rect(page, "api")
        f["drag"] = {"delta": _delta(before_drag, after_drag)}
        b, a = await _grip_pull(page, "api", 80, 60)
        f["resize_after_drag"] = {"delta": _delta(b, a), "resize_ok": _resize_ok(b, a)}
        if screenshots:
            p = out_dir / f"{label.lower()}_{engine}_1_after_resize.png"
            await page.screenshot(path = str(p))
            shots.append(p)
        await page.wait_for_timeout(3500)
        f["poll_after_resize"] = {"delta": _delta(a, await _rect(page, "api"))}
        b, a = await _grip_pull(page, "api", -60, -40)
        f["shrink_after_drag"] = {"delta": _delta(b, a), "resize_ok": _resize_ok(b, a)}

        h = await _handle_center(page, "api")
        during = await _drag(page, h["x"], h["y"], -5000, -5000, measure = "api")
        landed = await _rect(page, "api")
        f["drag_clamp"] = {"landed": _r(landed), "release_jump": _delta(during, landed)}

        # close + reopen
        await page.locator(API_CLOSE).click()
        await page.wait_for_timeout(800)
        await _toggle_api(page)
        f["reopen"] = {
            "rect": _r(await _rect(page, "api")),
            "vs_first_open": _delta(opened, await _rect(page, "api")),
        }
        # rapid close/reopen inside the exit animation
        await page.locator(API_CLOSE).click()
        await page.wait_for_timeout(60)
        await page.keyboard.press(await _chord(page))
        await page.wait_for_timeout(1500)
        f["rapid_reopen"] = {
            "panels": await page.locator(API_CLOSE).count(),
            "vs_first_open": _delta(opened, await _rect(page, "api")),
        }
        await ctx.close()

        # obstacle: hardware monitor first
        ctx, page = await _open_page(browser, session, True, vp)
        hw = await _rect(page, "hw")
        f["hw_rect"] = _r(hw)
        await _toggle_api(page)
        api = await _rect(page, "api")
        if hw and api:
            ox = max(
                0,
                min(hw["left"] + hw["width"], api["left"] + api["width"])
                - max(hw["left"], api["left"]),
            )
            oy = max(
                0,
                min(hw["top"] + hw["height"], api["top"] + api["height"])
                - max(hw["top"], api["top"]),
            )
            f["obstacle"] = {"api_rect": _r(api), "overlap_px2": round(ox * oy)}
        else:
            f["obstacle"] = {"api_rect": _r(api), "overlap_px2": None}
        # content growth after opening (first poll / new request rows), no user move
        await page.evaluate(f"""() => {{
          const p = ({_API_PANEL_JS})();
          const host = p.querySelector('.overflow-y-auto > div') || p;
          const d = document.createElement('div');
          d.style.height = '140px';
          d.setAttribute('data-scene-grow', '');
          host.insertBefore(d, host.children[1] || null);
        }}""")
        await page.wait_for_timeout(1200)
        grown = await _rect(page, "api")
        if hw and grown:
            ox = max(
                0,
                min(hw["left"] + hw["width"], grown["left"] + grown["width"])
                - max(hw["left"], grown["left"]),
            )
            oy = max(
                0,
                min(hw["top"] + hw["height"], grown["top"] + grown["height"])
                - max(hw["top"], grown["top"]),
            )
            f["grow_over_obstacle"] = {"api_rect": _r(grown), "overlap_px2": round(ox * oy)}
        if screenshots:
            p = out_dir / f"{label.lower()}_{engine}_2_both.png"
            await page.screenshot(path = str(p))
            shots.append(p)
        await page.locator(API_CLOSE).click()
        await page.wait_for_timeout(800)
        # hardware monitor alone: shared hook parity
        if hw:
            h = await _handle_center(page, "hw")
            b = await _rect(page, "hw")
            during = await _drag(page, h["x"], h["y"], -400, -250, measure = "hw")
            a = await _rect(page, "hw")
            f["hw_drag"] = {"delta": _delta(b, a), "release_jump": _delta(during, a)}
            b, a = await _grip_pull(page, "hw", 60, 40)
            f["hw_resize"] = {"delta": _delta(b, a), "resize_ok": _resize_ok(b, a)}
        await ctx.close()

        # short viewport
        ctx, page = await _open_page(browser, session, False, (900, 600))
        await _toggle_api(page)
        r = await _rect(page, "api")
        stop_btn = page.get_by_role("button", name = "Stop opening this automatically")
        vis = await stop_btn.bounding_box() if await stop_btn.count() else None
        f["short_viewport"] = {
            "rect": _r(r),
            "fits": bool(r) and r["top"] >= 0 and r["top"] + r["height"] <= 600,
            "footer_on_screen": bool(vis) and vis["y"] + vis["height"] <= 600,
        }
        if screenshots:
            p = out_dir / f"{label.lower()}_{engine}_3_short.png"
            await page.screenshot(path = str(p))
            shots.append(p)
        await ctx.close()
    finally:
        await browser.close()
    return f


async def drive(
    session: Session,
    out_dir: Path,
    label: str,
    engines: tuple[str, ...] = ("chromium", "firefox", "webkit"),
    **_: object,
) -> tuple[list[Path], dict]:
    out_dir.mkdir(parents = True, exist_ok = True)
    shots: list[Path] = []
    facts: dict = {}
    async with async_playwright() as pw:
        for engine in engines:
            try:
                facts[engine] = await _engine(
                    pw, engine, session, out_dir, label, shots, screenshots = (engine == "chromium")
                )
            except Exception as exc:  # one engine failing must not hide the others
                facts[engine] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
    (out_dir / f"{label.lower()}_facts.json").write_text(json.dumps(facts, indent = 2))
    return shots, facts


if __name__ == "__main__":
    import argparse

    from pr_ui_scenes._common import studio_session

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required = True)
    ap.add_argument("--home", type = Path, required = True)
    ap.add_argument("--password", default = "PrUiDiff-12204!x")
    ap.add_argument("--out", type = Path, required = True)
    ap.add_argument("--label", default = "SIDE")
    ap.add_argument("--engines", default = "chromium,firefox,webkit")
    a = ap.parse_args()
    s = studio_session(a.base_url, a.home, a.password)
    _, fx = asyncio.run(drive(s, a.out, a.label, engines = tuple(a.engines.split(","))))
    print(json.dumps(fx, indent = 2))
