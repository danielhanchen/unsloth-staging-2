"""Staging check for unslothai/unsloth#17004: with a background image on, an open model page must
not show the Model hub list through it. Runs against a live Studio (BASE_URL, STUDIO_PW) in
Chromium, Firefox and WebKit; screenshots + facts.json land in PW_ART_DIR. Exit 1 on any leak.

WebKit (Linux build) cannot keep a Blob in IndexedDB, so there the CSS rule the image turns on
is applied directly; Chromium and Firefox use a real stored image.
"""

import asyncio
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from playwright.async_api import async_playwright

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8888")
PW = os.environ["STUDIO_PW"]
OUT = Path(os.environ.get("PW_ART_DIR", "pw_out"))
REPO = "unsloth/Qwen3-0.6B-GGUF"
CLEAR = ".hub-page, .hub-page .hub-canvas { background: transparent !important; }"

SEED = r"""
async () => {
  const c = document.createElement('canvas'); c.width = 640; c.height = 400;
  const g = c.getContext('2d'); const gr = g.createLinearGradient(0, 0, 640, 400);
  gr.addColorStop(0, '#3b1d6e'); gr.addColorStop(1, '#0d3b4f'); g.fillStyle = gr; g.fillRect(0, 0, 640, 400);
  const blob = await new Promise(r => c.toBlob(r, 'image/png'));
  const db = await new Promise((res, rej) => {
    const q = indexedDB.open('unsloth-chat-background', 10);
    q.onupgradeneeded = () => q.result.createObjectStore('backgrounds', { keyPath: 'id' });
    q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error);
  });
  const value = { scope: 'all', effect: 'None', visibility: 70, conversation: 0, blur: 0, fade: true, outline: 30,
    sizing: 'cover', zoom: 100, position: '50% 50%', glassLightVisibility: 70, glassDarkVisibility: 40, glassFade: false, blob };
  await new Promise((res, rej) => {
    const tx = db.transaction('backgrounds', 'readwrite');
    tx.objectStore('backgrounds').put({ id: 'current', value, sha256: null });
    tx.oncomplete = res; tx.onerror = () => rej(tx.error);
  });
  db.close();
}
"""

MEASURE = r"""
() => {
  const pane = document.querySelector('[data-tour="hub-catalog"]');
  const ws = document.querySelector('[data-wallpaper-workspace]');
  let painted = 0;
  for (const el of pane.querySelectorAll('a, button, h3, li')) {
    const r = el.getBoundingClientRect();
    if (r.width && r.height && r.bottom > 0 && r.top < innerHeight && getComputedStyle(el).visibility === 'visible' && el.textContent.trim()) painted++;
  }
  return { wallpaper_ready: !!(ws && ws.hasAttribute('data-wallpaper-ready')),
           catalog_visibility: getComputedStyle(pane).visibility, painted_under_detail: painted };
}
"""


def login():
    req = urllib.request.Request(
        f"{BASE}/api/auth/login",
        json.dumps({"username": "unsloth", "password": PW}).encode(),
        {"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout = 30))


async def run(p, engine, view, width, tok):
    fake = engine == "webkit"
    browser = await getattr(p, engine).launch()
    ctx = await browser.new_context(viewport = {"width": width, "height": 1000})
    await ctx.add_init_script(
        "localStorage.setItem('unsloth_auth_token', %s); localStorage.setItem('unsloth_auth_refresh_token', %s);"
        "localStorage.setItem('unsloth.hub.allModelsView', %s);"
        % (
            json.dumps(tok["access_token"]),
            json.dumps(tok.get("refresh_token", "")),
            json.dumps(view),
        )
    )
    page = await ctx.new_page()
    await page.goto(f"{BASE}/hub", wait_until = "domcontentloaded")
    if not fake:
        await page.evaluate(SEED)
        await page.reload(wait_until = "domcontentloaded")
        await page.wait_for_selector(
            "[data-wallpaper-workspace][data-wallpaper-ready]", state = "attached", timeout = 60_000
        )
    await page.wait_for_selector(
        '[data-tour="hub-catalog"] a, [data-tour="hub-catalog"] button', timeout = 120_000
    )
    await page.wait_for_timeout(4000)
    await page.goto(f"{BASE}/hub?model={REPO}", wait_until = "domcontentloaded")
    await page.wait_for_selector(".hub-page .hub-canvas.z-20", timeout = 120_000)
    if fake:
        await page.add_style_tag(content = CLEAR)
    await (
        page.locator(".hub-page .hub-canvas.z-20")
        .get_by_text(REPO.split("/")[-1])
        .first.wait_for(timeout = 90_000)
    )
    await page.wait_for_timeout(6000)
    facts = await page.evaluate(MEASURE)
    facts.update(engine = engine, view = view, width = width, css_only = fake)
    await page.screenshot(path = str(OUT / f"hub_detail_{engine}_{view}_{width}.png"))
    await browser.close()
    return facts


async def main():
    OUT.mkdir(parents = True, exist_ok = True)
    subprocess.run(
        [sys.executable, "-m", "playwright", "install", "--with-deps", "firefox", "webkit"],
        check = True,
    )
    tok = login()
    results, bad = [], []
    async with async_playwright() as p:
        for engine in ("chromium", "firefox", "webkit"):
            for view, width in (("two", 1600), ("split", 900)):
                try:
                    f = await run(p, engine, view, width, tok)
                except Exception as exc:  # a crash is a failed cell, never a pass
                    f = {"engine": engine, "view": view, "width": width, "error": repr(exc)[:400]}
                print(json.dumps(f), flush = True)
                results.append(f)
                if (
                    f.get("error")
                    or f.get("painted_under_detail") != 0
                    or f.get("catalog_visibility") != "hidden"
                    or not (f.get("wallpaper_ready") or f.get("css_only"))
                ):
                    bad.append(f)
    (OUT / "facts.json").write_text(json.dumps(results, indent = 1))
    print(f"HUB_DETAIL_WALLPAPER cells={len(results)} failing={len(bad)}")
    sys.exit(1 if bad else 0)


asyncio.run(main())
