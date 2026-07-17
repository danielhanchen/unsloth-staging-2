"""
Real-browser validation for PR #7186 across all engines.

The bug this PR fixes makes Studio's StaticFiles path 500 on Python 3.14, so the
frontend never loads in ANY browser. This drives the GATED app (nest_asyncio NOT
applied on a plain CLI start) with Playwright's three engines:

  * chromium  -> Google Chrome + Microsoft Edge (both Chromium-based)
  * firefox   -> Mozilla Firefox (Gecko)
  * webkit    -> Apple Safari (WebKit)

For each engine it loads "/", asserts the page renders (the <h1>ok</h1> from
index.html) and that "/assets/app.js" (the previously-500ing StaticFiles thread-pool
path) returns 200 with the real JS body. Exit 0 iff every requested engine passes.

Usage:
  python playwright_browsers.py [--engines chromium,firefox,webkit] [--port N] [--shot-dir DIR]
"""
import argparse
import asyncio
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles


def _gate_apply_nest_asyncio():
    """The exact PR gate. On a plain CLI start (no running loop) -> does NOT apply."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    else:
        import nest_asyncio
        nest_asyncio.apply()
        return True


def _serve(frontend: Path, host: str, port: int):
    app = FastAPI()

    @app.get("/api/health")
    async def health():
        return {"ok": True}

    app.mount("/", StaticFiles(directory=str(frontend), html=True), name="frontend")

    ready = threading.Event()

    class _ReadyServer(uvicorn.Server):
        async def startup(self, *a, **k):
            await super().startup(*a, **k)
            ready.set()

    server = _ReadyServer(uvicorn.Config(app, host=host, port=port,
                                         log_level="warning", access_log=False))

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        finally:
            loop.close()

    threading.Thread(target=_run, daemon=True).start()
    ready.wait(timeout=10)
    return server


async def _drive(engines, base, shot_dir):
    from playwright.async_api import async_playwright

    results = {}
    async with async_playwright() as p:
        for name in engines:
            engine = getattr(p, name)
            browser = await engine.launch()
            try:
                page = await browser.new_page()
                asset_status = {}

                async def _capture(resp):
                    if resp.url.endswith("/assets/app.js"):
                        asset_status["code"] = resp.status

                page.on("response", _capture)
                resp = await page.goto(base + "/", wait_until="domcontentloaded")
                body = await page.content()
                # Fetch the asset explicitly too (belt and suspenders).
                asset = await page.evaluate(
                    "async () => { const r = await fetch('/assets/app.js');"
                    " return {status: r.status, text: await r.text()}; }"
                )
                if shot_dir:
                    await page.screenshot(path=str(Path(shot_dir) / f"{name}.png"))
                ok = (
                    resp is not None and resp.status == 200
                    and "ok" in body
                    and asset.get("status") == 200
                    and "studio asset" in asset.get("text", "")
                )
                results[name] = {
                    "page_status": resp.status if resp else None,
                    "asset_status": asset.get("status"),
                    "h1_ok": "ok" in body,
                    "pass": ok,
                }
            finally:
                await browser.close()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="chromium,firefox,webkit")
    ap.add_argument("--port", type=int, default=8993)
    ap.add_argument("--shot-dir", default="")
    args = ap.parse_args()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    py = "%d.%d.%d" % sys.version_info[:3]
    applied = _gate_apply_nest_asyncio()
    print(f"Python {py} | gate applied nest_asyncio={applied} (plain CLI start -> should be False)")

    if args.shot_dir:
        Path(args.shot_dir).mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as d:
        fe = Path(d)
        (fe / "index.html").write_text("<!doctype html><title>studio</title><h1>ok</h1>")
        (fe / "assets").mkdir()
        (fe / "assets" / "app.js").write_text("console.log('studio asset');")

        server = _serve(fe, "127.0.0.1", args.port)
        base = f"http://127.0.0.1:{args.port}"
        results = asyncio.run(_drive(engines, base, args.shot_dir))
        server.should_exit = True
        time.sleep(0.2)

    all_ok = True
    for name in engines:
        r = results.get(name, {"pass": False})
        all_ok = all_ok and r.get("pass", False)
        print(f"  {name:9s} page={r.get('page_status')} asset={r.get('asset_status')} "
              f"h1_ok={r.get('h1_ok')} -> {'PASS' if r.get('pass') else 'FAIL'}")
    print("RESULT=" + ("ALL PASS" if all_ok else "FAILURE"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    import os
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
