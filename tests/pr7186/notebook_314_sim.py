"""
Confirms Codex's P2 point on PR #7186 AND validates the version-gate fix.

Codex: on Python 3.14+, a notebook/IPython launch has a running loop, so the PR's
gate still calls nest_asyncio.apply(); that global patch also breaks the background
uvicorn loop, so StaticFiles still 500s for notebook starts on 3.14.

This simulates a NOTEBOOK launch (the gate runs while an event loop IS running, so
get_running_loop() succeeds) under two gate variants, then boots the real
run.py-style server (daemon thread + new_event_loop) and hits the StaticFiles path.

  --gate pr        : the shipped gate (apply iff a loop is running)
  --gate versioned : apply iff a loop is running AND sys.version_info < (3,14)

Expected on 3.14:
  pr        -> nest_asyncio applied -> GET / == 500   (Codex is right: still broken)
  versioned -> nest_asyncio skipped -> GET / == 200   (fix: notebook works on 3.14)
Expected on <=3.13: both -> 200 (nest_asyncio harmless there; versioned still applies it).
"""
import argparse
import asyncio
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import uvicorn
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles


def _gate(variant: str) -> bool:
    """Run the gate. Returns True iff nest_asyncio.apply() was called."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    else:
        if variant == "versioned" and sys.version_info >= (3, 14):
            return False
        import nest_asyncio
        nest_asyncio.apply()
        return True


def _run_gate_in_notebook(variant: str) -> bool:
    """Model a notebook: the gate executes while an event loop is running."""
    box = {}

    async def _cell():
        # get_running_loop() will succeed here, exactly like an in-cell run_server().
        box["applied"] = _gate(variant)

    asyncio.run(_cell())
    return box["applied"]


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
                                         log_level="critical", access_log=False))

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["pr", "versioned"], default="pr")
    ap.add_argument("--port", type=int, default=8981)
    args = ap.parse_args()

    py = "%d.%d.%d" % sys.version_info[:3]
    on_314 = sys.version_info[:2] >= (3, 14)

    applied = _run_gate_in_notebook(args.gate)
    print(f"Python {py} | notebook launch | gate={args.gate} | nest_asyncio applied={applied}")

    with TemporaryDirectory() as d:
        fe = Path(d)
        (fe / "index.html").write_text("<!doctype html><h1>ok</h1>")
        (fe / "assets").mkdir()
        (fe / "assets" / "app.js").write_text("console.log('x');")
        server = _serve(fe, "127.0.0.1", args.port)
        with httpx.Client(timeout=5.0) as c:
            try:
                root = c.get(f"http://127.0.0.1:{args.port}/").status_code
            except Exception as e:  # noqa: BLE001
                root = f"EXC {type(e).__name__}"
        server.should_exit = True
        time.sleep(0.2)

    print(f"  GET / -> {root}")

    # Expectation: broken only when nest_asyncio is applied on 3.14.
    if on_314 and applied:
        ok = root == 500
        note = "Codex-correct: notebook still 500s on 3.14 with nest_asyncio applied"
    else:
        ok = root == 200
        note = "works (notebook serves StaticFiles)"
    print(f"  [{note}] {'as expected' if ok else 'UNEXPECTED'}")
    print("RESULT=" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    import os
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
