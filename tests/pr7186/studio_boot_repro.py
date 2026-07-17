"""
Studio-realistic integration repro for PR #7186.

Mirrors studio/backend/run.py's real server architecture:
  * (optionally) nest_asyncio.apply() on the MAIN thread  -- global Task/loop patch
  * uvicorn in a daemon Thread with its own asyncio.new_event_loop() +
    run_until_complete(server.serve())  (NOT asyncio.run, NOT uvloop)
  * StaticFiles mounted at "/" (html=True), like setup_frontend
  * a threading.Event "ready" gate, like run_server's ready wait

Then it issues real HTTP requests to "/", "/assets/app.js" (the StaticFiles
thread-pool path that 500s on 3.14 when nest_asyncio is applied) and "/api/health".

Modes (simulate the three relevant states):
  --nest force  : always nest_asyncio.apply()   -> the OLD unconditional behavior
  --nest off    : never apply                    -> the NEW gate on a plain CLI start
  --nest auto   : the exact PR gate (apply iff a loop is already running) -> from a
                  plain CLI start this is identical to `off`

Expected:
  3.14 + force -> "/assets/app.js" 500 (cannot create weak reference to NoneType)
  3.14 + off/auto -> all 200
  <=3.13 + any -> all 200
Exit 0 iff observed == expected for the interpreter it runs under.
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


def _apply_nest(mode: str) -> bool:
    """Return True if nest_asyncio.apply() was called. `auto` = the PR's exact gate."""
    if mode == "force":
        import nest_asyncio
        nest_asyncio.apply()
        return True
    if mode == "off":
        return False
    # auto: byte-for-byte the run.py gate.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    else:
        import nest_asyncio
        nest_asyncio.apply()
        return True


def _build_app(frontend: Path) -> FastAPI:
    app = FastAPI()

    @app.get("/api/health")
    async def health():
        return {"ok": True}

    # Studio mounts the built frontend at "/" with html=True (setup_frontend).
    app.mount("/", StaticFiles(directory=str(frontend), html=True), name="frontend")
    return app


def _serve_in_thread(app, host, port):
    """Exactly run_server's pattern: daemon thread, fresh new_event_loop, serve()."""
    ready = threading.Event()
    failed = threading.Event()
    errbox = {}

    class _ReadyServer(uvicorn.Server):
        async def startup(self, *a, **k):
            await super().startup(*a, **k)
            ready.set()

    config = uvicorn.Config(app, host=host, port=port, log_level="warning",
                            access_log=False)
    server = _ReadyServer(config)

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        except BaseException as e:  # noqa: BLE001
            errbox["err"] = f"{type(e).__name__}: {e}"
            failed.set()
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    # run_server waits on a threading.Event, not asyncio.
    for _ in range(100):
        if ready.wait(timeout=0.1) or failed.is_set():
            break
    return server, ready, failed, errbox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nest", choices=["force", "off", "auto"], default="auto")
    ap.add_argument("--port", type=int, default=8971)
    args = ap.parse_args()

    py = "%d.%d.%d" % sys.version_info[:3]
    on_314 = sys.version_info[:2] >= (3, 14)
    applied = _apply_nest(args.nest)
    print(f"Python {py} | nest={args.nest} | nest_asyncio applied={applied}")

    with TemporaryDirectory() as d:
        fe = Path(d)
        (fe / "index.html").write_text("<!doctype html><title>studio</title><h1>ok</h1>")
        (fe / "assets").mkdir()
        (fe / "assets" / "app.js").write_text("console.log('studio asset');")

        app = _build_app(fe)
        server, ready, failed, errbox = _serve_in_thread(app, "127.0.0.1", args.port)

        if not ready.is_set():
            print(f"SERVER FAILED TO START: {errbox.get('err', 'timeout')}")
            print("RESULT=FAIL")
            return 1

        base = f"http://127.0.0.1:{args.port}"
        results = {}
        with httpx.Client(timeout=5.0) as c:
            for path in ("/", "/assets/app.js", "/api/health"):
                try:
                    r = c.get(base + path)
                    results[path] = r.status_code
                except Exception as e:  # noqa: BLE001
                    results[path] = f"EXC {type(e).__name__}"

        server.should_exit = True
        time.sleep(0.2)

        for p, s in results.items():
            print(f"  GET {p:16s} -> {s}")

        # Expectation model.
        breaks = on_314 and applied  # only 3.14 + nest_asyncio applied is expected broken
        if breaks:
            # StaticFiles goes through the anyio thread-pool -> 500 on 3.14.
            ok = results.get("/assets/app.js") == 500 or str(results.get("/")).startswith("5")
            verdict = "expected-500-on-3.14-with-nest_asyncio"
        else:
            ok = all(results.get(p) == 200 for p in ("/", "/assets/app.js", "/api/health"))
            verdict = "expected-all-200"

        print(f"  [{verdict}] observed={'as expected' if ok else 'UNEXPECTED'}")
        print("RESULT=" + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1


if __name__ == "__main__":
    import os
    rc = main()
    sys.stdout.flush()
    os._exit(rc)  # skip any lingering server-thread cleanup
