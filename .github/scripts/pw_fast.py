"""Fast headless Chromium on GPU hosts: launch with `--disable-gpu --disable-software-rasterizer`.

On the 8x B200 host, headless Chromium's GPU process spends ~150 s before the first frame
(measured: a one-line page's first screenshot took 147 s with default args, --in-process-gpu or
--use-angle=swiftshader, and 0.3 s with these two flags, rendering the same pixels). Every
journey, crawl route and upstream Playwright script paid that per page / context.

`patch()` wraps playwright's BrowserType.launch / launch_persistent_context (sync and async APIs
share the _impl class) to append the flags for Chromium unless the caller passed them.
PW_FAST_CHROMIUM=0 turns it off. `pw_fast_site/` holds a sitecustomize.py that calls patch(),
for Playwright scripts we run as subprocesses but do not own (put it first on PYTHONPATH).
"""

import os

ARGS = ["--disable-gpu", "--disable-software-rasterizer"]


def _with_args(kwargs):
    args = list(kwargs.get("args") or [])
    kwargs["args"] = args + [a for a in ARGS if a not in args]
    return kwargs


def patch():
    if os.environ.get("PW_FAST_CHROMIUM", "1") == "0":
        return False
    try:
        from playwright._impl._browser_type import BrowserType
    except Exception:  # noqa: BLE001 - playwright absent or reorganised: leave launches alone
        return False
    for name in ("launch", "launch_persistent_context"):
        orig = getattr(BrowserType, name, None)
        if orig is None or getattr(orig, "_pw_fast", False):
            continue

        def make(orig):
            async def wrapped(self, *a, **k):
                if getattr(self, "name", "") == "chromium":
                    k = _with_args(k)
                return await orig(self, *a, **k)

            wrapped._pw_fast = True
            return wrapped

        setattr(BrowserType, name, make(orig))
    return True


def site_env(env = None):
    """env with pw_fast_site first on PYTHONPATH, so a Playwright subprocess gets the flags too."""
    env = dict(os.environ if env is None else env)
    site = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pw_fast_site")
    env["PYTHONPATH"] = site + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env
