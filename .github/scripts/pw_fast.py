"""Fast headless Chromium on GPU hosts: launch with `--disable-gpu --disable-software-rasterizer`.

On the 8x B200 host, headless Chromium's GPU process spends ~150 s before the first frame
(measured: a one-line page's first screenshot took 147 s with default args, --in-process-gpu or
--use-angle=swiftshader, and 0.3 s with these two flags, rendering the same pixels). Every
journey, crawl route and upstream Playwright script paid that per page / context.

`patch()` wraps playwright's BrowserType.launch / launch_persistent_context (sync and async APIs
share the _impl class) to append the flags for Chromium unless the caller passed them.
PW_FAST_CHROMIUM=0 turns the flags off. The same wrapper swaps in a patched Chrome for Testing
when the bundled Chromium predates its security fixes (scripts/pw_secure.py; PW_PATCHED_CHROMIUM=0
turns that off). `pw_fast_site/` holds a sitecustomize.py that calls patch(), for Playwright
scripts we run as subprocesses but do not own (put it first on PYTHONPATH).
"""

import asyncio
import os
import sys

ARGS = ["--disable-gpu", "--disable-software-rasterizer"]


def _with_args(kwargs):
    args = list(kwargs.get("args") or [])
    kwargs["args"] = args + [a for a in ARGS if a not in args]
    return kwargs


async def _secure(kwargs):
    """Launch a patched Chrome for Testing (sandboxed where this host allows) unless the caller chose a browser."""
    if kwargs.get("executablePath") or kwargs.get("channel"):
        return kwargs
    try:
        import pw_secure
        exe = await asyncio.to_thread(
            pw_secure.patched_executable, kwargs.get("headless") is not False
        )
        sandbox = (
            exe
            and kwargs.get("chromiumSandbox") is None
            and "--no-sandbox" not in (kwargs.get("args") or [])
            and await asyncio.to_thread(pw_secure.sandbox_ok, exe)
        )
    except Exception:  # noqa: BLE001 - never fail a launch over this: keep the bundled build
        return kwargs
    if exe:
        kwargs["executablePath"] = exe
        if sandbox:
            kwargs["chromiumSandbox"] = True
    return kwargs


def patch():
    fast = os.environ.get("PW_FAST_CHROMIUM", "1") != "0"
    if not fast and os.environ.get("PW_PATCHED_CHROMIUM", "1") == "0":
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
                if getattr(self, "name", "") != "chromium":
                    return await orig(self, *a, **k)
                k = _with_args(k) if fast else k
                mine = await _secure(dict(k))
                if mine == k:
                    return await orig(self, *a, **k)
                try:
                    return await orig(self, *a, **mine)
                except Exception as e:  # noqa: BLE001 - patched build did not start: keep the flow working
                    print(
                        f"[pw_secure] WARNING: patched Chromium {mine.get('executablePath')} failed to launch "
                        f"({type(e).__name__}); retrying with Playwright's bundled build",
                        file = sys.stderr,
                    )
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
