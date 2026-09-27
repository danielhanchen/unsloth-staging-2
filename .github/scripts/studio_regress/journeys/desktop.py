#!/usr/bin/env python3
"""Journey 8b: Unsloth Desktop (Tauri) shell before/after, driven over WebDriver. Staging only.

Per PR, only when `studio/src-tauri/**` or `studio/frontend/**` change. Each side's checkout is
built with `tauri build --debug --no-bundle` (unsigned; no updater key needed) and launched
under Xvfb through `tauri-driver` + `WebKitWebDriver`, the pattern upstream's
desktop-app-clean-machine-ci.yml / tests/studio/appimage_model_download_webdriver.py use.
Playwright cannot attach to WebKitGTK, so this journey has its own tiny W3C client and its own
step loop, writing the SAME evidence layout as engine.run_journey
(<out>/<side>/desktop/<step>.png + <step>.facts.json) so diff.py pairs it like any journey.

    python desktop.py --bin studio/src-tauri/target/debug/unsloth-studio --side before \
        --out outputs/studio_regress/prN [--backend real|stub] [--driver-port 4444]

Backends: `real` = the Studio installed at ~/.unsloth/studio (the app spawns it with --api-only,
as on a user machine); `stub` = upstream's capability stub (shell and splash only, 10 s).
Env for determinism: UNSLOTH_STUDIO_FAKE_UPDATE=1 forces the update banner so it is compared.
Exit: 0 all steps ok, 1 a step failed, 2 driver / app would not start.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from studio_regress.contract import Journey, Step, StepFailed, StepUnreachable
except ImportError:  # run as a plain script on a staging runner
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from studio_regress.contract import Journey, Step, StepFailed, StepUnreachable

VIEW = (1440, 900)
ROUTES = ("/chat", "/hub", "/images", "/studio", "/export")
STUB = """case "$*" in
  "-h") exit 0 ;;
  *desktop-capabilities*)
    printf '%s\\n' '{"desktop_protocol_version":1,"desktop_manageability_version":2,"supports_api_only":true,"supports_provision_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_install_ok":true,"version":"__V__"}'
    exit 0 ;;
  *studio*--api-only*) exec sleep 600 ;;
esac
exit 1
"""


class WebDriver:
    """Minimal W3C client (urllib only), enough for navigate / execute / click / screenshot."""

    def __init__(self, base):
        self.base, self.sid = base.rstrip("/"), None

    def _req(self, method, path, body=None, timeout=60):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}").get("value")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"webdriver {method} {path}: {e.code} {e.read()[:300]!r}") from None

    def start(self, application, timeout=120):
        caps = {"capabilities": {"alwaysMatch": {"tauri:options": {"application": str(application)}}}}
        self.sid = self._req("POST", "/session", caps, timeout=timeout)["sessionId"]
        return self

    def s(self, path):
        return f"/session/{self.sid}{path}"

    def execute(self, script, args=(), asynchronous=False, timeout=60):
        return self._req("POST", self.s("/execute/async" if asynchronous else "/execute/sync"),
                         {"script": script, "args": list(args)}, timeout=timeout)

    def screenshot(self, path):
        Path(path).write_bytes(base64.b64decode(self._req("GET", self.s("/screenshot"))))

    def find(self, css):
        return self._req("POST", self.s("/elements"), {"using": "css selector", "value": css}) or []

    def click(self, element):
        eid = next(iter(element.values()))
        self._req("POST", self.s(f"/element/{eid}/click"), {})

    def quit(self):
        if self.sid:
            try:
                self._req("DELETE", self.s(""), timeout=15)
            except Exception:  # noqa: BLE001
                pass


class DCtx:
    """What desktop steps receive (the contract's Ctx fields that make sense here)."""

    def __init__(self, wd, out_dir, side, backend):
        self.page, self.out_dir, self.side, self.backend = wd, out_dir, side, backend
        self.state, self.base_url, self.api, self.models = {}, "", None, {}


def _wait(fn, timeout, every=0.5):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(every)
    return None


def _text(wd):
    return wd.execute("return document.body ? document.body.innerText : ''") or ""


def _freeze(wd):
    wd.execute("""const s = document.createElement('style'); s.textContent =
      '*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important}';
      document.head && document.head.appendChild(s);""")


def s_boot(ctx):
    wd = ctx.page
    ok = _wait(lambda: wd.execute("return document.readyState") == "complete" and len(_text(wd)) > 0, 120)
    if not ok:
        raise StepFailed("webview never rendered any text")
    has_tauri = wd.execute("return !!(window.__TAURI__ && window.__TAURI__.core)")
    _freeze(wd)
    return {"has_tauri_global": bool(has_tauri), "title": wd.execute("return document.title")}


def s_backend_ready(ctx):
    wd = ctx.page
    if ctx.backend == "stub":
        return {"skipped": "stub backend: shell only"}
    ready = _wait(lambda: "/login" in (wd.execute("return location.pathname") or "")
                  or "/chat" in (wd.execute("return location.pathname") or ""), 600, every=2)
    if not ready:
        raise StepFailed("app never left the startup screen for the Studio UI")
    _freeze(wd)
    return {"path": wd.execute("return location.pathname")}


def _nav(ctx, route):
    wd = ctx.page
    if ctx.backend == "stub":
        raise StepUnreachable("stub backend has no Studio routes")
    wd.execute("history.pushState({}, '', arguments[0]); dispatchEvent(new PopStateEvent('popstate'));", [route])
    time.sleep(1.5)
    _freeze(wd)
    return {"path": wd.execute("return location.pathname"), "chars": len(_text(wd)) > 0}


def s_settings(ctx):
    wd = ctx.page
    if ctx.backend == "stub":
        raise StepUnreachable("stub backend has no Settings")
    btn = [e for e in wd.find("button") if (wd.execute("return arguments[0].innerText", [e]) or "").strip() == "Settings"]
    if not btn:
        raise StepUnreachable("no Settings button")
    wd.click(btn[0])
    time.sleep(1)
    return {"dialog": bool(wd.find("[role=dialog]"))}


def s_update_banner(ctx):
    wd = ctx.page
    txt = _text(wd)
    return {"update_banner": "update" in txt.lower() and os.environ.get("UNSLOTH_STUDIO_FAKE_UPDATE") == "1"}


def s_ipc(ctx):
    """A harmless Tauri command round trip (updater check is refused on debug builds with a
    stable error string, which is itself a comparable fact)."""
    wd = ctx.page
    r = wd.execute("""const done = arguments[arguments.length - 1];
      if (!(window.__TAURI__ && window.__TAURI__.core)) { done({error: 'no __TAURI__'}); return; }
      window.__TAURI__.core.invoke('check_desktop_update').then(v => done({ok: v === null ? 'none' : 'update'}),
        e => done({error: String(e).slice(0, 120)}));""", asynchronous=True, timeout=90)
    return {"ipc": r}


# Step actions here are SYNCHRONOUS (WebDriver over urllib); run_steps below drives them, never
# engine.run_journey (which is Playwright + async).
JOURNEY = Journey(
    name="desktop", tier="fast", routes=("desktop",), serial=True,
    steps=tuple(Step(sid, fn) for sid, fn in (
        ("s01_boot", s_boot), ("s02_backend_ready", s_backend_ready),
        *[(f"s{3 + i:02d}_route_{r.strip('/')}", (lambda r: lambda ctx: _nav(ctx, r))(r)) for i, r in enumerate(ROUTES)],
        ("s08_settings", s_settings), ("s09_update_banner", s_update_banner), ("s10_ipc", s_ipc),
    )),
)


def run_steps(ctx, out_dir):
    """engine.run_journey's evidence contract, synchronous and WebDriver-backed."""
    jdir = Path(out_dir) / JOURNEY.name
    jdir.mkdir(parents=True, exist_ok=True)
    broken, timings = None, {}
    for step in JOURNEY.steps:
        facts, t0 = {"_step": step.id}, time.perf_counter()
        if broken:
            facts.update(_status="skipped_after_failure", _after=broken)
        else:
            try:
                facts.update(step.action(ctx) or {})
                facts["_status"] = "ok"
            except StepUnreachable as e:
                facts.update(_status="unreachable", _error=str(e)[:500])
                broken = step.id if ctx.backend != "stub" else broken
            except Exception as e:  # noqa: BLE001
                facts.update(_status="failed", _error=f"{type(e).__name__}: {e}"[:800])
                broken = step.id
            if step.shot and facts["_status"] == "ok":
                try:
                    ctx.page.screenshot(jdir / f"{step.id}.png")
                    dom = {"text": _text(ctx.page)[:20000]}
                    (jdir / f"{step.id}.dom.json").write_text(json.dumps(dom))
                except Exception as e:  # noqa: BLE001
                    facts["_capture_error"] = str(e)[:300]
        facts["_s"] = timings[step.id] = round(time.perf_counter() - t0, 2)
        (jdir / f"{step.id}.facts.json").write_text(json.dumps(facts, indent=1, default=str))
    return {"journey": JOURNEY.name, "broken": broken, "timings": timings}


def _stub_home(root, version):
    h = Path(root)
    (h / ".unsloth/studio/unsloth_studio/bin").mkdir(parents=True, exist_ok=True)
    stub = h / ".unsloth/studio/unsloth_studio/bin/unsloth"
    stub.write_text(STUB.replace("__V__", version))
    stub.chmod(0o755)
    return h


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bin", required=True)
    p.add_argument("--side", required=True, choices=("before", "after"))
    p.add_argument("--out", required=True)
    p.add_argument("--backend", default="real", choices=("real", "stub"))
    p.add_argument("--stub-version", default="")
    p.add_argument("--driver-port", type=int, default=4444)
    a = p.parse_args(argv)
    if not (shutil.which("tauri-driver") and shutil.which("WebKitWebDriver")):
        print("desktop: tauri-driver and WebKitWebDriver must be on PATH", file=sys.stderr)
        return 2
    out = Path(a.out) / a.side
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if a.backend == "stub":
        env["HOME"] = str(_stub_home(out / "stub_home", a.stub_version))
        env.pop("UNSLOTH_STUDIO_HOME", None)
    log = (out / "tauri-driver.log").open("wb")
    drv = subprocess.Popen(["tauri-driver", "--port", str(a.driver_port)], stdout=log, stderr=log, env=env)
    wd = WebDriver(f"http://127.0.0.1:{a.driver_port}")
    try:
        if not _wait(lambda: _ping(wd), 30):
            print("desktop: tauri-driver did not open its port", file=sys.stderr)
            return 2
        wd.start(Path(a.bin).resolve())
        r = run_steps(DCtx(wd, out, a.side, a.backend), out)
        print("DESKTOP " + json.dumps(r))
        return 1 if r["broken"] else 0
    except Exception as e:  # noqa: BLE001
        print(f"desktop: {e}", file=sys.stderr)
        return 2
    finally:
        wd.quit()
        drv.terminate()


def _ping(wd):
    try:
        wd._req("GET", "/status", timeout=2)
        return True
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    sys.exit(main())
