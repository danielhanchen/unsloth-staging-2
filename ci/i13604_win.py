"""Issue 13604 on a real Windows Unsloth Desktop (WebView2): is the chat composer on screen after
the first reply, when the chat is reopened, and with the window maximized?

The installed app is launched with WebView2 remote debugging on, and Playwright attaches over CDP.
Writes probe JSON + webview and full-screen screenshots to --out.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROBE_JS = r"""
() => {
  const pick = () => {
    const all = [...document.querySelectorAll('.aui-thread-composer-dock textarea, form textarea')];
    return all.find(t => t.offsetParent !== null) || all[0] || null;
  };
  const ta = pick();
  const r = (el) => { if (!el) return null; const b = el.getBoundingClientRect();
    return {top: Math.round(b.top), bottom: Math.round(b.bottom), height: Math.round(b.height), left: Math.round(b.left), width: Math.round(b.width)}; };
  let hit = null;
  if (ta) { const b = ta.getBoundingClientRect(); const x = b.left + b.width / 2, y = b.top + b.height / 2;
    const el = (y >= 0 && y < innerHeight) ? document.elementFromPoint(x, y) : null;
    hit = el ? (el === ta || ta.contains(el) ? 'textarea' : (String(el.className || el.tagName)).slice(0, 120)) : 'offscreen'; }
  const q = (s) => r(document.querySelector(s));
  const chain = [];
  for (let el = document.querySelector('.aui-thread-root'); el; el = el.parentElement) {
    const b = el.getBoundingClientRect(); const cs = getComputedStyle(el);
    chain.push({tag: el.tagName, cls: String(el.className || '').slice(0, 90), h: Math.round(b.height), top: Math.round(b.top), bottom: Math.round(b.bottom), display: cs.display, overflow: cs.overflow});
  }
  return {
    innerWidth, innerHeight, outerWidth, outerHeight, dpr: devicePixelRatio,
    screen: {w: screen.width, h: screen.height, availW: screen.availWidth, availH: screen.availHeight},
    visualViewport: window.visualViewport ? {w: visualViewport.width, h: visualViewport.height, scale: visualViewport.scale} : null,
    htmlClass: document.documentElement.className,
    htmlAttrs: [...document.documentElement.attributes].map(a => a.name).filter(n => n.startsWith('data-')),
    docScrollHeight: document.documentElement.scrollHeight, bodyScrollHeight: document.body.scrollHeight,
    textarea: r(ta), dock: q('.aui-thread-composer-dock'), root: q('.aui-thread-root'), viewport: q('.aui-thread-viewport'),
    hitAtTextarea: hit, path: location.pathname + location.search, chain,
    messages: document.querySelectorAll('[data-message-id]').length,
    text: (document.querySelector('.aui-thread-viewport') || document.body).innerText.slice(-300),
  };
}
"""


def log(*a):
    print(*a, flush=True)


def grab_screen(path):
    try:
        from PIL import ImageGrab
        ImageGrab.grab(all_screens=True).save(path)
    except Exception as e:  # noqa: BLE001
        log("screen grab failed:", e)


def wait_cdp(port, timeout):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=3) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--cdp", type=int, default=9222)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"label": a.label}

    env = dict(os.environ)
    env["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = f"--remote-debugging-port={a.cdp}"
    app_log = open(out / "app-stdout.log", "wb")
    proc = subprocess.Popen([a.app], env=env, stdout=app_log, stderr=subprocess.STDOUT)
    log("app pid", proc.pid)
    ver = wait_cdp(a.cdp, 180)
    if not ver:
        log("::error::WebView2 CDP endpoint never opened")
        grab_screen(out / f"{a.label}_nocdp.png")
        return 2
    results["webview2"] = ver.get("Browser")
    log("webview2", ver.get("Browser"))

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(f"http://127.0.0.1:{a.cdp}")
        ctx = browser.contexts[0]
        end = time.time() + 120
        page = None
        while time.time() < end and page is None:
            page = next((pg for pg in ctx.pages if not pg.url.startswith("devtools")), None)
            if page is None:
                time.sleep(1)
        consoles = []
        page.on("console", lambda m: consoles.append(f"{m.type}: {m.text}"[:300]))
        page.on("pageerror", lambda e: consoles.append(f"pageerror: {e}"[:400]))

        def snap(name):
            r = page.evaluate(PROBE_JS)
            results[name] = r
            log(name, json.dumps({k: r.get(k) for k in ("innerHeight", "outerHeight", "screen", "dpr", "textarea", "dock", "root", "hitAtTextarea", "path")}))
            try:
                page.screenshot(path=str(out / f"{a.label}_{name}.png"))
            except Exception as e:  # noqa: BLE001
                log("page shot failed", e)
            grab_screen(out / f"{a.label}_{name}_screen.png")

        # Startup: preflight, managed server, desktop auth, then the chat route.
        end = time.time() + 900
        while time.time() < end:
            try:
                path = page.evaluate("location.pathname")
                ready = page.evaluate("!!document.querySelector('form textarea') && !!localStorage.getItem('unsloth_auth_token')")
            except Exception:  # noqa: BLE001
                path, ready = "?", False
            if ready:
                break
            time.sleep(3)
        log("ready at", page.url)
        grab_screen(out / f"{a.label}_00_ready_screen.png")
        page.screenshot(path=str(out / f"{a.label}_00_ready.png"))

        port = page.evaluate("""async () => {
          for (let p = 8888; p <= 8908; p++) {
            try { const r = await fetch(`http://127.0.0.1:${p}/api/health`); if (r.ok) return p; } catch (e) {}
          }
          return null; }""")
        results["api_port"] = port
        log("api port", port)
        load = page.evaluate("""async (port) => {
          const tok = localStorage.getItem('unsloth_auth_token');
          const h = {'Content-Type': 'application/json', Authorization: 'Bearer ' + tok};
          const r = await fetch(`http://127.0.0.1:${port}/api/inference/load`, {method: 'POST', headers: h,
            body: JSON.stringify({model_path: 'unsloth/Qwen3-0.6B-GGUF', gguf_variant: 'Q4_K_M', max_seq_length: 4096})});
          return r.status + ' ' + (await r.text()).slice(0, 300); }""", port)
        log("load:", load)
        for _ in range(200):
            st = page.evaluate("""async (port) => {
              const tok = localStorage.getItem('unsloth_auth_token');
              const r = await fetch(`http://127.0.0.1:${port}/api/inference/status`, {headers: {Authorization: 'Bearer ' + tok}});
              return (await r.json()).loaded; }""", port)
            if st:
                break
            time.sleep(3)
        log("model loaded:", st)
        page.evaluate("localStorage.setItem('unsloth_chat_reasoning_enabled', 'false')")
        page.reload()
        page.wait_for_selector("form textarea", timeout=120_000)
        time.sleep(3)
        snap("01_new_chat")

        box = page.locator("form textarea").first
        box.click()
        box.fill("Reply Hello")
        box.press("Enter")
        time.sleep(2)
        snap("02_streaming")
        try:
            page.wait_for_function(
                "() => ![...document.querySelectorAll('button[aria-label=\"Stop generating\"]')].some(b => b.offsetParent !== null)",
                timeout=600_000, polling=1000)
        except Exception as e:  # noqa: BLE001
            log("stop wait:", e)
        time.sleep(5)
        snap("03_after_reply")

        page.evaluate("window.__TAURI__.window.getCurrentWindow().maximize()")
        time.sleep(4)
        snap("04_maximized")

        url = page.url
        page.reload()
        time.sleep(10)
        snap("05_reloaded_maximized")
        if "thread=" not in url:
            try:
                page.locator('[data-sidebar="menu-button"]').filter(has_text="Hello").first.click(timeout=5000)
            except Exception:  # noqa: BLE001
                try:
                    page.locator('[data-sidebar="menu-button"]').filter(has_text="New Chat").first.click(timeout=5000)
                except Exception as e:  # noqa: BLE001
                    log("reopen click:", e)
            time.sleep(5)
            snap("06_reopened_maximized")

        page.evaluate("window.__TAURI__.window.getCurrentWindow().unmaximize()")
        time.sleep(4)
        snap("07_unmaximized")
        results["console"] = consoles[-60:]
    (out / f"{a.label}_results.json").write_text(json.dumps(results, indent=1))
    proc.terminate()
    # Verdict line for the log: composer bottom inside the visible webview after the reply.
    bad = []
    for k in ("03_after_reply", "04_maximized", "05_reloaded_maximized", "06_reopened_maximized", "07_unmaximized"):
        r = results.get(k)
        if not r:
            continue
        ta = r.get("textarea") or {}
        if not ta or ta.get("bottom", 1e9) > r["innerHeight"] or r.get("hitAtTextarea") != "textarea":
            bad.append(k)
    log(f"VERDICT {a.label}: composer off-screen or covered in {bad}" if bad else f"VERDICT {a.label}: composer visible in every state")
    return 0


if __name__ == "__main__":
    sys.exit(main())
