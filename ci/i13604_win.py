"""Issue 13604 on a real Windows Unsloth Desktop (WebView2): is the chat composer on screen after
the first reply, when the chat is reopened, and with the window maximized?

Release builds keep WebView2 DevTools off, so there is no CDP. The app is driven like a user:
native keyboard / mouse through Windows UI Automation (WebView2 exposes its accessibility tree
there, with screen coordinates), the model is loaded over the backend API with the desktop
secret the app itself uses, and every state gets a full-screen grab.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8888"


def log(*a):
    print(*a, flush=True)


def http(method, path, body=None, token=None, timeout=60):
    req = urllib.request.Request(API + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def grab(path):
    from PIL import ImageGrab
    ImageGrab.grab(all_screens=True).save(path)


def rect(r):
    return {"left": r.left, "top": r.top, "right": r.right, "bottom": r.bottom}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompt", default="Reply Hello")
    ap.add_argument("--seed-only", action="store_true", help="chat once, maximize, quit (the pre-update session)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"label": a.label}

    import ctypes
    from ctypes import wintypes
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    work = wintypes.RECT()
    ctypes.windll.user32.SystemParametersInfoW(0x30, 0, ctypes.byref(work), 0)  # SPI_GETWORKAREA
    res["work_area"] = {"left": work.left, "top": work.top, "right": work.right, "bottom": work.bottom}
    res["screen"] = [ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)]
    log("work area", res["work_area"], "screen", res["screen"])

    proc = subprocess.Popen([a.app], stdout=open(out / "app-stdout.log", "wb"), stderr=subprocess.STDOUT)
    secret_file = Path.home() / ".unsloth" / "studio" / "auth" / ".desktop_secret"
    end = time.time() + 600
    token = None
    while time.time() < end and token is None:
        try:
            if secret_file.exists():
                token = http("POST", "/api/auth/desktop-login", {"secret": secret_file.read_text().strip()})["access_token"]
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    if not token:
        log("::error::no desktop token")
        grab(out / f"{a.label}_notoken.png")
        return 2
    log("desktop token ok")
    print(http("POST", "/api/inference/load", {"model_path": "unsloth/Qwen3-0.6B-GGUF", "gguf_variant": "Q4_K_M",
                                               "max_seq_length": 8192}, token, timeout=900))
    for _ in range(200):
        if http("GET", "/api/inference/status", token=token).get("loaded"):
            break
        time.sleep(3)
    log("model loaded")
    time.sleep(15)

    from pywinauto import Desktop
    from pywinauto.keyboard import send_keys
    win = None
    for _ in range(60):
        wins = [w for w in Desktop(backend="uia").windows() if w.process_id() == proc.pid
                or "Unsloth" in (w.window_text() or "")]
        wins = [w for w in wins if w.class_name() != "ConsoleWindowClass"]
        if wins:
            win = wins[0]
            break
        time.sleep(2)
    if win is None:
        log("::error::app window not found")
        grab(out / f"{a.label}_nowindow.png")
        return 2
    log("window", win.window_text(), win.class_name(), rect(win.rectangle()))
    win.set_focus()
    # The app reloads its chat once the model is resident; let it settle.
    time.sleep(5)

    def edits():
        found = []
        for e in win.descendants(control_type="Edit"):
            try:
                found.append({"name": e.window_text()[:60], "rect": rect(e.rectangle()),
                              "offscreen": bool(e.element_info.element.CurrentIsOffscreen)})
            except Exception as ex:  # noqa: BLE001
                found.append({"error": str(ex)[:100]})
        return found

    def composer():
        cands = []
        for e in win.descendants(control_type="Edit"):
            try:
                r = e.rectangle()
            except Exception:  # noqa: BLE001
                continue
            if r.width() > 300:
                cands.append((r.bottom, e))
        return max(cands, key=lambda c: c[0])[1] if cands else None

    def snap(name):
        wr = win.rectangle()
        state = {"window": rect(wr), "maximized": win.is_maximized(), "edits": edits()}
        c = composer()
        state["composer"] = rect(c.rectangle()) if c else None
        bottom_limit = min(wr.bottom, work.bottom)
        state["composer_visible"] = bool(c) and c.rectangle().bottom <= bottom_limit and c.rectangle().top >= wr.top
        res[name] = state
        log(name, json.dumps({k: state[k] for k in ("window", "maximized", "composer", "composer_visible")}))
        grab(out / f"{a.label}_{name}.png")

    for label in ("Remind me later", "Later", "Dismiss"):
        try:
            win.child_window(title=label, control_type="Button").click_input()
            log("dismissed banner via", label)
            break
        except Exception:  # noqa: BLE001
            pass
    time.sleep(2)
    for _ in range(30):
        if composer() is not None:
            break
        time.sleep(2)
    snap("01_new_chat")

    c = composer()
    if c is None:
        log("::error::no composer edit found")
        return 2
    c.click_input()
    send_keys(a.prompt.replace(" ", "{SPACE}") + "{ENTER}", with_spaces=False)
    time.sleep(3)
    snap("02_streaming")
    end = time.time() + 600
    while time.time() < end:
        stops = [b for b in win.descendants(control_type="Button") if "Stop generating" in (b.window_text() or "")]
        if not stops:
            break
        time.sleep(3)
    time.sleep(6)
    snap("03_after_reply")

    win.maximize()
    time.sleep(5)
    snap("04_maximized")
    if a.seed_only:
        (out / f"{a.label}_results.json").write_text(json.dumps(res, indent=1))
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], check=False)
        time.sleep(10)
        log(f"SEEDED {a.label}")
        return 0

    win.restore()
    time.sleep(5)
    snap("05_restored")

    # Reopen: a new chat, then back to the first chat in Recents.
    send_keys("^+o")
    time.sleep(3)
    try:
        items = [i for i in win.descendants() if (i.window_text() or "").strip() in ("New Chat", "Hello", "Reply Hello")]
        log("recents candidates", [(i.window_text(), i.element_info.control_type) for i in items][:10])
        for i in items:
            if i.rectangle().left < 300:
                i.click_input()
                break
    except Exception as e:  # noqa: BLE001
        log("reopen:", e)
    time.sleep(5)
    snap("06_reopened")
    win.maximize()
    time.sleep(5)
    snap("07_reopened_maximized")

    # Interface zoom (Ctrl+=, native WebView2 zoom), as a user on a high-DPI or zoomed setup has.
    win.restore()
    time.sleep(3)
    for step in range(3):
        win.set_focus()
        send_keys("^=")
        time.sleep(3)
    snap("08_zoomed_restored")
    win.maximize()
    time.sleep(5)
    snap("09_zoomed_maximized")
    c = composer()
    if c is not None:
        c.click_input()
        send_keys("Reply{SPACE}Hello{SPACE}again{ENTER}")
        time.sleep(3)
        end = time.time() + 600
        while time.time() < end:
            if not [b for b in win.descendants(control_type="Button") if "Stop generating" in (b.window_text() or "")]:
                break
            time.sleep(3)
        time.sleep(6)
        snap("10_zoomed_maximized_after_reply")
    win.restore()
    time.sleep(5)
    snap("11_zoomed_restored_after_reply")

    (out / f"{a.label}_results.json").write_text(json.dumps(res, indent=1))
    bad = [k for k, v in res.items() if isinstance(v, dict) and "composer_visible" in v and not v["composer_visible"]]
    log(f"VERDICT {a.label}: composer off-screen in {bad}" if bad else f"VERDICT {a.label}: composer visible in every state")
    proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
