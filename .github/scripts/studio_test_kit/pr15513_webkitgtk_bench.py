"""PR 15513: the real Studio frontend in real WebKitGTK, Settings dialog scrolled over a seeded long chat.

Arms (fresh web process each, interleaved): base = main's build, css = PR build without the CPU-paint class,
image = PR build with the image shadows started (what the Linux app does when it paints on the CPU).
Every arm gets the Linux desktop classes (`tauri render-linux`). Serves the build with the long-chat gate's
in-process API (tests/studio/_area_matrix.py), so no backend is needed.

  orchestrate:  python pr15513_webkitgtk_bench.py run --base-dist D --head-dist D --cps cps.js --out out.json
  one arm:      python pr15513_webkitgtk_bench.py arm <dist> <arm> <theme> <turns> <cps.js>   (under xvfb-run)
"""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = Path(os.environ.get("REPO_ROOT", Path.cwd()))
sys.path.insert(0, str(REPO / "tests" / "studio"))

SCROLL_S = float(os.environ.get("BENCH_SCROLL_S", "6"))

# Large surfaces keep only a blur-free 1px edge: what a "no shadows on CPU painting" PR would ship.
_FLAT_SURFACES = (
    '[data-slot="dialog-content"], [data-slot="alert-dialog-content"], [data-slot="sheet-content"], '
    ".chat-composer-surface, .unsloth-composer-surface, [data-sonner-toast][data-styled='true'], .unsloth-files-panel"
)
FLAT_CSS = (
    f":is({_FLAT_SURFACES}) {{ --popup-shadow: 0 0 0 1px rgb(0 0 0 / 0.08) !important;"
    " --composer-shadow: 0 0 0 1px rgb(0 0 0 / 0.08) !important; --composer-focus-shadow: 0 0 0 1px rgb(0 0 0 / 0.12) !important; }"
    f" .dark :is({_FLAT_SURFACES}) {{ --popup-shadow: 0 0 0 1px rgb(255 255 255 / 0.08) !important;"
    " --composer-shadow: 0 0 0 1px rgb(255 255 255 / 0.08) !important; }"
)
# Upper bound: every shadow in the app gone.
ALLFLAT_CSS = (
    "*, *::before, *::after { box-shadow: none !important; text-shadow: none !important; }"
)
# Ablations on top of allflat, each switching one more paint feature off, to find what the CPU spends frames on.
ABLATIONS = {
    "noradius": "*, *::before, *::after { border-radius: 0 !important; }",
    "noalpha": "*, *::before, *::after { opacity: 1 !important; backdrop-filter: none !important; filter: none !important;"
    " -webkit-backdrop-filter: none !important; }",
    "nofontaa": "* { -webkit-font-smoothing: none !important; text-rendering: optimizeSpeed !important; }",
    "contain": '[data-slot="dialog-content"] *, [data-role] { contain: paint; }',
    "nobg": "*, *::before, *::after { background-image: none !important; }",
    "noborder": "*, *::before, *::after { border-color: transparent !important; outline: none !important; }",
}
ARM_CSS = {
    "flat": FLAT_CSS,
    "allflat": ALLFLAT_CSS,
    **{k: ALLFLAT_CSS + " " + v for k, v in ABLATIONS.items()},
}
# Candidate fixes for the dialog's rounded overflow clip, each on top of the flat edges.
CLIP_FIXES = {
    "squaredialog": ':is([data-slot="dialog-content"], [data-slot="alert-dialog-content"]) { border-radius: 0 !important; }',
    "noclip": ':is([data-slot="dialog-content"], [data-slot="alert-dialog-content"]) { overflow: visible !important; }',
    "layer": '[data-slot="dialog-content"] .overflow-y-auto { will-change: transform; }',
    "flatnoradius": "*, *::before, *::after { border-radius: 0 !important; }",
}
ARM_CSS.update({k: FLAT_CSS + " " + v for k, v in CLIP_FIXES.items()})

MEASURE_JS = r"""
return (async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const frame = () => new Promise((r) => requestAnimationFrame(r));
  const out = {};
  const t0 = performance.now();
  for (;;) {
    const rows = document.querySelectorAll('[data-role]');
    const last = rows[rows.length - 1];
    if (rows.length >= __N__ || (last && Number(last.getAttribute('aria-setsize')) >= __N__)) break;
    if (performance.now() - t0 > 120000) { out.error = 'thread never loaded'; return out; }
    await sleep(100);
  }
  out.thread_ms = Math.round(performance.now() - t0);
  if (__IMAGE__) {
    const s0 = performance.now();
    window.__stop = CPS.startCpuPaintShadows({ interfaceZoom: () => 1, subscribeInterfaceZoom: () => () => {} });
    for (;;) {
      const v = getComputedStyle(document.documentElement).getPropertyValue('--cpu-popup');
      if (v.startsWith('url')) break;
      if (performance.now() - s0 > 20000) { out.error = 'image never published'; return out; }
      await sleep(20);
    }
    out.images_ms = Math.round(performance.now() - s0);
  }
  await sleep(1500);
  const scrollFor = async (sc, secs, step) => {
    const deltas = [];
    let dir = -1, last = await frame();
    const end = last + secs * 1000;
    for (;;) {
      sc.scrollTop += step * dir;
      if (sc.scrollTop <= 0 || sc.scrollTop + sc.clientHeight >= sc.scrollHeight - 1) dir = -dir;
      const t = await frame();
      deltas.push(t - last);
      last = t;
      if (t >= end) break;
    }
    const sorted = [...deltas].sort((a, b) => a - b);
    const q = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
    return { frames: deltas.length, fps: deltas.length / (deltas.reduce((a, b) => a + b, 0) / 1000),
             p50: q(0.5), p90: q(0.9), max: sorted[sorted.length - 1], over25: deltas.filter((d) => d > 25).length };
  };
  let chat = document.querySelector('[data-role]');
  while (chat && !(/(auto|scroll)/.test(getComputedStyle(chat).overflowY) && chat.scrollHeight > chat.clientHeight + 200)) chat = chat.parentElement;
  if (chat) {
    const comp = document.querySelector('.unsloth-composer-surface, .chat-composer-surface');
    out.composer_image = comp ? getComputedStyle(comp).borderImageSource.slice(0, 16) : null;
    const c = await scrollFor(chat, __S__, 40);
    for (const k in c) out['chat_' + k] = c[k];
    chat.scrollTop = chat.scrollHeight;
    await sleep(800);
  } else {
    out.chat_error = 'no chat scroller';
  }
  const open0 = performance.now();
  const ev = { key: ',', code: 'Comma', ctrlKey: true, bubbles: true, cancelable: true };
  (document.activeElement || document.body).dispatchEvent(new KeyboardEvent('keydown', ev));
  let dialog = null;
  for (;;) {
    dialog = document.querySelector('[data-slot="dialog-content"]');
    if (dialog && dialog.getBoundingClientRect().width > 100) break;
    if (performance.now() - open0 > 10000) { out.error = 'settings never opened'; return out; }
    await frame();
  }
  await frame();
  out.open_ms = Math.round(performance.now() - open0);
  out.classes = document.documentElement.className;
  out.radius = getComputedStyle(dialog).borderTopLeftRadius;
  await sleep(800);
  const cs = getComputedStyle(dialog);
  out.border_image = cs.borderImageSource.slice(0, 16);
  out.box_shadow = cs.boxShadow.slice(0, 60);
  let scrollers = [];
  const s1 = performance.now();
  for (;;) {
    dialog = document.querySelector('[data-slot="dialog-content"]') || dialog;
    scrollers = [...dialog.querySelectorAll('*')].filter((e) => {
      const s = getComputedStyle(e);
      return /(auto|scroll)/.test(s.overflowY) && e.scrollHeight > e.clientHeight + 200;
    }).sort((a, b) => b.clientHeight - a.clientHeight);
    if (scrollers.length) break;
    if (performance.now() - s1 > 15000) { out.error = 'no scroller in settings'; return out; }
    await sleep(200);
  }
  const sc = scrollers[0];
  out.scroller = { h: sc.clientHeight, sh: sc.scrollHeight };
  const deltas = [];
  let dir = 1, last = await frame();
  const end = last + __S__ * 1000;
  for (;;) {
    sc.scrollTop += 12 * dir;
    if (sc.scrollTop <= 0 || sc.scrollTop + sc.clientHeight >= sc.scrollHeight - 1) dir = -dir;
    const t = await frame();
    deltas.push(t - last);
    last = t;
    if (t >= end) break;
  }
  const sorted = [...deltas].sort((a, b) => a - b);
  const q = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
  out.frames = deltas.length;
  out.fps = deltas.length / (deltas.reduce((a, b) => a + b, 0) / 1000);
  out.p50 = q(0.5); out.p90 = q(0.9); out.max = sorted[sorted.length - 1];
  out.over25 = deltas.filter((d) => d > 25).length;
  return out;
})().then((o) => JSON.stringify(o));
"""


def serve(dist: Path, turns: int):
    import functools
    import http.server
    import threading
    from urllib.parse import urlparse

    import _area_matrix as am

    api = am.Api(turns)

    class Route:
        def __init__(self, handler, method, body):
            self.h, self.request = handler, type("R", (), {})()
            self.request.url, self.request.method, self.request.post_data = (
                handler.path,
                method,
                body,
            )

        def fulfill(
            self,
            status = 200,
            content_type = "application/json",
            body = "",
        ):
            data = body.encode() if isinstance(body, str) else body
            self.h.send_response(status)
            self.h.send_header("Content-Type", content_type)
            self.h.send_header("Content-Length", str(len(data)))
            self.h.end_headers()
            self.h.wfile.write(data)

    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _api(self, method):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else b""
            api.handle(Route(self, method, body))

        def do_GET(self):
            if urlparse(self.path).path.startswith("/api/"):
                return self._api("GET")
            return super().do_GET()

        def do_POST(self):
            return (
                self._api("POST")
                if urlparse(self.path).path.startswith("/api/")
                else self.send_error(404)
            )

        def do_PUT(self):
            return (
                self._api("PUT")
                if urlparse(self.path).path.startswith("/api/")
                else self.send_error(404)
            )

        def send_head(self):
            p = urlparse(self.path).path
            if p == "/" or not (Path(self.directory) / p.lstrip("/")).is_file():
                self.path = "/index.html"
            return super().send_head()

    srv = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(H, directory = str(dist))
    )
    threading.Thread(target = srv.serve_forever, daemon = True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}", api, am


def arm(dist, name, theme, turns, cps_path):
    import gi

    gi.require_version("Gtk", "3.0")
    gi.require_version("WebKit2", "4.1")
    from gi.repository import GLib, Gtk, WebKit2

    srv, base, api, am = serve(Path(dist), int(turns))
    mode = next(m for m in am.app_modes() if m["name"] == "default")
    init = {
        "unsloth_auth_token": am.auth_token(),
        "unsloth_auth_refresh_token": "r",
        "unsloth_window_material": am.material_storage(mode["set"]),
        "theme": theme,
    }
    start = (
        "(() => { const v = %s; for (const k in v) localStorage.setItem(k, v[k]);"
        " document.documentElement.classList.add('tauri', 'render-linux'); })()" % json.dumps(init)
    )
    ucm = WebKit2.UserContentManager()
    ucm.add_script(
        WebKit2.UserScript(
            start,
            WebKit2.UserContentInjectedFrames.TOP_FRAME,
            WebKit2.UserScriptInjectionTime.START,
            None,
            None,
        )
    )
    image = name == "image"
    if image:
        ucm.add_script(
            WebKit2.UserScript(
                Path(cps_path).read_text(),
                WebKit2.UserContentInjectedFrames.TOP_FRAME,
                WebKit2.UserScriptInjectionTime.END,
                None,
                None,
            )
        )
    if name == "cls":
        ucm.add_script(
            WebKit2.UserScript(
                "document.documentElement.classList.add('tauri', 'render-linux', 'webkit-cpu-paint')",
                WebKit2.UserContentInjectedFrames.TOP_FRAME,
                WebKit2.UserScriptInjectionTime.END,
                None,
                None,
            )
        )
    extra_css = ARM_CSS.get(name)
    if extra_css:
        ucm.add_style_sheet(
            WebKit2.UserStyleSheet(
                extra_css,
                WebKit2.UserContentInjectedFrames.TOP_FRAME,
                WebKit2.UserStyleLevel.AUTHOR,
                None,
                None,
            )
        )
    win = Gtk.Window()
    win.set_default_size(1440, 900)
    view = WebKit2.WebView.new_with_user_content_manager(ucm)
    win.add(view)
    result = {
        "arm": name,
        "theme": theme,
        "turns": int(turns),
        "webkit": f"{WebKit2.get_major_version()}.{WebKit2.get_minor_version()}.{WebKit2.get_micro_version()}",
        "scale": os.environ.get("GDK_SCALE", "1"),
    }
    js = MEASURE_JS.replace("__N__", str(2 * int(turns))).replace(
        "__IMAGE__", "true" if image else "false"
    )
    js = js.replace("__S__", str(SCROLL_S))

    def done(v, res):
        try:
            val = v.call_async_javascript_function_finish(res)
            result.update(json.loads(val.to_string()))
        except Exception as e:  # noqa: BLE001
            result["error"] = f"js: {e}"
        Gtk.main_quit()

    def on_load(v, ev):
        if ev == WebKit2.LoadEvent.FINISHED:
            v.call_async_javascript_function(js, -1, None, None, None, None, done)

    view.connect("load-changed", on_load)
    GLib.timeout_add_seconds(300, Gtk.main_quit)
    view.load_uri(f"{base}/chat?thread={api.rec['id']}")
    win.show_all()
    Gtk.main()
    srv.shutdown()
    print("RESULT " + json.dumps(result), flush = True)


def run(argv):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dist", required = True)
    ap.add_argument("--head-dist", required = True)
    ap.add_argument("--cps", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--reps", type = int, default = 4)
    ap.add_argument("--turns", type = int, default = 150)
    ap.add_argument("--themes", default = "dark,light")
    ap.add_argument("--scales", default = "1,2")
    ap.add_argument("--python", default = "/usr/bin/python3")
    ap.add_argument("--arms", default = os.environ.get("BENCH_ARMS", "base,image,flat,allflat"))
    a = ap.parse_args(argv)
    rows = []
    dists = {"base": a.base_dist}
    arms = [(n, dists.get(n, a.head_dist)) for n in a.arms.split(",")]
    for scale in a.scales.split(","):
        w, h = (1440 * int(scale) + 100, 900 * int(scale) + 100)
        for rep in range(a.reps):
            for theme in a.themes.split(","):
                for name, dist in arms:
                    env = {**os.environ, "GDK_SCALE": scale, "REPO_ROOT": str(REPO)}
                    cmd = [
                        "timeout",
                        "330",
                        "xvfb-run",
                        "-a",
                        "-s",
                        f"-screen 0 {w}x{h}x24",
                        a.python,
                        "-I",
                        str(HERE),
                        "arm",
                        dist,
                        name,
                        theme,
                        str(a.turns),
                        a.cps,
                    ]
                    t0 = time.time()
                    p = subprocess.run(cmd, capture_output = True, text = True, env = env)
                    line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
                    r = (
                        json.loads(line[-1][7:])
                        if line
                        else {
                            "arm": name,
                            "theme": theme,
                            "scale": scale,
                            "error": "no result: " + p.stderr[-600:],
                        }
                    )
                    r["rep"], r["wall_s"] = rep, round(time.time() - t0, 1)
                    rows.append(r)
                    print(json.dumps(r), flush = True)
                    Path(a.out).write_text(json.dumps(rows, indent = 1))
    summary = {}
    for r in rows:
        if "fps" not in r:
            continue
        k = f"webkit {r['webkit']} scale {r['scale']} {r['theme']}"
        summary.setdefault(k, {}).setdefault(r["arm"], []).append(r)
    print(
        "\n| setup | arm | Settings scroll fps median (runs) | p50 / p90 / max ms | > 25 ms | "
        "chat scroll fps median (runs) | p50 / p90 / max ms | > 25 ms | Settings open ms |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for k, arms_ in summary.items():
        for name in dict.fromkeys(r["arm"] for r in rows):
            rs = arms_.get(name, [])
            if not rs:
                continue
            med = lambda f: statistics.median(x.get(f, float("nan")) for x in rs)  # noqa: E731
            runs = ", ".join("%.1f" % x["fps"] for x in rs)
            cruns = ", ".join("%.1f" % x.get("chat_fps", float("nan")) for x in rs)
            print(
                f"| {k} | {name} | {med('fps'):.1f} ({runs}) | "
                f"{med('p50'):.1f} / {med('p90'):.1f} / {med('max'):.0f} | {med('over25'):.0f} | "
                f"{med('chat_fps'):.1f} ({cruns}) | {med('chat_p50'):.1f} / {med('chat_p90'):.1f} / "
                f"{med('chat_max'):.0f} | {med('chat_over25'):.0f} | {med('open_ms'):.0f} |"
            )
    errs = [r for r in rows if "fps" not in r]
    print(f"\nerrors: {len(errs)}")
    for r in errs[:5]:
        print(json.dumps(r)[:400])
    return len(errs), len(rows)


def sh(
    cmd,
    cwd = None,
    check = True,
):
    print(f"+ {cmd}", flush = True)
    return subprocess.run(cmd, shell = True, cwd = cwd, check = check)


BASE = Path("/tmp/pr15513_base")


def ci():
    """Staging CI entry: builds main and the PR on the runner, measures inside Ubuntu 24.04 (the AppImage's base)."""
    fe = REPO / "studio" / "frontend"
    out = Path(os.environ.get("PW_ART_DIR", "logs/playwright_staging")).resolve()
    out.mkdir(parents = True, exist_ok = True)
    sh("npm ci --no-audit --no-fund --loglevel=error", cwd = fe)
    sh("npx vite build --logLevel error", cwd = fe)
    sh(
        f"npx rolldown src/lib/cpu-paint-shadows.ts --format iife --name CPS -o {out}/cps.js",
        cwd = fe,
    )
    sh("git fetch -q --depth 1 origin upstream-main", cwd = REPO)
    sh(f"git worktree add -q --detach {BASE} FETCH_HEAD", cwd = REPO)
    sh(f"git -C {BASE} log -1 --format='base %H %s'; git -C {REPO} log -1 --format='head %H %s'")
    sh(f"ln -s {fe}/node_modules {BASE}/studio/frontend/node_modules")
    sh("npx vite build --logLevel error", cwd = BASE / "studio" / "frontend")
    script = HERE.relative_to(REPO)
    sh(
        f"docker run --rm -v {REPO}:/repo -v {BASE}:{BASE} -v {out}:/out -e BENCH_REPS -e REPO_ROOT=/repo "
        f"-w /repo ubuntu:24.04 bash -c 'apt-get -qq update && apt-get -qq install -y python3 >/dev/null "
        f"&& python3 -I /repo/{script} incontainer'"
    )


def incontainer():
    env = "DEBIAN_FRONTEND=noninteractive "
    sh(
        "apt-get -qq update && " + env + "apt-get -qq install -y xvfb xauth curl python3-gi "
        "gir1.2-webkit2-4.1 gir1.2-gtk-3.0 fonts-dejavu-core >/dev/null"
    )
    out = Path("/out")
    common = [
        "--base-dist",
        str(BASE / "studio/frontend/dist"),
        "--head-dist",
        "/repo/studio/frontend/dist",
        "--cps",
        str(out / "cps.js"),
        "--reps",
        os.environ.get("BENCH_REPS", "3"),
    ]
    sh("nproc; free -g | head -2; dpkg -l libwebkit2gtk-4.1-0 | tail -1")
    bad, total = (
        run(common + ["--out", str(out / "apt.json"), "--scales", "2"])
        if os.environ.get("BENCH_APT_PASS")
        else (0, 0)
    )
    ver = "2.50.4-0ubuntu0.24.04.1"
    url = "https://launchpad.net/ubuntu/+archive/primary/+files"
    debs = [
        f"/tmp/{n}_{ver}_amd64.deb"
        for n in (
            "libwebkit2gtk-4.1-0",
            "libjavascriptcoregtk-4.1-0",
            "gir1.2-webkit2-4.1",
            "gir1.2-javascriptcoregtk-4.1",
        )
    ]
    for d in debs:
        sh(f"curl -sSfL -o {d} {url}/{Path(d).name}")
    sh(env + "apt-get -qq install -y --allow-downgrades " + " ".join(debs) + " >/dev/null")
    sh("dpkg -l libwebkit2gtk-4.1-0 | tail -1")
    b2, t2 = run(
        common
        + [
            "--out",
            str(out / "webkit_2_50_4.json"),
            "--scales",
            "1,2",
            "--themes",
            "dark,light",
            "--arms",
            "base,cls",
        ]
    )
    if (bad + b2) * 2 > total + t2:
        sys.exit(f"too many failed measurements: {bad + b2} of {total + t2}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        ci()
    elif sys.argv[1] == "incontainer":
        incontainer()
    elif sys.argv[1] == "arm":
        arm(*sys.argv[2:7])
    else:
        run(sys.argv[2:])
