"""Determinism fixture shared by every journey: one browser-context recipe, one settle
barrier, one screenshot + DOM capture. Both sides of a pair go through exactly this code.

What is pinned: viewport 1440x900 @1x, en-US, UTC, light scheme, reduced motion, a frozen
displayed clock (page.clock, wall time only; timers still run so auth / polling behave),
CSS that kills animations / transitions / caret blink / scrollbar fade, fonts loaded + two
stable animation frames before any capture.

What is normalised in DOM text (never in pixels): UUIDs, hex ids, ISO / clock times, dates,
durations ("12.3s", "4 min ago"), throughput ("12.5 tok/s"), byte sizes with decimals,
ports. Pixels for those regions are handled by per-step mask selectors instead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

VIEWPORT = {"width": 1440, "height": 900}
FIXED_TIME = "2026-01-15T12:00:00Z"
KILL_MOTION_CSS = """
*, *::before, *::after {
  animation-duration: 0s !important; animation-delay: 0s !important;
  animation-iteration-count: 1 !important; transition: none !important;
  caret-color: transparent !important; scroll-behavior: auto !important;
}
::-webkit-scrollbar { display: none !important; }
[data-sonner-toaster] li { transition: none !important; }
"""
# Always-volatile regions masked in every shot (reason: live counters, clocks).
# .unsloth-welcome-greeting: the empty chat's greeting and mascot are picked at random per load
# (thread.tsx buildWelcome); the seeded Math.random below does not pin them, since how many
# draws happen before it depends on timing.
# Success / info toasts ("<model> loaded") expire on a timer, so whether one is still up at capture
# is timing. Error and warning toasts stay visible: they are the regression signal.
GLOBAL_MASKS = ("[data-testid='tokens-per-second']", "[data-volatile]", "time", ".unsloth-welcome-greeting",
                "[data-sonner-toast]:not([data-type='error']):not([data-type='warning'])")

_NORMALISE = [
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<uuid>"),
    (re.compile(r"\b[0-9a-f]{12,64}\b", re.I), "<hex>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?Z?\b"), "<datetime>"),
    (re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\s?(AM|PM|am|pm)?\b"), "<time>"),
    (re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(, \d{4})?\b"), "<date>"),
    (re.compile(r"\b\d+(\.\d+)?\s?(tok/s|tokens/s|it/s|t/s)\b"), "<rate>"),
    (re.compile(r"\b\d+(\.\d+)?\s?(ms|s|sec|secs|seconds|min|mins|minutes|h|hours)\b( ago)?"), "<dur>"),
    (re.compile(r"\b(just now|\d+ (second|minute|hour|day)s? ago)\b"), "<ago>"),
    (re.compile(r"\b\d+(,\d{3})*(\.\d+)?\s?(B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)\b"), "<size>"),
    (re.compile(r"\b\d+(\.\d+)?\s?(GHz|MHz|°C|W)\b"), "<hw>"),
    (re.compile(r"\b\d+ lines\b"), "<n> lines"),
    (re.compile(r"(127\.0\.0\.1|localhost):\d+"), r"\1:<port>"),
    (re.compile(r"\bport[=: ]\d+"), "port=<port>"),
    (re.compile(r"\bpid[=: ]?\d+"), "pid<pid>"),
    (re.compile(r"\b\d{8}-\d{6}\b"), "<stamp>"),
    (re.compile(r"\b\d+(\.\d+)?%"), "<pct>"),
]


# A line that is only a compact duration ("9m", "1h 5m", "2d 3h"): Settings > System "Uptime"
# (host uptime, so it differs between two runners of the same OS). Whole-line only: "270m" inside
# a model name must survive.
_COMPACT_DUR = re.compile(r"^(\d+d)?\s?(\d+h)?\s?(\d+m)?\s?(\d+s)?$")


def normalise_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if any(c.isdigit() for c in s) and _COMPACT_DUR.match(s):
        return "<dur>"
    for rx, rep in _NORMALISE:
        s = rx.sub(rep, s)
    return re.sub(r"\s+", " ", s).strip()


# One definition of "control identity", shared by the coverage inventory and the click
# recorder so the two always agree: "<role>:<accessible name>", name normalised.
CONTROL_KEY_JS = r"""
window.__srCtl = window.__srCtl || {
  sel: 'button,a[href],[role=button],[role=tab],[role=menuitem],[role=switch],[role=checkbox],' +
       '[role=combobox],[role=radio],[role=option],select,input[type=checkbox],input[type=radio]',
  key(el) {
    const role = el.getAttribute('role') || ({A: 'link', SELECT: 'combobox', INPUT: el.type}[el.tagName]) || 'button';
    let name = (el.getAttribute('aria-label') || el.innerText || el.getAttribute('title') || '')
                 .split('\n')[0].trim().replace(/\s+/g, ' ').replace(/[0-9]+/g, '#').slice(0, 60);
    return name ? role + ':' + name : null;
  },
  visible(el) { const r = el.getBoundingClientRect(); const cs = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && cs.display !== 'none'; },
};
"""
# Deterministic Math.random (Studio picks random greetings / tips) and an in-flight request
# counter the settle barrier waits on (async panels render "Loading..." placeholders first).
DETERMINISM_JS = r"""
(() => {
  let s = 0x2F6E2B1;
  Math.random = () => { s |= 0; s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  // In-flight requests by start tick (tick = 100ms of setInterval; Date may be frozen).
  // Streaming endpoints (SSE, long-poll, chat generation) never count; anything else counts for
  // up to 150 ticks: a cold /api/system probe takes ~10 s and the page renders "No GPU" until it
  // answers, so a 2 s cut-off let the shot race it.
  const LONG = /stream|events|completions|chat-runs|\/logs|monitor|progress|sse/i;
  window.__srTick = 0; setInterval(() => { window.__srTick++; }, 100);
  const open = new Map(); let seq = 0;
  Object.defineProperty(window, '__srInflight', { get() {
    let n = 0; for (const t0 of open.values()) if (window.__srTick - t0 < 150) n++; return n; } });
  // Every open (non-streaming) request whatever its age: for waits that must see a slow answer land.
  Object.defineProperty(window, '__srOpenAll', { get() { return open.size; } });
  const urlOf = a => { try { return String(a[0] && a[0].url || a[0] || ''); } catch (e) { return ''; } };
  const of = window.fetch;
  window.fetch = function (...a) {
    if (LONG.test(urlOf(a))) return of.apply(this, a);
    const id = ++seq; open.set(id, window.__srTick);
    return of.apply(this, a).finally(() => { open.delete(id); }); };
  const oo = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function (...a) { this.__srUrl = String(a[1] || ''); return oo.apply(this, a); };
  const os = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.send = function (...a) {
    if (LONG.test(this.__srUrl || '')) return os.apply(this, a);
    const id = ++seq; open.set(id, window.__srTick);
    this.addEventListener('loadend', () => { open.delete(id); }); return os.apply(this, a); };
})();
"""
CLICK_RECORDER_JS = CONTROL_KEY_JS + r"""
window.__srClicks = window.__srClicks || [];
document.addEventListener('click', ev => {
  const el = ev.target && ev.target.closest && ev.target.closest(window.__srCtl.sel);
  if (!el) return;
  const k = window.__srCtl.key(el);
  if (k) window.__srClicks.push(location.pathname + '|' + k);
}, true);
"""


async def drain_clicks(page):
    try:
        return await page.evaluate("() => { const c = window.__srClicks || []; window.__srClicks = []; return c; }")
    except Exception:
        return []


async def new_context(browser, **extra):
    kw = dict(viewport=VIEWPORT, device_scale_factor=1, locale="en-US", timezone_id="UTC",
              color_scheme="light", reduced_motion="reduce", bypass_csp=True)
    kw.update(extra)
    ctx = await browser.new_context(**kw)
    await ctx.add_init_script(
        "(() => { const s = document.createElement('style'); s.setAttribute('data-sr-fixture','1');"
        f" s.textContent = {json.dumps(KILL_MOTION_CSS)};"
        " const add = () => (document.head || document.documentElement).appendChild(s);"
        " if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', add); else add();"
        # Studio prefers light theme when unset; pin it so an OS dark default never leaks in.
        " try { localStorage.setItem('theme', 'light'); localStorage.setItem('vite-ui-theme', 'light'); } catch (e) {}"
        "})();")
    await ctx.add_init_script(DETERMINISM_JS)
    await ctx.add_init_script(CLICK_RECORDER_JS)
    return ctx


async def warm_browser(browser, timeout_s=300.0) -> float:
    """Pay a fresh browser's first-frame cost once, before any step, instead of inside the first
    journey's first steps (their screenshots and click "stable" checks). pw_fast's launch flags
    removed the minutes-long first frame on GPU hosts; this keeps whatever is left out of the
    steps and is ~0.3 s otherwise. Returns seconds."""
    import time
    from studio_regress import timeouts
    t0 = time.perf_counter()
    ctx = await new_context(browser)
    try:
        page = await ctx.new_page()
        await page.set_content("<html><body><h1 style='font-family:sans-serif'>warm</h1>"
                               "<button>ok</button></body></html>")
        await page.screenshot(timeout=timeouts.scaled(timeout_s) * 1000, animations="disabled", caret="hide")
        await page.locator("button").click(timeout=timeouts.scaled(timeout_s) * 1000)
    except Exception:
        pass   # the steps' own timeouts decide; this only moves the cost out of them
    finally:
        await ctx.close()
    return round(time.perf_counter() - t0, 1)


async def prepare_page(page, freeze_clock=True):
    if freeze_clock:
        try:
            # Displayed time only: pause-less fixed wall clock; setTimeout / intervals still fire.
            await page.clock.set_fixed_time(FIXED_TIME)
        except Exception:
            pass
    return page


LOADING_RX = r"(Loading\.\.\.|Loading…|Checking [A-Za-z ]{1,60}(\.\.\.|…))"
LOADING_MS = 30_000   # placeholders mean "not rendered yet": waited out even past the request-quiet budget


async def settle(page, timeout_ms=10_000, quiet_ms=16_000, loading_ms=LOADING_MS):
    """Requests quiet + no loading placeholders (bounded), fonts ready, no pending image
    decode, then two identical layout frames."""
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    except Exception:
        pass
    try:
        await page.evaluate(
            """async ([quietMs, loadingMs, rx]) => {
              // iteration-bounded: Date.now / performance.now may be frozen by page.clock
              const re = new RegExp(rx); let calm = 0;
              for (let i = 0; i < Math.max(quietMs, loadingMs) / 120; i++) {
                const loading = re.test(document.body ? document.body.innerText : '');
                const inflight = i < quietMs / 120 && (window.__srInflight || 0) > 0;
                calm = (loading || inflight) ? 0 : calm + 1;
                if (calm >= 3) return;
                await new Promise(r => setTimeout(r, 120));
              }
            }""", [quiet_ms, loading_ms, LOADING_RX])
    except Exception:
        pass
    try:
        await page.evaluate(
            """async () => {
              if (document.fonts && document.fonts.ready) await document.fonts.ready;
              await Promise.all([...document.images].filter(i => !i.complete)
                 .map(i => new Promise(r => { i.onload = i.onerror = r; setTimeout(r, 3000); })));
              const sig = () => document.body ? document.body.scrollHeight + ':' +
                  document.body.getElementsByTagName('*').length : '';
              let last = sig();
              for (let i = 0; i < 20; i++) {
                await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
                const now = sig();
                if (now === last) return; last = now;
              }
            }""")
    except Exception:
        pass
    # Blur the focused element: a focus ring on whichever element won a race is noise.
    try:
        await page.evaluate("() => document.activeElement && document.activeElement.blur && document.activeElement.blur()")
    except Exception:
        pass


async def dom_snapshot(page, hide=(), instance=None):
    """Normalised visible-text lines plus interactive element roles/names. Elements matching
    `hide` (the step's masks) are excluded from the text, like they are from the pixels.
    `instance`: its token / port are replaced by fixed text first (engine.canon_text)."""
    from studio_regress.engine import canon_text
    data = await page.evaluate(
        """(hide) => {
          const hidden = [];
          for (const sel of hide) { try { for (const el of document.querySelectorAll(sel)) {
            hidden.push([el, el.style.visibility]); el.style.visibility = 'hidden'; } } catch (e) {} }
          const vis = el => { const r = el.getBoundingClientRect(); const cs = getComputedStyle(el);
            return r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && cs.display !== 'none'; };
          const text = (document.body ? document.body.innerText : '').split('\\n');
          const sel = 'button,a[href],[role=button],[role=tab],[role=menuitem],[role=switch],'+
                      '[role=checkbox],[role=combobox],input,select,textarea,[role=link]';
          const ctrls = [...document.querySelectorAll(sel)].filter(vis).map(el => {
            const role = el.getAttribute('role') || el.tagName.toLowerCase();
            const name = (el.getAttribute('aria-label') || el.innerText || el.getAttribute('placeholder') ||
                          el.getAttribute('title') || el.getAttribute('name') || '').trim().slice(0, 80);
            const st = [el.disabled ? 'disabled' : '', el.getAttribute('aria-checked') === 'true' ? 'checked' : '',
                        el.getAttribute('aria-selected') === 'true' ? 'selected' : '',
                        el.getAttribute('aria-expanded') === 'true' ? 'expanded' : ''].filter(Boolean).join(',');
            return role + ':' + name + (st ? '[' + st + ']' : '');
          });
          for (const [el, v] of hidden) el.style.visibility = v;
          return {text, ctrls, url: location.pathname};
        }""", list(hide))
    lines = [normalise_text(canon_text(t, instance)) for t in data["text"]]
    lines = [l for l in lines if l]
    ctrls = [normalise_text(canon_text(c, instance)) for c in data["ctrls"]]
    return {"url": data["url"], "lines": lines + ["#ctrl " + c for c in ctrls]}


async def mask_rects(page, selectors):
    rects = []
    for sel in selectors:
        try:
            boxes = await page.evaluate(
                """sel => [...document.querySelectorAll(sel)].map(e => { const r = e.getBoundingClientRect();
                     return [r.x, r.y, r.width, r.height]; }).filter(b => b[2] > 0 && b[3] > 0)""", sel)
            rects.extend(boxes)
        except Exception:
            pass
    return rects


# Readouts that move run to run whatever the build: throughput, wall-clock times, request
# durations. Tagged by their OWN text (not a container's), so only the figure is masked, and
# tagged data-volatile, which GLOBAL_MASKS covers. The DOM snapshot normalises the same text.
TAG_VOLATILE_JS = r"""() => {
  const rx = [/\b\d+(\.\d+)?\s*tok\/s\b/i, /\b\d{1,2}:\d{2}(:\d{2})?\s?(AM|PM)\b/i,
              /^\s*\d+(\.\d+)?\s?(ms|s)\s*$/, /(127\.0\.0\.1|localhost):\d{2,5}\b/,
              /\d{6}-pid\d+\.log/, /^\s*(\d+d\s?)?(\d+h\s?)?\d+[mh]\s*$/];
  let n = 0;
  for (const el of document.querySelectorAll('body *')) {
    if (el.hasAttribute('data-volatile')) continue;
    const own = [...el.childNodes].filter(c => c.nodeType === 3).map(c => c.textContent).join('');
    if (own.trim() && rx.some(r => r.test(own))) { el.setAttribute('data-volatile', ''); n++; }
  }
  return n;
}"""


# Parallel mode (engine.instance_info): each unit's Studio has its own home state/<root>-iNN and
# port, both the same width across instances. Their text reaches the page (auth reset hint, Settings
# > Logs paths, window.location.origin in agent / API panels). Mask ONLY the glyphs that differ:
# the "iNN" of the token and the port digits (after ':' / '=' / 'port '), found by text-node
# Ranges, clipped to every overflow ancestor (text scrolled out of a panel must not mask what is
# drawn over it), painted as data-volatile boxes (a GLOBAL_MASK, so Playwright paints them and
# mask_rects records them for the diff). Nothing in the app's own DOM is changed. Input / textarea
# values have no text Ranges: such a field is masked whole and reported.
INSTANCE_MASK_JS = r"""(inst) => {
  const out = []; const tok = inst.token; const from = inst.mask_from; const port = String(inst.port);
  const portRx = new RegExp('(?:[:=]|[Pp]ort )(' + port + ')(?!\\d)', 'g');
  const clip = (r, el) => {
    let x0 = r.left, y0 = r.top, x1 = r.right, y1 = r.bottom;
    for (let e = el; e && e !== document.body && e !== document.documentElement; e = e.parentElement) {
      const cs = getComputedStyle(e);
      if (cs.overflowX !== 'visible' || cs.overflowY !== 'visible') {
        const b = e.getBoundingClientRect();
        x0 = Math.max(x0, b.left); y0 = Math.max(y0, b.top); x1 = Math.min(x1, b.right); y1 = Math.min(y1, b.bottom);
      }
    }
    return (x1 > x0 && y1 > y0) ? [x0, y0, x1 - x0, y1 - y0] : null;
  };
  const box = (r, what) => {
    const d = document.createElement('div');
    d.setAttribute('data-volatile', ''); d.setAttribute('data-sr-instance-mask', what);
    d.style.cssText = 'position:absolute;pointer-events:none;z-index:2147483647;margin:0;padding:0;border:0;' +
      `left:${r[0] + scrollX}px;top:${r[1] + scrollY}px;width:${r[2]}px;height:${r[3]}px`;
    document.body.appendChild(d);
  };
  const mark = (node, a, b, what) => {
    const rg = document.createRange(); rg.setStart(node, a); rg.setEnd(node, b);
    let n = 0;
    for (const r of rg.getClientRects()) { const c = clip(r, node.parentElement); if (c) { box(c, what); n++; } }
    if (n) out.push(what + ':' + node.nodeValue.slice(Math.max(0, a - 24), b + 8));
  };
  if (!document.body) return out;
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let node = w.nextNode(); node; node = w.nextNode()) {
    const v = node.nodeValue || '';
    const el = node.parentElement;
    if (!el || el.closest('[data-sr-instance-mask],script,style,noscript')) continue;
    for (let i = v.indexOf(tok); i >= 0; i = v.indexOf(tok, i + 1)) mark(node, i + from, i + tok.length, 'home');
    if (v.indexOf(port) >= 0) {
      portRx.lastIndex = 0; let m;
      while ((m = portRx.exec(v))) { const b = m.index + m[0].length; mark(node, b - port.length, b, 'port'); }
    }
  }
  for (const el of document.querySelectorAll('input,textarea')) {
    const v = el.value || '';
    if ((v.includes(tok) || new RegExp('(?:[:=])' + port + '(?!\\d)').test(v)) && !el.hasAttribute('data-volatile')) {
      el.setAttribute('data-volatile', ''); el.setAttribute('data-sr-instance-field', '');
      out.push('field:' + (el.getAttribute('name') || el.getAttribute('aria-label') || el.tagName));
    }
  }
  return out;
}"""
CLEAR_INSTANCE_MASK_JS = r"""() => {
  for (const d of document.querySelectorAll('[data-sr-instance-mask]')) d.remove();
  for (const el of document.querySelectorAll('[data-sr-instance-field]')) {
    el.removeAttribute('data-sr-instance-field'); el.removeAttribute('data-volatile'); }
}"""


async def mask_instance(page, instance):
    """Place the instance-literal masks (see INSTANCE_MASK_JS); returns what was masked."""
    if not instance:
        return []
    try:
        return await page.evaluate(INSTANCE_MASK_JS, instance)
    except Exception:
        return []


async def clear_instance_masks(page):
    try:
        await page.evaluate(CLEAR_INSTANCE_MASK_JS)
    except Exception:
        pass


async def tag_volatile(page):
    try:
        return await page.evaluate(TAG_VOLATILE_JS)
    except Exception:
        return 0


async def capture(page, out_dir: Path, step_id: str, masks=(), full_page=False, shot=True, quick=False,
                  instance=None):
    """Write <step>.png (masked regions painted by Playwright) and <step>.dom.json.
    Returns (mask_rects, dom). Masks are recorded so the diff ignores the same rectangles.
    quick: evidence of a failed step (never pixel-compared): short settle, one frame.
    dom["_settle_s"] / dom["_shot_s"] carry the time spent (not written to dom.json).
    `instance` (per-unit Studio, engine.instance_info): mask the instance token / port glyphs and
    replace them in the DOM text; what was masked is returned as dom["_instance_masked"]."""
    import time
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if quick:
        await settle(page, timeout_ms=3_000, quiet_ms=1_500, loading_ms=1_500)
    else:
        await settle(page)
    t1 = time.perf_counter()
    await tag_volatile(page)
    inst_masked = await mask_instance(page, instance) if shot else []
    sels = tuple(GLOBAL_MASKS) + tuple(masks)
    rects = await mask_rects(page, sels) if shot else []
    if shot:
        locs = [page.locator(s) for s in sels]
        try:
            await page.mouse.move(VIEWPORT["width"] - 1, VIEWPORT["height"] - 1)
        except Exception:
            pass
        kw = dict(full_page=full_page, animations="disabled", caret="hide", mask=locs,
                  mask_color="#FF00FF", scale="css")
        # Visual stability barrier: JS-driven fades / counters are invisible to the layout
        # barrier, so re-shoot until two consecutive frames are byte-identical (bounded).
        prev = await page.screenshot(**kw)
        for _ in range(0 if quick else 4):
            await page.wait_for_timeout(250)
            cur = await page.screenshot(**kw)
            if cur == prev:
                break
            prev = cur
        (out_dir / f"{step_id}.png").write_bytes(prev)
    if instance:
        await clear_instance_masks(page)
    dom = await dom_snapshot(page, sels, instance=instance)
    (out_dir / f"{step_id}.dom.json").write_text(json.dumps(dom, indent=1))
    if inst_masked:
        dom["_instance_masked"] = inst_masked[:20]
    dom["_settle_s"], dom["_shot_s"] = round(t1 - t0, 2), round(time.perf_counter() - t1, 2)
    area = VIEWPORT["width"] * VIEWPORT["height"]
    masked = sum(r[2] * r[3] for r in rects)
    if masked > 0.05 * area:
        # Recorded, not silently accepted: a mask budget overrun hides too much.
        dom["_mask_overrun"] = round(masked / area, 3)
    return rects, dom


# Hub listings the browser fetches (the picker's Recommended list, the Hub page) come in live
# trendingScore order, which can move between the two sides. The first answer per URL is replayed
# to every later request in this process (both sides of a run), so both list the same repos in
# the same order.
HUB_REPLAY: dict[str, tuple[int, dict, bytes]] = {}


async def replay_hub(route):
    url = route.request.url
    if url not in HUB_REPLAY:
        try:
            r = await route.fetch()
            body = await r.body()
        except Exception:
            await route.continue_()
            return
        if r.status != 200:
            await route.fulfill(response=r)
            return
        # setdefault: parallel units share this process; the first answer wins for all of them
        HUB_REPLAY.setdefault(url, (r.status, dict(r.headers), body))
    status, headers, body = HUB_REPLAY[url]
    # body() is already decoded: drop the transfer headers that described the wire form.
    headers = {k: v for k, v in headers.items()
               if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")}
    # The Hub echoes the requesting origin in its CORS header, and the other side's Studio may
    # sit on another port: answer for this request's origin, or the browser drops the body.
    origin = await route.request.header_value("origin")
    if origin:
        headers = {k: v for k, v in headers.items() if k.lower() != "access-control-allow-origin"}
        headers["access-control-allow-origin"] = origin
    await route.fulfill(status=status, headers=headers, body=body)
