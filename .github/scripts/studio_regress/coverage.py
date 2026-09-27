"""Surface inventory + safe click-all crawl, and the coverage number.

The crawl is a Journey ("crawl") with one frozen step per route. On the BEFORE side each
route step discovers its interactive controls (role + accessible name, via the same
CONTROL_KEY_JS the click recorder uses) and writes them to <root>/<side>/crawl_manifest/<route>.json; the
AFTER side replays exactly that list, so a control gone on AFTER is recorded missing (and
shows as a facts delta), never silently dropped. Each safe control is clicked, the outcome
recorded (url change / dialog / menu / popover opened), then Escape + route restore.
Overlays that opened get an extra screenshot (`_extra_png`), diffed like step PNGs.

Coverage = |exercised ∩ inventory| / |inventory|, overall and per route, where exercised =
controls the crawl clicked OK ∪ controls any journey clicked (facts["_clicked"]).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from studio_regress import fixture
from studio_regress.contract import Journey, Step, StepUnreachable

ROUTES = ("/chat", "/hub", "/projects", "/images", "/video", "/audio", "/studio", "/export",
          "/data-recipes", "/api-monitor", "/settings")
# Never clicked by the crawl (irreversible, long-running, or leaves the app). Exercised only
# inside their own journeys.
DENY = re.compile(
    r"(?i)\b(delete|remove|shutdown|shut down|log ?out|sign ?out|uninstall|reset|clear|revoke|"
    r"stop|cancel|update|install|download|upload|start|train|export|send|dictate|record|"
    r"connect|apply|save|restart|reload models|create token|new chat|new project|import|"
    r"purge|wipe|discard|close sidebar|resize or collapse)\b")
MAX_PER_ROUTE = 80
# Regions fed by live external data (HF trending / search) or wall-clock state: masked in
# pixels, hidden from DOM text, and their controls left out of the inventory.
VOLATILE = {
    # The stat pills' row as a whole: the pills change width with their numbers and shift the
    # transfer-mode toggles beside them, so masking pill by pill still leaves a diff.
    "/hub": (".hub-stat-pill", "*:has(> .hub-stat-pill)", "section[class*='group/carousel']", "[aria-label^='Get ']",
             "[aria-label^='More options for ']", "table tbody"),
    # History, not UI: the stat cards and the request list count and list the requests earlier
    # journeys made, which differ between sides whenever a journey does (a rollback is a load).
    "/api-monitor": ("main > section.grid", "main section > div.grid", "main section > div.flex.flex-wrap"),
    "/data-recipes": (),
}
SHELL_MIN_ROUTES = 3   # a control on >= this many routes is sidebar / shell, counted once


def route_step_id(route):
    return "r_" + (route.strip("/").replace("/", "_").replace("-", "_") or "root")


INVENTORY_JS = "(volatile) => {" + fixture.CONTROL_KEY_JS + r"""
  const seen = new Set(), out = [];
  const skip = el => volatile.some(s => { try { return el.closest(s); } catch (e) { return false; } });
  for (const el of document.querySelectorAll(window.__srCtl.sel)) {
    if (!window.__srCtl.visible(el) || el.disabled || skip(el)) continue;
    const k = window.__srCtl.key(el); if (!k || seen.has(k)) continue;
    seen.add(k); out.push(k);
  }
  return out;
}"""

OVERLAY_JS = "() => document.querySelectorAll('[role=dialog],[role=menu],[role=listbox],[data-radix-popper-content-wrapper]').length"
# Union box of the visible overlays, padded and clipped to the viewport. A crawl extra shot is
# about the overlay a click opened; the page behind it carries state from earlier clicks (which
# hub row is selected, what a list scrolled to) that differs between sides by timing alone, so
# the shot is clipped to the overlay.
OVERLAY_BOX_JS = """() => {
  const els = [...document.querySelectorAll('[role=dialog],[role=menu],[role=listbox],[data-radix-popper-content-wrapper]')];
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const e of els) {
    const r = e.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    x0 = Math.min(x0, r.left); y0 = Math.min(y0, r.top); x1 = Math.max(x1, r.right); y1 = Math.max(y1, r.bottom);
  }
  if (x0 === Infinity) return null;
  const pad = 8, W = window.innerWidth, H = window.innerHeight;
  x0 = Math.max(0, Math.floor(x0 - pad)); y0 = Math.max(0, Math.floor(y0 - pad));
  x1 = Math.min(W, Math.ceil(x1 + pad)); y1 = Math.min(H, Math.ceil(y1 + pad));
  if (x1 - x0 < 4 || y1 - y0 < 4) return null;
  return {x: x0, y: y0, width: x1 - x0, height: y1 - y0};
}"""


_CSS_FOR_ROLE = {"button": "button,[role=button]", "link": "a[href]", "combobox": "[role=combobox],select",
                 "tab": "[role=tab]", "menuitem": "[role=menuitem]", "switch": "[role=switch]",
                 "checkbox": "[role=checkbox],input[type=checkbox]", "radio": "[role=radio],input[type=radio]",
                 "option": "[role=option]"}


async def _locator(page, key):
    """Accessible-name match first; the CONTROL_KEY_JS name is innerText-based, so fall back
    to a role-scoped CSS locator filtered by visible text (comboboxes / aria-labelledby)."""
    role, name = key.split(":", 1)
    rx = re.compile("^" + re.escape(name).replace("\\#", r"\d+"))
    loc = page.get_by_role({"text": "textbox"}.get(role, role), name=rx)
    try:
        if await loc.count():
            return loc
    except Exception:
        pass
    return page.locator(_CSS_FOR_ROLE.get(role, "[role=%s]" % role)).filter(has_text=rx)


async def _click(loc):
    """Visible-on-hover controls (e.g. 'Show workflows') need the row hovered first."""
    try:
        await loc.first.click(timeout=2000)
    except Exception:
        await loc.first.hover(timeout=2000, force=True)
        await loc.first.click(timeout=2000, force=True)


# The AFTER crawl replays the control list the BEFORE crawl saved: the parallel scheduler runs the
# two crawl arms back to back, before first (scheduler.build_plan), never together.
AFTER_REPLAYS_BEFORE = True


def _manifest_dir(root, side="before"):
    """Inside the side dir so staging shards (one side per job) merge without collisions."""
    return Path(root) / side / "crawl_manifest"


def _load_route(ctx, route):
    p = _manifest_dir(Path(ctx.out_dir).parent) / f"{route_step_id(route)}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _save_route(ctx, route, controls):
    d = _manifest_dir(Path(ctx.out_dir).parent, ctx.side)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{route_step_id(route)}.json").write_text(json.dumps({"route": route, "controls": controls}, indent=1))


def load_manifest(root):
    """{route: {"controls": [...]}} from the per-route files (written concurrently)."""
    out = {}
    d = _manifest_dir(root)
    for f in sorted((d if d.is_dir() else _manifest_dir(root, "after")).glob("*.json")):
        d = json.loads(f.read_text())
        out[d["route"]] = {"controls": d["controls"]}
    return out


async def _restore(page, ctx, route):
    await page.goto(ctx.base_url + route, wait_until="domcontentloaded")
    await fixture.settle(page, quiet_ms=1500)


async def _no_hub_datasets(route):
    await route.fulfill(json=[])


CLICK_OUTCOME_JS = """(before) => location.href !== before || document.querySelectorAll(
  '[role=dialog],[role=menu],[role=listbox],[data-radix-popper-content-wrapper]').length > 0"""
CLICK_SETTLE_MS = 250


async def _click_outcome(page, before_url):
    """Was a blind 250 ms after every click. Same ceiling, but a click that opened an overlay or
    navigated returns as soon as that shows; a click with no visible effect still waits it out
    (nothing observable marks "nothing happened")."""
    try:
        await page.wait_for_function(CLICK_OUTCOME_JS, arg=before_url, timeout=CLICK_SETTLE_MS, polling=25)
    except Exception:
        pass


def _route_action(route):
    vol = VOLATILE.get(route, ())

    async def act(ctx):
        page = ctx.page
        # Dataset pickers query the Hub from the browser; live search results reorder between
        # the two sides, so the crawl sees a fixed empty listing instead.
        await page.route("https://huggingface.co/api/datasets*", _no_hub_datasets)
        await page.route("https://huggingface.co/api/models*", fixture.replay_hub)
        await _restore(page, ctx, route)
        if route not in page.url.replace(ctx.base_url, ""):
            raise StepUnreachable(f"{route} redirected to {page.url}")
        found = await page.evaluate(INVENTORY_JS, list(vol))
        prior = _load_route(ctx, route)
        replay = ctx.side != "before" and prior is not None
        planned = prior["controls"] if replay else list(found)[:200]
        if not replay:
            _save_route(ctx, route, planned)
        safe = [k for k in planned if not DENY.search(k.split(":", 1)[1])][:MAX_PER_ROUTE]
        outcomes, extras = {}, []
        jdir = Path(ctx.out_dir) / "crawl" / route_step_id(route)
        dirty = False
        for i, key in enumerate(safe):
            try:
                loc = await _locator(page, key)
                # Lazy restore: only reload the route when the control is not reachable in the
                # current state (an overlay / nav from a previous click is still in the way).
                if dirty or await loc.count() == 0:
                    await _restore(page, ctx, route)
                    dirty = False
                    loc = await _locator(page, key)
                    if await loc.count() == 0:
                        outcomes[key] = "missing"
                        continue
                before_url = page.url
                await _click(loc)
                await _click_outcome(page, before_url)
                opened = await page.evaluate(OVERLAY_JS)
                moved = page.url != before_url
                outcomes[key] = "nav" if moved else ("overlay" if opened else "ok")
                if opened:
                    jdir.mkdir(parents=True, exist_ok=True)
                    name = f"{i:02d}"
                    await fixture.settle(page, quiet_ms=3000)
                    await page.mouse.move(fixture.VIEWPORT["width"] - 1, fixture.VIEWPORT["height"] - 1)
                    await fixture.tag_volatile(page)
                    inst = (ctx.state or {}).get("_instance")
                    await fixture.mask_instance(page, inst)
                    box = await page.evaluate(OVERLAY_BOX_JS)
                    await page.screenshot(path=str(jdir / f"{name}.png"), animations="disabled", caret="hide",
                                          mask=[page.locator(s) for s in fixture.GLOBAL_MASKS + tuple(vol)],
                                          **({"clip": box} if box else {}))
                    if inst:
                        await fixture.clear_instance_masks(page)
                    extras.append(f"crawl/{route_step_id(route)}/{name}.png")
                    await page.keyboard.press("Escape")
                    await page.wait_for_timeout(150)
                    dirty = await page.evaluate(OVERLAY_JS) > 0
                dirty = dirty or moved
            except Exception as e:
                outcomes[key] = "error:" + type(e).__name__
                dirty = True
        await _restore(page, ctx, route)
        # found_now depends on what had rendered when the crawl looked (memory readouts arrive
        # with the hardware probe): advisory, like _outcomes.
        return {"inventory": planned, "_found_now": sorted(set(found) - set(planned))[:50],
                # Click outcomes (ok / overlay / nav / error) depend on popover timing: advisory.
                # A control that vanished is the stable signal, so `missing` is the diffed fact.
                "missing": sorted(k for k, v in outcomes.items() if v == "missing"),
                "_outcomes": outcomes, "_extra_png": extras}
    act.__name__ = f"crawl_{route_step_id(route)}"
    return act


def crawl_journey(routes=ROUTES):
    return Journey(name="crawl", tier="fast", routes=tuple(routes),
                   steps=tuple(Step(route_step_id(r), _route_action(r), timeout_s=420,
                                    masks=VOLATILE.get(r, ()), mask_reason="live HF data" if VOLATILE.get(r) else "")
                               for r in routes))


def compute(side_dir: Path, man: dict | None = None):
    """Coverage for one side from its facts files."""
    side_dir = Path(side_dir)
    man = load_manifest(side_dir.parent) if man is None else man
    inventory = {r: set(v["controls"]) for r, v in man.items()}
    counts = {}
    for inv in inventory.values():
        for k in inv:
            counts[k] = counts.get(k, 0) + 1
    shell = {k for k, n in counts.items() if n >= SHELL_MIN_ROUTES}
    if shell:
        inventory = {r: inv - shell for r, inv in inventory.items()}
        inventory["shell"] = shell
    exercised = {r: set() for r in inventory}
    for f in side_dir.glob("*/*.facts.json"):
        facts = json.loads(f.read_text())
        for k, v in (facts.get("_outcomes") or facts.get("outcomes") or {}).items():
            route = facts.get("_route") or _route_for_step(f.name, man)
            if v in ("ok", "nav", "overlay"):
                if k in shell:
                    exercised["shell"].add(k)
                elif route in exercised:
                    exercised[route].add(k)
        for c in facts.get("_clicked") or []:
            path, _, key = c.partition("|")
            if key in shell:
                exercised["shell"].add(key)
                continue
            for r in inventory:
                if path.startswith(r):
                    exercised[r].add(key)
    per = {}
    tot_inv = tot_ex = 0
    for r, inv in inventory.items():
        if not inv:
            continue
        ex = len(inv & exercised[r])
        per[r] = round(100 * ex / len(inv), 1)
        tot_inv += len(inv)
        tot_ex += ex
    safe_inv = {r: {k for k in inv if not DENY.search(k.split(":", 1)[1])} for r, inv in inventory.items()}
    s_inv = sum(len(v) for v in safe_inv.values())
    s_ex = sum(len(safe_inv[r] & exercised[r]) for r in safe_inv)
    per_safe = {r: round(100 * len(v & exercised[r]) / len(v), 1) for r, v in safe_inv.items() if v}
    return {"overall": round(100 * tot_ex / tot_inv, 1) if tot_inv else 0.0,
            "overall_safe": round(100 * s_ex / s_inv, 1) if s_inv else 0.0, "per_route_safe": per_safe,
            "per_route": per, "inventory": tot_inv, "exercised": tot_ex,
            "denied": sum(1 for inv in inventory.values() for k in inv if DENY.search(k.split(":", 1)[1]))}


def _route_for_step(fname, man):
    step = fname[: -len(".facts.json")]
    for r in man:
        if route_step_id(r) == step:
            return r
    return None
