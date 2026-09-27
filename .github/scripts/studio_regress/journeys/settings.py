"""Journey 8a: every Settings dialog tab, plus one change -> reload -> persisted -> reset.

Tab list is frozen (names as of unsloth main 2026-09); a tab missing on one side is
StepUnreachable (DIVERGED), a new tab on one side shows up in the dialog DOM diff of s01.
"""

from __future__ import annotations

import re

from studio_regress import timeouts
from studio_regress.contract import Journey, Step, StepFailed, StepUnreachable

TABS = ("General", "Profile", "Appearance", "System", "Chat", "API", "Remote & LAN",
        "Connections", "Accounts", "Agents", "Voice", "Data", "Shortcuts", "Logs", "About")
# Live readouts that change between identical runs (never the tab's layout / controls).
TAB_MASKS = {
    "System": ("[role=dialog] div.grid.gap-2.py-3",             # CPU / RAM / disk / VRAM live cards
               "[role=dialog] div[class*='w-[min(calc(392px']",   # per-GPU used / free bars
               "[role=dialog] span.tabular-nums.rounded-full"),   # per-GPU "N% VRAM" badge
    "Logs": ("[role=dialog] pre", "[role=dialog] code", "[role=dialog] button[role=combobox]",
             "[role=dialog] div.ml-auto.flex.items-center.gap-2"),  # file name, line count
}
PERSIST_SWITCH = "Show context window usage"   # Chat tab, harmless, persisted server-side


def _dialog(ctx):
    return ctx.page.locator("[role=dialog]")


async def open_settings(ctx):
    p = ctx.page
    if "/chat" not in p.url:
        await p.goto(ctx.base_url + "/chat", wait_until="domcontentloaded")
    if await _dialog(ctx).count() == 0:
        try:
            await p.get_by_role("button", name="Settings", exact=True).click(timeout=timeouts.scaled_ms(15_000))
            await _dialog(ctx).wait_for(state="visible", timeout=timeouts.scaled_ms(10_000))
        except Exception as e:
            raise StepUnreachable(f"Settings dialog: {e}") from None
    return {}


async def _tab(ctx, name):
    loc = _dialog(ctx).get_by_role("button", name=re.compile(r"^" + re.escape(name) + r"\b"))
    if await loc.count() == 0:
        raise StepUnreachable(f"settings tab {name!r} not present")
    await loc.first.click()
    # No blind 300 ms here: tab steps wait for the dialog to settle (_settled_dialog) and the
    # persist steps wait for their switch to be visible (_switch).


# Live readouts on the System tab (per-GPU "N GiB used / N GiB free", usage bars, "N% VRAM",
# CPU / RAM / disk) move with whatever else runs on the box. TAB_MASKS' class selectors miss
# some of them, so the step also tags them by content: any element whose OWN text carries a
# memory / percent figure, and every width-styled bar fill, gets data-volatile (a GLOBAL_MASK).
TAG_LIVE_READOUTS_JS = r"""() => {
  // Re-tags on every DOM change: the rows re-render on each poll, and React's fresh nodes do
  // not carry the attribute, so a one-shot tag is gone by the time the shot is taken.
  const tag = () => {
    const d = document.querySelector('[role=dialog]');
    if (!d) return 0;
    const rx = /\d[\d.,]*\s*(GiB|GB|MiB|MB|TiB|TB|%)/;
    let n = 0;
    for (const el of d.querySelectorAll('*')) {
      const own = [...el.childNodes].filter(c => c.nodeType === 3).map(c => c.textContent).join('');
      const bar = el.style && (/%/.test(el.style.width || '') || /scaleX|translateX/.test(el.style.transform || ''));
      const target = bar ? (el.parentElement || el) : el;
      if ((rx.test(own) || bar || el.getAttribute('role') === 'progressbar') && !target.hasAttribute('data-volatile')) {
        target.setAttribute('data-volatile', '');
        n++;
      }
    }
    return n;
  };
  if (!window.__srLiveTag) {
    window.__srLiveTag = new MutationObserver(() => tag());
    window.__srLiveTag.observe(document.body, {childList: true, subtree: true, characterData: true,
                                               attributes: true, attributeFilter: ['style']});
  }
  return tag();
}"""


DIALOG_SIGNATURE_JS = r"""() => {
  const d = document.querySelector('[role=dialog]');
  if (!d) return '';
  const els = [...d.querySelectorAll('button,[role=switch],[role=combobox],input,textarea,select')];
  return els.length + '|' + els.map(e => (e.disabled ? 'd' : '') + (e.getAttribute('aria-checked') || '')
    + (e.getAttribute('data-state') || '')).join(',') + '|' + (d.innerText || '').length;
}"""


async def _settled_dialog(page, quiet_ms=1200, max_ms=10_000):
    """A tab renders in stages (switches arrive disabled until their setting loads, sections
    below the fold mount late), so a shot taken on the first frame catches a different stage
    on each side. Wait until the dialog's controls, their states and its text length hold
    still for `quiet_ms`."""
    last, stable, waited = None, 0, 0
    while waited < max_ms:
        sig = await page.evaluate(DIALOG_SIGNATURE_JS)
        stable = stable + 200 if sig == last else 0
        if stable >= quiet_ms:
            return
        last = sig
        await page.wait_for_timeout(200)
        waited += 200


def _tab_step(name):
    async def act(ctx):
        await open_settings(ctx)
        await _tab(ctx, name)
        # A starved Studio answers a tab's own requests late (seen: /api/settings/caches 451 s,
        # /api/inference/status 355 s, the System tab's GPU probe past 2 minutes), and a shot taken
        # before they land catches "Loading...", "Checking for GPUs...", "Measuring cache sizes...",
        # "Unavailable" or disabled controls on one side only. Waiting for Studio is a shared
        # prerequisite, not the tab's subject: outside the step budget (timeouts.unbudgeted), capped.
        async with timeouts.unbudgeted(ctx.state):
            try:
                await ctx.page.wait_for_function("() => (window.__srOpenAll || 0) === 0",
                                                 timeout=timeouts.scaled_ms(120_000), polling=250)
            except Exception:
                pass
            try:
                await ctx.page.wait_for_function(
                    "() => !/(Checking for GPUs|Measuring cache sizes)/.test(document.body.innerText)",
                    timeout=timeouts.scaled_ms(300_000), polling=250)
            except Exception:
                pass
        await _settled_dialog(ctx.page)
        if name in TAB_MASKS:
            await ctx.page.evaluate(TAG_LIVE_READOUTS_JS)
        n = await _dialog(ctx).locator(
            "button,[role=switch],[role=combobox],input,textarea,select").count()
        # Counted before capture's settle, so late-rendering rows move it; the DOM diff is the signal.
        return {"_controls": n}
    act.__name__ = f"tab_{name}"
    return act


async def _switch(ctx):
    sw = _dialog(ctx).get_by_role("switch", name=PERSIST_SWITCH)
    try:   # the tab's content mounts after the click
        await sw.first.wait_for(state="visible", timeout=timeouts.scaled_ms(10_000))
    except Exception:
        pass
    if await sw.count() == 0:
        raise StepUnreachable(f"switch {PERSIST_SWITCH!r} not present")
    return sw.first


async def _checked_becomes(ctx, sw, value, changed=True, timeout_ms=5_000):
    """Bounded wait for the switch's aria-checked to differ from (changed) / equal `value`.
    On timeout it returns; the caller's assertion reports what the switch shows."""
    js = ("([el, v]) => el.getAttribute('aria-checked') !== v" if changed
          else "([el, v]) => el.getAttribute('aria-checked') === v")
    try:
        await ctx.page.wait_for_function(js, arg=[await sw.element_handle(), value],
                                         timeout=timeouts.scaled_ms(timeout_ms), polling=50)
    except Exception:
        pass


async def toggle_persist(ctx):
    await open_settings(ctx)
    await _tab(ctx, "Chat")
    sw = await _switch(ctx)
    before = await sw.get_attribute("aria-checked")
    await sw.click()
    # Was a blind 800 ms. The preference is a zustand store persisted to localStorage in the
    # same update that flips aria-checked, so the flip is the whole condition.
    await _checked_becomes(ctx, sw, before, changed=True)
    after = await sw.get_attribute("aria-checked")
    if before == after:
        raise StepFailed("switch did not change")
    ctx.state["persist_from"], ctx.state["persist_to"] = before, after
    return {"from": before, "to": after}


async def reload_persisted(ctx):
    await ctx.page.reload(wait_until="domcontentloaded")
    # Was a blind 800 ms: open_settings waits for the Settings button and the dialog itself.
    await open_settings(ctx)
    await _tab(ctx, "Chat")
    now = await (await _switch(ctx)).get_attribute("aria-checked")
    if now != ctx.state.get("persist_to"):
        raise StepFailed(f"setting not persisted: {now} != {ctx.state.get('persist_to')}")
    return {"state": now}


async def reset_setting(ctx):
    sw = await _switch(ctx)
    before = await sw.get_attribute("aria-checked")
    await sw.click()
    await _checked_becomes(ctx, sw, before, changed=True)   # was a blind 600 ms
    now = await sw.get_attribute("aria-checked")
    if now != ctx.state.get("persist_from"):
        raise StepFailed("reset did not restore the original value")
    await ctx.page.keyboard.press("Escape")
    return {"state": now}


JOURNEY = Journey(
    name="settings", tier="fast", routes=("settings",),
    steps=(Step("s00_open", open_settings),)
    + tuple(Step(f"s{i + 1:02d}_tab_{t.lower().replace(' & ', '_').replace(' ', '_')}", _tab_step(t),
                 masks=TAB_MASKS.get(t, ()), mask_reason="live hardware / log readouts" if t in TAB_MASKS else "")
            for i, t in enumerate(TABS))
    + (Step("s20_toggle_persist", toggle_persist),
       Step("s21_reload_persisted", reload_persisted),
       Step("s22_reset", reset_setting)),
)
