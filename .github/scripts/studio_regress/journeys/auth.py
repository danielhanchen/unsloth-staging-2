"""Journey 1: first-run auth through the UI.

Runs on a FRESH state home (bootstrap password present). Leaves the side's password at
ctx.state["password"], so the runner's later API logins use it. Must run before any other
journey on that Studio (runner orders it first; Journey.serial).
"""

from __future__ import annotations

import re
from pathlib import Path

from studio_regress import timeouts
from studio_regress.contract import Journey, Step, StepFailed, StepUnreachable

_WRONG = "Definitely-Wrong-Pass-1"


async def _goto(ctx, path):
    await ctx.page.goto(ctx.base_url + path, wait_until="domcontentloaded")


async def _wait_route(ctx, rx, timeout=20_000):
    try:
        await ctx.page.wait_for_url(re.compile(rx), timeout=timeouts.scaled_ms(timeout))
    except Exception:
        raise StepFailed(f"expected route {rx}, at {ctx.page.url}") from None
    return ctx.page.url.replace(ctx.base_url, "")


async def _field(ctx, sel, timeout=20_000):
    loc = ctx.page.locator(sel)
    try:
        await loc.wait_for(state="visible", timeout=timeouts.scaled_ms(timeout))
    except Exception:
        raise StepUnreachable(f"{sel} not shown at {ctx.page.url}") from None
    return loc


async def root_redirect(ctx):
    await _goto(ctx, "/")
    route = await _wait_route(ctx, r"/(change-password|login)")
    await _field(ctx, "#new-password, #password")
    return {"route": route.split("?")[0]}


async def change_password_form(ctx):
    await _goto(ctx, "/change-password")
    await _field(ctx, "#new-password")
    has_current = await ctx.page.locator("#current-password").count() > 0
    return {"has_current_field": has_current}


async def _fill_change(ctx, new, confirm):
    if await ctx.page.locator("#current-password").count():
        boot = (Path(ctx.state["home"]) / "auth" / ".bootstrap_password").read_text().strip()
        await ctx.page.fill("#current-password", boot)
    await ctx.page.fill("#new-password", new)
    await ctx.page.fill("#confirm-password", confirm)


async def mismatch_rejected(ctx):
    """A mismatched confirmation must not be submittable (disabled submit) or, if the form
    allows submit, must keep the user on the page."""
    pw = ctx.state["password"]
    await _fill_change(ctx, pw, pw + "x")
    btn = ctx.page.locator('button[type="submit"]')
    disabled = await btn.is_disabled()
    if not disabled:
        await btn.click()
        # The good outcome (stay on the page) has no event to wait for: a navigation away is
        # the failure, so give it this long to happen.
        await ctx.page.wait_for_timeout(600)
    if "/change-password" not in ctx.page.url:
        raise StepFailed("mismatched confirmation was accepted")
    return {"submit_disabled": disabled}


async def set_new_password(ctx):
    pw = ctx.state["password"]
    await _fill_change(ctx, pw, pw)
    async with ctx.page.expect_response(lambda r: "/api/auth/change-password" in r.url,
                                        timeout=timeouts.scaled_ms(30_000)) as ri:
        await ctx.page.locator('button[type="submit"]').click()
    status = (await ri.value).status
    if status >= 400:
        raise StepFailed(f"change-password POST {status}")
    route = await _wait_route(ctx, r"^(?!.*change-password).*$")
    # The pages after this (chat shell: Train, GPU labels) render off the cold GPU probe; pay it
    # now, like authed_context does for every other journey, so no shot catches it half done.
    import asyncio
    from studio_regress import engine
    # Outside the step's budget (timeouts.unbudgeted): on a busy GPU host the probe alone ran
    # 60-500 s, which is Studio's hardware detection, not this step's subject.
    async with timeouts.unbudgeted(ctx.state):
        try:
            tok = await asyncio.to_thread(engine.api_login, ctx.base_url, pw)
            await asyncio.to_thread(engine.warm, ctx.base_url, tok["access_token"])
        except Exception:
            pass
    return {"status": status, "route": route.split("?")[0]}


async def logout(ctx):
    p = ctx.page
    # Two attempts: on a loaded host the menu item's click can land while the menu is still
    # animating in and do nothing, which is not what this step is about.
    for attempt in range(2):
        try:
            await p.get_by_role("button", name="Unsloth account menu").click(timeout=timeouts.scaled_ms(15_000))
            await p.get_by_role("menuitem", name=re.compile(r"^Log out")).click(timeout=timeouts.scaled_ms(5_000))
        except Exception as e:
            if attempt:
                raise StepUnreachable(f"account menu / Log out: {e}") from None
            await p.keyboard.press("Escape")
            continue
        try:
            await p.wait_for_url(re.compile(r"/login"), timeout=timeouts.scaled_ms(10_000))
            break
        except Exception:
            if attempt:
                break      # _wait_route below reports the failure
            await p.keyboard.press("Escape")
    route = await _wait_route(ctx, r"/login")
    await _field(ctx, "#password")   # form rendered, not the blank route shell
    return {"route": route.split("?")[0]}


async def old_password_rejected(ctx):
    boot = ctx.state["bootstrap_password"]
    await (await _field(ctx, "#password")).fill(boot)
    async with ctx.page.expect_response(lambda r: "/api/auth/login" in r.url,
                                        timeout=timeouts.scaled_ms(20_000)) as ri:
        await ctx.page.locator('button[type="submit"]').click()
    status = (await ri.value).status
    # Staying on /login is the good outcome and has no event; a navigation away is the failure.
    await ctx.page.wait_for_timeout(400)
    if status < 400 or "/login" not in ctx.page.url:
        raise StepFailed(f"old bootstrap password accepted ({status})")
    await ctx.page.fill("#password", "")   # per-home random length: not the subject
    return {"status": status}


async def wrong_password_rejected(ctx):
    await ctx.page.fill("#password", _WRONG)
    async with ctx.page.expect_response(lambda r: "/api/auth/login" in r.url,
                                        timeout=timeouts.scaled_ms(20_000)) as ri:
        await ctx.page.locator('button[type="submit"]').click()
    status = (await ri.value).status
    if status < 400:
        raise StepFailed("wrong password accepted")
    await ctx.page.fill("#password", "")
    return {"status": status}


async def new_password_login(ctx):
    await ctx.page.fill("#password", ctx.state["password"])
    await ctx.page.locator('button[type="submit"]').click()
    route = await _wait_route(ctx, r"^(?!.*/login).*$")
    return {"route": route.split("?")[0]}


async def reload_keeps_session(ctx):
    await ctx.page.reload(wait_until="domcontentloaded")
    # Was a blind 1 s: wait until the router decided (the auth guard redirects to /login or
    # /change-password, or the signed-in shell's account menu renders), bounded.
    try:
        await ctx.page.wait_for_function(
            "() => /\\/(login|change-password)/.test(location.pathname)"
            " || !!document.querySelector('button[aria-label$=\"account menu\"], [data-testid^=\"nav-row-\"]')",
            timeout=timeouts.scaled_ms(15_000), polling=100)
    except Exception:
        pass
    # A token the server no longer honours is only found out by the shell's first API calls
    # (401 -> refresh -> /login): let those finish before judging the route.
    from studio_regress import fixture
    await fixture.settle(ctx.page, timeout_ms=5_000, quiet_ms=2_000, loading_ms=3_000)
    if "/login" in ctx.page.url or "/change-password" in ctx.page.url:
        raise StepFailed(f"session lost on reload: {ctx.page.url}")
    return {"route": ctx.page.url.replace(ctx.base_url, "").split("?")[0]}


JOURNEY = Journey(
    name="auth", tier="fast", serial=True, routes=("/login", "/change-password"),
    steps=(
        Step("s01_root_redirect", root_redirect),
        Step("s02_change_password_form", change_password_form),
        Step("s03_mismatch_rejected", mismatch_rejected),
        Step("s04_set_new_password", set_new_password, timeout_s=300),  # pays the cold GPU probe
        Step("s05_logout", logout),
        Step("s06_old_password_rejected", old_password_rejected),
        Step("s07_wrong_password_rejected", wrong_password_rejected),
        Step("s08_new_password_login", new_password_login),
        Step("s09_reload_keeps_session", reload_keeps_session),
    ),
)
