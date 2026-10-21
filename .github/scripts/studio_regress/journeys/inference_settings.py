"""Journey 2: GGUF inference settings. Load at ctx 2048, change the context in the run-settings
panel and re-apply (effective ctx read back from /api/inference/status), custom sampling +
system prompt, save a preset, reload the page and prove it persisted, one short reply, Stop, then
re-apply a context while the load reply is lost in transit (no rollback of a load that landed).

Model: unsloth/gemma-3-270m-it-GGUF UD-Q4_K_XL (CPU-capable, 254 MiB)."""

from __future__ import annotations

from .. import timeouts
from ..contract import Journey, Step, StepFailed
from . import _b_common as b
from . import _b_ui as ui

PRESET = "regress-preset"
SYSTEM = "You are a terse assistant. Answer in one short sentence."


async def load_2048(ctx):
    repo, variant = b.model(ctx, "gguf_small", b.GGUF_270M)
    await b.unload_all(ctx)
    st = await b.load_gguf(ctx, repo, variant, 2048)
    eff = b.effective_ctx(st)
    if eff != 2048:
        raise StepFailed(f"loaded with max_seq_length 2048, status reports context {eff}")
    await b.goto(ctx, "/chat")
    await ui.need(ctx.page.get_by_role("textbox", name = "Message input"), "chat composer", 60_000)
    return {
        "model": repo,
        "variant": variant,
        "context_length": eff,
        "requested_context_length": st.get("requested_context_length"),
    }


async def open_settings(ctx):
    await ui.open_run_settings(ctx.page)
    return {
        "context_field": await ui.field(ctx.page, "Context Length"),
        "temperature_field": await ui.field(ctx.page, "Temperature"),
    }


async def reapply_4096(ctx):
    import time

    typed = await ui.set_field(ctx.page, "Context Length", 4096)
    btn = await ui.need(ctx.page.get_by_role("button", name = "Reload model"), "Reload model")
    answered = {}

    def on_response(r):
        if "/api/inference/load" in r.url and r.request.method == "POST" and "at" not in answered:
            answered.update(at = time.monotonic(), status = r.status)

    ctx.page.on("response", on_response)
    try:
        await btn.click()
        # Done when the server runs the requested 4096; or, fail fast, once the load request has
        # been answered and the server has sat idle on some other context for a few seconds (a
        # reload that kept or reverted the context used to wait out the whole 300 s poll).
        grace = timeouts.scaled(5)
        st = await b.poll(
            lambda: b.status(ctx),
            lambda s: not s.get("loading")
            and (
                (s.get("loaded") and s.get("requested_context_length") == 4096)
                or ("at" in answered and time.monotonic() - answered["at"] > grace)
            ),
            300,
            state = ctx.state,
        )
    finally:
        ctx.page.remove_listener("response", on_response)
    if not st.get("loaded"):
        raise StepFailed(
            f"re-applied 4096, no model loaded afterwards (load answered HTTP {answered.get('status')})"
        )
    if st.get("requested_context_length") != 4096:
        raise StepFailed(
            f"re-applied 4096, server runs requested context {st.get('requested_context_length')} "
            f"(load answered HTTP {answered.get('status')})"
        )
    eff = b.effective_ctx(st)
    if eff != 4096:
        raise StepFailed(f"re-applied 4096, effective context is {eff}")
    # Was a blind 800 ms: wait (bounded) for the panel to show the context it now runs with.
    await _field_settles(ctx.page, "Context Length", 4096)
    return {
        "typed": typed,
        "context_length": eff,
        "field_after": await ui.field(ctx.page, "Context Length"),
    }


async def custom_sampling(ctx):
    t = await ui.set_field(ctx.page, "Temperature", "0.3")
    p = await ui.set_field(ctx.page, "Top P", "0.9")
    seed = await ui.set_field(ctx.page, "Seed", "3407")  # fixed seed: the reply is comparable
    sp = await ui.need(ctx.page.get_by_role("textbox", name = "System prompt"), "System prompt")
    await sp.fill(SYSTEM)
    if t not in ("0.3", "0.30") or p not in ("0.9", "0.90"):
        raise StepFailed(f"fields did not keep the values: temperature={t} top_p={p}")
    return {"temperature": t, "top_p": p, "seed": seed}


async def save_preset(ctx):
    name = await ui.need(
        ctx.page.get_by_role("textbox", name = "Inference preset name"), "preset name"
    )
    await name.fill(PRESET)
    await (
        await ui.need(
            ctx.page.get_by_role("button", name = "Save current settings as"), "Save preset"
        )
    ).click()
    s = (
        await b.poll(
            lambda: b.get(ctx, "/api/chat/settings"),
            lambda d: any(
                p.get("name") == PRESET
                for p in (d.get("settings") or {}).get("customPresets") or []
            ),
            30,
            1.0,
            state = ctx.state,
        )
    )["settings"]
    pre = next(p for p in s["customPresets"] if p["name"] == PRESET)["params"]
    if (pre.get("temperature"), pre.get("topP"), pre.get("systemPrompt")) != (0.3, 0.9, SYSTEM):
        raise StepFailed(f"saved preset params differ: {pre}")
    # The panel's scroll offset after the save depends on when the list re-rendered; pin it.
    await ctx.page.get_by_role("textbox", name = "Inference preset name").evaluate(
        "e => e.scrollIntoView({block: 'center'})"
    )
    return {
        "active_preset": s.get("activePreset"),
        "preset_params": {
            k: pre.get(k) for k in ("temperature", "topP", "systemPrompt", "maxSeqLength")
        },
    }


async def reload_persisted(ctx):
    await ctx.page.reload(wait_until = "domcontentloaded")
    await ui.need(ctx.page.get_by_role("textbox", name = "Message input"), "chat composer", 60_000)
    await ui.open_run_settings(ctx.page)
    t, p = await ui.field(ctx.page, "Temperature"), await ui.field(ctx.page, "Top P")
    sp = await ctx.page.get_by_role("textbox", name = "System prompt").first.input_value()
    if (t, p, sp) != ("0.3", "0.9", SYSTEM):
        raise StepFailed(f"after reload: temperature={t} top_p={p} system={sp!r}")
    return {"temperature": t, "top_p": p, "system_prompt_kept": True}


async def short_reply(ctx):
    close = ctx.page.get_by_role("button", name = "Close run settings")
    if await close.count():
        await close.first.click()
    n = await ctx.page.locator(ui.ASSISTANT).count()
    await ui.send(ctx.page, "Say hello in three words.")
    txt = await ui.wait_reply(ctx.page, n, state = ctx.state)
    return {"reply_nonempty": bool(txt), "assistant_rows": n + 1}


async def stop_generation(ctx):
    n = await ctx.page.locator(ui.ASSISTANT).count()
    await ui.send(ctx.page, "Count from 1 to 500, one number per line.")
    stop = ctx.page.locator(ui.STOP).first
    await ui.need(ctx.page.locator(ui.STOP), "Stop button", 60_000)
    await stop.click()
    await stop.wait_for(state = "hidden", timeout = timeouts.scaled_ms(30_000))
    st = await b.status(ctx)
    if not st.get("loaded"):
        raise StepFailed("model unloaded after Stop")
    await ctx.page.wait_for_timeout(500)
    return {
        "stopped": True,
        "assistant_rows": await ctx.page.locator(ui.ASSISTANT).count(),
        "rows_before": n,
    }


async def lost_load_reply(ctx):
    """Re-apply a new context while the load's HTTP reply is lost in transit. The request reaches
    the server and the load completes there; only the answer is cut (connection reset), which is
    what a page reload mid-load or a dropped keepalive body looks like to the client. A client
    that treats that unknown outcome as a failure rolls the server back to the previous context
    (unsloth#11729); one that does not keeps the context it asked for."""
    page = ctx.page
    before = b.effective_ctx(await b.status(ctx))
    target = 3072 if before != 3072 else 2560
    seen = {"n": 0, "cut": False}

    async def cut_first_reply(route):
        if seen["cut"] and route.request.method == "POST":
            # A load after the cut is the client's rollback: pass it through, and note when the
            # server has answered it so the outcome is read after the rollback, not before.
            seen["after_cut"] = seen.get("after_cut", 0) + 1
            try:
                await route.fulfill(response = await route.fetch(timeout = 300_000))
            except Exception:
                try:
                    await route.continue_()
                except Exception:
                    pass
            seen["rollback_done"] = seen.get("rollback_done", 0) + 1
            return
        if seen["n"] == 0 and route.request.method == "POST":
            seen["n"] += 1
            try:
                await route.fetch(timeout = 300_000)  # the server performs the load
            except Exception:
                pass
            await route.abort("connectionreset")
            seen["cut"] = True
        else:
            await route.continue_()

    await page.route("**/api/inference/load", cut_first_reply)
    try:
        await ui.open_run_settings(page)
        await ui.set_field(page, "Context Length", target)
        await (
            await ui.need(page.get_by_role("button", name = "Reload model"), "Reload model")
        ).click()
        await b.poll(lambda: b.status(ctx), lambda s: seen["cut"], 330, state = ctx.state)
        if not seen["cut"]:
            raise StepFailed("Reload model sent no /api/inference/load within 330s")
        # A rollback, if the client sends one, starts right after the failed reply. Was a blind
        # 8 s; now: up to the same 8 s (load-scaled) for the client to show it handled the failure
        # (its error toast) or to send the rollback load itself, then a short grace for a rollback
        # issued by that same handler. Without either signal the full window is still waited.
        seen["signal"] = await _await_failure_handled(page, seen)
        if seen.get("after_cut"):
            await b.poll(
                lambda: b.status(ctx),
                lambda s: seen.get("rollback_done", 0) >= seen["after_cut"],
                300,
                0.5,
                state = ctx.state,
            )
        st = await b.poll(
            lambda: b.status(ctx),
            lambda s: s.get("loaded") and not s.get("loading"),
            300,
            state = ctx.state,
        )
    finally:
        await page.unroute("**/api/inference/load")
    await page.wait_for_timeout(800)
    # The cut load surfaces a "not running" error toast on its own timer; whether it is still up
    # at capture is timing. The facts below carry the outcome, so dismiss it.
    for btn in await page.get_by_role("button", name = "Close toast").all():
        try:
            await btn.click(timeout = 2_000)
        except Exception:
            pass
    final = b.effective_ctx(st)
    return {
        "previous_context": before,
        "requested_context": target,
        "final_context": final,
        "rolled_back": final != target,
        "load_requests_cut": seen["n"],
        "_failure_signal": seen.get("signal"),
        "_rollback_requests": seen.get("after_cut", 0),
    }


async def _field_settles(
    page,
    name,
    want,
    timeout_ms = 5_000,
):
    """Bounded wait for a labelled field to read `want` (the panel re-renders after a reload)."""
    try:
        box = page.get_by_role("textbox", name = name).first
        await page.wait_for_function(
            "([el, want]) => el && (el.value || '').replace(/,/g, '') === String(want)",
            arg = [await box.element_handle(), want],
            timeout = timeouts.scaled_ms(timeout_ms),
            polling = 100,
        )
    except Exception:
        pass  # the facts record what the field shows; the assertion is on /api/inference/status


ERROR_TOAST = "[data-sonner-toast][data-type='error']"


async def _await_failure_handled(
    page,
    seen,
    window_ms = 8_000,
    grace_ms = 3_000,
):
    """Return once the client has visibly processed the cut reply (error toast, or a rollback
    POST already sent) plus `grace_ms`, else after `window_ms` (both load-scaled)."""
    import time

    t0 = time.monotonic()
    window_s = timeouts.scaled(window_ms / 1000)
    while time.monotonic() - t0 < window_s:
        if seen.get("after_cut"):
            return "rollback_sent"
        try:
            if await page.locator(ERROR_TOAST).count():
                await page.wait_for_timeout(timeouts.scaled_ms(grace_ms))
                return "toast"
        except Exception:
            pass
        await page.wait_for_timeout(100)
    return "window"


JOURNEY = Journey(
    name = "inference_settings",
    tier = "model",
    needs = ("gguf_small",),
    routes = ("/chat",),
    steps = (
        Step("load_2048", load_2048, timeout_s = 400),
        Step("open_settings", open_settings),
        Step("reapply_4096", reapply_4096, timeout_s = 400),
        Step("custom_sampling", custom_sampling),
        Step("save_preset", save_preset),
        Step("reload_persisted", reload_persisted),
        Step(
            "short_reply",
            short_reply,
            masks = b.VOLATILE_MASKS,
            mask_reason = "tok/s, elapsed",
            timeout_s = 240,
        ),
        Step(
            "stop_generation",
            stop_generation,
            masks = b.VOLATILE_MASKS + b.GENERATED_TEXT_MASKS,
            mask_reason = "partial stream length varies",
            timeout_s = 240,
        ),
        # Last: it may leave the two sides on different contexts on purpose.
        Step(
            "lost_load_reply",
            lost_load_reply,
            masks = b.VOLATILE_MASKS + b.GENERATED_TEXT_MASKS,
            mask_reason = "earlier replies",
            timeout_s = 700,
        ),
    ),
)
