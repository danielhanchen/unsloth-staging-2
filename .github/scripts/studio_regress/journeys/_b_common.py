"""Shared helpers for the model / GPU journeys (inference_settings, parallel, compaction,
full_access, train_responses_only, export_reload). Async, over ctx.api (httpx.AsyncClient
already carrying the Bearer token) and ctx.page (Playwright).

Everything returns plain JSON-safe values so it can land in facts. Positive assertions raise
StepFailed; a route that 404s where the step needs it raises StepUnreachable (DIVERGED)."""

from __future__ import annotations

import asyncio
import math
import os
import time
from pathlib import Path

from .. import fixture, timeouts
from ..contract import StepFailed, StepUnreachable

GGUF_270M = ("unsloth/gemma-3-270m-it-GGUF", "UD-Q4_K_XL")
GGUF_QWEN35_08B = ("unsloth/Qwen3.5-0.8B-MTP-GGUF", "UD-Q4_K_XL")
GGUF_QWEN35_2B = ("unsloth/Qwen3.5-2B-MTP-GGUF", "UD-Q4_K_XL")
TRAIN_MODEL = "unsloth/gemma-3-270m-it"
FLAPPY = "Create a Flappy Bird game in HTML, then Javascript and fix bugs"
# Volatile regions: throughput / elapsed labels. Never the subject of a step.
VOLATILE_MASKS = ("[data-testid*='tokens-per-second']", "[data-testid*='elapsed']",
                  "[class*='tok-s']", "time")
# Model-written text: differs across kernels / GPUs even at temperature 0. Masked only where the
# step's subject is NOT the text (a notice, a banner, a control state).
GENERATED_TEXT_MASKS = tuple(f"[data-role=assistant] {t}" for t in
                             ("p", "li", "h1", "h2", "h3", "h4", "pre", "table", "blockquote")) + (
    "button[aria-label^='Context usage']",)


def model(ctx, key, default):
    """Fixture override from ctx.models (switchboard.toml [models] by the same key), else the default."""
    v = (ctx.models or {}).get(key)
    if isinstance(v, dict) and v.get("repo"):      # switchboard [models] spec
        return (v["repo"], v.get("variant"))
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return tuple(v)
    return default


async def req(ctx, method, path, body=None, timeout=600.0, ok=None):
    """(status, json|text). `ok` = accepted statuses; others raise StepFailed. 404 on a route
    the step needs is StepUnreachable."""
    # HTTP timeouts stretch with the host like every other wait: a starved Studio answered
    # /api/inference/status in over 60 s, and a ReadTimeout there failed a load step outright.
    # Clamped to the step budget left (callers passing timeouts.inner() are not stretched twice).
    left = timeouts.remaining(getattr(ctx, "state", None), 2.0)
    timeout = timeouts.scaled(timeout) if left is None else min(timeouts.scaled(timeout), left)
    r = await ctx.api.request(method, path, json=body, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = r.text[:2000]
    if ok is not None and r.status_code not in ok:
        if r.status_code == 404 and not isinstance(data, dict) or (
                r.status_code == 404 and "Not Found" == (data or {}).get("detail")):
            raise StepUnreachable(f"{method} {path} -> 404")
        raise StepFailed(f"{method} {path} -> {r.status_code}: {str(data)[:400]}")
    return r.status_code, data


async def get(ctx, path, timeout=60.0):
    return (await req(ctx, "GET", path, timeout=timeout, ok=(200,)))[1]


async def poll(probe, accept, deadline_s, interval_s=1.0, state=None):
    """Probe until accept(result). `deadline_s` is the idle-host deadline: it stretches with the
    host load (timeouts.inner) and, given the step's ctx.state, ends 5 s before the step budget so
    the StepFailed below (with its last= diagnostic) wins over the engine's bare cancel. Status
    probes are cheap, so poll at 1 s (was 2 s): a condition met is seen up to a second sooner."""
    deadline_s = timeouts.inner(deadline_s, state)
    t0 = time.monotonic()
    last = None
    while time.monotonic() - t0 < deadline_s:
        try:
            last = await probe()
            if accept(last):
                return last
        except (StepFailed, StepUnreachable):
            raise
        except Exception as e:  # transient transport errors while the server is busy
            last = f"{type(e).__name__}: {e}"
        await asyncio.sleep(interval_s)
    raise StepFailed(f"timed out after {deadline_s:.0f}s; last={str(last)[:400]}")


async def status(ctx):
    return await get(ctx, "/api/inference/status")


async def load_gguf(ctx, repo, variant, ctx_len, n_parallel=None, deadline_s=600):
    body = {"model_path": repo, "gguf_variant": variant, "max_seq_length": int(ctx_len)}
    if n_parallel:
        body["n_parallel"] = int(n_parallel)
    await req(ctx, "POST", "/api/inference/load", body, timeout=timeouts.inner(deadline_s, ctx.state), ok=(200,))
    st = await poll(lambda: status(ctx),
                    lambda s: isinstance(s, dict) and s.get("loaded") and not s.get("loading"),
                    deadline_s, state=ctx.state)
    return st


def gguf_matches(st, repo, variant, ctx_len, n_parallel=None):
    """The active load is exactly this GGUF (repo, variant, effective AND requested context,
    requested slots when asked) and nothing is loading. Anything unknown -> False (reload)."""
    if not isinstance(st, dict) or not st.get("loaded") or st.get("loading") or not st.get("is_gguf"):
        return False
    if len(st.get("loaded") or []) != 1:
        return False
    ident = st.get("model_identifier") or st.get("active_model")
    if ident != repo or (variant and (st.get("gguf_variant") or "").lower() != str(variant).lower()):
        return False
    if effective_ctx(st) != int(ctx_len) or st.get("requested_context_length") not in (int(ctx_len), None):
        return False
    if n_parallel and st.get("requested_parallel_slots") != int(n_parallel):
        return False
    return True


async def ensure_gguf(ctx, repo, variant, ctx_len, n_parallel=None, deadline_s=600):
    """(status, reused). A GGUF spawn is minutes on a loaded host (llama-server start); when an
    earlier journey on this Studio left exactly this load in place, keep it. For steps whose
    subject is NOT the load itself (the load is setup). `reused` goes into facts so both sides
    record which path they took."""
    st = await status(ctx)
    if gguf_matches(st, repo, variant, ctx_len, n_parallel):
        return st, True
    await unload_all(ctx)
    return await load_gguf(ctx, repo, variant, ctx_len, n_parallel=n_parallel, deadline_s=deadline_s), False


def effective_ctx(st):
    """Resolved context of the active load, from /api/inference/status."""
    for k in ("context_length", "n_ctx"):
        v = st.get(k)
        if isinstance(v, int) and v > 0:
            return v
    inf = st.get("inference") or {}
    for k in ("context_length", "n_ctx", "max_seq_length"):
        v = inf.get(k)
        if isinstance(v, int) and v > 0:
            return v
    return None


async def unload_all(ctx):
    st = await status(ctx)
    for m in st.get("loaded") or []:
        await req(ctx, "POST", "/api/inference/unload", {"model_path": m}, ok=(200, 404, 409))


async def chat(ctx, messages, max_tokens=32, stream=False, timeout=600.0, **extra):
    """Non-stream OpenAI-form completion. Deterministic: temperature 0, seed 3407."""
    body = {"model": "default", "messages": messages, "stream": stream, "max_tokens": max_tokens,
            "temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 3407, **extra}
    return await req(ctx, "POST", "/api/inference/chat/completions", body, timeout=timeout)


def reply_text(body):
    if not isinstance(body, dict):
        return ""
    ch = (body.get("choices") or [{}])[0]
    return ((ch.get("message") or {}).get("content") or "")


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


ROOT_RENDERED_JS = "() => { const r = document.getElementById('root'); return !!(r && r.children.length); }"


async def goto(ctx, route, wait_text=None, timeout_ms=60_000):
    timeout_ms = timeouts.scaled_ms(timeout_ms)
    await ctx.page.goto(ctx.base_url + route, wait_until="domcontentloaded", timeout=timeout_ms)
    if wait_text:
        await ctx.page.get_by_text(wait_text, exact=False).first.wait_for(timeout=timeout_ms)
    else:
        # Not "networkidle": the chat page keeps polling / streaming, so it may never settle.
        await ctx.page.wait_for_load_state("load", timeout=timeout_ms)
        # Was a blind 1.2 s: wait for the SPA to have mounted instead, then for the page's own
        # requests to go quiet (bounded; capture() runs the full settle before any shot).
        try:
            await ctx.page.wait_for_function(ROOT_RENDERED_JS, timeout=timeouts.scaled_ms(10_000), polling=100)
        except Exception:
            pass
        await fixture.settle(ctx.page, timeout_ms=5_000, quiet_ms=1_500, loading_ms=5_000)


async def open_train_page(ctx):
    """/studio is reached from the sidebar; a cold deep link lands on /chat first."""
    await goto(ctx, "/chat")
    await ctx.page.get_by_test_id("nav-row-train").first.click()
    await ctx.page.get_by_role("heading", name="Train", exact=True).first.wait_for(
        timeout=timeouts.scaled_ms(60_000))
    # Was a blind 800 ms after the heading: wait for the form's requests to go quiet instead.
    await fixture.settle(ctx.page, timeout_ms=5_000, quiet_ms=1_500, loading_ms=5_000)


# Absolute paths differ between the two sides (each side has its own Studio home), so any text
# showing one is volatile by construction. Playwright selector syntax (text=/re/) is allowed.
PATH_TEXT_MASKS = ("text=/\\/(mnt|home|tmp|Users)\\//",)


def studio_home(ctx):
    h = (ctx.models or {}).get("_studio_home") or os.environ.get("UNSLOTH_STUDIO_HOME")
    return Path(h) if h else None
