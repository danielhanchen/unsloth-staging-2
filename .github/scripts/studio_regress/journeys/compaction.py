"""Journey 4: context compaction at ctx 2048 on unsloth/Qwen3.5-0.8B-MTP-GGUF (UD-Q4_K_XL).

API half (deterministic, tokenizer-measured): a seeded history sized with the loaded model's own
tokenizer (/api/inference/chat/count_tokens) so that history alone fits the prompt budget and
history + the Flappy Bird prompt does not; budget = ctx - min(max_tokens, ctx // 4), the
server's own formula (core/inference/context_window.py). Then the pair upstream's GPU harness
insists on (tests/kaggle/studio_gpu/run_studio_gpu.py:assert_compaction):
  * context_overflow=truncate_oldest -> 200 with context_truncated.dropped_messages > 0,
  * default / "error" policy -> 400 code=context_length_exceeded (negative control),
  * a two-message chat drops nothing (length control).
UI half: the Flappy Bird prompt plus two follow-ups in a real chat with Max Tokens pinned, until
the compaction notice ([data-testid=compaction-notice]) renders, then one more answer succeeds."""

from __future__ import annotations

from ..contract import Journey, Step, StepFailed
from . import _b_common as b
from . import _b_ui as ui

CTX = 2048
REPLY_TOKENS = 64  # max_tokens of the API probes
# max_tokens of the budget-straddling probe (seed + api_compacts). Any value >= CTX // 4 gives the
# same prompt budget (the server reserves min(max_tokens, ctx // 4)); it was REPLY_TOKENS * 100
# (6400), which let the compacted reply run on to the end of the window. CTX // 4 keeps the
# budget, the seeded history and the dropped count identical and caps the reply at 512 tokens.
API_MAX_TOKENS = CTX // 4
UI_MAX_TOKENS = (
    256  # x (1 + AUTO_CONTINUE_LIMIT 3) auto-continued rounds per turn stays under the window
)
FOLLOW_UPS = (
    "Now write the Javascript part in full.",
    "Fix any bugs and print the whole file again.",
    "Add a score counter.",
    "Add a restart button.",
    "Make the pipes move faster over time.",
)


def prompt_budget(ctx_len, max_tokens):
    return ctx_len - min(max_tokens, ctx_len // 4)


def filler(i, words = 40):
    """Distinct per-turn filler: nothing dedupes or caches into fitting."""
    return f"Design note {i}: " + " ".join(f"n{i}w{j}" for j in range(words))


async def seed_history(
    count,
    budget,
    prompt,
    step_words = 40,
    max_turns = 400,
):
    """History h (user/assistant pairs) with count(h) <= budget < count(h + prompt).

    `count` is an async callable messages -> tokens. Grows by whole pairs until history + prompt
    passes the budget, then binary-searches the word count of the LAST user turn for the longest
    history that still fits alone, so the final prompt is what tips it over."""
    ask = {"role": "user", "content": prompt}
    hist = []
    for i in range(max_turns):
        if await count(hist + [ask]) > budget:
            break
        hist += [
            {"role": "user", "content": filler(i, step_words)},
            {"role": "assistant", "content": f"Noted {i}."},
        ]
    else:
        raise StepFailed(f"{max_turns} turns never passed the {budget}-token budget")
    if hist:
        k = len(hist) // 2 - 1
        lo, hi = 0, step_words  # invariant: words=lo fits, or lo == 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            hist[-2] = {"role": "user", "content": filler(k, mid)}
            if await count(hist) <= budget:
                lo = mid
            else:
                hi = mid - 1
        hist[-2] = {"role": "user", "content": filler(k, lo)}
        if await count(hist) > budget:  # even an empty last turn is too long: drop the pair
            hist = hist[:-2]
    h, hp = await count(hist), await count(hist + [ask])
    if not (h <= budget < hp):
        raise StepFailed(
            f"could not straddle the budget: history {h}, with prompt {hp}, budget {budget}"
        )
    return hist, h, hp


def dropped(body):
    t = (body or {}).get("context_truncated") if isinstance(body, dict) else None
    return int((t or {}).get("dropped_messages") or 0) if isinstance(t, dict) else 0


async def _count(ctx, messages):
    _, d = await b.req(
        ctx,
        "POST",
        "/api/inference/chat/count_tokens",
        {"model": "default", "messages": messages},
        ok = (200,),
    )
    return int(d["input_tokens"])


async def load_2048(ctx):
    repo, variant = b.model(ctx, "gguf_compaction", b.GGUF_QWEN35_08B)
    # Setup, not the subject: keep an identical load an earlier journey left in place.
    st, reused = await b.ensure_gguf(ctx, repo, variant, CTX)
    if b.effective_ctx(st) != CTX:
        raise StepFailed(f"context is {b.effective_ctx(st)}, not {CTX}")
    await b.goto(ctx, "/chat")
    await ui.need(ctx.page.get_by_role("textbox", name = "Message input"), "chat composer", 60_000)
    return {"model": repo, "variant": variant, "context_length": CTX, "_reused_load": reused}


async def seed(ctx):
    budget = prompt_budget(CTX, API_MAX_TOKENS)  # the probes' max_tokens path below
    hist, h, hp = await seed_history(lambda m: _count(ctx, m), budget, b.FLAPPY)
    ctx.state["history"] = hist
    return {
        "budget": budget,
        "history_messages": len(hist),
        "history_tokens": h,
        "with_prompt_tokens": hp,
    }


async def api_compacts(ctx):
    msgs = ctx.state["history"] + [{"role": "user", "content": b.FLAPPY}]
    code, body = await b.chat(
        ctx, msgs, max_tokens = API_MAX_TOKENS, context_overflow = "truncate_oldest"
    )
    n = dropped(body)
    if code != 200 or n <= 0:
        raise StepFailed(f"truncate_oldest: HTTP {code}, dropped_messages={n}: {str(body)[:300]}")
    return {
        "status": code,
        "dropped_messages": n,
        "reply_nonempty": bool(b.reply_text(body).strip()),
        "truncation": (body or {}).get("context_truncated"),
    }


async def over_window(ctx):
    """A conversation past the WHOLE window: the budget trigger above compacts before this point,
    but only a prompt that cannot physically fit makes the "error" policy refuse."""
    hist, i = [], 0
    while await _count(ctx, hist) <= CTX + 256:
        hist += [
            {"role": "user", "content": filler(1000 + i)},
            {"role": "assistant", "content": f"Noted {i}."},
        ]
        i += 1
    ctx.state["over"] = hist + [{"role": "user", "content": b.FLAPPY}]
    code, body = await b.chat(
        ctx, ctx.state["over"], max_tokens = REPLY_TOKENS, context_overflow = "truncate_oldest"
    )
    if code != 200 or dropped(body) <= 0:
        raise StepFailed(f"over-window + truncate_oldest: HTTP {code}, dropped={dropped(body)}")
    return {
        "messages": len(ctx.state["over"]),
        "tokens": await _count(ctx, ctx.state["over"]),
        "dropped_messages": dropped(body),
    }


async def api_refuses_when_off(ctx):
    code, body = await b.chat(
        ctx, ctx.state["over"], max_tokens = REPLY_TOKENS, context_overflow = "error"
    )
    # /api answers HTTPException-shaped ({"detail": {"error": ...}}), /v1 OpenAI-shaped.
    b0 = body if isinstance(body, dict) else {}
    err = (
        b0.get("error") or (b0.get("detail") or {}).get("error") or {}
        if isinstance(b0.get("detail", {}), dict)
        else b0.get("error") or {}
    )
    if code != 400 or err.get("code") != "context_length_exceeded":
        raise StepFailed(
            f"compaction off: expected 400 context_length_exceeded, got {code} {str(body)[:300]}"
        )
    return {"status": code, "code": err.get("code")}


async def api_short_not_compacted(ctx):
    code, body = await b.chat(
        ctx,
        [{"role": "user", "content": "Say hi."}],
        max_tokens = 16,
        context_overflow = "truncate_oldest",
    )
    if code != 200 or dropped(body):
        raise StepFailed(f"short chat: HTTP {code}, dropped={dropped(body)}")
    return {"status": code, "dropped_messages": 0}


async def ui_settings(ctx):
    await ui.open_run_settings(ctx.page)
    got = {
        k: await ui.set_field(ctx.page, k, v)
        for k, v in (("Max Tokens", UI_MAX_TOKENS), ("Seed", 3407), ("Temperature", 0))
    }
    await ctx.page.get_by_role("button", name = "Close run settings").first.click()
    return got


async def ui_flappy_until_compacted(ctx):
    page = ctx.page
    turns = [b.FLAPPY, *FOLLOW_UPS]
    for i, text in enumerate(turns):
        n = await page.locator(ui.ASSISTANT).count()
        await ui.send(page, text)
        await ui.wait_reply(page, n, timeout_s = 240, state = ctx.state)
        notice = page.locator("[data-testid=compaction-notice]")
        if await notice.count():
            await notice.first.scroll_into_view_if_needed()
            return {
                "turns_sent": i + 1,
                "dropped": int(await notice.first.get_attribute("data-dropped") or 0),
                "archived": await notice.first.get_attribute("data-archived"),
            }
    raise StepFailed(f"{len(turns)} turns at ctx {CTX} never showed the compaction notice")


async def ui_answer_after_compaction(ctx):
    n = await ctx.page.locator(ui.ASSISTANT).count()
    await ui.send(ctx.page, "In one sentence, what game are we building?")
    txt = await ui.wait_reply(ctx.page, n, timeout_s = 240, state = ctx.state)
    return {"reply_nonempty": bool(txt.strip()), "mentions_flappy": "flappy" in txt.lower()}


JOURNEY = Journey(
    name = "compaction",
    tier = "gpu",
    needs = ("gguf_compaction",),
    routes = ("/chat",),
    steps = (
        Step("load_2048", load_2048, timeout_s = 600),
        Step("seed_history", seed, shot = False, timeout_s = 300),
        Step("api_compacts", api_compacts, shot = False, timeout_s = 300),
        Step("over_window_compacts", over_window, shot = False, timeout_s = 300),
        Step("api_refuses_when_off", api_refuses_when_off, shot = False, timeout_s = 120),
        Step("api_short_not_compacted", api_short_not_compacted, shot = False, timeout_s = 120),
        Step("ui_settings", ui_settings),
        # The generated game differs by kernel / GPU; the NOTICE is the subject, the code is not.
        Step(
            "ui_compacted",
            ui_flappy_until_compacted,
            timeout_s = 1200,
            masks = b.VOLATILE_MASKS + b.GENERATED_TEXT_MASKS,
            mask_reason = "generated code / prose and token counts vary by kernel",
        ),
        Step(
            "ui_answer_after",
            ui_answer_after_compaction,
            timeout_s = 300,
            masks = b.VOLATILE_MASKS + b.GENERATED_TEXT_MASKS,
            mask_reason = "generated prose varies by kernel",
        ),
    ),
)
