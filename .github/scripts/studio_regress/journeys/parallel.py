"""Journey 3: parallel chats, then chat while a tiny training run is live.

API half: gemma-3-270m GGUF loaded with n_parallel=2; two streaming completions started together
must BOTH stream (their token windows overlap), and cancelling one by cancel_id must leave the
other finishing normally. UI half: two browser tabs send at once and both get replies. Then a
3-step training run starts and a chat is sent while it is live: the documented outcomes are a
served reply (the chat fits beside training) or the 409 "while training is running" refusal for a
NEW load; either is recorded, anything else fails. Training is stopped once observed."""

from __future__ import annotations

import asyncio
import json
import time
import uuid

from .. import timeouts
from ..contract import Journey, Step, StepFailed
from . import _b_common as b
from . import _b_ui as ui
from .train_responses_only import dataset_rows, train_body

LONG = "List the numbers from 1 to 120 separated by commas."


async def stream(ctx, prompt, cancel_id=None, max_tokens=256, started=None):
    body = {"model": "default", "stream": True, "max_tokens": max_tokens, "temperature": 0.0,
            "seed": 3407, "messages": [{"role": "user", "content": prompt}]}
    if cancel_id:
        body["cancel_id"] = cancel_id
    t0 = time.monotonic()
    first = end = None
    chunks, finish, status = 0, None, None
    async with ctx.api.stream("POST", "/api/inference/chat/completions", json=body, timeout=600) as r:
        status = r.status_code
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            ch = (ev.get("choices") or [{}])[0]
            if (ch.get("delta") or {}).get("content"):
                chunks += 1
                first = first or time.monotonic() - t0
                if started is not None:
                    started.set()
            finish = ch.get("finish_reason") or finish
    end = time.monotonic() - t0
    return {"status": status, "first_s": first, "end_s": end, "chunks": chunks, "finish": finish}


async def load_parallel(ctx):
    repo, variant = b.model(ctx, "gguf_small", b.GGUF_270M)
    # The load is setup here (the subject is what two slots do), so an identical load left by an
    # earlier journey on this Studio is kept: a llama-server spawn is minutes on a loaded host.
    st, reused = await b.ensure_gguf(ctx, repo, variant, 4096, n_parallel=2)
    await b.goto(ctx, "/chat")
    await ui.need(ctx.page.get_by_role("textbox", name="Message input"), "chat composer", 60_000)
    # parallel_slots: what llama-server really runs. A build without --kv-unified falls back to 1
    # slot, and two_streams below then fails on "did not overlap"; recorded here so the cause is
    # in the facts next to the failure.
    return {"model": repo, "context_length": b.effective_ctx(st), "n_parallel": 2,
            "parallel_slots": st.get("parallel_slots"), "_reused_load": reused}


async def two_streams(ctx):
    a, c = await asyncio.gather(stream(ctx, LONG), stream(ctx, "Name ten colours, one per line."))
    for name, r in (("A", a), ("B", c)):
        if r["status"] != 200 or not r["chunks"]:
            raise StepFailed(f"stream {name}: {r}")
    overlap = a["first_s"] < c["end_s"] and c["first_s"] < a["end_s"]
    if not overlap:
        raise StepFailed(f"streams did not overlap (serialised): A={a} B={c}")
    return {"overlap": overlap, "a_chunks": a["chunks"], "b_chunks": c["chunks"],
            "a_finish": a["finish"], "b_finish": c["finish"]}


async def cancel_one(ctx):
    cid = str(uuid.uuid4())
    started = asyncio.Event()
    ta = asyncio.create_task(stream(ctx, LONG, cancel_id=cid, max_tokens=512, started=started))
    tb = asyncio.create_task(stream(ctx, LONG, max_tokens=512))
    # Was a blind 0.8 s: cancel once the cancellable stream is really generating (its id is
    # registered and tokens flow), bounded by the old 0.8 s scaled for load when nothing arrives.
    try:
        await asyncio.wait_for(started.wait(), timeout=timeouts.scaled(0.8))
    except asyncio.TimeoutError:
        pass
    _, d = await b.req(ctx, "POST", "/api/inference/cancel", {"cancel_id": cid}, ok=(200,))
    a, c = await ta, await tb
    if c["status"] != 200 or c["finish"] not in ("stop", "length"):
        raise StepFailed(f"the uncancelled stream did not finish normally: {c}")
    if a["chunks"] >= c["chunks"]:
        raise StepFailed(f"cancelled stream was not cut short: cancelled={a} other={c}")
    return {"cancel_ack": d.get("cancelled"), "cancelled_chunks": a["chunks"],
            "other_chunks": c["chunks"], "other_finish": c["finish"]}


async def two_tabs_ui(ctx):
    other = await ctx.page.context.new_page()
    try:
        await other.goto(ctx.base_url + "/chat", wait_until="load", timeout=timeouts.scaled_ms(30_000))
        await ui.need(other.get_by_role("textbox", name="Message input"), "second tab composer", 60_000)
        n1 = await ctx.page.locator(ui.ASSISTANT).count()
        await asyncio.gather(ui.send(ctx.page, "Name three fruits."), ui.send(other, "Name three animals."))
        r1, r2 = await asyncio.gather(ui.wait_reply(ctx.page, n1, state=ctx.state),
                                      ui.wait_reply(other, 0, state=ctx.state))
    finally:
        await other.close()
    return {"tab1_reply": bool(r1.strip()), "tab2_reply": bool(r2.strip())}


async def train_and_chat(ctx):
    home = b.studio_home(ctx)
    ds = home / "assets" / "datasets" / "uploads" / "studio_regress_chat.jsonl"
    ds.parent.mkdir(parents=True, exist_ok=True)
    ds.write_text("\n".join(json.dumps(r) for r in dataset_rows()) + "\n")
    _, d = await b.req(ctx, "POST", "/api/train/start",
                       train_body(ds, max_steps=50, project_name="studio-regress-parallel"),
                       ok=(200,), timeout=300)
    if d.get("status") not in ("queued", "pending"):
        raise StepFailed(f"train start: {d}")
    ctx.state["job_id"] = d.get("job_id")
    st = await b.poll(lambda: b.get(ctx, "/api/train/status"),
                      lambda s: s.get("phase") in ("training", "completed", "error", "stopped"), 900, 1.0,
                      state=ctx.state)
    if st.get("phase") != "training":
        raise StepFailed(f"training never became live: {st.get('phase')} {st.get('error')}")
    code, body = await b.chat(ctx, [{"role": "user", "content": "Say ok."}], max_tokens=8)
    served = code == 200 and bool(b.reply_text(body).strip())
    refused = code == 409 and "training" in json.dumps(body).lower()
    if not (served or refused):
        raise StepFailed(f"chat during training: HTTP {code} {str(body)[:300]}")
    n = await ctx.page.locator(ui.ASSISTANT).count()
    await ui.send(ctx.page, "Say ok again.")
    ui_ok = True
    try:
        await ui.wait_reply(ctx.page, n, timeout_s=120, state=ctx.state)
    except StepFailed:
        ui_ok = False
    ctx.state["training_live"] = True
    return {"chat_during_training": "served" if served else "refused_409", "ui_reply": ui_ok}


LIVE_DONE = ("completed", "stopped", "error", "idle")


async def stop_training(ctx):
    await b.req(ctx, "POST", "/api/train/stop", {"save": False, "expected_job_id": ctx.state.get("job_id")},
                ok=(200,))
    st = await b.poll(lambda: b.get(ctx, "/api/train/status"),
                      lambda s: s.get("phase") in LIVE_DONE, 300, 1.0, state=ctx.state)
    if st.get("phase") == "error":
        raise StepFailed(f"stop ended in error: {st.get('error')}")
    return {"phase": st.get("phase")}


async def teardown(ctx):
    """A step failing after train/start (the chat, the UI reply) skips stop_training, and the job
    would keep the GPU busy under the side's remaining journeys: stop it the same way."""
    if not ctx.state.get("job_id") or ctx.api is None:
        return
    st = await b.get(ctx, "/api/train/status")
    if st.get("phase") in LIVE_DONE:
        return
    await b.req(ctx, "POST", "/api/train/stop", {"save": False, "expected_job_id": ctx.state["job_id"]})


JOURNEY = Journey(
    name="parallel", tier="gpu", needs=("gguf_small", "train_small"), routes=("/chat",), teardown=teardown,
    steps=(
        Step("load_parallel", load_parallel, timeout_s=400),
        Step("two_streams", two_streams, shot=False, timeout_s=300),
        Step("cancel_one", cancel_one, shot=False, timeout_s=300),
        Step("two_tabs_ui", two_tabs_ui, masks=b.VOLATILE_MASKS + b.GENERATED_TEXT_MASKS,
             mask_reason="generated text", timeout_s=300),
        Step("train_and_chat", train_and_chat, masks=b.VOLATILE_MASKS + b.GENERATED_TEXT_MASKS,
             mask_reason="generated text", timeout_s=1100),
        Step("stop_training", stop_training, shot=False, timeout_s=400),
    ),
)
