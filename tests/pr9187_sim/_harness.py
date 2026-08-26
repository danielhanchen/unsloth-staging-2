"""Shared building blocks for the PR 9187 scenarios.

The producer is scripted rather than a real model: every scenario needs to control
exactly when a chunk lands, when the stream ends, and where a disconnect falls, and a
real generator cannot be steered that finely. The end-to-end reality check is a
separate live-backend probe (scripts/pr9187_durable_probe.py), which is what proves
the scripted contract matches a real generation.
"""

import asyncio
import json
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

OWNER = "alice"
OTHER = "bob"


# --- seeding -----------------------------------------------------------------------


def seed_thread(thread_id = "thread-1", user_id = "user-1", model = "local.gguf"):
    from storage import studio_db

    studio_db.upsert_chat_thread({
        "id": thread_id, "title": "Chat", "modelType": "base",
        "modelId": model, "createdAt": 1,
    })
    studio_db.upsert_chat_message({
        "id": user_id, "threadId": thread_id, "role": "user",
        "content": [{"type": "text", "text": "Hello"}], "createdAt": 2,
    })


def seed_run(
    run_id = "run-1",
    thread_id = "thread-1",
    user_id = "user-1",
    assistant_id = "assistant-1",
    owner = OWNER,
    model = "local.gguf",
    seed_the_thread = True,
):
    from storage import chat_generation_runs_db as runs_db

    if seed_the_thread:
        seed_thread(thread_id, user_id, model)
    run, created = runs_db.create_run(
        run_id = run_id,
        owner_subject = owner,
        thread_id = thread_id,
        user_message_id = user_id,
        assistant_message_id = assistant_id,
        request_payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "cancel_id": run_id,
            "thread_id": thread_id,
            "generation_run_id": run_id,
        },
    )
    return run, created


# --- scripted producers ------------------------------------------------------------


def text_chunk(text):
    return {"choices": [{"delta": {"content": text}, "finish_reason": None}]}


def stop_chunk(reason = "stop"):
    return {"choices": [{"delta": {}, "finish_reason": reason}]}


def script(monkeypatch, chunks, *, done = True, gate = None, after_gate = None, fail = None):
    """Install a scripted producer.

    `gate` is a threading.Event the stream waits on midway, so a scenario can hold a
    generation open exactly as long as it needs.
    """
    from routes import inference

    async def body():
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
        if gate is not None:
            while not gate.is_set():
                await asyncio.sleep(0.01)
            for chunk in after_gate or []:
                yield f"data: {json.dumps(chunk)}\n\n"
        if fail is not None:
            raise fail
        if done:
            yield "data: [DONE]\n\n"

    async def fake(_payload, _request, _subject, *, cancel_on_disconnect):
        fake.cancel_on_disconnect = cancel_on_disconnect
        return SimpleNamespace(status_code = 200, body_iterator = body())

    fake.cancel_on_disconnect = None
    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    return fake


def cancellable_script(monkeypatch, *, chunks_before = 3):
    """A producer that streams forever until the request's cancel event fires.

    Mirrors what a real generator does: it observes
    `request.state.generation_cancel_event`, which is how the supervisor stops it.
    """
    from routes import inference

    state = SimpleNamespace(started = threading.Event(), emitted = 0)

    async def body(cancel_event):
        for i in range(chunks_before):
            state.emitted += 1
            yield f"data: {json.dumps(text_chunk(f'c{i}'))}\n\n"
        state.started.set()
        while cancel_event is None or not cancel_event.is_set():
            await asyncio.sleep(0.01)
            state.emitted += 1
            yield f"data: {json.dumps(text_chunk('.'))}\n\n"

    async def fake(_payload, request, _subject, *, cancel_on_disconnect):
        event = getattr(getattr(request, "state", None), "generation_cancel_event", None)
        return SimpleNamespace(status_code = 200, body_iterator = body(event))

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    return state


# --- driving the routes ------------------------------------------------------------


def supervisor():
    from core.inference.chat_generation_runs import ChatGenerationSupervisor

    return ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))


def route_request(sup):
    return SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(chat_generation_supervisor = sup))
    )


async def _one_subscription(run_id, after, last_event_id, subject):
    from routes import chat_generation_runs as run_routes

    response = await run_routes.chat_generation_events(
        run_id,
        # is_disconnected True means the stream serves what it has and stops rather than
        # long-polling for more, which is what makes these scenarios finish in millisecs.
        SimpleNamespace(is_disconnected = AsyncMock(return_value = True)),
        after = after,
        last_event_id = last_event_id,
        current_subject = subject,
    )
    raw = ""
    async for part in response.body_iterator:
        raw += part.decode() if isinstance(part, bytes) else part
    seqs = [int(line[4:]) for line in raw.splitlines() if line.startswith("id: ")]
    return seqs, raw


async def subscribe(run_id, after = 0, last_event_id = None, subject = OWNER):
    """Drain a subscription the way the real client does: page until nothing new.

    `list_events` pages at 1000, so one pass over a longer run stops at the page
    boundary. The browser client reconnects from its cursor; so does this.
    """
    seqs, raw = [], ""
    cursor = after
    for _ in range(100):
        page, page_raw = await _one_subscription(run_id, cursor, last_event_id, subject)
        last_event_id = None
        raw += page_raw
        if not page:
            break
        seqs.extend(page)
        cursor = page[-1]
    return seqs, raw


# --- oracles -----------------------------------------------------------------------


def db_path():
    import os
    from pathlib import Path

    return Path(os.environ["UNSLOTH_STUDIO_HOME"]) / "studio.db"


def raw_query(sql, params = ()):
    conn = sqlite3.connect(f"file:{db_path()}?mode=ro", uri = True)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def all_events(run_id):
    """Every event, bypassing list_events' 1000-row page limit."""
    return raw_query(
        "SELECT seq, event_type AS type, payload_json FROM chat_generation_events "
        "WHERE run_id=? ORDER BY seq", (run_id,)
    )


def assert_ledger_is_sound(run_id):
    """Every invariant the replay protocol depends on, read straight from SQLite."""
    from storage import chat_generation_runs_db as runs_db

    events = all_events(run_id)
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs), f"{run_id}: sequences out of order"
    assert len(seqs) == len(set(seqs)), f"{run_id}: duplicate sequence"
    assert seqs == list(range(1, len(seqs) + 1)), f"{run_id}: gap in {seqs[:20]}..."

    run = runs_db.get_run(run_id)
    if run is not None:
        assert run["lastEventSeq"] == (seqs[-1] if seqs else 0), (
            f"{run_id}: lastEventSeq {run['lastEventSeq']} != max seq {seqs[-1] if seqs else 0}"
        )
        if run["status"] in ("completed", "cancelled", "failed"):
            terminal = [e for e in events if e["type"].startswith("run.")
                        and e["type"] not in ("run.created", "run.started", "run.cancelling")]
            assert len(terminal) == 1, f"{run_id}: {len(terminal)} terminal events, want 1"

    assert raw_query("PRAGMA integrity_check")[0]["integrity_check"] == "ok"
    assert raw_query("PRAGMA foreign_key_check") == []


def text_of(run_id):
    out = []
    for event in all_events(run_id):
        if event["type"] != "chunk":
            continue
        for choice in json.loads(event["payload_json"]).get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                out.append(delta["content"])
    return "".join(out)
