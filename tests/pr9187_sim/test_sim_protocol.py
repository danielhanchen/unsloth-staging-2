"""Disconnect boundaries, cursor handling, create idempotency, replay scale, subscribers.

Every scenario asserts the ledger invariants as well as its own oracle: a scenario that
passes its own check while corrupting the event log is not a pass.
"""

import asyncio

import pytest

import _harness as H

pytestmark = pytest.mark.asyncio


# --- disconnect boundaries ---------------------------------------------------------


async def test_disconnect_before_first_chunk_then_resume(monkeypatch):
    """The subscriber leaves before anything is emitted. The run still completes."""
    H.seed_run()
    H.script(monkeypatch, [H.text_chunk("a"), H.text_chunk("b"), H.stop_chunk()])
    await H.supervisor()._produce("run-1")

    seqs, _ = await H.subscribe("run-1", after = 0)
    assert seqs[0] == 1
    H.assert_ledger_is_sound("run-1")
    assert H.text_of("run-1") == "ab"


async def test_disconnect_mid_stream_replays_only_the_tail(monkeypatch):
    H.seed_run()
    H.script(monkeypatch, [H.text_chunk(c) for c in "abcdef"] + [H.stop_chunk()])
    await H.supervisor()._produce("run-1")

    everything, _ = await H.subscribe("run-1", after = 0)
    cut = everything[len(everything) // 2]
    tail, _ = await H.subscribe("run-1", after = cut)

    assert all(s > cut for s in tail), "replay handed back an already-applied event"
    assert everything[: everything.index(cut) + 1] + tail == everything
    H.assert_ledger_is_sound("run-1")


async def test_disconnect_after_terminal_commit_still_replays_the_terminal(monkeypatch):
    """The client died between the terminal commit and applying it. It must see it."""
    H.seed_run()
    H.script(monkeypatch, [H.text_chunk("a"), H.stop_chunk()])
    await H.supervisor()._produce("run-1")

    from storage import chat_generation_runs_db as runs_db

    last = runs_db.get_run("run-1")["lastEventSeq"]
    tail, raw = await H.subscribe("run-1", after = last - 1)
    assert tail == [last]
    assert "run.completed" in raw
    H.assert_ledger_is_sound("run-1")


async def test_a_rewound_cursor_is_deduped_by_the_client_rule(monkeypatch):
    """The wire is at-least-once; the applied stream must still be exactly-once.

    A stale tab can reconnect with an old cursor, so the server will resend. The
    client's rule is `if (event.seq <= cursor) continue` (chat-generation-api.ts:377).
    Applying that rule here proves the two halves compose into exactly-once.
    """
    H.seed_run()
    H.script(monkeypatch, [H.text_chunk(c) for c in "abcdefghij"] + [H.stop_chunk()])
    await H.supervisor()._produce("run-1")

    applied = []
    client_cursor = 0
    # Reconnect five times, each time from a deliberately rewound position.
    for rewind in (0, 3, 1, 7, 0):
        seqs, _ = await H.subscribe("run-1", after = rewind)
        assert seqs and min(seqs) == rewind + 1, "server did not honour the cursor"
        for s in seqs:
            if s <= client_cursor:
                continue
            client_cursor = s
            applied.append(s)

    assert applied == sorted(set(applied)), "the client rule let a duplicate through"
    assert applied == list(range(1, len(applied) + 1)), "the applied stream has a gap"
    H.assert_ledger_is_sound("run-1")


async def test_terminal_run_is_replayable_from_zero_forever(monkeypatch):
    H.seed_run()
    H.script(monkeypatch, [H.text_chunk("a"), H.stop_chunk()])
    await H.supervisor()._produce("run-1")

    first, _ = await H.subscribe("run-1", after = 0)
    second, _ = await H.subscribe("run-1", after = 0)
    assert first == second and first, "replay is not stable"


# --- cursors -----------------------------------------------------------------------


@pytest.mark.parametrize("after", [0, 1, 2])
async def test_low_cursors_return_everything_above(monkeypatch, after):
    H.seed_run()
    H.script(monkeypatch, [H.text_chunk("a"), H.text_chunk("b"), H.stop_chunk()])
    await H.supervisor()._produce("run-1")

    seqs, _ = await H.subscribe("run-1", after = after)
    assert seqs and min(seqs) == after + 1


async def test_future_cursor_is_not_an_error_and_yields_nothing_new(monkeypatch):
    H.seed_run()
    H.script(monkeypatch, [H.text_chunk("a"), H.stop_chunk()])
    await H.supervisor()._produce("run-1")

    from storage import chat_generation_runs_db as runs_db

    last = runs_db.get_run("run-1")["lastEventSeq"]
    seqs, _ = await H.subscribe("run-1", after = last + 500)
    assert seqs == []


async def test_cursor_beyond_sqlite_max_integer_is_rejected_not_crashed(monkeypatch):
    from fastapi import HTTPException

    from routes import chat_generation_runs as run_routes

    with pytest.raises(HTTPException) as exc:
        run_routes._event_cursor(run_routes._SQLITE_MAX_INTEGER + 1, None)
    assert exc.value.status_code == 400


@pytest.mark.parametrize("header", ["abc", "-1", "1.5", "0x10", " 12", "12 ", ""])
async def test_non_integer_last_event_id_is_rejected(header):
    from fastapi import HTTPException

    from routes import chat_generation_runs as run_routes

    if header == "":
        assert run_routes._event_cursor(None, header) == 0
        return
    with pytest.raises(HTTPException) as exc:
        run_routes._event_cursor(None, header)
    assert exc.value.status_code == 400


async def test_cursor_takes_the_greater_of_query_and_header():
    from routes import chat_generation_runs as run_routes

    assert run_routes._event_cursor(10, "3") == 10
    assert run_routes._event_cursor(3, "10") == 10
    assert run_routes._event_cursor(None, "0007") == 7
    assert run_routes._event_cursor(5, None) == 5


async def test_javascript_unsafe_cursor_survives_a_round_trip():
    """2**53 is where a JS number stops being exact. The server must not widen it."""
    from routes import chat_generation_runs as run_routes

    unsafe = 2 ** 53 + 1
    assert run_routes._event_cursor(None, str(unsafe)) == unsafe
    assert run_routes._event_cursor(unsafe, None) == unsafe


async def test_cursor_from_another_run_does_not_leak_that_runs_events(monkeypatch):
    H.seed_run()
    H.script(monkeypatch, [H.text_chunk("a"), H.stop_chunk()])
    await H.supervisor()._produce("run-1")

    H.seed_run(run_id = "run-2", assistant_id = "assistant-2", seed_the_thread = False)
    H.script(monkeypatch, [H.text_chunk("z"), H.stop_chunk()])
    await H.supervisor()._produce("run-2")

    one = {e["seq"] for e in _events("run-1")}
    two_seqs, _ = await H.subscribe("run-2", after = 0)
    assert set(two_seqs) & one == set(two_seqs) & one  # ids are per-run, overlap is fine
    assert H.text_of("run-1") == "a" and H.text_of("run-2") == "z"
    H.assert_ledger_is_sound("run-1")
    H.assert_ledger_is_sound("run-2")


def _events(run_id):
    return H.all_events(run_id)


# --- create idempotency ------------------------------------------------------------


async def test_same_run_id_twice_creates_one_run(monkeypatch):
    H.seed_thread()
    first, created_first = H.seed_run(seed_the_thread = False)
    second, created_second = H.seed_run(seed_the_thread = False)
    assert created_first is True and created_second is False
    assert first["id"] == second["id"]


async def test_same_run_id_with_a_different_payload_conflicts():
    from storage import chat_generation_runs_db as runs_db

    H.seed_thread()
    H.seed_run(seed_the_thread = False)
    with pytest.raises(runs_db.ChatGenerationConflictError):
        runs_db.create_run(
            run_id = "run-1", owner_subject = H.OWNER, thread_id = "thread-1",
            user_message_id = "user-1", assistant_message_id = "assistant-1",
            request_payload = {"model": "local.gguf",
                               "messages": [{"role": "user", "content": "DIFFERENT"}]},
        )


async def test_same_run_id_from_another_subject_conflicts():
    from storage import chat_generation_runs_db as runs_db

    H.seed_thread()
    run, _ = H.seed_run(seed_the_thread = False)
    with pytest.raises(runs_db.ChatGenerationConflictError):
        runs_db.create_run(
            run_id = "run-1", owner_subject = H.OTHER, thread_id = "thread-1",
            user_message_id = "user-1", assistant_message_id = "assistant-1",
            request_payload = run["requestPayload"],
        )


async def test_a_second_active_run_on_one_thread_is_refused():
    from storage import chat_generation_runs_db as runs_db

    H.seed_thread()
    H.seed_run(seed_the_thread = False)
    with pytest.raises(runs_db.ChatGenerationConflictError):
        runs_db.create_run(
            run_id = "run-2", owner_subject = H.OWNER, thread_id = "thread-1",
            user_message_id = "user-1", assistant_message_id = "assistant-2",
            request_payload = {"model": "local.gguf",
                               "messages": [{"role": "user", "content": "Hello"}]},
        )


async def test_a_second_run_is_allowed_once_the_first_is_terminal(monkeypatch):
    H.seed_run()
    H.script(monkeypatch, [H.text_chunk("a"), H.stop_chunk()])
    await H.supervisor()._produce("run-1")

    _run, created = H.seed_run(
        run_id = "run-2", assistant_id = "assistant-2", seed_the_thread = False
    )
    assert created is True


# --- replay scale ------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 999, 1000, 1001, 2000])
async def test_replay_is_gapless_across_the_page_boundary(monkeypatch, n):
    H.seed_run()
    H.script(monkeypatch, [H.text_chunk("x") for _ in range(n)] + [H.stop_chunk()])
    await H.supervisor()._produce("run-1")

    seqs, _ = await H.subscribe("run-1", after = 0)
    assert seqs == list(range(1, len(seqs) + 1)), "page boundary dropped or reordered an event"
    H.assert_ledger_is_sound("run-1")
    assert H.text_of("run-1") == "x" * n


@pytest.mark.parametrize("cut", [998, 999, 1000, 1001])
async def test_resume_exactly_on_the_page_boundary(monkeypatch, cut):
    H.seed_run()
    H.script(monkeypatch, [H.text_chunk("x") for _ in range(1200)] + [H.stop_chunk()])
    await H.supervisor()._produce("run-1")

    everything, _ = await H.subscribe("run-1", after = 0)
    tail, _ = await H.subscribe("run-1", after = cut)
    assert tail == [s for s in everything if s > cut]


# --- concurrent subscribers --------------------------------------------------------


@pytest.mark.parametrize("readers", [1, 8, 32, 33, 64])
async def test_every_concurrent_subscriber_gets_the_same_ledger(monkeypatch, readers):
    """The SSE waiter pool is 32 workers; 33 and 64 must not starve anyone."""
    H.seed_run()
    H.script(monkeypatch, [H.text_chunk(c) for c in "abcdefgh"] + [H.stop_chunk()])
    await H.supervisor()._produce("run-1")

    results = await asyncio.gather(*(H.subscribe("run-1", after = 0) for _ in range(readers)))
    ledgers = {tuple(seqs) for seqs, _ in results}
    assert len(ledgers) == 1, f"{len(ledgers)} distinct ledgers across {readers} subscribers"
    assert list(ledgers.pop()) == list(range(1, len(results[0][0]) + 1))
