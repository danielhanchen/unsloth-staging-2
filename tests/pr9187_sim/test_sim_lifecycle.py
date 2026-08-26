"""Cancellation, terminal truthfulness, crash recovery, and SQLite contention.

The question every scenario here answers is the same: can a run be left active forever?
A run stuck in queued/running is the failure mode that costs a user their GPU and their
answer, so each trigger asserts a terminal state, not just "no exception".
"""

import asyncio
import sqlite3
import threading
import time

import pytest

import _harness as H

pytestmark = pytest.mark.asyncio


def _row(run_id = "run-1"):
    from storage import chat_generation_runs_db as runs_db

    return runs_db.get_run(run_id)


# --- terminal truthfulness ---------------------------------------------------------


async def test_clean_stop_is_completed(monkeypatch):
    H.seed_run()
    H.script(monkeypatch, [H.text_chunk("a"), H.stop_chunk("stop")])
    await H.supervisor()._produce("run-1")
    assert (_row()["status"], _row()["finishReason"]) == ("completed", "stop")


async def test_token_cap_is_completed_with_length(monkeypatch):
    """Length is a truthful completion, not a failure. The UI reads finishReason."""
    H.seed_run()
    H.script(monkeypatch, [H.text_chunk("a"), H.stop_chunk("length")])
    await H.supervisor()._produce("run-1")
    assert (_row()["status"], _row()["finishReason"]) == ("completed", "length")


async def test_eof_without_done_is_interrupted_not_completed(monkeypatch):
    """This is the #8925 lie: a cut stream must not read as a finished answer."""
    H.seed_run()
    H.script(monkeypatch, [H.text_chunk("a")], done = False)
    await H.supervisor()._produce("run-1")
    assert _row()["status"] == "failed"
    assert _row()["finishReason"] == "interrupted"


async def test_an_error_chunk_cannot_be_followed_by_a_clean_completion(monkeypatch):
    H.seed_run()
    H.script(monkeypatch, [{"error": {"message": "boom"}}, H.stop_chunk("stop")])
    await H.supervisor()._produce("run-1")
    assert _row()["status"] == "failed"


async def test_a_raising_producer_is_failed_not_left_running(monkeypatch):
    H.seed_run()
    H.script(monkeypatch, [H.text_chunk("a")], fail = RuntimeError("engine died"))
    await H.supervisor()._produce("run-1")
    assert _row()["status"] == "failed"
    H.assert_ledger_is_sound("run-1")


async def test_buffered_chunks_survive_a_crash(monkeypatch):
    """Chunks still in the batch buffer when the producer dies must be flushed."""
    H.seed_run()
    H.script(monkeypatch, [H.text_chunk("keep-me")], fail = RuntimeError("boom"))
    await H.supervisor()._produce("run-1")
    assert "keep-me" in H.text_of("run-1")


# --- cancellation ------------------------------------------------------------------


async def test_cancel_while_running_wins_over_completion(monkeypatch):
    """request_cancel commits first, so the worker's own verdict must not overwrite it."""
    from storage import chat_generation_runs_db as runs_db

    H.seed_run()
    gate = threading.Event()
    H.script(monkeypatch, [H.text_chunk("a")], gate = gate, after_gate = [H.stop_chunk()])

    task = asyncio.create_task(H.supervisor()._produce("run-1"))
    await asyncio.sleep(0.2)
    runs_db.request_cancel("run-1")
    gate.set()
    await task

    assert _row()["status"] == "cancelled", "a cancel that committed first was overwritten"
    H.assert_ledger_is_sound("run-1")


async def test_cancel_while_queued_never_starts_a_generation(monkeypatch):
    from storage import chat_generation_runs_db as runs_db

    H.seed_run()
    started = []

    from routes import inference

    async def fake(*_a, **_k):
        started.append(1)
        raise AssertionError("a cancelled run must not reach the engine")

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    runs_db.request_cancel("run-1")
    await H.supervisor()._produce("run-1")

    assert started == []
    assert _row()["status"] == "cancelled"


async def test_double_cancel_is_idempotent():
    from storage import chat_generation_runs_db as runs_db

    H.seed_run()
    first = runs_db.request_cancel("run-1")
    second = runs_db.request_cancel("run-1")
    assert first["status"] == "cancelled"
    assert second["status"] == "cancelled"
    H.assert_ledger_is_sound("run-1")


async def test_cancel_after_terminal_does_not_reopen_the_run(monkeypatch):
    from storage import chat_generation_runs_db as runs_db

    H.seed_run()
    H.script(monkeypatch, [H.text_chunk("a"), H.stop_chunk()])
    await H.supervisor()._produce("run-1")

    runs_db.request_cancel("run-1")
    assert _row()["status"] == "completed", "a terminal run was reopened by a late cancel"
    H.assert_ledger_is_sound("run-1")


async def test_cancel_reaches_a_generator_that_watches_the_event(monkeypatch):
    """The supervisor's Event is the same object the request sees. Prove the wiring."""
    H.seed_run()
    state = H.cancellable_script(monkeypatch)
    sup = H.supervisor()

    sup.start("run-1", thread_id = "thread-1", model = "local.gguf")
    for _ in range(200):
        if state.started.is_set():
            break
        await asyncio.sleep(0.02)
    assert state.started.is_set(), "the scripted generator never started"

    sup.cancel("run-1")
    await asyncio.wait_for(asyncio.gather(*sup._tasks.values(), return_exceptions = True), 15)
    assert _row()["status"] in ("cancelled", "failed")


async def test_the_admission_slot_is_released_on_every_outcome(monkeypatch):
    """A leaked reservation blocks idle-unload and model swap forever."""
    from core.inference import llama_keepwarm
    from state import active_generations

    # A delta, not an absolute: these counters are module globals shared with every
    # other test in the process, so only the change across this run is ours to assert.
    base_pending = llama_keepwarm.other_non_preview_pending_count()
    base_admitted = llama_keepwarm.other_admitted_inference_count()
    base_active = active_generations.count()

    for i, chunks in enumerate((
        [H.text_chunk("a"), H.stop_chunk()],      # completed
        [H.text_chunk("a")],                       # interrupted
    )):
        run_id = f"run-{i}"
        H.seed_run(run_id = run_id, assistant_id = f"assistant-{i}",
                   seed_the_thread = (i == 0))
        H.script(monkeypatch, chunks, done = (i == 0))
        sup = H.supervisor()
        sup.start(run_id, thread_id = "thread-1", model = "local.gguf")
        await asyncio.wait_for(
            asyncio.gather(*sup._tasks.values(), return_exceptions = True), 20
        )
        assert active_generations.count() == base_active, (
            f"{run_id} leaked an active generation"
        )
        assert llama_keepwarm.other_non_preview_pending_count() == base_pending, (
            f"{run_id} leaked a lifecycle-gate reservation"
        )
        assert llama_keepwarm.other_admitted_inference_count() == base_admitted, (
            f"{run_id} leaked an admitted inference slot"
        )


# --- crash recovery ----------------------------------------------------------------


async def test_reconcile_terminates_an_orphaned_run():
    from storage import chat_generation_runs_db as runs_db

    H.seed_run()
    runs_db.mark_running("run-1", _worker_token())
    assert _row()["status"] == "running"

    runs_db.reconcile_orphaned_runs()

    assert _row()["status"] == "failed"
    assert _row()["finishReason"] == "interrupted"
    H.assert_ledger_is_sound("run-1")


async def test_reconcile_is_idempotent_across_repeated_boots():
    from storage import chat_generation_runs_db as runs_db

    H.seed_run()
    runs_db.mark_running("run-1", _worker_token())
    for _ in range(3):
        runs_db.reconcile_orphaned_runs()
    terminal = [e for e in H.all_events("run-1") if e["type"] == "run.failed"]
    assert len(terminal) == 1, f"{len(terminal)} terminal events after three boots"
    H.assert_ledger_is_sound("run-1")


async def test_reconcile_leaves_a_completed_run_alone(monkeypatch):
    from storage import chat_generation_runs_db as runs_db

    H.seed_run()
    H.script(monkeypatch, [H.text_chunk("a"), H.stop_chunk()])
    await H.supervisor()._produce("run-1")
    before = dict(_row())

    runs_db.reconcile_orphaned_runs()

    assert _row()["status"] == before["status"] == "completed"
    assert _row()["lastEventSeq"] == before["lastEventSeq"]


async def test_reconcile_marks_the_assistant_message_interrupted():
    from storage import chat_generation_runs_db as runs_db

    H.seed_run()
    runs_db.mark_running("run-1", _worker_token())
    runs_db.reconcile_orphaned_runs()

    rows = H.raw_query("SELECT metadata_json FROM chat_messages WHERE id='assistant-1'")
    assert rows, "the assistant placeholder vanished"
    assert "interrupted" in (rows[0]["metadata_json"] or "")


async def test_a_stale_worker_token_cannot_append_after_reconcile():
    from storage import chat_generation_runs_db as runs_db

    H.seed_run()
    stale = _worker_token()
    runs_db.mark_running("run-1", stale)
    runs_db.reconcile_orphaned_runs()

    before = len(H.all_events("run-1"))
    # Either refusal shape is fine; writing the chunk is not.
    try:
        runs_db.append_events("run-1", stale, [("chunk", H.text_chunk("ghost"))])
    except KeyError:
        pass
    assert len(H.all_events("run-1")) == before, "a pre-restart worker wrote after recovery"
    assert "ghost" not in H.text_of("run-1")


async def test_a_wrong_worker_token_is_refused():
    from storage import chat_generation_runs_db as runs_db

    H.seed_run()
    runs_db.mark_running("run-1", _worker_token())
    before = len(H.all_events("run-1"))
    try:
        runs_db.append_events("run-1", "not-the-token", [("chunk", H.text_chunk("ghost"))])
    except KeyError:
        pass
    assert len(H.all_events("run-1")) == before


def _worker_token():
    return H.raw_query("SELECT worker_token t FROM chat_generation_runs WHERE id='run-1'")[0]["t"]


# --- SQLite contention -------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("hold_seconds", [1.0, 4.0])
async def test_a_held_writer_lock_does_not_strand_a_run(monkeypatch, hold_seconds):
    """Another writer holds the DB. The run may fail, but it may not stay active."""
    H.seed_run()
    H.script(monkeypatch, [H.text_chunk("a"), H.stop_chunk()])

    holder = sqlite3.connect(str(H.db_path()), timeout = 30)
    holder.execute("BEGIN IMMEDIATE")
    release_at = time.monotonic() + hold_seconds

    async def release_later():
        while time.monotonic() < release_at:
            await asyncio.sleep(0.05)
        holder.rollback()
        holder.close()

    releaser = asyncio.create_task(release_later())
    try:
        await H.supervisor()._produce("run-1")
    except Exception:
        pass
    finally:
        await releaser

    assert _row()["status"] in ("completed", "failed", "cancelled"), (
        "the run is still active after writer contention"
    )
    H.assert_ledger_is_sound("run-1")
