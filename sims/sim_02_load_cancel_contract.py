"""H1-H4: the scoped load-cancellation contract, from an old client's point of view.

This is the change that touches every model type's /load and /unload, so the
questions are: does a client that never sends load_request_id behave exactly as
it did on main, can a cancel ever unload a resident model, and can /load block
forever waiting for a cancel handshake that never completes.

Driven against the real routes module with the backends stubbed, so the control
flow under test is production code.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

# PR_BACKEND lets the identical suite run against a checkout on any OS runner.
BACKEND = Path(
    os.environ.get(
        "PR_BACKEND",
        Path(__file__).resolve().parents[2] / "unsloth_pr7984" / "studio" / "backend",
    )
)
sys.path.insert(0, str(BACKEND))

import routes.inference as inf  # noqa: E402
from models.inference import LoadRequest, UnloadRequest  # noqa: E402


@pytest.fixture(autouse = True)
def _clean_attempt_registry():
    with inf._scoped_load_attempts_lock:
        inf._scoped_load_attempts.clear()
        inf._scoped_load_cancel_tombstones.clear()
        inf._running_load_attempt = None
    yield
    with inf._scoped_load_attempts_lock:
        inf._scoped_load_attempts.clear()
        inf._scoped_load_cancel_tombstones.clear()
        inf._running_load_attempt = None


# ── H1: a client that predates the feature ────────────────────────────────────

def test_old_client_load_registers_nothing():
    """No load_request_id means no registry entry, so nothing about the old
    single-load path can change shape."""
    attempt = inf._begin_load_attempt(LoadRequest(model_path = "org/model"), "subject")
    assert attempt.request_id is None
    assert inf._scoped_load_attempts == {}
    inf._finish_load_attempt(attempt)
    assert inf._scoped_load_attempts == {}


def test_old_client_unload_never_takes_the_cancel_path():
    """cancel_load_request_id defaults to None, and the cancel branch is gated
    on it being set, so an old /unload cannot land in the new code at all."""
    request = UnloadRequest(model_path = "org/model")
    assert request.cancel_load_request_id is None
    assert inf._cancel_scoped_load_attempt(request, "subject") == (None, False)
    assert inf._scoped_load_cancel_tombstones == {}


def test_two_old_clients_can_load_concurrently_as_before():
    """Unregistered attempts must not collide with each other: on main two
    loads with no id were serialised by the lifecycle gate, not rejected."""
    first = inf._begin_load_attempt(LoadRequest(model_path = "org/a"), "subject")
    second = inf._begin_load_attempt(LoadRequest(model_path = "org/a"), "subject")
    assert first.token != second.token


# ── H3: the new client's contract ─────────────────────────────────────────────

def test_duplicate_request_id_is_rejected():
    inf._begin_load_attempt(LoadRequest(model_path = "org/a", load_request_id = "r1"), "s")
    with pytest.raises(HTTPException) as excinfo:
        inf._begin_load_attempt(LoadRequest(model_path = "org/a", load_request_id = "r1"), "s")
    assert excinfo.value.status_code == 409


def test_the_same_request_id_is_free_for_a_different_subject():
    """Keyed per subject, so one user cannot deny another user an id."""
    inf._begin_load_attempt(LoadRequest(model_path = "org/a", load_request_id = "r1"), "alice")
    attempt = inf._begin_load_attempt(
        LoadRequest(model_path = "org/a", load_request_id = "r1"), "bob"
    )
    assert attempt.request_id == "r1"


def test_cancel_arriving_before_the_load_is_remembered():
    """A cancel that beats its load leaves a tombstone, and the load then starts
    already-cancelled rather than running to completion unobserved."""
    cancel = UnloadRequest(model_path = "org/a", cancel_load_request_id = "r1")
    assert inf._cancel_scoped_load_attempt(cancel, "s") == (None, False)
    attempt = inf._begin_load_attempt(
        LoadRequest(model_path = "org/a", load_request_id = "r1"), "s"
    )
    assert attempt.cancel_event.is_set()
    assert attempt.cancel_complete.is_set()


def test_a_tombstone_for_another_model_does_not_cancel_this_load():
    inf._cancel_scoped_load_attempt(
        UnloadRequest(model_path = "org/OTHER", cancel_load_request_id = "r1"), "s"
    )
    attempt = inf._begin_load_attempt(
        LoadRequest(model_path = "org/a", load_request_id = "r1"), "s"
    )
    assert not attempt.cancel_event.is_set()


def test_model_path_match_is_case_insensitive_both_ways():
    """Hub ids differ in case between the picker and the load path, and the
    cancel must still bind. Guards the Windows/macOS case-folding story too."""
    inf._begin_load_attempt(
        LoadRequest(model_path = "Org/Model-GGUF", load_request_id = "r1"), "s"
    )
    attempt, _ = inf._cancel_scoped_load_attempt(
        UnloadRequest(model_path = "org/model-gguf", cancel_load_request_id = "r1"), "s"
    )
    assert attempt is not None


def test_cancel_for_a_different_model_is_refused():
    """The whole point of the scoping: a late cancel must never stop a newer
    load of a different model."""
    inf._begin_load_attempt(LoadRequest(model_path = "org/a", load_request_id = "r1"), "s")
    attempt, running = inf._cancel_scoped_load_attempt(
        UnloadRequest(model_path = "org/b", cancel_load_request_id = "r1"), "s"
    )
    assert (attempt, running) == (None, False)


def test_tombstones_expire_and_stay_bounded():
    for index in range(inf._SCOPED_LOAD_CANCEL_TOMBSTONE_LIMIT_PER_SUBJECT + 25):
        inf._cancel_scoped_load_attempt(
            UnloadRequest(model_path = "org/a", cancel_load_request_id = f"r{index}"), "s"
        )
    assert (
        len(inf._scoped_load_cancel_tombstones)
        <= inf._SCOPED_LOAD_CANCEL_TOMBSTONE_LIMIT_PER_SUBJECT
    )
    inf._prune_scoped_load_cancel_tombstones(time.monotonic() + inf._SCOPED_LOAD_CANCEL_TOMBSTONE_TTL_S + 1)
    assert inf._scoped_load_cancel_tombstones == {}


# ── H4: can /load block forever on the cancel handshake ───────────────────────

@pytest.mark.timeout(60)
def test_load_does_not_block_forever_when_the_cancel_never_completes():
    """_run_tracked_load_model_impl waits on cancel_complete in its finally.
    A cancel that sets cancel_event and then dies before setting cancel_complete
    (crashed unload task, client gone, server shutting down) must not strand the
    load thread. An unbounded wait here also blocks interpreter exit, because
    asyncio.to_thread runs on non-daemon executor threads."""
    request = LoadRequest(model_path = "org/a", load_request_id = "r1")

    async def scenario():
        async def fake_impl(*args, **kwargs):
            attempt, _ = inf._cancel_scoped_load_attempt(
                UnloadRequest(model_path = "org/a", cancel_load_request_id = "r1"), "s"
            )
            assert attempt is not None
            # Simulate the teardown handler dying before its finally runs.
            attempt.cancel_complete.clear()
            return {"status": "loaded"}

        inf._load_model_impl_original = inf._load_model_impl
        inf._load_model_impl = fake_impl
        try:
            await asyncio.wait_for(
                inf._run_tracked_load_model_impl(request, None, "s"),
                timeout = 20,
            )
        finally:
            inf._load_model_impl = inf._load_model_impl_original

    asyncio.run(scenario())


# ── H2: a cancel must never unload a model that is already resident ───────────

def test_a_completed_load_is_no_longer_cancellable():
    """Once /load's finally has run, the attempt is deregistered, so a late
    cancel finds nothing and cannot evict the now-resident model."""
    attempt = inf._begin_load_attempt(
        LoadRequest(model_path = "org/a", load_request_id = "r1"), "s"
    )
    inf._finish_load_attempt(attempt)
    found, running = inf._cancel_scoped_load_attempt(
        UnloadRequest(model_path = "org/a", cancel_load_request_id = "r1"), "s"
    )
    assert (found, running) == (None, False)


def test_cancelling_a_queued_attempt_reports_not_running():
    """Only the attempt the dispatcher is actually running gets a backend
    teardown; a queued one just gets its event set."""
    attempt = inf._begin_load_attempt(
        LoadRequest(model_path = "org/a", load_request_id = "r1"), "s"
    )
    found, running = inf._cancel_scoped_load_attempt(
        UnloadRequest(model_path = "org/a", cancel_load_request_id = "r1"), "s"
    )
    assert found is attempt and running is False
    assert attempt.cancel_event.is_set() and attempt.cancel_complete.is_set()


def test_registry_is_thread_safe_under_concurrent_load_and_cancel():
    """Loads and cancels arrive from different request threads; the registry
    must not corrupt or deadlock."""
    errors: list[BaseException] = []

    def loader(index: int):
        try:
            attempt = inf._begin_load_attempt(
                LoadRequest(model_path = "org/a", load_request_id = f"r{index}"), "s"
            )
            time.sleep(0.001)
            inf._finish_load_attempt(attempt)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def canceller(index: int):
        try:
            inf._cancel_scoped_load_attempt(
                UnloadRequest(model_path = "org/a", cancel_load_request_id = f"r{index}"), "s"
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = []
    for index in range(60):
        threads.append(threading.Thread(target = loader, args = (index,)))
        threads.append(threading.Thread(target = canceller, args = (index,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout = 30)
    assert not errors, errors[:3]
    assert all(not thread.is_alive() for thread in threads)


# ── request id validation, since it reaches a dict key ────────────────────────

@pytest.mark.parametrize(
    "value",
    ["../../etc/passwd", "a" * 129, "", "with space", "semi;colon", "nul\x00byte"],
)
def test_hostile_request_ids_are_rejected_by_the_model(value):
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        LoadRequest(model_path = "org/a", load_request_id = value)


@pytest.mark.parametrize("value", ["r1", "A-b_c.1:2", "0", "a" * 128])
def test_reasonable_request_ids_are_accepted(value):
    assert LoadRequest(model_path = "org/a", load_request_id = value).load_request_id == value
