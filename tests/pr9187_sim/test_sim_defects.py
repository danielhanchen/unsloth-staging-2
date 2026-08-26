"""The four defects, each as a scenario that is red before its fix and green after.

F1 and F2 are frontend and live in temp/pr9187_sim/frontend/. F3 and F4 are here.
"""

import asyncio
import threading
import time

import pytest

import _harness as H

pytestmark = pytest.mark.asyncio


# --- F3: server-side media eligibility ----------------------------------------------


@pytest.mark.parametrize("field", ["image_base64", "audio_base64", "video_base64"])
async def test_f3_media_payloads_never_reach_durable_storage(field):
    """A direct caller must not be able to persist a base64 blob into request_json."""
    from fastapi import HTTPException

    from routes import chat_generation_runs as run_routes

    payload = run_routes.CreateChatGenerationRun(
        runId = "run-1", threadId = "thread-1",
        userMessageId = "user-1", assistantMessageId = "assistant-1",
        requestPayload = {
            "model": "local.gguf",
            "messages": [{"role": "user", "content": "describe this"}],
            "stream": True,
            field: "iVBORw0KGgo=" * 200,
        },
    )
    with pytest.raises(HTTPException) as exc:
        run_routes._sanitize_request(payload)
    assert exc.value.status_code == 400
    assert "local inference" in str(exc.value.detail) or "media" in str(exc.value.detail).lower()


# --- F4: bounded shutdown ------------------------------------------------------------


async def test_f4_shutdown_is_bounded_when_a_worker_ignores_cancellation(monkeypatch):
    """A generator parked in a thread cannot be cancelled. Shutdown must still return.

    Without a bound on the post-cancel gather, `await supervisor.stop()` waits forever
    and takes the whole uvicorn shutdown with it.
    """
    from core.inference import chat_generation_runs as cgr
    from routes import inference
    from types import SimpleNamespace

    monkeypatch.setattr(cgr, "_SHUTDOWN_GRACE_SECONDS", 0.3)

    wedged = threading.Event()
    released = threading.Event()

    async def body():
        yield 'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        wedged.set()
        try:
            while not released.is_set():
                await asyncio.sleep(0.05)
        except (asyncio.CancelledError, GeneratorExit):
            # A real engine's teardown drains its subprocess here. If that drain
            # blocks, `await iterator.aclose()` in the producer's finally never
            # returns, and neither does the gather that waits on the task.
            while not released.is_set():
                await asyncio.sleep(0.05)
            raise
        yield "data: [DONE]\n\n"

    async def fake(_payload, _request, _subject, *, cancel_on_disconnect):
        return SimpleNamespace(status_code = 200, body_iterator = body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)

    H.seed_run()
    sup = H.supervisor()
    sup.start("run-1", thread_id = "thread-1", model = "local.gguf")
    assert await asyncio.to_thread(wedged.wait, 10), "the wedged producer never started"

    started = time.monotonic()
    try:
        await asyncio.wait_for(sup.stop(), timeout = 15)
    except asyncio.TimeoutError:
        pytest.fail(
            "supervisor.stop() did not return within 15s with a wedged worker; "
            "the post-cancel gather is unbounded"
        )
    finally:
        released.set()
        await asyncio.sleep(0.2)

    elapsed = time.monotonic() - started
    assert elapsed < 15, f"shutdown took {elapsed:.1f}s"
