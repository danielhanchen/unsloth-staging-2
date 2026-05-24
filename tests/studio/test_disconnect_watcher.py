"""Behavioral unit tests for ``_await_disconnect_then_close``.

PR #5749 added this watcher to close the upstream httpx response when the
Starlette client disconnects mid-prefill. The review found that closing
the response without first setting ``cancel_event`` causes the streamer's
``except (RemoteProtocolError, ReadError, CloseError)`` clause to re-raise
(it only suppresses when ``cancel_event.is_set()``), which leaks a synthetic
``data: {"error": ...}`` SSE chunk in the OpenAI path. These tests lock in
the fix: ``cancel_event`` MUST be set before ``resp.aclose()`` runs.

AST-loads the function in isolation (no full-module import) so the test
stays light and matches the project's existing pattern in
``test_stream_cancel_registration_timing.py`` / ``test_cancel_atomicity.py``.
"""

from __future__ import annotations

import ast
import asyncio
import threading
from pathlib import Path

import pytest

_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "backend"
    / "routes"
    / "inference.py"
)


def _load_disconnect_helper():
    src = _SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_await_disconnect_then_close"
    )
    body = ast.get_source_segment(src, fn)
    assert body is not None
    ns: dict = {}

    class _Logger:
        def debug(self, *a, **k):
            pass

    preamble = "import asyncio\n"
    exec(preamble + body, ns)  # noqa: S102 - controlled exec on first-party source
    ns["logger"] = _Logger()
    return ns["_await_disconnect_then_close"], ns


class _FakeRequest:
    """Yields a sequence of is_disconnected() results, then True forever."""

    def __init__(self, sequence):
        self._seq = list(sequence)
        self.calls = 0

    async def is_disconnected(self):
        self.calls += 1
        return self._seq.pop(0) if self._seq else True


class _FakeResp:
    """Tracks aclose() ordering vs. a cancel_event."""

    def __init__(self, cancel_event=None):
        self.closed = False
        self.close_calls = 0
        self.cancel_event_was_set_when_closed = None
        self._cancel_event = cancel_event

    async def aclose(self):
        self.close_calls += 1
        self.closed = True
        if self._cancel_event is not None:
            self.cancel_event_was_set_when_closed = self._cancel_event.is_set()


def test_signature_accepts_cancel_event():
    """Function must accept a cancel_event positional/keyword."""

    fn, _ = _load_disconnect_helper()
    import inspect

    params = list(inspect.signature(fn).parameters)
    assert params[:3] == ["request", "resp", "cancel_event"], (
        "Expected signature (request, resp, cancel_event); got "
        f"{params!r}. The disconnect watcher must accept cancel_event so it "
        "can signal cancellation before closing the upstream response."
    )


def test_watcher_closes_response_after_disconnect():
    fn, _ = _load_disconnect_helper()

    async def _run():
        req = _FakeRequest([False, False, True])
        resp = _FakeResp()
        cancel_event = threading.Event()
        await asyncio.wait_for(fn(req, resp, cancel_event), timeout=2.0)
        assert resp.close_calls == 1
        assert req.calls >= 3

    asyncio.run(_run())


def test_watcher_sets_cancel_event_before_closing_response():
    """Regression lock for PR 5749 code-review fix #1.

    The streamer's ``except (RemoteProtocolError, ReadError, CloseError)``
    clause only suppresses when ``cancel_event.is_set()``. If the watcher
    closes ``resp`` without first setting ``cancel_event``, the resulting
    ``RemoteProtocolError`` re-raises and the OpenAI passthrough emits a
    synthetic ``data: {"error": ...}`` chunk to an already-gone client.
    """

    fn, _ = _load_disconnect_helper()

    async def _run():
        req = _FakeRequest([True])
        cancel_event = threading.Event()
        resp = _FakeResp(cancel_event=cancel_event)
        await asyncio.wait_for(fn(req, resp, cancel_event), timeout=2.0)
        assert resp.close_calls == 1
        assert cancel_event.is_set(), "cancel_event must be set after disconnect"
        assert resp.cancel_event_was_set_when_closed is True, (
            "cancel_event must be set BEFORE resp.aclose() so the streamer's "
            "RemoteProtocolError handler treats the close as cancellation, "
            "not as an upstream error worth re-raising."
        )

    asyncio.run(_run())


def test_watcher_handles_aclose_failure_quietly():
    """aclose() raising should not propagate out of the watcher."""

    fn, _ = _load_disconnect_helper()

    class _BoomResp:
        async def aclose(self):
            raise RuntimeError("simulated transport teardown race")

    async def _run():
        req = _FakeRequest([True])
        cancel_event = threading.Event()
        # Must not raise: silent-but-logged contract preserved.
        await asyncio.wait_for(fn(req, _BoomResp(), cancel_event), timeout=2.0)
        assert cancel_event.is_set()

    asyncio.run(_run())


def test_watcher_handles_cancellation():
    """CancelledError from the surrounding task must exit cleanly."""

    fn, _ = _load_disconnect_helper()

    async def _run():
        # request.is_disconnected() never returns True; we cancel the watcher.
        req = _FakeRequest([False] * 1000)
        resp = _FakeResp()
        cancel_event = threading.Event()
        task = asyncio.create_task(fn(req, resp, cancel_event))
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pytest.fail("watcher must absorb CancelledError, not re-raise it")
        assert resp.close_calls == 0
        assert not cancel_event.is_set()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# AST-level checks: the watcher must be wired into both passthrough streamers
# with cancel_event passed through, and the streamers must remain the only
# call sites. Catches regressions where someone adds a third unsynced call.
# ---------------------------------------------------------------------------


def test_both_passthrough_streamers_pass_cancel_event_to_watcher():
    src = _SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(src)

    call_sites = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_await_disconnect_then_close"
        ):
            call_sites.append(node)

    assert len(call_sites) >= 2, (
        f"Expected at least 2 call sites for _await_disconnect_then_close "
        f"(anthropic + openai passthrough); found {len(call_sites)}."
    )

    for call in call_sites:
        # Must pass exactly (request, resp, cancel_event) positionally; the
        # cancel_event arg is the regression we care about.
        arg_names = [
            a.id if isinstance(a, ast.Name) else ast.unparse(a) for a in call.args
        ]
        assert "cancel_event" in arg_names, (
            f"Call site at line {call.lineno} does not pass cancel_event; "
            f"args = {arg_names}. Without cancel_event the watcher cannot "
            "signal cancellation before closing resp."
        )
