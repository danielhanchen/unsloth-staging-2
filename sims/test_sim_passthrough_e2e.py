# Simulation suite 2: drive Studio's REAL passthrough functions against a
# scripted fake llama-server over HTTP (CPU-only, no GPU, OS-portable).
# Covers the overflow retry loop (streaming + non-streaming), default vs
# opt-in vs env-var policy, non-context errors, retry budgets, and a fuzz
# pass over the truncation invariants. Requires the full backend deps
# (fastapi/pydantic/httpx), i.e. the CI-recipe environment.

from __future__ import annotations

import asyncio
import http.server
import json
import os
import random
import sys
import threading
import types
from pathlib import Path

import pytest

_THIS = Path(__file__).resolve()
_BACKEND_DIR = os.environ.get("STUDIO_BACKEND_DIR")
if not _BACKEND_DIR:
    for cand in [_THIS.parents[2] / "unsloth_main" / "studio" / "backend",
                 _THIS.parents[1] / "studio" / "backend",
                 _THIS.parents[2] / "studio" / "backend"]:
        if cand.is_dir():
            _BACKEND_DIR = str(cand)
            break
assert _BACKEND_DIR, "set STUDIO_BACKEND_DIR"
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastapi import HTTPException

from models.inference import ChatCompletionRequest
from routes.inference import (
    _apply_overflow_truncation,
    _CLIP_MARKER,
    _estimate_message_tokens,
    _openai_passthrough_non_streaming,
    _openai_passthrough_stream,
    _truncate_middle_messages,
)


# ---------------------------------------------------------------------------
# Scripted fake llama-server
# ---------------------------------------------------------------------------


def _overflow_body(n_prompt=70494, n_ctx=24576) -> bytes:
    return json.dumps({"error": {
        "code": 400,
        "message": f"request ({n_prompt} tokens) exceeds the available context "
                   f"size ({n_ctx} tokens), try increasing it",
        "type": "exceed_context_size_error",
        "n_prompt_tokens": n_prompt, "n_ctx": n_ctx}}).encode()


_OK_JSON = json.dumps({
    "id": "chatcmpl-fake", "object": "chat.completion", "model": "fake",
    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "c9", "type": "function", "function": {
            "name": "bash", "arguments": "{\"command\":\"ls\"}"}}]}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}).encode()

_OK_SSE = (
    b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},'
    b'"finish_reason":null}],"object":"chat.completion.chunk"}\n\n'
    b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
    b'"object":"chat.completion.chunk"}\n\n'
    b"data: [DONE]\n\n"
)


class _FakeLlama(http.server.BaseHTTPRequestHandler):
    """Scripted upstream: a queue of responses, records request bodies."""
    script: list = []
    seen_bodies: list = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        type(self).seen_bodies.append(body)
        step = self.script.pop(0) if self.script else ("json", 200, _OK_JSON)
        kind, status, payload = step
        self.send_response(status)
        if kind == "sse":
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

    def log_message(self, *a):
        pass


@pytest.fixture()
def fake_llama():
    _FakeLlama.script = []
    _FakeLlama.seen_bodies = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeLlama)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()


def _backend(srv):
    return types.SimpleNamespace(
        base_url=f"http://127.0.0.1:{srv.server_address[1]}",
        context_length=24576,
    )


def _payload(n_tool_turns=10, result_chars=3000, stream=False, **extra):
    msgs = [
        {"role": "system", "content": "You are an agent." * 10},
        {"role": "user", "content": "Do the task." * 10},
    ]
    for i in range(n_tool_turns):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": f"c{i}", "type": "function",
            "function": {"name": "read", "arguments": f'{{"f":"/x{i}"}}'}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "content": "y" * result_chars})
    msgs.append({"role": "user", "content": "continue"})
    return ChatCompletionRequest(
        messages=msgs,
        tools=[{"type": "function", "function": {"name": "read", "parameters": {}}}],
        stream=stream,
        max_tokens=32000,
        **extra,
    )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Non-streaming retry behavior
# ---------------------------------------------------------------------------


def test_default_mode_surfaces_400_without_retry(fake_llama, monkeypatch):
    monkeypatch.delenv("UNSLOTH_CONTEXT_OVERFLOW", raising=False)
    _FakeLlama.script = [("json", 400, _overflow_body())]
    with pytest.raises(HTTPException) as ei:
        _run(_openai_passthrough_non_streaming(_backend(fake_llama), _payload(), "m"))
    assert ei.value.status_code == 400
    assert "context_length_exceeded" in json.dumps(ei.value.detail)
    assert len(_FakeLlama.seen_bodies) == 1  # no retry in default mode


def test_opt_in_truncates_and_retries_to_success(fake_llama, monkeypatch):
    monkeypatch.delenv("UNSLOTH_CONTEXT_OVERFLOW", raising=False)
    _FakeLlama.script = [("json", 400, _overflow_body()), ("json", 200, _OK_JSON)]
    resp = _run(_openai_passthrough_non_streaming(
        _backend(fake_llama), _payload(context_overflow="truncate_middle"), "m"))
    assert resp.status_code == 200
    assert len(_FakeLlama.seen_bodies) == 2
    first, second = _FakeLlama.seen_bodies
    assert len(second["messages"]) < len(first["messages"])
    assert second["max_tokens"] <= max(1024, int(24576 * 0.25))
    # System prompt and newest message survive.
    assert second["messages"][0]["role"] == "system"
    assert second["messages"][-1] == first["messages"][-1]
    # Tool pairing intact after truncation.
    ids = {tc["id"] for m in second["messages"] if m.get("role") == "assistant"
           for tc in (m.get("tool_calls") or [])}
    for m in second["messages"]:
        if m.get("role") == "tool":
            assert m["tool_call_id"] in ids


def test_env_var_default_enables_without_request_field(fake_llama, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CONTEXT_OVERFLOW", "truncate_middle")
    _FakeLlama.script = [("json", 400, _overflow_body()), ("json", 200, _OK_JSON)]
    resp = _run(_openai_passthrough_non_streaming(_backend(fake_llama), _payload(), "m"))
    assert resp.status_code == 200
    assert len(_FakeLlama.seen_bodies) == 2


def test_explicit_error_beats_env_var(fake_llama, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CONTEXT_OVERFLOW", "truncate_middle")
    _FakeLlama.script = [("json", 400, _overflow_body())]
    with pytest.raises(HTTPException):
        _run(_openai_passthrough_non_streaming(
            _backend(fake_llama), _payload(context_overflow="error"), "m"))
    assert len(_FakeLlama.seen_bodies) == 1


def test_non_context_400_is_never_retried(fake_llama, monkeypatch):
    monkeypatch.delenv("UNSLOTH_CONTEXT_OVERFLOW", raising=False)
    bad = json.dumps({"error": {"code": 400, "message": "invalid tool schema",
                                "type": "invalid_request_error"}}).encode()
    _FakeLlama.script = [("json", 400, bad)]
    with pytest.raises(HTTPException) as ei:
        _run(_openai_passthrough_non_streaming(
            _backend(fake_llama), _payload(context_overflow="truncate_middle"), "m"))
    assert ei.value.status_code == 400
    assert len(_FakeLlama.seen_bodies) == 1


def test_500_is_never_retried(fake_llama, monkeypatch):
    monkeypatch.delenv("UNSLOTH_CONTEXT_OVERFLOW", raising=False)
    _FakeLlama.script = [("json", 500, b'{"error":"boom"}')]
    with pytest.raises(HTTPException):
        _run(_openai_passthrough_non_streaming(
            _backend(fake_llama), _payload(context_overflow="truncate_middle"), "m"))
    assert len(_FakeLlama.seen_bodies) == 1


def test_persistent_overflow_exhausts_bounded_budget(fake_llama, monkeypatch):
    monkeypatch.delenv("UNSLOTH_CONTEXT_OVERFLOW", raising=False)
    _FakeLlama.script = [("json", 400, _overflow_body()) for _ in range(10)]
    with pytest.raises(HTTPException) as ei:
        _run(_openai_passthrough_non_streaming(
            _backend(fake_llama), _payload(context_overflow="truncate_middle"), "m"))
    assert ei.value.status_code == 400
    # 1 initial + at most 3 retries; must terminate, never loop.
    assert 2 <= len(_FakeLlama.seen_bodies) <= 4


def test_two_overflows_then_success(fake_llama, monkeypatch):
    monkeypatch.delenv("UNSLOTH_CONTEXT_OVERFLOW", raising=False)
    _FakeLlama.script = [("json", 400, _overflow_body()),
                         ("json", 400, _overflow_body(n_prompt=30000)),
                         ("json", 200, _OK_JSON)]
    resp = _run(_openai_passthrough_non_streaming(
        _backend(fake_llama), _payload(context_overflow="truncate_middle"), "m"))
    assert resp.status_code == 200
    assert len(_FakeLlama.seen_bodies) == 3


def test_empty_tool_content_flows_through_passthrough(fake_llama, monkeypatch):
    """F5 end to end: empty tool results validate and reach the upstream."""
    monkeypatch.delenv("UNSLOTH_CONTEXT_OVERFLOW", raising=False)
    _FakeLlama.script = [("json", 200, _OK_JSON)]
    payload = ChatCompletionRequest(
        messages=[
            {"role": "user", "content": "t"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": ""},
        ],
        tools=[{"type": "function", "function": {"name": "bash", "parameters": {}}}],
        max_tokens=128,
    )
    resp = _run(_openai_passthrough_non_streaming(_backend(fake_llama), payload, "m"))
    assert resp.status_code == 200
    sent = _FakeLlama.seen_bodies[0]["messages"]
    assert any(m.get("role") == "tool" and m.get("content") == "" for m in sent)


# ---------------------------------------------------------------------------
# Streaming retry behavior
# ---------------------------------------------------------------------------


class _FakeRequest:
    async def is_disconnected(self):
        return False


async def _consume_stream(resp):
    chunks = []
    async for chunk in resp.body_iterator:
        chunks.append(chunk if isinstance(chunk, (bytes, bytearray)) else chunk.encode())
    return b"".join(chunks)


def test_stream_overflow_retry_then_relay(fake_llama, monkeypatch):
    monkeypatch.delenv("UNSLOTH_CONTEXT_OVERFLOW", raising=False)
    _FakeLlama.script = [("json", 400, _overflow_body()), ("sse", 200, _OK_SSE)]

    async def go():
        resp = await _openai_passthrough_stream(
            _FakeRequest(), asyncio.Event(), _backend(fake_llama),
            _payload(stream=True, context_overflow="truncate_middle"), "m", "cmpl-x")
        return await _consume_stream(resp)

    body = _run(go())
    assert b'"content":"hi"' in body and b"[DONE]" in body
    assert len(_FakeLlama.seen_bodies) == 2


def test_stream_default_mode_raises_mapped_400(fake_llama, monkeypatch):
    monkeypatch.delenv("UNSLOTH_CONTEXT_OVERFLOW", raising=False)
    _FakeLlama.script = [("json", 400, _overflow_body())]

    async def go():
        await _openai_passthrough_stream(
            _FakeRequest(), asyncio.Event(), _backend(fake_llama),
            _payload(stream=True), "m", "cmpl-x")

    with pytest.raises(HTTPException) as ei:
        _run(go())
    assert "context_length_exceeded" in json.dumps(ei.value.detail)
    assert len(_FakeLlama.seen_bodies) == 1


def test_stream_persistent_overflow_bounded(fake_llama, monkeypatch):
    monkeypatch.delenv("UNSLOTH_CONTEXT_OVERFLOW", raising=False)
    _FakeLlama.script = [("json", 400, _overflow_body()) for _ in range(10)]

    async def go():
        await _openai_passthrough_stream(
            _FakeRequest(), asyncio.Event(), _backend(fake_llama),
            _payload(stream=True, context_overflow="truncate_middle"), "m", "cmpl-x")

    with pytest.raises(HTTPException):
        _run(go())
    assert 2 <= len(_FakeLlama.seen_bodies) <= 4


# ---------------------------------------------------------------------------
# Truncation fuzz: invariants over randomized conversations
# ---------------------------------------------------------------------------


def _random_conversation(rng):
    msgs = []
    if rng.random() < 0.9:
        msgs.append({"role": "system", "content": "s" * rng.randint(1, 2000)})
    msgs.append({"role": "user", "content": "u" * rng.randint(1, 4000)})
    for i in range(rng.randint(0, 30)):
        kind = rng.random()
        if kind < 0.5:
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{
                "id": f"r{i}", "type": "function",
                "function": {"name": "t", "arguments": "{}"}}]})
            for _ in range(rng.randint(1, 3)):
                msgs.append({"role": "tool", "tool_call_id": f"r{i}",
                             "content": "x" * rng.randint(0, 20000)})
        elif kind < 0.7:
            msgs.append({"role": "user", "content": [
                {"type": "text", "text": "m" * rng.randint(1, 3000)}]})
        elif kind < 0.9:
            msgs.append({"role": "assistant", "content": "a" * rng.randint(1, 5000)})
        else:
            msgs.append({"role": "tool", "tool_call_id": "orphan",
                         "content": "o" * rng.randint(0, 5000)})
    msgs.append({"role": "user", "content": "final"})
    return msgs


@pytest.mark.parametrize("seed", range(60))
def test_truncation_fuzz_invariants(seed):
    rng = random.Random(seed)
    msgs = _random_conversation(rng)
    keep = rng.choice([0.1, 0.3, 0.5, 0.75, 0.9])
    new, dropped = _truncate_middle_messages([dict(m) for m in msgs], keep)
    assert dropped >= 0 and len(new) == len(msgs) - dropped
    if msgs[0]["role"] == "system":
        assert new[0] == msgs[0]
    assert new[-1] == msgs[-1]
    # No orphaned tool results among messages that had a paired call.
    ids = {tc["id"] for m in new if m.get("role") == "assistant"
           for tc in (m.get("tool_calls") or [])}
    for m in new:
        if m.get("role") == "tool" and m.get("tool_call_id", "").startswith("r"):
            assert m["tool_call_id"] in ids
    if dropped:
        assert sum(_estimate_message_tokens(m) for m in new) < \
            sum(_estimate_message_tokens(m) for m in msgs)


@pytest.mark.parametrize("seed", range(30))
def test_apply_overflow_terminates_under_repeated_use(seed):
    """Simulate the retry loop's worst case: apply until no progress; must
    terminate quickly and never raise, including multimodal contents."""
    rng = random.Random(1000 + seed)
    body = {"messages": _random_conversation(rng), "max_tokens": 32000}
    err = _overflow_body().decode()
    for _ in range(10):
        if not _apply_overflow_truncation(body, err):
            break
    else:
        pytest.fail("truncation made progress forever; retry loop would spin")
    # Structure still valid: roles only from the known set, final msg present.
    assert body["messages"][-1]["content"] == "final"
    assert all(m.get("role") in ("system", "user", "assistant", "tool")
               for m in body["messages"])


def test_clip_skips_multimodal_list_content():
    msgs = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1",
         "content": [{"type": "text", "text": "z" * 50000}]},
        {"role": "user", "content": "final"},
    ]
    body = {"messages": msgs, "max_tokens": 32000}
    # Must not raise on list contents; may or may not reduce.
    _apply_overflow_truncation(body, _overflow_body().decode())
    assert isinstance(body["messages"][2]["content"], list)  # untouched shape
