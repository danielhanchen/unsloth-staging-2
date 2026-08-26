"""Which requests may become durable, and which must stay on the old stream.

The PR keeps external providers, tool loops, diffusion, audio, incognito, deep research
and the public API on the subscriber-owned path. The frontend enforces that; this asks
whether the *server* does, because a stale tab, a Tauri build mid-update, or a direct API
caller only meets the server.
"""

import pytest
from fastapi import HTTPException

import _harness as H
from routes import chat_generation_runs as run_routes

BASE = {
    "model": "local.gguf",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": True,
}

TINY_B64 = "iVBORw0KGgo="


def _create(**overrides):
    payload = dict(BASE)
    payload.update(overrides)
    return run_routes.CreateChatGenerationRun(
        runId = "run-1", threadId = "thread-1",
        userMessageId = "user-1", assistantMessageId = "assistant-1",
        requestPayload = payload,
    )


def _sanitize(**overrides):
    return run_routes._sanitize_request(_create(**overrides))


# --- must be admitted --------------------------------------------------------------


def test_a_plain_local_text_turn_is_admitted():
    sanitized = _sanitize()
    assert sanitized["stream"] is True
    assert sanitized["cancel_id"] == "run-1"
    assert sanitized["generation_run_id"] == "run-1"


def test_the_server_pins_its_own_fields_over_client_values():
    sanitized = _sanitize(stream = False, thread_id = "somewhere-else", cancel_id = "spoof")
    assert sanitized["stream"] is True
    assert sanitized["thread_id"] == "thread-1"
    assert sanitized["cancel_id"] == "run-1"


def test_benign_key_shaped_argument_names_are_not_credentials():
    """'monkey' contains 'key'. A blunt substring scan would reject this."""
    sanitized = _sanitize(messages = [{"role": "user", "content": "tell me about a monkey"}])
    assert sanitized["messages"][0]["content"] == "tell me about a monkey"


# --- must be refused ---------------------------------------------------------------


@pytest.mark.parametrize("overrides, fragment", [
    ({"enable_tools": True}, "legacy streaming path"),
    ({"tools": [{"type": "function", "function": {"name": "f"}}]}, "legacy streaming path"),
    ({"mcp_enabled": True}, "legacy streaming path"),
    ({"provider_id": "openai"}, "local inference"),
    ({"provider_type": "anthropic"}, "local inference"),
    ({"external_model": "gpt-4o"}, "local inference"),
    ({"encrypted_api_key": "abc"}, "local inference"),
    ({"provider_base_url": "https://example.test"}, "local inference"),
    ({"n": 2}, "n=1"),
    ({"totally_made_up_field": 1}, "Unsupported durable request fields"),
])
def test_ineligible_requests_are_refused(overrides, fragment):
    with pytest.raises(HTTPException) as exc:
        _sanitize(**overrides)
    assert exc.value.status_code in (400, 422)
    assert fragment in str(exc.value.detail)


@pytest.mark.parametrize("field", ["image_base64", "audio_base64", "video_base64"])
def test_media_turns_are_refused(field):
    """The frontend keeps media off the durable path; the server must agree.

    Recovery replays text and reasoning deltas only, and the payload is persisted
    verbatim, so admitting a base64 blob both stores it forever and enters a replay
    path nothing was designed for.
    """
    with pytest.raises(HTTPException) as exc:
        _sanitize(**{field: TINY_B64})
    assert exc.value.status_code == 400


def test_a_credential_in_the_envelope_is_refused():
    with pytest.raises(HTTPException) as exc:
        _sanitize(messages = [{"role": "user", "content": "hi", "api_key": "sk-secret"}])
    assert "Credentials cannot be persisted" in str(exc.value.detail)


def test_message_text_is_data_not_configuration():
    """A user may legitimately paste the words 'api_key' into a prompt."""
    sanitized = _sanitize(messages = [{"role": "user", "content": "what is an api_key?"}])
    assert "api_key" in sanitized["messages"][0]["content"]


# --- the public path must not become durable ---------------------------------------


@pytest.mark.asyncio
async def test_public_chat_completions_keeps_cancel_on_disconnect(monkeypatch):
    from models.inference import ChatCompletionRequest
    from routes import inference

    seen = []

    async def fake(_payload, _request, _subject, *, cancel_on_disconnect):
        seen.append(cancel_on_disconnect)
        return "ok"

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    payload = ChatCompletionRequest(model = "local", messages = [{"role": "user", "content": "x"}])
    assert await inference.openai_chat_completions(payload, object(), H.OWNER) == "ok"
    assert seen == [True], "the public route stopped cancelling on disconnect"


@pytest.mark.asyncio
async def test_the_durable_producer_disables_cancel_on_disconnect(monkeypatch):
    H.seed_run()
    fake = H.script(monkeypatch, [H.text_chunk("a"), H.stop_chunk()])
    await H.supervisor()._produce("run-1")
    assert fake.cancel_on_disconnect is False, "the durable run would die with its subscriber"


@pytest.mark.asyncio
async def test_no_durable_row_exists_for_a_public_request(monkeypatch):
    from models.inference import ChatCompletionRequest
    from routes import inference

    H.seed_thread()

    async def fake(_payload, _request, _subject, *, cancel_on_disconnect):
        return "ok"

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    await inference.openai_chat_completions(
        ChatCompletionRequest(model = "local", messages = [{"role": "user", "content": "x"}]),
        object(), H.OWNER,
    )
    assert H.raw_query("SELECT id FROM chat_generation_runs") == []


def test_the_disconnect_policy_wrapper_still_looks_like_a_request():
    """Every downstream caller reads attributes off this proxy."""
    from types import SimpleNamespace

    from routes import inference

    real = SimpleNamespace(
        scope = {"type": "http"}, state = SimpleNamespace(x = 1),
        method = "POST", url = "http://test/x", app = SimpleNamespace(),
    )
    wrapped = inference._DisconnectPolicyRequest(real, cancel_on_disconnect = False)
    assert wrapped.scope == {"type": "http"}
    assert wrapped.state.x == 1
    assert wrapped.method == "POST"
    assert wrapped.app is real.app
