import os
import sys
import threading
from pathlib import Path

import pytest

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from routes.inference import _CANCEL_REGISTRY, _TrackedCancel, _cancel_by_keys, router


@pytest.fixture()
def client():
    _CANCEL_REGISTRY.clear()
    app = FastAPI()
    app.include_router(router, prefix="/inference")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_registry():
    _CANCEL_REGISTRY.clear()
    yield
    _CANCEL_REGISTRY.clear()


def test_tracked_cancel_same_session_concurrent_requests_coexist():
    ev1, ev2 = threading.Event(), threading.Event()
    with _TrackedCancel(ev1, "thread-A"):
        with _TrackedCancel(ev2, "thread-A"):
            bucket = _CANCEL_REGISTRY.get("thread-A")
            assert bucket is not None
            assert ev1 in bucket and ev2 in bucket
            assert _cancel_by_keys(["thread-A"]) == 2
            assert ev1.is_set() and ev2.is_set()


def test_tracked_cancel_exit_preserves_siblings_and_cleans_empty_bucket():
    ev1, ev2 = threading.Event(), threading.Event()
    t1 = _TrackedCancel(ev1, "thread-B")
    t2 = _TrackedCancel(ev2, "thread-B")
    t1.__enter__()
    t2.__enter__()
    t1.__exit__(None, None, None)
    assert _CANCEL_REGISTRY["thread-B"] == {ev2}
    t2.__exit__(None, None, None)
    assert "thread-B" not in _CANCEL_REGISTRY


def test_tracked_cancel_filters_falsy_keys():
    ev = threading.Event()
    with _TrackedCancel(ev, None, "", "real-key"):
        assert list(_CANCEL_REGISTRY.keys()) == ["real-key"]


def test_cancel_by_keys_empty_list_does_not_fan_out():
    ev1, ev2 = threading.Event(), threading.Event()
    with _TrackedCancel(ev1, "sess-A"):
        with _TrackedCancel(ev2, "sess-B"):
            assert _cancel_by_keys([]) == 0
            assert not ev1.is_set() and not ev2.is_set()


def test_cancel_by_keys_dedups_same_event_under_multiple_keys():
    ev = threading.Event()
    with _TrackedCancel(ev, "sess-x", "cmpl-x"):
        assert _cancel_by_keys(["sess-x", "cmpl-x"]) == 1
    assert ev.is_set()


def test_cancel_endpoint_empty_body_returns_zero(client):
    ev = threading.Event()
    with _TrackedCancel(ev, "sess-live"):
        r = client.post("/inference/cancel", json={})
    assert r.status_code == 200
    assert r.json() == {"cancelled": 0}
    assert not ev.is_set()


@pytest.mark.parametrize(
    "value",
    [["not", "a", "string"], 12345, None, "", {"nested": "dict"}],
)
def test_cancel_endpoint_non_string_session_id_returns_zero(client, value):
    ev = threading.Event()
    with _TrackedCancel(ev, "sess-live"):
        r = client.post("/inference/cancel", json={"session_id": value})
    assert r.status_code == 200
    assert r.json() == {"cancelled": 0}
    assert not ev.is_set()


def test_cancel_endpoint_mixes_valid_and_invalid_keys(client):
    ev = threading.Event()
    with _TrackedCancel(ev, "sess-real", "cmpl-real"):
        r = client.post(
            "/inference/cancel",
            json={"session_id": 42, "completion_id": "cmpl-real"},
        )
    assert r.json() == {"cancelled": 1}
    assert ev.is_set()


def test_cancel_endpoint_array_body_treated_as_no_keys(client):
    ev = threading.Event()
    with _TrackedCancel(ev, "sess-live"):
        r = client.post("/inference/cancel", json=["session_id", "sess-live"])
    assert r.json() == {"cancelled": 0}
    assert not ev.is_set()


def test_cancel_endpoint_by_session_id_cancels_event(client):
    ev = threading.Event()
    with _TrackedCancel(ev, "sess-x", "cmpl-x"):
        r = client.post("/inference/cancel", json={"session_id": "sess-x"})
    assert r.json() == {"cancelled": 1}
    assert ev.is_set()


def test_cancel_endpoint_unknown_key_returns_zero(client):
    ev = threading.Event()
    with _TrackedCancel(ev, "sess-known"):
        r = client.post("/inference/cancel", json={"session_id": "sess-missing"})
    assert r.json() == {"cancelled": 0}
    assert not ev.is_set()


def test_llama_cpp_t_max_predict_ms_always_gated_on_max_tokens():
    src_path = _backend_root / "core" / "inference" / "llama_cpp.py"
    lines = src_path.read_text().splitlines()
    assign_lines = [
        i
        for i, line in enumerate(lines)
        if 't_max_predict_ms"]' in line
        and "=" in line
        and "DEFAULT_T_MAX_PREDICT_MS" in line
    ]
    assert len(assign_lines) >= 3, f"expected >=3 sites, got {len(assign_lines)}"
    for i in assign_lines:
        prev = None
        for j in range(i - 1, -1, -1):
            stripped = lines[j].strip()
            if not stripped or stripped.startswith("#"):
                continue
            prev = stripped
            break
        assert prev == "if max_tokens is None:", (
            f"line {i + 1}: t_max_predict_ms not guarded by 'if max_tokens is None:'"
        )
