import os
import sys
import threading

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.join(_HERE, "studio", "backend")
if not os.path.isdir(_BACKEND):
    _BACKEND = os.path.join(_HERE, "..", "studio", "backend")
sys.path.insert(0, _BACKEND)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from routes.inference import _CANCEL_REGISTRY, _TrackedCancel, router


@pytest.fixture()
def client():
    _CANCEL_REGISTRY.clear()
    app = FastAPI()
    app.include_router(router, prefix="/inference")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    return TestClient(app)


def test_list_session_id_returns_zero_not_500(client):
    ev = threading.Event()
    with _TrackedCancel(ev, "sess-live"):
        r = client.post(
            "/inference/cancel",
            json={"session_id": ["not", "a", "string"]},
        )
    assert r.status_code == 200
    assert r.json() == {"cancelled": 0}
    assert not ev.is_set()


def test_integer_session_id_returns_zero(client):
    ev = threading.Event()
    with _TrackedCancel(ev, "sess-live"):
        r = client.post("/inference/cancel", json={"session_id": 12345})
    assert r.status_code == 200
    assert r.json() == {"cancelled": 0}
    assert not ev.is_set()


def test_null_session_id_returns_zero(client):
    ev = threading.Event()
    with _TrackedCancel(ev, "sess-live"):
        r = client.post("/inference/cancel", json={"session_id": None})
    assert r.status_code == 200
    assert r.json() == {"cancelled": 0}
    assert not ev.is_set()


def test_empty_string_session_id_returns_zero(client):
    ev = threading.Event()
    with _TrackedCancel(ev, "sess-live"):
        r = client.post("/inference/cancel", json={"session_id": ""})
    assert r.status_code == 200
    assert r.json() == {"cancelled": 0}
    assert not ev.is_set()


def test_dict_session_id_returns_zero(client):
    ev = threading.Event()
    with _TrackedCancel(ev, "sess-live"):
        r = client.post(
            "/inference/cancel",
            json={"session_id": {"nested": "dict"}},
        )
    assert r.status_code == 200
    assert r.json() == {"cancelled": 0}
    assert not ev.is_set()


def test_mixed_valid_and_invalid_keys_cancels_only_valid(client):
    ev = threading.Event()
    with _TrackedCancel(ev, "sess-real", "cmpl-real"):
        r = client.post(
            "/inference/cancel",
            json={"session_id": 42, "completion_id": "cmpl-real"},
        )
    assert r.status_code == 200
    assert r.json() == {"cancelled": 1}
    assert ev.is_set()


def test_json_array_body_treated_as_no_keys(client):
    ev = threading.Event()
    with _TrackedCancel(ev, "sess-live"):
        r = client.post("/inference/cancel", json=["session_id", "sess-live"])
    assert r.status_code == 200
    assert r.json() == {"cancelled": 0}
    assert not ev.is_set()
