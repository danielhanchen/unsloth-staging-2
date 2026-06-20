# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Idle-TTL settings + backend activity tracking that drive auto-eviction."""

import sys
import time
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.model_ttl_settings as ttl  # noqa: E402


def test_coerce_bounds_and_types():
    assert ttl._coerce_ttl_seconds(60) == 60
    assert ttl._coerce_ttl_seconds("120") == 120
    assert ttl._coerce_ttl_seconds(0) == 0
    assert ttl._coerce_ttl_seconds(-5) is None
    assert ttl._coerce_ttl_seconds(ttl.MAX_MODEL_IDLE_TTL_SECONDS + 1) is None
    assert ttl._coerce_ttl_seconds(True) is None
    assert ttl._coerce_ttl_seconds("abc") is None


def test_validate_raises_on_invalid():
    with pytest.raises(ValueError):
        ttl.validate_model_idle_ttl_seconds(-1)
    assert ttl.validate_model_idle_ttl_seconds(300) == 300


def test_default_from_env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MODEL_IDLE_TTL", raising = False)
    assert ttl.default_model_idle_ttl_seconds() == 0  # disabled by default
    monkeypatch.setenv("UNSLOTH_MODEL_IDLE_TTL", "900")
    assert ttl.default_model_idle_ttl_seconds() == 900
    monkeypatch.setenv("UNSLOTH_MODEL_IDLE_TTL", "bad")
    assert ttl.default_model_idle_ttl_seconds() == 0


def test_get_set_roundtrip(monkeypatch):
    store: dict = {}
    monkeypatch.setattr("storage.studio_db.get_app_setting", lambda k, d = None: store.get(k, d))
    monkeypatch.setattr("storage.studio_db.upsert_app_settings", lambda d: store.update(d))
    monkeypatch.delenv("UNSLOTH_MODEL_IDLE_TTL", raising = False)

    assert ttl.get_model_idle_ttl_seconds() == 0  # default disabled
    assert ttl.set_model_idle_ttl_seconds(600) == 600
    assert store[ttl.MODEL_IDLE_TTL_SETTING_KEY] == 600
    assert ttl.get_model_idle_ttl_seconds() == 600


def test_backend_idle_seconds_and_activity(monkeypatch):
    from routes.inference import get_llama_cpp_backend

    backend = get_llama_cpp_backend()

    # No model loaded -> idle is undefined (None), so the evictor never fires.
    monkeypatch.setattr(type(backend), "is_loaded", property(lambda self: False))
    assert backend.idle_seconds is None

    # Loaded -> idle measured from the last activity timestamp.
    monkeypatch.setattr(type(backend), "is_loaded", property(lambda self: True))
    backend._last_activity = time.monotonic() - 5.0
    assert backend.idle_seconds >= 4.0
    backend.note_activity()
    assert backend.idle_seconds < 1.0
