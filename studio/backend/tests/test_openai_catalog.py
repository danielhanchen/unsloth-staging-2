# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GET /v1/models lists the full server catalog (loaded + locally available)."""

import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import routes.inference as inf  # noqa: E402


class _Info:
    def __init__(self, id, display_name, model_id=None):
        self.id = id
        self.display_name = display_name
        self.model_id = model_id


class _FakeLlama:
    is_loaded = True
    model_identifier = "/srv/models/Qwen3-Q4.gguf"
    context_length = 4096
    max_context_length = None
    native_context_length = None


class _FakeUnsloth:
    active_model_name = None
    models: dict = {}
    context_length = None
    max_seq_length = None


def test_catalog_lists_loaded_and_available(monkeypatch):
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _FakeLlama())
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _FakeUnsloth())
    monkeypatch.setattr(
        inf,
        "_cached_local_catalog",
        lambda: [
            _Info("/data/models/Qwen3-Q4.gguf", "Qwen3-Q4"),  # same as loaded -> dedup
            _Info("/data/models/Llama-8B-Q8.gguf", "Llama-8B-Q8"),  # available, not loaded
            _Info("models--org--Foo", "Foo", model_id="org/Foo"),  # hf cache repo id
        ],
    )

    data = inf._openai_catalog_objects()
    ids = {m["id"]: m for m in data}

    # Loaded model is present, marked loaded, and keeps context fields.
    assert ids["Qwen3-Q4"]["loaded"] is True
    assert ids["Qwen3-Q4"]["context_length"] == 4096
    # Available-but-not-loaded models are listed too.
    assert ids["Llama-8B-Q8"]["loaded"] is False
    assert ids["org/Foo"]["loaded"] is False
    # The loaded gguf and the on-disk copy collapse to one clean id.
    assert [m["id"] for m in data].count("Qwen3-Q4") == 1
    # No absolute paths or .gguf suffixes leak anywhere.
    blob = json.dumps(data)
    assert ".gguf" not in blob
    assert "/srv/" not in blob
    assert "/data/" not in blob
