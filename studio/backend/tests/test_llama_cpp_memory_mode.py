# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for GGUF memory placement mode and explicit GPU selection (#7164)."""

from __future__ import annotations

import struct
import subprocess
import sys
import types as _types
from pathlib import Path
from unittest.mock import patch

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

_httpx_stub = _types.ModuleType("httpx")
for _exc_name in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))
_httpx_stub.Timeout = type("T", (), {"__init__": lambda s, *a, **k: None})
_httpx_stub.Client = type(
    "Client",
    (),
    {
        "__init__": lambda self, **kw: None,
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: None,
    },
)
sys.modules.setdefault("httpx", _httpx_stub)

import pytest

from core.inference.llama_cpp import LlamaCppBackend

_GGUF_MAGIC = 0x46554747
_VTYPE_STRING = 8


def _enc_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _enc_kv_string(key: str, value: str) -> bytes:
    return _enc_string(key) + struct.pack("<I", _VTYPE_STRING) + _enc_string(value)


def _write_minimal_gguf(path: Path, *, arch: str = "llama") -> Path:
    body = _enc_kv_string("general.architecture", arch)
    header = struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, 1)
    path.write_bytes(header + body)
    return path


class _FakeProcess:
    """Minimal stand-in so is_loaded returns True."""

    def terminate(self):
        pass

    def wait(self, timeout = None):
        return 0

    def kill(self):
        pass

    def poll(self):
        return 0


def _loaded_backend(**overrides):
    backend = LlamaCppBackend()
    backend._process = _FakeProcess()
    backend._healthy = True
    backend._model_identifier = "owner/repo"
    backend._hf_variant = "Q4_K_M"
    backend._requested_n_ctx = 8192
    backend._cache_type_kv = None
    backend._speculative_type = None
    backend._requested_spec_mode = "auto"
    backend._chat_template_override = None
    backend._is_vision = False
    backend._extra_args = None
    backend._extra_args_source = None
    backend._gguf_path = None
    backend._gpu_ids = None
    backend._memory_mode = None
    for key, value in overrides.items():
        setattr(backend, key, value)
    return backend


# ── _memory_mode_flags ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mode,expected",
    [
        (None, []),
        ("auto", []),
        ("AUTO", []),
        ("pinned", ["--mlock"]),
        ("PINNED", ["--mlock"]),
        ("resident", ["--no-mmap", "--mlock"]),
        ("RESIDENT", ["--no-mmap", "--mlock"]),
        ("", []),
    ],
)
def test_memory_mode_flags_maps_modes(mode, expected):
    assert LlamaCppBackend._memory_mode_flags(mode) == expected


# ── _already_in_target_state ─────────────────────────────────────────────────


def _base_target_state_kwargs(backend):
    return {
        "model_identifier": "owner/repo",
        "hf_variant": "Q4_K_M",
        "n_ctx": 8192,
        "cache_type_kv": None,
        "speculative_type": None,
        "chat_template_override": None,
        "extra_args": None,
        "is_vision": False,
        "gpu_ids": backend.gpu_ids,
        "memory_mode": backend.memory_mode,
    }


def test_already_in_target_state_matches_same_gpu_ids():
    backend = _loaded_backend(_gpu_ids = [0, 1])
    kwargs = _base_target_state_kwargs(backend)
    assert backend._already_in_target_state(**kwargs) is True


def test_already_in_target_state_rejects_different_gpu_ids():
    backend = _loaded_backend(_gpu_ids = [0, 1])
    kwargs = _base_target_state_kwargs(backend)
    kwargs["gpu_ids"] = [0]
    assert backend._already_in_target_state(**kwargs) is False


def test_already_in_target_state_matches_same_memory_mode():
    backend = _loaded_backend(_memory_mode = "resident")
    kwargs = _base_target_state_kwargs(backend)
    assert backend._already_in_target_state(**kwargs) is True


def test_already_in_target_state_rejects_different_memory_mode():
    backend = _loaded_backend(_memory_mode = "resident")
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = "pinned"
    assert backend._already_in_target_state(**kwargs) is False


def test_already_in_target_state_normalizes_empty_gpu_ids():
    backend = _loaded_backend(_gpu_ids = None)
    kwargs = _base_target_state_kwargs(backend)
    kwargs["gpu_ids"] = []
    assert backend._already_in_target_state(**kwargs) is True


# ── GPU selection filtering ──────────────────────────────────────────────────


def _backend_for_gpu_selection_test(tmp_path, visible_gpus, selected_gpu_ids):
    """Return a real LlamaCppBackend with the GPU selection path stubbed out."""
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)

    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None: list(visible_gpus)
    backend._read_gguf_metadata = lambda _p: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _p: 1024
    backend._mmproj_vram_bytes = lambda _p: 0
    backend._resolve_launch_mmproj_path = lambda **k: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"

    seen_gpus = []

    def capturing_select_gpus(requested_total, gpus, **kwargs):
        seen_gpus.extend(gpus)
        return ([gpus[0][0]], False)

    backend._select_gpus = capturing_select_gpus
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _d: True

    class _FakePopen:
        pid = 12345

        def poll(self):
            return None

    return backend, gguf, seen_gpus, _FakePopen


def test_gpu_ids_filter_restricts_visible_gpus(tmp_path):
    backend, gguf, seen_gpus, fake_popen = _backend_for_gpu_selection_test(
        tmp_path,
        visible_gpus = [(0, 10000, 16000), (1, 8000, 16000), (2, 6000, 16000)],
        selected_gpu_ids = [1, 2],
    )

    with patch.object(subprocess, "Popen", return_value = fake_popen()):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [1, 2],
        )

    assert seen_gpus == [(1, 8000), (2, 6000)]


def test_gpu_ids_filter_raises_when_no_visible_match(tmp_path):
    backend, gguf, _seen, _fake = _backend_for_gpu_selection_test(
        tmp_path,
        visible_gpus = [(0, 10000, 16000)],
        selected_gpu_ids = [9],
    )

    with pytest.raises(ValueError, match = "gpu_ids"):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [9],
        )
