# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for GGUF memory placement mode and explicit GPU selection (#7164)."""

from __future__ import annotations

import os
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


def test_gpu_ids_preserved_on_fit_fallback(tmp_path):
    """When _select_gpus falls back to --fit on, still pin CUDA_VISIBLE_DEVICES."""
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)

    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None: [
        (0, 10000, 16000),
        (1, 8000, 16000),
        (2, 6000, 16000),
    ]
    backend._read_gguf_metadata = lambda _p: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _p: 1024
    backend._mmproj_vram_bytes = lambda _p: 0
    backend._resolve_launch_mmproj_path = lambda **k: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    # Force a --fit on fallback.
    backend._select_gpus = lambda *a, **k: (None, True)
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _d: True

    captured_envs = []

    def _make_fake_popen(cmd, **kwargs):
        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured_envs.append(kwargs.get("env") or dict(os.environ))

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with patch.object(subprocess, "Popen", side_effect = _make_fake_popen):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [1, 2],
        )

    assert captured_envs, "llama-server was not spawned"
    assert captured_envs[-1]["CUDA_VISIBLE_DEVICES"] == "1,2"


def test_memory_mode_clears_inherited_mmap_env_vars(tmp_path):
    """An explicit memory_mode must clear stale LLAMA_ARG_* env vars."""
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)

    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None: [(0, 10000, 16000)]
    backend._read_gguf_metadata = lambda _p: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _p: 1024
    backend._mmproj_vram_bytes = lambda _p: 0
    backend._resolve_launch_mmproj_path = lambda **k: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._select_gpus = lambda *a, **k: ([0], False)
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _d: True

    base_env = dict(os.environ)
    base_env.update(
        {
            "LLAMA_ARG_MLOCK": "1",
            "LLAMA_ARG_MMAP": "1",
            "LLAMA_ARG_NO_MMAP": "1",
        }
    )
    backend._llama_server_env_for_binary = lambda _binary: dict(base_env)

    captured_envs = []

    def _make_fake_popen(cmd, **kwargs):
        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured_envs.append(kwargs.get("env") or dict(os.environ))

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with patch.object(subprocess, "Popen", side_effect = _make_fake_popen):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            memory_mode = "auto",
        )

    assert captured_envs, "llama-server was not spawned"
    env = captured_envs[-1]
    assert "LLAMA_ARG_MLOCK" not in env
    assert "LLAMA_ARG_MMAP" not in env
    assert "LLAMA_ARG_NO_MMAP" not in env


def test_vulkan_gpu_ids_translated_to_compact_ordinals(tmp_path):
    """Physical gpu_ids are mapped to Vulkan0..N ordinals for --device pinning."""
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)

    backend = LlamaCppBackend()
    backend._is_vulkan_backend = lambda _binary = None: True
    # Vulkan reports one device at ordinal 0.
    backend._get_gpu_memory = lambda _binary = None: [(0, 10000, 16000)]
    backend._get_gpu_free_memory = lambda _binary = None: [(0, 10000)]
    backend._read_gguf_metadata = lambda _p: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _p: 1024
    backend._mmproj_vram_bytes = lambda _p: 0
    backend._resolve_launch_mmproj_path = lambda **k: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._select_gpus = lambda *a, **k: ([0], False)
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _d: True

    captured_cmds = []

    def _make_fake_popen(cmd, **kwargs):
        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured_cmds.append(list(cmd))

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with (
        patch.object(
            subprocess,
            "Popen",
            side_effect = _make_fake_popen,
        ),
        patch(
            "utils.hardware.get_parent_visible_gpu_ids",
            return_value = [1],
        ),
    ):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [1],
        )

    assert captured_cmds, "llama-server was not spawned"
    cmd = captured_cmds[-1]
    assert "--device" in cmd
    assert cmd[cmd.index("--device") + 1] == "Vulkan0"


def test_memory_mode_auto_matches_none_in_target_state():
    """An explicit 'auto' request should not reload a load that omitted the field."""
    backend = _loaded_backend(_memory_mode = None)
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = "auto"
    assert backend._already_in_target_state(**kwargs) is True


def test_memory_mode_pinned_does_not_match_none():
    backend = _loaded_backend(_memory_mode = None)
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = "pinned"
    assert backend._already_in_target_state(**kwargs) is False


# ── inherited LLAMA_ARG_* mmap/mlock env is scrubbed when a mode is set ───────


def _mem_env_backend(gguf):
    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None: [(0, 10000, 16000)]
    backend._read_gguf_metadata = lambda _p: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _p: 1024
    backend._mmproj_vram_bytes = lambda _p: 0
    backend._resolve_launch_mmproj_path = lambda **k: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._select_gpus = lambda *a, **k: ([0], False)
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _d: True
    return backend


@pytest.mark.parametrize(
    "mode,scrubbed",
    [("auto", True), ("pinned", True), ("resident", True), (None, False)],
)
def test_memory_mode_scrubs_inherited_mmap_env(tmp_path, monkeypatch, mode, scrubbed):
    """An explicit memory_mode strips inherited LLAMA_ARG_MLOCK/NO_MMAP/MMAP so
    llama-server can't silently run a placement Studio did not select (#7164).
    memory_mode=None (no opinion) leaves any operator env untouched."""
    monkeypatch.setenv("LLAMA_ARG_MLOCK", "1")
    monkeypatch.setenv("LLAMA_ARG_NO_MMAP", "1")
    monkeypatch.setenv("LLAMA_ARG_MMAP", "true")

    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)
    backend = _mem_env_backend(gguf)

    captured_envs = []

    def _make_fake_popen(cmd, **kwargs):
        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured_envs.append(kwargs.get("env") or {})

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with patch.object(subprocess, "Popen", side_effect = _make_fake_popen):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            memory_mode = mode,
        )

    assert captured_envs, "llama-server was not spawned"
    env = captured_envs[-1]
    for var in ("LLAMA_ARG_MLOCK", "LLAMA_ARG_NO_MMAP", "LLAMA_ARG_MMAP"):
        assert (var not in env) == scrubbed


# ── gpu_ids is rejected for diffusion GGUFs (single-GPU DG_GPU runner) ────────


def test_load_model_rejects_gpu_ids_for_diffusion_gguf(tmp_path):
    """Diffusion GGUFs serve via the single-GPU DG_GPU runner and ignore gpu_ids;
    load_model must reject a gpu_ids selection (the route maps this ValueError to a
    400) so the training guard's budget can't diverge from where the runner starts."""
    gguf = tmp_path / "diffusion.gguf"
    _write_minimal_gguf(gguf, arch = "diffusion-gemma")

    def _make_backend():
        b = LlamaCppBackend()
        # Force diffusion routing without depending on the full metadata parser.
        b._read_gguf_metadata = lambda _p: setattr(b, "_is_diffusion", True)
        b._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
        b._is_vulkan_backend = lambda _binary = None: False
        return b

    started = []
    b1 = _make_backend()
    b1._start_diffusion_server = lambda **kw: (started.append(kw) or True)
    with pytest.raises(ValueError, match = "gpu_ids"):
        b1.load_model(gguf_path = str(gguf), model_identifier = "d", gpu_ids = [1])
    assert started == []  # runner never launched

    # The same diffusion GGUF WITHOUT gpu_ids still reaches the runner.
    b2 = _make_backend()
    started2 = []
    b2._start_diffusion_server = lambda **kw: (started2.append(kw) or True)
    assert b2.load_model(gguf_path = str(gguf), model_identifier = "d") is True
    assert len(started2) == 1
