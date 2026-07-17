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


def _install_stub_if_absent(name: str, build):
    """Install a lightweight stub for ``name`` only when the real module is not
    importable. Never shadow a real module: this test file was collected before
    a real-httpx importer once installed an incomplete httpx stub via
    sys.modules.setdefault, which then broke unrelated tests in a combined pytest
    run (order-dependent failures). Preferring the real module keeps the stub as a
    pure fallback for minimal environments without changing global import state."""
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except Exception:
        sys.modules[name] = build()


def _build_loggers_stub():
    mod = _types.ModuleType("loggers")
    mod.get_logger = lambda name: __import__("logging").getLogger(name)
    return mod


def _build_structlog_stub():
    mod = _types.ModuleType("structlog")
    mod.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
    return mod


def _build_httpx_stub():
    mod = _types.ModuleType("httpx")
    for _exc_name in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
    ):
        setattr(mod, _exc_name, type(_exc_name, (Exception,), {}))
    mod.Timeout = type("T", (), {"__init__": lambda s, *a, **k: None})
    mod.Client = type(
        "Client",
        (),
        {
            "__init__": lambda self, **kw: None,
            "__enter__": lambda self: self,
            "__exit__": lambda self, *a: None,
        },
    )
    return mod


def _build_jwt_stub():
    mod = _types.ModuleType("jwt")
    mod.decode = lambda *a, **k: {}
    mod.ExpiredSignatureError = type("ExpiredSignatureError", (Exception,), {})
    mod.InvalidTokenError = type("InvalidTokenError", (Exception,), {})
    return mod


_install_stub_if_absent("loggers", _build_loggers_stub)
_install_stub_if_absent("structlog", _build_structlog_stub)
_install_stub_if_absent("httpx", _build_httpx_stub)
_install_stub_if_absent("jwt", _build_jwt_stub)

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


def _fit_fallback_backend(
    tmp_path,
    gpu_memory,
    *,
    vulkan = False,
):
    """Backend stubbed like the --fit-fallback test but with a configurable probe."""
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)

    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None: list(gpu_memory)
    backend._read_gguf_metadata = lambda _p: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _p: 1024
    backend._mmproj_vram_bytes = lambda _p: 0
    backend._resolve_launch_mmproj_path = lambda **k: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: vulkan
    backend._get_gpu_free_memory = lambda _binary = None: list(gpu_memory)
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _d: True
    return backend, gguf


def test_empty_probe_preserves_explicit_gpu_ids(tmp_path):
    """A CUDA/ROCm build whose telemetry probe returns nothing (no nvidia-smi +
    CPU-only torch) must still honor a route-validated explicit gpu_ids rather than
    raising: auto-selection loads here via --fit on, so the explicit request pins the
    same GPU via CUDA_VISIBLE_DEVICES + --fit on instead of failing (#7164)."""
    backend, gguf = _fit_fallback_backend(tmp_path, gpu_memory = [])  # empty probe

    captured = {}

    def _make_fake_popen(cmd, **kwargs):
        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured["env"] = kwargs.get("env") or dict(os.environ)
                captured["cmd"] = list(cmd)

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with (
        patch.object(subprocess, "Popen", side_effect = _make_fake_popen),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = [0, 1]),
    ):
        assert backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [1],
        )

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    cmd = captured["cmd"]
    assert "--fit" in cmd and cmd[cmd.index("--fit") + 1] == "on"


def test_empty_probe_still_rejects_vulkan_gpu_ids(tmp_path):
    """A Vulkan build cannot map physical ids to ggml ordinals without a probe, so an
    empty Vulkan probe must keep rejecting the selection (tracked as #7201) rather than
    silently spreading the load across every device."""
    backend, gguf = _fit_fallback_backend(tmp_path, gpu_memory = [], vulkan = True)

    with (
        patch.object(subprocess, "Popen"),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = []),
    ):
        with pytest.raises(ValueError, match = "do not match any visible GPUs"):
            backend.load_model(
                gguf_path = str(gguf),
                model_identifier = "test",
                gpu_ids = [0],
            )


def test_torchless_vulkan_populated_probe_uses_identity_ordinals(tmp_path):
    """A torch-less Vulkan host has no CUDA/HIP physical namespace (empty parent-visible
    mask), so gpu_ids ARE the Vulkan ordinals: with a POPULATED probe the selection must
    load (pinned via --device Vulkan<i>) instead of raising -- otherwise explicit GPU
    selection is unreachable on exactly the AMD/Vulkan hosts #7164 targets (#7188)."""
    backend, gguf = _fit_fallback_backend(
        tmp_path, gpu_memory = [(0, 10000, 16000), (1, 8000, 16000)], vulkan = True
    )
    backend._select_gpus = lambda *a, **k: ([1], False)

    captured = {}

    def _make_fake_popen(cmd, **kwargs):
        class _FakePopen:
            pid = 321

            def __init__(self, cmd, **kwargs):
                captured["cmd"] = list(cmd)

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with (
        patch.object(subprocess, "Popen", side_effect = _make_fake_popen),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = []),
    ):
        assert backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [1],
        )
    cmd = captured["cmd"]
    assert "--device" in cmd and cmd[cmd.index("--device") + 1] == "Vulkan1"


def test_vulkan_rejects_duplicate_gpu_ids(tmp_path):
    """The CUDA resolver's duplicate check is skipped for the Vulkan path, so the branch
    must reject duplicates itself rather than let set() silently collapse them (#7188)."""
    backend, gguf = _fit_fallback_backend(
        tmp_path, gpu_memory = [(0, 10000, 16000), (1, 8000, 16000)], vulkan = True
    )
    with (
        patch.object(subprocess, "Popen"),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = []),
    ):
        with pytest.raises(ValueError, match = "unique and non-negative"):
            backend.load_model(
                gguf_path = str(gguf),
                model_identifier = "test",
                gpu_ids = [0, 0],
            )


def test_empty_probe_rejects_gpu_ids_without_gpu_backend(tmp_path):
    """A non-Vulkan build with an empty probe AND an empty parent-visible mask has no
    GPU backend at all (CPU / Metal build). The route now skips resolve_requested_gpu_ids
    for GGUF on non-CUDA hosts, so the backend must reject gpu_ids here rather than pin a
    non-existent device and silently run on CPU while reporting backend.gpu_ids (#7188)."""
    backend, gguf = _fit_fallback_backend(tmp_path, gpu_memory = [])  # empty probe

    with (
        patch.object(subprocess, "Popen"),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = []),
    ):
        with pytest.raises(ValueError, match = "no GPU backend"):
            backend.load_model(
                gguf_path = str(gguf),
                model_identifier = "test",
                gpu_ids = [0],
            )


def test_empty_probe_rejects_gpu_ids_outside_parent_mask(tmp_path):
    """Empty non-Vulkan probe but a set parent-visible mask (real GPU, no telemetry):
    a request outside that mask is a genuine 'no such GPU', so it must still raise
    rather than pin an id the host can't offer (#7188)."""
    backend, gguf = _fit_fallback_backend(tmp_path, gpu_memory = [])  # empty probe

    with (
        patch.object(subprocess, "Popen"),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = [0, 1]),
    ):
        with pytest.raises(ValueError, match = "do not match any visible GPUs"):
            backend.load_model(
                gguf_path = str(gguf),
                model_identifier = "test",
                gpu_ids = [9],
            )


def test_vulkan_gpu_ids_strips_conflicting_user_device(tmp_path):
    """On Vulkan the --device flag is the only device pin. When explicit gpu_ids is set,
    a user --device in extras must be stripped so it can't last-wins-override Studio's pin
    and offload to a GPU the training guard never budgeted (#7188). Studio's --device
    survives; unrelated extras pass through."""
    backend, gguf = _fit_fallback_backend(
        tmp_path, gpu_memory = [(0, 10000, 16000), (1, 8000, 16000)], vulkan = True
    )
    backend._select_gpus = lambda *a, **k: ([0], False)

    captured = {}

    def _make_fake_popen(cmd, **kwargs):
        class _FakePopen:
            pid = 999

            def __init__(self, cmd, **kwargs):
                captured["cmd"] = list(cmd)

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with (
        patch.object(subprocess, "Popen", side_effect = _make_fake_popen),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = [0, 1]),
    ):
        assert backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [0],
            extra_args = ["--device", "Vulkan1", "--top-k", "5"],
        )

    cmd = captured["cmd"]
    # Studio pinned Vulkan0 (gpu_indices=[0]); the user's Vulkan1 override is gone.
    assert "Vulkan1" not in cmd
    device_idxs = [i for i, tok in enumerate(cmd) if tok == "--device"]
    assert len(device_idxs) == 1
    assert cmd[device_idxs[0] + 1] == "Vulkan0"
    # Unrelated user extras still pass through.
    assert "--top-k" in cmd and cmd[cmd.index("--top-k") + 1] == "5"


def test_populated_probe_nonmatching_gpu_ids_still_raises(tmp_path):
    """When the probe DID enumerate GPUs but none match the request, the id is genuinely
    absent and the load must still raise (the empty-probe relaxation must not swallow
    a real 'no such GPU' error)."""
    backend, gguf = _fit_fallback_backend(tmp_path, gpu_memory = [(0, 10000, 16000)])

    with (
        patch.object(subprocess, "Popen"),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = [0]),
    ):
        with pytest.raises(ValueError, match = "do not match any visible GPUs"):
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


@pytest.mark.parametrize(
    "mode,user_flag,winning",
    [
        ("resident", "--mmap", "--no-mmap"),
        ("pinned", "--no-mmap", None),
        ("auto", "--mlock", None),
    ],
)
def test_memory_mode_strips_conflicting_extra_args(tmp_path, mode, user_flag, winning):
    """When a placement mode is applied, a conflicting --mmap/--no-mmap/--mlock left
    in extra_args must be stripped so llama.cpp's last-wins parsing can't run a
    placement that disagrees with the stored memory_mode (#7164). auto emits no
    memory flag, so the user's --mlock is dropped rather than pinning the child."""
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)
    backend = _mem_env_backend(gguf)

    captured_cmds = []

    def _make_fake_popen(cmd, **kwargs):
        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured_cmds.append(list(cmd))

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with patch.object(subprocess, "Popen", side_effect = _make_fake_popen):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            memory_mode = mode,
            extra_args = [user_flag],
        )

    assert captured_cmds, "llama-server was not spawned"
    cmd = captured_cmds[-1]
    # The caller's conflicting flag is gone.
    assert user_flag not in cmd
    # Only Studio's own memory flags (if any) remain, and the last mmap/no-mmap
    # flag reflects Studio's mode, not the stripped user flag.
    mmap_flags = [a for a in cmd if a in ("--mmap", "--no-mmap")]
    if winning is None:
        assert "--mmap" not in cmd  # user --mmap/--no-mmap fully stripped
    else:
        assert mmap_flags[-1] == winning


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


def test_explicit_auto_reloads_when_child_inherited_mem_env():
    """When the live child was launched with no mode but inherited operator
    LLAMA_ARG_* placement flags, an explicit 'auto' request must reload so the
    scrub runs -- otherwise it dedups to already-loaded and stays mlocked (#7164)."""
    backend = _loaded_backend(_memory_mode = None, _launched_with_inherited_mem_env = True)
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = "auto"
    assert backend._already_in_target_state(**kwargs) is False


def test_omitted_mode_does_not_reload_child_with_inherited_mem_env():
    """A request that also omits the mode (memory_mode=None) does NOT reload the
    inherited-env child: omitting keeps the operator env, so there's nothing to
    scrub and no spurious reload (which would be rejected during training)."""
    backend = _loaded_backend(_memory_mode = None, _launched_with_inherited_mem_env = True)
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = None
    assert backend._already_in_target_state(**kwargs) is True


def test_explicit_auto_matches_scrubbed_child():
    """Once the child was launched clean (no inherited env), an explicit 'auto'
    re-Apply dedups to already-loaded -- no needless reload."""
    backend = _loaded_backend(_memory_mode = None, _launched_with_inherited_mem_env = False)
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = "auto"
    assert backend._already_in_target_state(**kwargs) is True


def test_memory_mode_pinned_does_not_match_none():
    backend = _loaded_backend(_memory_mode = None)
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = "pinned"
    assert backend._already_in_target_state(**kwargs) is False


def test_load_response_and_status_round_trip_placement_fields():
    """gpu_ids and gguf_memory_mode are accepted by the response schemas so
    status-hydrated requests can preserve explicit placement settings."""
    from models.inference import InferenceStatusResponse, LoadResponse

    load_resp = LoadResponse(
        status = "loaded",
        model = "m",
        display_name = "m",
        is_gguf = True,
        inference = {},
        gpu_ids = [0, 1],
        gguf_memory_mode = "resident",
    )
    assert load_resp.gpu_ids == [0, 1]
    assert load_resp.gguf_memory_mode == "resident"

    status_resp = InferenceStatusResponse(
        is_gguf = True,
        gpu_ids = [0, 1],
        gguf_memory_mode = "pinned",
    )
    assert status_resp.gpu_ids == [0, 1]
    assert status_resp.gguf_memory_mode == "pinned"


@pytest.mark.parametrize(
    "mode,expected_requested,expected_canonical",
    [
        ("auto", "auto", None),
        ("AUTO", "auto", None),
        ("pinned", "pinned", "pinned"),
        (None, None, None),
    ],
)
def test_requested_memory_mode_preserves_explicit_auto(
    tmp_path, mode, expected_requested, expected_canonical
):
    """The response echoes requested_memory_mode, not the canonical placement. An explicit
    "auto" must survive as "auto" (not collapse to null) so the UI can restore it and a
    later reload re-runs the inherited-env scrub instead of letting LLAMA_ARG_MLOCK creep
    back; canonical _memory_mode still maps "auto" -> None for placement (#7188)."""
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)
    backend = _mem_env_backend(gguf)

    with patch.object(subprocess, "Popen"):
        assert backend.load_model(gguf_path = str(gguf), model_identifier = "t", memory_mode = mode)
    assert backend.requested_memory_mode == expected_requested
    assert backend.memory_mode == expected_canonical


def test_already_in_target_state_gpu_ids_order_insensitive():
    """[0,1] and [1,0] are the same selection (Studio sorts before pinning), so a
    reordered echo must dedupe instead of forcing a reload / training 409 (#7188)."""
    backend = _loaded_backend(_gpu_ids = [0, 1])
    kwargs = _base_target_state_kwargs(backend)
    kwargs["gpu_ids"] = [1, 0]
    assert backend._already_in_target_state(**kwargs) is True


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
    memory_mode=None (no opinion) leaves any operator env untouched for backwards
    compatibility; the reload-dedup instead reloads a None-loaded-with-inherited-env
    child when a later explicit 'auto' arrives (see the target-state tests below)."""
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
    b1._start_diffusion_server = lambda **kw: started.append(kw) or True
    with pytest.raises(ValueError, match = "gpu_ids"):
        b1.load_model(gguf_path = str(gguf), model_identifier = "d", gpu_ids = [1])
    assert started == []  # runner never launched

    # The same diffusion GGUF WITHOUT gpu_ids still reaches the runner.
    b2 = _make_backend()
    started2 = []
    b2._start_diffusion_server = lambda **kw: started2.append(kw) or True
    assert b2.load_model(gguf_path = str(gguf), model_identifier = "d") is True
    assert len(started2) == 1


@pytest.mark.parametrize("mode", ["pinned", "PINNED", "resident", "RESIDENT"])
def test_load_model_rejects_explicit_memory_mode_for_diffusion_gguf(tmp_path, mode):
    """The diffusion runner has no --mlock/--no-mmap plumbing, so an explicit
    pinned/resident memory_mode would be silently dropped yet recorded as honored.
    load_model must reject it (route -> 400) rather than mislead the user."""
    gguf = tmp_path / "diffusion.gguf"
    _write_minimal_gguf(gguf, arch = "diffusion-gemma")

    backend = LlamaCppBackend()
    backend._read_gguf_metadata = lambda _p: setattr(backend, "_is_diffusion", True)
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    started = []
    backend._start_diffusion_server = lambda **kw: started.append(kw) or True

    with pytest.raises(ValueError, match = "memory_mode"):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "d",
            memory_mode = mode,
        )
    assert started == []  # runner never launched


@pytest.mark.parametrize("mode", [None, "auto", "AUTO", ""])
def test_diffusion_load_allows_default_memory_mode_and_clears_stale_state(tmp_path, mode):
    """auto/blank/None placement is the no-op default and is allowed for diffusion.
    A successful diffusion load must also clear any _gpu_ids/_memory_mode left by a
    prior llama-server load so reload-dedup reflects the runner's real (unset) state
    and doesn't force a needless kill+restart of the healthy diffusion server."""
    gguf = tmp_path / "diffusion.gguf"
    _write_minimal_gguf(gguf, arch = "diffusion-gemma")

    backend = LlamaCppBackend()
    backend._read_gguf_metadata = lambda _p: setattr(backend, "_is_diffusion", True)
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._start_diffusion_server = lambda **kw: True

    # Simulate leftover placement state from a previous llama-server GGUF load.
    backend._gpu_ids = [0, 1]
    backend._memory_mode = "resident"

    assert (
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "d",
            memory_mode = mode,
        )
        is True
    )
    assert backend._gpu_ids is None
    assert backend._memory_mode is None


@pytest.mark.parametrize(
    "name,expected",
    [
        ("google/diffusiongemma-27b-it", True),  # canonical form: no separator before "gemma"
        ("google/diffusion-gemma-27b", True),  # separated form
        ("DiffusionGemma-2B", True),
        ("/models/diffusion/llama.gguf", True),  # standalone "diffusion" segment
        ("owner/llama-3-8b", False),
        ("my-diffusionfoo-model", False),  # "diffusion" run into unrelated text
    ],
)
def test_is_likely_diffusion_model_name_matches_diffusiongemma(name, expected):
    """The pre-kill heuristic must match the canonical google/diffusiongemma-* form
    (no separator before "gemma"), not just a standalone "diffusion" segment (#7188)."""
    assert LlamaCppBackend._is_likely_diffusion_model_name(model_identifier = name) == expected


def test_local_chat_gguf_in_diffusion_path_not_prekilled(tmp_path):
    """A local chat GGUF whose path contains "diffusion" (e.g. /models/diffusion/x.gguf)
    is NOT a diffusion model -- its header proves it. The Phase 0 name heuristic runs only
    for HF loads, so an explicit gpu_ids on such a local chat GGUF must load normally
    instead of being rejected on the path name before the header read (#7188)."""
    backend, gguf = _fit_fallback_backend(tmp_path, gpu_memory = [(0, 10000, 16000)])
    backend._select_gpus = lambda *a, **k: ([0], False)

    with patch.object(subprocess, "Popen"):
        assert (
            backend.load_model(
                gguf_path = str(gguf),
                model_identifier = "/models/diffusion/chat.gguf",
                gpu_ids = [0],
            )
            is True
        )
