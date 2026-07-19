# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for two GGUF residency/GPU-pin correctness gaps (#7239):

  * FIX 2: an explicit Vulkan ordinal absent from the ggml probe must NOT be
    silently redirected to a different device (fail-open) and misreported; the
    load rejects with an actionable error instead.
  * FIX 3: the diffusion runner ignores llama-server residency flags, so a prior
    keep-in-VRAM chat load must not leak keep_resident/mlock/gpu_ids into it
    (else it opts out of the idle TTL unload and /status misreports); and
    unload_model must reset those fields so they always describe the active runner.

No GPU, network, or real subprocesses are used.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from core.inference.llama_residency import should_idle_unload  # noqa: E402


# ── FIX 2: explicit Vulkan ordinal absent from the probe rejects ──────────────


def test_explicit_vulkan_ordinal_absent_from_probe_rejects(tmp_path):
    # Probe enumerates only Vulkan0; the user pins ordinal [3]. filter_selected_gpus
    # fail-opens to the full list, so before the fix the fit picked Vulkan0, pinned
    # --device Vulkan0, but recorded gpu_ids=[3] -> /status + /load misreport. The
    # load must instead reject with an actionable "not present" error.
    backend = LlamaCppBackend()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\0" * 1024)

    def _fake_metadata(self, path):
        self._is_diffusion = False
        self._context_length = 4096

    def _spawn_reached(*_a, **_k):
        # If the reject regressed, the load proceeds to spawn: fail loudly here so
        # the test never hangs and never silently passes on the wrong device.
        raise RuntimeError("SPAWN_REACHED: explicit-absent Vulkan ordinal was not rejected")

    with (
        mock.patch.object(
            LlamaCppBackend,
            "_find_llama_server_binary",
            staticmethod(lambda **_k: "/fake/llama-server"),
        ),
        mock.patch.object(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda binary = None: True)
        ),
        mock.patch.object(LlamaCppBackend, "_read_gguf_metadata", _fake_metadata),
        mock.patch.object(LlamaCppBackend, "_already_in_target_state", lambda self, **_k: False),
        mock.patch.object(LlamaCppBackend, "_wait_for_vram_settle", lambda self, **_k: None),
        mock.patch.object(LlamaCppBackend, "_find_free_port", staticmethod(lambda: 12345)),
        mock.patch.object(LlamaCppBackend, "_kill_process", lambda self: None),
        mock.patch.object(LlamaCppBackend, "_get_gguf_size_bytes", staticmethod(lambda p: 1024)),
        # Only Vulkan0 is present (idx, free_mib, total_mib).
        mock.patch.object(
            LlamaCppBackend,
            "_get_gpu_memory",
            staticmethod(lambda binary = None: [(0, 40000, 48000)]),
        ),
        mock.patch.object(LlamaCppBackend, "_start_llama_process", _spawn_reached),
        mock.patch("core.inference.llama_cpp.subprocess.Popen", side_effect = _spawn_reached),
    ):
        with pytest.raises(ValueError, match = "not present"):
            backend.load_model(gguf_path = str(gguf), model_identifier = "m", gpu_ids = [3])

    # The raw, unpinnable ordinal was never recorded as the active selection.
    assert backend.gpu_ids != [3]


def test_partially_unmatched_vulkan_ordinals_reject(tmp_path):
    # Probe enumerates Vulkan0 and Vulkan1; the user pins [0, 99]. Ordinal 0
    # matches (so the old "none match" guard passed), 99 is absent -> the load
    # was launched on Vulkan0 while silently dropping 99, breaking the explicit
    # selection. Any absent requested ordinal must now reject.
    backend = LlamaCppBackend()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\0" * 1024)

    def _fake_metadata(self, path):
        self._is_diffusion = False
        self._context_length = 4096

    def _spawn_reached(*_a, **_k):
        raise RuntimeError("SPAWN_REACHED: partially-unmatched Vulkan pin was not rejected")

    with (
        mock.patch.object(
            LlamaCppBackend,
            "_find_llama_server_binary",
            staticmethod(lambda **_k: "/fake/llama-server"),
        ),
        mock.patch.object(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda binary = None: True)
        ),
        mock.patch.object(LlamaCppBackend, "_read_gguf_metadata", _fake_metadata),
        mock.patch.object(LlamaCppBackend, "_already_in_target_state", lambda self, **_k: False),
        mock.patch.object(LlamaCppBackend, "_wait_for_vram_settle", lambda self, **_k: None),
        mock.patch.object(LlamaCppBackend, "_find_free_port", staticmethod(lambda: 12345)),
        mock.patch.object(LlamaCppBackend, "_kill_process", lambda self: None),
        mock.patch.object(LlamaCppBackend, "_get_gguf_size_bytes", staticmethod(lambda p: 1024)),
        # Vulkan0 and Vulkan1 present (idx, free_mib, total_mib).
        mock.patch.object(
            LlamaCppBackend,
            "_get_gpu_memory",
            staticmethod(lambda binary = None: [(0, 40000, 48000), (1, 20000, 24000)]),
        ),
        mock.patch.object(LlamaCppBackend, "_start_llama_process", _spawn_reached),
        mock.patch("core.inference.llama_cpp.subprocess.Popen", side_effect = _spawn_reached),
    ):
        with pytest.raises(ValueError, match = "not present"):
            backend.load_model(gguf_path = str(gguf), model_identifier = "m", gpu_ids = [0, 99])

    # The partially-unpinnable selection was never recorded as the active pin.
    assert backend.gpu_ids != [0, 99]


def test_all_present_vulkan_ordinals_do_not_reject(tmp_path):
    # A request where ALL ordinals are present must pass the absent-ordinal guard,
    # even if the fitter later narrows the set -- that is narrowing of valid
    # ordinals, not an absent ordinal. Reaching the spawn (SPAWN_REACHED) proves
    # the guard did not falsely reject.
    backend = LlamaCppBackend()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\0" * 1024)

    def _fake_metadata(self, path):
        self._is_diffusion = False
        self._context_length = 4096

    def _spawn_reached(*_a, **_k):
        raise RuntimeError("SPAWN_REACHED")

    with (
        mock.patch.object(
            LlamaCppBackend,
            "_find_llama_server_binary",
            staticmethod(lambda **_k: "/fake/llama-server"),
        ),
        mock.patch.object(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda binary = None: True)
        ),
        mock.patch.object(LlamaCppBackend, "_read_gguf_metadata", _fake_metadata),
        mock.patch.object(LlamaCppBackend, "_already_in_target_state", lambda self, **_k: False),
        mock.patch.object(LlamaCppBackend, "_wait_for_vram_settle", lambda self, **_k: None),
        mock.patch.object(LlamaCppBackend, "_find_free_port", staticmethod(lambda: 12345)),
        mock.patch.object(LlamaCppBackend, "_kill_process", lambda self: None),
        mock.patch.object(LlamaCppBackend, "_get_gguf_size_bytes", staticmethod(lambda p: 1024)),
        mock.patch.object(
            LlamaCppBackend,
            "_get_gpu_memory",
            staticmethod(lambda binary = None: [(0, 40000, 48000), (1, 20000, 24000)]),
        ),
        mock.patch.object(LlamaCppBackend, "_start_llama_process", _spawn_reached),
        mock.patch("core.inference.llama_cpp.subprocess.Popen", side_effect = _spawn_reached),
    ):
        # Not a ValueError("not present"): the guard let a fully-present [0, 1] pass.
        with pytest.raises(RuntimeError, match = "SPAWN_REACHED"):
            backend.load_model(gguf_path = str(gguf), model_identifier = "m", gpu_ids = [0, 1])


# ── FIX 3: diffusion path clears leaked residency; unload_model resets it ──────


def _drive_diffusion(backend: LlamaCppBackend, tmp_path: Path) -> bool:
    """Run _start_diffusion_server with the shim/process fully stubbed."""
    fake_proc = mock.MagicMock()
    fake_proc.stdout = None

    class _NoopThread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

        def join(self, timeout = None):
            pass

        def is_alive(self):
            return False

    with (
        mock.patch.object(
            LlamaCppBackend,
            "_find_diffusion_assets",
            lambda self: (["python", "-m", "shim"], "/fake/visual-bin", None),
        ),
        mock.patch.object(LlamaCppBackend, "_kill_process", lambda self: None),
        mock.patch.object(LlamaCppBackend, "_find_free_port", staticmethod(lambda: 12345)),
        mock.patch.object(LlamaCppBackend, "_effective_gpu_count", staticmethod(lambda: 1)),
        mock.patch.object(LlamaCppBackend, "_drain_stdout", lambda self: None),
        mock.patch.object(LlamaCppBackend, "_wait_for_health", lambda self, timeout = 0.0: True),
        mock.patch("core.inference.llama_cpp._swa_cache_path", lambda: tmp_path / "swa" / "x"),
        mock.patch("core.inference.llama_cpp.subprocess.Popen", return_value = fake_proc),
        mock.patch("core.inference.llama_cpp.threading.Thread", _NoopThread),
    ):
        return backend._start_diffusion_server(
            model_path = str(tmp_path / "diffusion.gguf"),
            gguf_path = str(tmp_path / "diffusion.gguf"),
            hf_repo = None,
            hf_variant = None,
            model_identifier = "diffusion-gemma",
            n_ctx = 0,
            extra_args = None,
        )


def test_diffusion_start_clears_leaked_keep_in_vram_state(tmp_path):
    backend = LlamaCppBackend()
    # A prior keep-in-VRAM chat load set these fields on the shared backend.
    backend._keep_resident = True
    backend._mlock = True
    backend._gpu_ids = [0]

    healthy = _drive_diffusion(backend, tmp_path)
    assert healthy is True

    # The diffusion shim honors none of these, so they must be cleared.
    assert backend.keep_resident is False
    assert backend.mlock is False
    assert backend.gpu_ids is None
    # With keep_resident cleared, the idle keep-warm loop can TTL-unload the runner.
    assert (
        should_idle_unload(is_loaded = True, is_idle = True, keep_resident = backend.keep_resident)
        is True
    )


def test_unload_model_resets_residency_fields(tmp_path):
    backend = LlamaCppBackend()
    backend._keep_resident = True
    backend._mlock = True
    backend._gpu_ids = [0]

    with mock.patch.object(LlamaCppBackend, "_kill_process", lambda self: None):
        assert backend.unload_model() is True

    assert backend.keep_resident is False
    assert backend.mlock is False
    assert backend.gpu_ids is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
