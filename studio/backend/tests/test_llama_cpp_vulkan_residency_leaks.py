# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for two GGUF residency/GPU-pin correctness gaps (#7239):

  * FIX 2: an explicit Vulkan ordinal absent from the ggml probe must reject with an
    actionable error, not be silently redirected (fail-open) and misreported.
  * FIX 3: the diffusion runner ignores residency flags, so a prior keep-in-VRAM chat
    load must not leak keep_resident/mlock/gpu_ids into it; and unload_model must reset
    those fields so they always describe the active runner.

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
    # Probe enumerates only Vulkan0; the user pins [3]. filter_selected_gpus fail-opens,
    # so before the fix the fit pinned Vulkan0 but recorded gpu_ids=[3] (misreport). The
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


# ── FIX 3 (#7239): an AUTO Vulkan pick is recorded as None, not an explicit pin ─


def test_auto_vulkan_pick_not_recorded_as_explicit_pin(tmp_path):
    """An auto Vulkan load (gpu_ids omitted) whose fitter narrows to Vulkan0 must still
    PIN the child (--device Vulkan0) yet record gpu_ids as None, not [0]: recording an
    auto pick as an explicit pin misreports /status and makes dedupe miss the loaded
    server. Mirrors the CUDA/ROCm ``elif gpu_ids`` branch (Codex #7239)."""
    backend = LlamaCppBackend()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\0" * 1024)

    def _fake_metadata(self, path):
        self._is_diffusion = False
        self._context_length = 4096

    captured = {}

    def _capture_popen(cmd, *_a, **_k):
        # The recording branch (self._gpu_ids) runs before any spawn, so capture the
        # launched argv here; backend.gpu_ids is asserted from the outer scope.
        captured["cmd"] = list(cmd)
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
            staticmethod(lambda binary = None: [(0, 40000, 48000)]),
        ),
        # File-size-only auto path narrows to Vulkan0 (the fitter's real internals
        # are irrelevant to the recording branch under test).
        mock.patch.object(LlamaCppBackend, "_select_gpus", lambda self, *a, **k: ([0], False)),
        mock.patch("core.inference.llama_cpp.subprocess.Popen", side_effect = _capture_popen),
    ):
        with pytest.raises(RuntimeError, match = "SPAWN_REACHED"):
            backend.load_model(gguf_path = str(gguf), model_identifier = "m")  # gpu_ids omitted

    # The child is still pinned to the auto-picked device ...
    assert "--device" in captured["cmd"]
    assert "Vulkan0" in captured["cmd"]
    # ... but the RECORDED selection for /status + dedupe is None (auto), not [0].
    assert backend.gpu_ids is None


# ── FIX A: invalid Vulkan ordinal rejects BEFORE the Phase 1 kill ─────────────


def test_invalid_vulkan_ordinal_rejects_before_kill(tmp_path):
    # An invalid Vulkan pin used to be validated only after Phase 1 killed the live
    # server, leaving nothing running (the CUDA path is range-checked at the route
    # before any kill; Vulkan ordinals are not). The preflight must reject the
    # absent ordinal ABOVE the kill, so a healthy running model is left untouched.
    backend = LlamaCppBackend()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\0" * 1024)

    kill_mock = mock.MagicMock()

    def _fake_metadata(self, path):
        self._is_diffusion = False
        self._context_length = 4096

    def _spawn_reached(*_a, **_k):
        raise RuntimeError("SPAWN_REACHED: invalid Vulkan ordinal was not rejected pre-kill")

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
        # Record kills: the preflight must fire BEFORE _kill_process is invoked.
        mock.patch.object(LlamaCppBackend, "_kill_process", kill_mock),
        mock.patch.object(LlamaCppBackend, "_get_gguf_size_bytes", staticmethod(lambda p: 1024)),
        # Only Vulkan0 present; the user pins the absent ordinal [99].
        mock.patch.object(
            LlamaCppBackend,
            "_get_gpu_memory",
            staticmethod(lambda binary = None: [(0, 40000, 48000)]),
        ),
        mock.patch.object(LlamaCppBackend, "_start_llama_process", _spawn_reached),
        mock.patch("core.inference.llama_cpp.subprocess.Popen", side_effect = _spawn_reached),
    ):
        with pytest.raises(ValueError, match = "not present"):
            backend.load_model(gguf_path = str(gguf), model_identifier = "m", gpu_ids = [99])

    # The live process was NOT killed: the invalid pin never reached Phase 1.
    kill_mock.assert_not_called()
    # The raw, unpinnable ordinal was never recorded as the active selection.
    assert backend.gpu_ids != [99]


# ── FIX B: a Vulkan ggml ordinal must not feed the CUDA-mask diffusion runner ──


def test_diffusion_on_vulkan_with_gpu_ids_rejects_before_spawn(tmp_path):
    # A valid (present) Vulkan ordinal passes the Fix A preflight, so this isolates
    # the diffusion-specific rejection: the diffusion runner pins its visual server
    # by CUDA physical index, which cannot honor a ggml Vulkan ordinal. It must
    # reject rather than silently launch the runner on the wrong / no device.
    backend = LlamaCppBackend()
    gguf = tmp_path / "diffusion.gguf"
    gguf.write_bytes(b"\0" * 1024)

    def _fake_metadata(self, path):
        self._is_diffusion = True
        self._context_length = 4096

    def _diffusion_spawn_reached(self, **_k):
        raise RuntimeError("DIFFUSION_SPAWN_REACHED: Vulkan ordinal fed to the CUDA-mask runner")

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
        mock.patch.object(LlamaCppBackend, "_find_free_port", staticmethod(lambda: 12345)),
        mock.patch.object(LlamaCppBackend, "_kill_process", lambda self: None),
        # Vulkan0 is present, so the pin [0] survives the Fix A preflight.
        mock.patch.object(
            LlamaCppBackend,
            "_get_gpu_memory",
            staticmethod(lambda binary = None: [(0, 40000, 48000)]),
        ),
        mock.patch.object(LlamaCppBackend, "_start_diffusion_server", _diffusion_spawn_reached),
    ):
        with pytest.raises(ValueError, match = "not supported for a DiffusionGemma"):
            backend.load_model(gguf_path = str(gguf), model_identifier = "diffusion-m", gpu_ids = [0])


def test_non_diffusion_vulkan_valid_ordinal_still_proceeds(tmp_path):
    # The #7239 feature: a valid Vulkan pin for the NORMAL llama-server path must
    # NOT be falsely rejected by either fix. The load must reach Phase 1 (kill) and
    # then the spawn, proving neither the preflight nor the diffusion guard fired.
    backend = LlamaCppBackend()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\0" * 1024)

    kill_mock = mock.MagicMock()

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
        mock.patch.object(LlamaCppBackend, "_kill_process", kill_mock),
        mock.patch.object(LlamaCppBackend, "_get_gguf_size_bytes", staticmethod(lambda p: 1024)),
        mock.patch.object(
            LlamaCppBackend,
            "_get_gpu_memory",
            staticmethod(lambda binary = None: [(0, 40000, 48000), (1, 20000, 24000)]),
        ),
        mock.patch.object(LlamaCppBackend, "_start_llama_process", _spawn_reached),
        mock.patch("core.inference.llama_cpp.subprocess.Popen", side_effect = _spawn_reached),
    ):
        # Reaching the spawn proves the valid Vulkan pin was accepted, not rejected.
        with pytest.raises(RuntimeError, match = "SPAWN_REACHED"):
            backend.load_model(gguf_path = str(gguf), model_identifier = "m", gpu_ids = [0, 1])

    # The normal path DID reach Phase 1 (contrast: the invalid-ordinal test does not).
    kill_mock.assert_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ── BUG A (#7239): the RAW requested pin is recorded and drives load dedupe ─────


def test_narrowed_explicit_pin_records_raw_and_effective(tmp_path):
    """An explicit [0, 1] pin narrowed to [0] must record BOTH gpu_ids=[0] (effective,
    for /status) and requested_gpu_ids=[0, 1] (raw, for dedupe). Otherwise a re-Apply of
    the still-[0, 1] selection compares [0, 1] against [0] and needlessly reloads (#7239)."""
    backend = LlamaCppBackend()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\0" * 1024)

    def _fake_metadata(self, path):
        self._is_diffusion = False
        self._context_length = 4096

    def _spawn_reached(*_a, **_k):
        # The recording branch runs before any spawn; assert backend state after.
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
            # Both ordinals present, so the explicit [0, 1] passes the absent-ordinal
            # guard; the fitter (patched below) then narrows the valid set to [0].
            staticmethod(lambda binary = None: [(0, 40000, 48000), (1, 20000, 24000)]),
        ),
        mock.patch.object(LlamaCppBackend, "_select_gpus", lambda self, *a, **k: ([0], False)),
        mock.patch.object(LlamaCppBackend, "_start_llama_process", _spawn_reached),
        mock.patch("core.inference.llama_cpp.subprocess.Popen", side_effect = _spawn_reached),
    ):
        with pytest.raises(RuntimeError, match = "SPAWN_REACHED"):
            backend.load_model(gguf_path = str(gguf), model_identifier = "m", gpu_ids = [0, 1])

    # /status echoes the effective (narrowed) pin ...
    assert backend.gpu_ids == [0]
    # ... while the raw request is preserved for the load dedupe.
    assert backend.requested_gpu_ids == [0, 1]


def _match_kwargs(**overrides):
    """Non-GPU kwargs matching a fresh backend so only the pin decides the dedupe."""
    base = dict(
        model_identifier = "m",
        hf_variant = None,
        n_ctx = 0,
        cache_type_kv = None,
        speculative_type = None,
        chat_template_override = None,
        extra_args = None,
        is_vision = False,
        gguf_path = None,
    )
    base.update(overrides)
    return base


def test_already_in_target_state_dedupes_raw_requested_pin():
    """_already_in_target_state (backend mirror of the route dedupe) compares the RAW
    requested pin, not the fit-narrowed self._gpu_ids: [0, 1]->[0] re-sent as [0, 1]
    MATCHES; [0]-only does NOT; auto None dedupes None-vs-None (Codex #7239)."""
    backend = LlamaCppBackend()
    # Make the backend appear as a live, healthy server serving model "m" so the
    # dedupe reaches the pin comparison (is_loaded + model_identifier gates first).
    backend._process = mock.MagicMock()
    backend._healthy = True
    backend._model_identifier = "m"
    backend._requested_spec_mode = "auto"  # matches an omitted speculative_type
    backend._is_diffusion = False
    # Narrowed explicit pin: raw [0, 1], effective [0].
    backend._requested_gpu_ids = [0, 1]
    backend._gpu_ids = [0]

    # Re-sent raw [0, 1] dedupes (no reload) ...
    assert backend._already_in_target_state(**_match_kwargs(gpu_ids = [0, 1])) is True
    # ... a real change to [0] does not (raw [0] != raw [0, 1]) ...
    assert backend._already_in_target_state(**_match_kwargs(gpu_ids = [0])) is False
    # ... and /status still reports the effective pin.
    assert backend.gpu_ids == [0]

    # Auto load records None on both, and dedupes None-vs-None.
    backend._requested_gpu_ids = None
    backend._gpu_ids = None
    assert backend._already_in_target_state(**_match_kwargs(gpu_ids = None)) is True
