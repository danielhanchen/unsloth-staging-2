# SPDX-License-Identifier: AGPL-3.0-only
"""Unit tests for issue #7164: keep-model-in-VRAM (--no-mmap / --mlock),
explicit per-GPU selection (CUDA/HIP/Vulkan), and idle-unload suppression.

These exercise the pure arg-building + idle-unload-decision helpers factored out
of llama_cpp.py / llama_keepwarm.py so the exact llama-server flag list and child
env are asserted without a live llama.cpp binary or a GPU.
"""

from core.inference import llama_residency as R


# ── (1) keep-resident / RAM flags ────────────────────────────────────────────


def test_residency_flags_default_off_is_noop():
    # Backwards-compat: both off -> no flags -> launch command unchanged.
    assert R.build_residency_flags() == []
    assert R.build_residency_flags(keep_model_in_vram = False, mlock = False) == []


def test_residency_flags_keep_in_vram_emits_no_mmap():
    assert R.build_residency_flags(keep_model_in_vram = True) == ["--no-mmap"]


def test_residency_flags_mlock_only():
    assert R.build_residency_flags(mlock = True) == ["--mlock"]


def test_residency_flags_both():
    # --no-mmap before --mlock; both are independent opt-ins.
    assert R.build_residency_flags(keep_model_in_vram = True, mlock = True) == [
        "--no-mmap",
        "--mlock",
    ]


# ── (2) per-GPU selection: filter + pin-id resolution ────────────────────────


def test_filter_selected_gpus_single():
    gpus = [(0, 45000), (1, 8000)]
    assert R.filter_selected_gpus(gpus, [1]) == [(1, 8000)]


def test_filter_selected_gpus_multi_preserves_order():
    gpus = [(0, 45000), (1, 8000), (2, 24000)]
    assert R.filter_selected_gpus(gpus, [2, 0]) == [(0, 45000), (2, 24000)]


def test_filter_selected_gpus_none_or_empty_is_passthrough():
    gpus = [(0, 45000), (1, 8000)]
    assert R.filter_selected_gpus(gpus, None) == gpus
    assert R.filter_selected_gpus(gpus, []) == gpus


def test_filter_selected_gpus_no_match_fails_open():
    # A stale selection that matches nothing must NOT strand the load on CPU.
    gpus = [(0, 45000), (1, 8000)]
    assert R.filter_selected_gpus(gpus, [7]) == gpus


def test_resolve_pin_ids_prefers_fit_subset():
    # When the fit narrowed the set, pin to that subset.
    assert R.resolve_pin_ids([0], [0, 1]) == [0]


def test_resolve_pin_ids_falls_back_to_user_selection():
    # Fit could not size the model (gpu_indices None) -> honor the raw selection.
    assert R.resolve_pin_ids(None, [1]) == [1]


def test_resolve_pin_ids_none_when_no_selection():
    assert R.resolve_pin_ids(None, None) is None
    assert R.resolve_pin_ids(None, []) is None


# ── (2b) per-GPU selection: backend-specific pin mechanism ───────────────────


def test_pin_nvidia_single_gpu_sets_only_cuda_visible_devices():
    env, pop, dev = R.resolve_gpu_pin([1], is_rocm = False, is_vulkan = False)
    assert env == {"CUDA_VISIBLE_DEVICES": "1"}
    assert pop == []  # NVIDIA: no ROCR to clear
    assert dev == []  # env-based pin, no --device


def test_pin_nvidia_multi_gpu():
    env, pop, dev = R.resolve_gpu_pin([0, 2], is_rocm = False, is_vulkan = False)
    assert env == {"CUDA_VISIBLE_DEVICES": "0,2"}
    assert pop == []
    assert dev == []


def test_pin_amd_rocm_sets_hip_and_clears_rocr():
    # AMD ROCm (the reporter's W7900 + W7500): HIP mask required; ROCR cleared so
    # the mask is not applied twice (which would re-index a "1" pin out of range).
    env, pop, dev = R.resolve_gpu_pin([1], is_rocm = True, is_vulkan = False)
    assert env == {"CUDA_VISIBLE_DEVICES": "1", "HIP_VISIBLE_DEVICES": "1"}
    assert pop == ["ROCR_VISIBLE_DEVICES"]
    assert dev == []


def test_pin_amd_rocm_multi_gpu():
    env, pop, dev = R.resolve_gpu_pin([0, 1], is_rocm = True, is_vulkan = False)
    assert env == {"CUDA_VISIBLE_DEVICES": "0,1", "HIP_VISIBLE_DEVICES": "0,1"}
    assert pop == ["ROCR_VISIBLE_DEVICES"]


def test_pin_vulkan_uses_device_arg_not_env():
    env, pop, dev = R.resolve_gpu_pin([0, 2], is_rocm = False, is_vulkan = True)
    assert env == {}
    assert pop == []
    assert dev == ["--device", "Vulkan0,Vulkan2"]


def test_pin_empty_selection_is_noop_on_every_backend():
    for rocm in (True, False):
        for vulkan in (True, False):
            assert R.resolve_gpu_pin(None, is_rocm = rocm, is_vulkan = vulkan) == ({}, [], [])
            assert R.resolve_gpu_pin([], is_rocm = rocm, is_vulkan = vulkan) == ({}, [], [])


# ── (1b) idle-unload suppression ─────────────────────────────────────────────


def test_idle_unload_fires_when_idle_and_not_resident():
    assert R.should_idle_unload(is_loaded = True, is_idle = True, keep_resident = False) is True


def test_idle_unload_suppressed_when_keep_resident():
    # keep_model_in_vram opts the model out of the TTL auto-unload entirely.
    assert R.should_idle_unload(is_loaded = True, is_idle = True, keep_resident = True) is False


def test_idle_unload_never_when_not_idle_or_not_loaded():
    assert R.should_idle_unload(is_loaded = True, is_idle = False, keep_resident = False) is False
    assert R.should_idle_unload(is_loaded = False, is_idle = True, keep_resident = False) is False


# ── end-to-end arg-list assertions (mirrors load_model's compose order) ──────


def _compose_launch(
    *,
    base_cmd,
    probed_gpus,
    gpu_ids,
    keep_model_in_vram,
    mlock,
    is_rocm,
    is_vulkan,
    fit_gpu_indices = "__auto__",
):
    """Reproduce the arg/env composition load_model performs, so a single test
    can assert the final llama-server flag list + child env for a scenario.

    ``fit_gpu_indices="__auto__"`` simulates the auto-fit selecting exactly the
    (post-filter) probed GPUs; pass an explicit list/None to model other fits.
    Returns (cmd, env_updates, env_pop).
    """
    cmd = list(base_cmd)
    gpus = R.filter_selected_gpus(probed_gpus, gpu_ids)
    if fit_gpu_indices == "__auto__":
        gpu_indices = [idx for idx, _free in gpus] if gpu_ids else None
    else:
        gpu_indices = fit_gpu_indices

    # residency flags (before user extras)
    cmd += R.build_residency_flags(keep_model_in_vram = keep_model_in_vram, mlock = mlock)

    pin_ids = R.resolve_pin_ids(gpu_indices, gpu_ids)
    env_updates, env_pop, dev_args = R.resolve_gpu_pin(
        pin_ids, is_rocm = is_rocm, is_vulkan = is_vulkan
    )
    if is_vulkan:
        cmd += dev_args
        return cmd, {}, []
    return cmd, env_updates, env_pop


BASE = ["llama-server", "-m", "/models/m.gguf", "--port", "8080", "-c", "4096"]


def test_e2e_defaults_unchanged():
    # Nothing selected, no residency -> command + env identical to today.
    cmd, env, pop = _compose_launch(
        base_cmd = BASE,
        probed_gpus = [(0, 45000), (1, 8000)],
        gpu_ids = None,
        keep_model_in_vram = False,
        mlock = False,
        is_rocm = False,
        is_vulkan = False,
    )
    assert cmd == BASE
    assert env == {}
    assert pop == []


def test_e2e_keep_resident_nvidia_single_gpu():
    cmd, env, pop = _compose_launch(
        base_cmd = BASE,
        probed_gpus = [(0, 45000), (1, 8000)],
        gpu_ids = [0],
        keep_model_in_vram = True,
        mlock = False,
        is_rocm = False,
        is_vulkan = False,
    )
    assert cmd == BASE + ["--no-mmap"]
    assert env == {"CUDA_VISIBLE_DEVICES": "0"}
    assert pop == []


def test_e2e_amd_rocm_second_gpu_only_keep_resident_and_mlock():
    # Reporter's exact case: force the small W7500 (physical id 1) only, pinned in
    # VRAM without a RAM mmap copy, host pages locked.
    cmd, env, pop = _compose_launch(
        base_cmd = BASE,
        probed_gpus = [(0, 45000), (1, 8000)],
        gpu_ids = [1],
        keep_model_in_vram = True,
        mlock = True,
        is_rocm = True,
        is_vulkan = False,
    )
    assert cmd == BASE + ["--no-mmap", "--mlock"]
    assert env == {"CUDA_VISIBLE_DEVICES": "1", "HIP_VISIBLE_DEVICES": "1"}
    assert pop == ["ROCR_VISIBLE_DEVICES"]


def test_e2e_amd_rocm_both_gpus():
    cmd, env, pop = _compose_launch(
        base_cmd = BASE,
        probed_gpus = [(0, 45000), (1, 8000)],
        gpu_ids = [0, 1],
        keep_model_in_vram = False,
        mlock = False,
        is_rocm = True,
        is_vulkan = False,
    )
    assert cmd == BASE  # no residency flags
    assert env == {"CUDA_VISIBLE_DEVICES": "0,1", "HIP_VISIBLE_DEVICES": "0,1"}
    assert pop == ["ROCR_VISIBLE_DEVICES"]


def test_e2e_vulkan_pin_via_device_arg():
    cmd, env, pop = _compose_launch(
        base_cmd = BASE,
        probed_gpus = [(0, 45000), (1, 8000)],
        gpu_ids = [1],
        keep_model_in_vram = True,
        mlock = False,
        is_rocm = False,
        is_vulkan = True,
    )
    assert cmd == BASE + ["--no-mmap", "--device", "Vulkan1"]
    assert env == {}
    assert pop == []


class _FakeBackend:
    """Minimal stand-in for LlamaCppBackend exposing what the keep-warm loop reads."""

    def __init__(
        self,
        *,
        is_loaded = True,
        keep_resident = False,
    ):
        self.is_loaded = is_loaded
        self.keep_resident = keep_resident


def _keepwarm_would_unload(backend, *, is_idle):
    """Reproduce llama_keepwarm.idle_unload_loop's guarded decision exactly."""
    return R.should_idle_unload(
        is_loaded = backend.is_loaded,
        is_idle = is_idle,
        keep_resident = bool(getattr(backend, "keep_resident", False)),
    )


def test_keepwarm_loop_unloads_ordinary_idle_model():
    assert _keepwarm_would_unload(_FakeBackend(keep_resident = False), is_idle = True) is True


def test_keepwarm_loop_never_unloads_keep_resident_model():
    # The headline #7164 guarantee: a keep-resident model survives the idle TTL.
    assert _keepwarm_would_unload(_FakeBackend(keep_resident = True), is_idle = True) is False


def test_keepwarm_missing_attr_defaults_to_unload_eligible():
    # An old backend object without the attribute behaves exactly as before.
    class _Old:
        is_loaded = True

    assert _keepwarm_would_unload(_Old(), is_idle = True) is True


def test_e2e_user_selection_honored_when_fit_returns_none():
    # Fit could not size the model (gpu_indices None), but the user explicitly
    # chose GPU 1 -> the child is still pinned to it.
    cmd, env, pop = _compose_launch(
        base_cmd = BASE,
        probed_gpus = [(0, 45000), (1, 8000)],
        gpu_ids = [1],
        keep_model_in_vram = False,
        mlock = False,
        is_rocm = False,
        is_vulkan = False,
        fit_gpu_indices = None,
    )
    assert env == {"CUDA_VISIBLE_DEVICES": "1"}
