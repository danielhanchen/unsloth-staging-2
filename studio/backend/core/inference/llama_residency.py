# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pure helpers for the keep-model-in-VRAM load options and explicit per-GPU
selection for the local llama.cpp (GGUF) backend (issue #7164). No llama.cpp /
torch dependency, so flag-building and idle-unload logic stay unit-testable.
"""

from __future__ import annotations

from typing import Iterable, Optional


# ── (1) keep-resident / RAM control flags ────────────────────────────────────


def build_residency_flags(*, keep_model_in_vram: bool = False, mlock: bool = False) -> list[str]:
    """llama-server flags for the keep-resident / RAM options.

    keep_model_in_vram -> --no-mmap: don't memory-map the GGUF into system RAM,
    so under full GPU offload the redundant RAM copy is removed (LM Studio-style
    save-RAM). --no-mmap does not itself place or pin weights in VRAM (that follows
    from full offload) nor stop the driver-level VRAM paging some GPUs do when idle.
    mlock -> --mlock pins host pages; a separate opt-in since it grows RAM residency.
    Both default off, so an omitted pair yields [] (launch command unchanged).
    """
    flags: list[str] = []
    if keep_model_in_vram:
        flags.append("--no-mmap")
    if mlock:
        flags.append("--mlock")
    return flags


def derive_residency_state(
    cmd: Iterable[str],
    *,
    keep_model_in_vram: bool = False,
    mlock: bool = False,
) -> tuple[bool, bool]:
    """Return the (keep_resident, mlock) actually in effect for the FINAL argv.

    User llama_extra_args are appended AFTER build_residency_flags and llama.cpp
    parses last-wins, so a user --mmap / --no-mlock overrides Unsloth's flag.
    Scan the command in order so the last occurrence of each toggle pair wins;
    fall back to the requested boolean when neither flag of a pair is present.
    """
    keep_resident = bool(keep_model_in_vram)
    mlock_state = bool(mlock)
    for arg in cmd:
        a = str(arg)
        if a == "--no-mmap":
            keep_resident = True
        elif a == "--mmap":
            keep_resident = False
        elif a == "--mlock":
            mlock_state = True
        elif a == "--no-mlock":
            mlock_state = False
    return keep_resident, mlock_state


# ── (2) per-GPU selection ────────────────────────────────────────────────────


def _as_id_list(gpu_ids: Optional[Iterable[int]]) -> Optional[list[int]]:
    """Normalise a selection to list[int], or None. [] -> None (auto)."""
    if gpu_ids is None:
        return None
    ids = [int(x) for x in gpu_ids]
    return ids or None


def filter_selected_gpus(
    gpus: list[tuple[int, int]], gpu_ids: Optional[Iterable[int]]
) -> list[tuple[int, int]]:
    """Restrict the probed (index, free_mib) list to the user's selection.

    None/empty selection -> unchanged (auto). A selection matching none of the
    probed GPUs returns the original list (fail-open) so a stale UI choice can
    never strand the load on CPU. Probed order is preserved.
    """
    ids = _as_id_list(gpu_ids)
    if ids is None:
        return gpus
    wanted = set(ids)
    filtered = [g for g in gpus if g[0] in wanted]
    return filtered if filtered else gpus


def resolve_pin_ids(
    gpu_indices: Optional[Iterable[int]], user_gpu_ids: Optional[Iterable[int]]
) -> Optional[list[int]]:
    """Physical GPU ids to pin the child to: the fit-selected subset when the
    auto-fit ran, else the raw user selection so an explicit choice is honoured
    even when the fit could not size the model. None -> no explicit pin.
    """
    if gpu_indices is not None:
        return [int(x) for x in gpu_indices]
    ids = _as_id_list(user_gpu_ids)
    return ids


def resolve_gpu_pin(
    pin_ids: Optional[Iterable[int]], *, is_rocm: bool, is_vulkan: bool
) -> tuple[dict[str, str], list[str], list[str]]:
    """Map a physical GPU id list to the launch mechanism for the active backend.

    Returns (env_updates, env_pop_keys, device_args). NVIDIA: CUDA_VISIBLE_DEVICES.
    ROCm: also HIP_VISIBLE_DEVICES, and pop ROCR_VISIBLE_DEVICES so the mask is not
    applied twice (ROCR reduces + re-indexes, then a non-zero HIP pin points out of
    range -> CPU fallback). Vulkan: pin by ordinal via --device Vulkan<i>. None/empty
    pin -> no-op.
    """
    ids = _as_id_list(pin_ids)
    if ids is None:
        return {}, [], []

    if is_vulkan:
        return {}, [], ["--device", ",".join(f"Vulkan{i}" for i in ids)]

    pinned = ",".join(str(i) for i in ids)
    env_updates: dict[str, str] = {"CUDA_VISIBLE_DEVICES": pinned}
    env_pop: list[str] = []
    if is_rocm:
        env_updates["HIP_VISIBLE_DEVICES"] = pinned
        env_pop.append("ROCR_VISIBLE_DEVICES")
    return env_updates, env_pop, []


# ── (1b) idle-unload suppression ─────────────────────────────────────────────


def should_idle_unload(*, is_loaded: bool, is_idle: bool, keep_resident: bool) -> bool:
    """Whether the idle keep-warm loop should unload this tick. A model loaded
    with keep_model_in_vram opts out of the TTL auto-unload for its session, so
    unload only when loaded, past the idle TTL, and not pinned resident.
    """
    return bool(is_loaded and is_idle and not keep_resident)
