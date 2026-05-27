# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Early CPU thread-pool configuration for Studio processes."""

import os
from typing import MutableMapping, Optional


_THREAD_POOL_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# Idle Studio almost never benefits from more than ~8 native OpenMP / BLAS
# workers. On large-core hosts the default OpenMP behaviour spawns one
# worker per logical core, which inflates kernel thread count and RSS for
# no throughput win (see issue #5630). The cap deliberately stays small
# enough that even tiny VMs (>=2 cores) are unaffected.
_DEFAULT_IDLE_CAP = 8

# Sentinel values that disable the cap entirely. Useful for HPC users who
# explicitly want one thread per core for CPU-heavy training.
_OPT_OUT_VALUES = frozenset({"off", "none", "disabled", "unlimited"})

# Sentinel value that asks for the same auto-default that an unset env
# would produce. Lets users be explicit in scripts.
_AUTO_VALUES = frozenset({"auto", "default"})


def _resolve_cpu_count() -> int:
    """Best-effort logical CPU count, respecting cgroup affinity on Linux."""
    try:
        return len(os.sched_getaffinity(0))  # Linux only
    except (AttributeError, OSError):
        return os.cpu_count() or _DEFAULT_IDLE_CAP


def configure_cpu_threads(env: Optional[MutableMapping[str, str]] = None) -> None:
    """Seed native CPU thread-pool env vars from ``UNSLOTH_CPU_THREADS``.

    Behaviour:

    * unset / empty / ``"auto"`` / ``"default"``: seed each native
      thread-pool variable with ``min(cpu_count(), 8)`` so idle Studio
      does not spawn one OpenMP / BLAS worker per logical core.
    * ``"off"`` / ``"none"`` / ``"disabled"`` / ``"unlimited"`` (case
      insensitive): skip seeding entirely. Restores the pre-cap OS
      defaults for users who explicitly want one thread per core.
    * positive integer: seed with that exact value.
    * anything else (``"0"``, ``"-3"``, ``"1.5"``, ``"abc"``, ...): raise
      ``ValueError``.

    This must run before importing libraries that initialize an OpenMP or
    BLAS thread pool. ``os.environ.setdefault`` is used so that explicit
    per-library env vars (``OMP_NUM_THREADS`` etc.) always win over the
    Studio-level knob.

    The mechanism is cross-platform: ``os.environ.setdefault`` is
    identical on Linux / macOS / Windows, and the four target variables
    are honoured by libgomp / LLVM OpenMP / MSVC OpenMP, Intel MKL,
    OpenBLAS, and NumExpr on every platform PyTorch ships wheels for.
    macOS MLX uses its own Metal-backed pool and is unaffected by these
    env vars by design.
    """
    environ = os.environ if env is None else env
    raw = environ.get("UNSLOTH_CPU_THREADS", "").strip()
    lowered = raw.lower()

    if lowered in _OPT_OUT_VALUES:
        return

    if not raw or lowered in _AUTO_VALUES:
        thread_count = min(_resolve_cpu_count(), _DEFAULT_IDLE_CAP)
    else:
        try:
            thread_count = int(raw)
        except ValueError:
            raise ValueError(
                "UNSLOTH_CPU_THREADS must be a positive integer, "
                "'auto', or an opt-out sentinel ('off' / 'none' / 'disabled' / 'unlimited')"
            ) from None
        if thread_count < 1:
            raise ValueError(
                "UNSLOTH_CPU_THREADS must be a positive integer, "
                "'auto', or an opt-out sentinel ('off' / 'none' / 'disabled' / 'unlimited')"
            )

    value = str(thread_count)
    for variable in _THREAD_POOL_ENV_VARS:
        environ.setdefault(variable, value)
