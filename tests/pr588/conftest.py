# SPDX-License-Identifier: AGPL-3.0-only
# Per-directory conftest for the PR-588 sim suite.
#
# Hardens the CPU-only test harness on macOS / Windows runners:
#   * shim torch.compile to a no-op so the @torch.compile decorator at
#     unsloth_zoo import time doesn't pull in the full inductor chain
#     that demands a real triton.
#   * stub torch._C._cuda_getCurrentRawStream so unsloth.kernels.utils'
#     `_CUDA_STREAMS = {i: torch._C._cuda_getCurrentRawStream(i) ...}`
#     module-level dict-comp doesn't crash on a runner with no CUDA.
#   * apply tests/_zoo_aggressive_cuda_spoof so the device-detection
#     chain in unsloth_zoo treats this process as a CUDA host.
#
# Idempotent — safe to apply on Linux too.
import os
import sys
from pathlib import Path


def _shim_torch_compile() -> None:
    try:
        import torch
    except Exception:
        return
    if getattr(torch.compile, "_pr588_shim", False):
        return
    def _noop_compile(fn=None, *a, **k):
        if fn is None:
            return lambda f: f
        return fn
    _noop_compile._pr588_shim = True
    torch.compile = _noop_compile


def _shim_cuda_raw_stream() -> None:
    try:
        import torch
    except Exception:
        return
    if not hasattr(torch._C, "_cuda_getCurrentRawStream"):
        torch._C._cuda_getCurrentRawStream = lambda index = 0: 0


def _apply_aggressive_spoof() -> None:
    tests_dir = Path(__file__).resolve().parent.parent
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    try:
        import _zoo_aggressive_cuda_spoof as _spoof
        _spoof.apply()
    except Exception:
        pass


_shim_torch_compile()
_shim_cuda_raw_stream()
_apply_aggressive_spoof()
