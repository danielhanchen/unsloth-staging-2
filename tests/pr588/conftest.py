# SPDX-License-Identifier: AGPL-3.0-only
# Per-directory conftest for the PR-588 sim suite.
#
# The simulation tests import unsloth_zoo's PR-modified modules
# (empty_model, hf_utils, vllm_utils). On macOS / Windows runners with no
# real triton, the @torch.compile decorator inside
# unsloth_zoo.temporary_patches.gemma3n runs at import time and pulls in
# the full torch._inductor backend chain that ultimately tries to
# `from triton.compiler import CompiledKernel`. Stub torch.compile to a
# no-op so the inductor chain stays cold.
#
# Idempotent — safe to apply on Linux too, where the sims neither need
# nor benefit from real compilation.
import os
import sys


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


_shim_torch_compile()
