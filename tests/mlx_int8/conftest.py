# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Shared fixtures.

These tests run on any MLX backend, including the Linux/CUDA build, because everything
they assert is either dispatch logic or algorithm arithmetic. The Metal kernels are not
exercised here -- see the macOS CI job for those.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

# The Metal backend cannot run here; the portable one implements identical arithmetic.
os.environ.setdefault("UNSLOTH_MLX_INT8_BACKEND", "portable")


@pytest.fixture(autouse=True)
def clean_state():
    """Every test starts with MLX unpatched and the allow-list empty."""
    import unsloth_mlx_int8
    from unsloth_mlx_int8 import backends, capability

    unsloth_mlx_int8.disable()
    capability.reset()
    backends.reset()
    yield
    unsloth_mlx_int8.disable()
    capability.reset()
    backends.reset()


@pytest.fixture
def make_ql():
    def _make(in_dims=1024, out_dims=2048, bits=4, group_size=64, bias=False):
        lin = nn.Linear(in_dims, out_dims, bias=bias)
        return nn.QuantizedLinear.from_linear(lin, group_size=group_size, bits=bits)

    return _make


class TinyModel(nn.Module):
    """Two eligible projections plus one too small to qualify."""

    def __init__(self, hidden=1024, inter=2048):
        super().__init__()
        self.mlp_gate = nn.Linear(hidden, inter, bias=False)
        self.mlp_down = nn.Linear(inter, hidden, bias=False)
        self.tiny = nn.Linear(64, 128, bias=False)

    def __call__(self, x):
        return self.mlp_down(self.mlp_gate(x))


@pytest.fixture
def quantized_model():
    model = TinyModel()
    nn.quantize(model, group_size=64, bits=4)
    return model
