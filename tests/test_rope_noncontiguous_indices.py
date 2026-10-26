# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Strided int32 position ids already on the device must rotate by their logical values in every
indexed RoPE entry point (the kernels read indices as a dense pointer)."""

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason = "Triton RoPE kernel needs CUDA"),
]
pytest.importorskip("triton")

import unsloth  # noqa: F401
from unsloth.kernels import rope_embedding as R
from unsloth.kernels.qk_norm_rope import fast_qk_norm_rope

BSZ, SEQ, N_HEADS, N_KV, HEAD_DIM = 2, 24, 8, 2, 64


def _strided_ids():
    ids = torch.arange(2 * BSZ * SEQ, device = "cuda", dtype = torch.int32).view(BSZ, 2 * SEQ)[:, ::2]
    # Precondition: flatten + cast keep the stride-2 view, else the test is vacuous.
    assert not ids.reshape(-1).to(dtype = torch.int32, device = "cuda").is_contiguous()
    return ids


def _cos_sin(dtype):
    rows = 4 * BSZ * SEQ
    inv = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2, device = "cuda").float() / HEAD_DIM))
    emb = torch.outer(torch.arange(rows, device = "cuda").float(), inv).repeat(1, 2)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _inputs(dtype, layout):
    g = torch.Generator(device = "cuda").manual_seed(3407)
    shape_q = (BSZ, N_HEADS, SEQ, HEAD_DIM) if layout == "bhsd" else (BSZ, SEQ, N_HEADS, HEAD_DIM)
    shape_k = (BSZ, N_KV, SEQ, HEAD_DIM) if layout == "bhsd" else (BSZ, SEQ, N_KV, HEAD_DIM)
    q = torch.randn(shape_q, device = "cuda", generator = g).to(dtype)
    k = torch.randn(shape_k, device = "cuda", generator = g).to(dtype)
    dq = torch.randn(BSZ, N_HEADS, SEQ, HEAD_DIM, device = "cuda", generator = g).to(dtype)
    dk = torch.randn(BSZ, N_KV, SEQ, HEAD_DIM, device = "cuda", generator = g).to(dtype)
    return q, k, dq, dk


def _run(fn, q, k, dq, dk, ids):
    a, b = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
    qo, ko = fn(a, b, ids)
    torch.autograd.backward((qo, ko), (dq, dk))
    return qo.detach(), ko.detach(), a.grad, b.grad


def _assert_same(fn, dtype, layout):
    q, k, dq, dk = _inputs(dtype, layout)
    strided = _strided_ids()
    dense = strided.contiguous()
    for x, y in zip(_run(fn, q, k, dq, dk, strided), _run(fn, q, k, dq, dk, dense)):
        assert torch.equal(x, y)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_eager_rope_strided_indices(dtype):
    cos, sin = _cos_sin(dtype)
    _assert_same(lambda q, k, ids: R.fast_rope_embedding(q, k, cos, sin, ids), dtype, "bhsd")


@pytest.mark.skipif(not R._TRACEABLE, reason = "needs torch.library.triton_op")
def test_compiled_rope_strided_indices():
    cos, sin = _cos_sin(torch.float16)
    fn = torch.compile(lambda q, k, ids: R.fast_rope_embedding(q, k, cos, sin, ids), fullgraph = True)
    _assert_same(fn, torch.float16, "bhsd")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_qk_norm_rope_strided_indices(dtype):
    cos, sin = _cos_sin(dtype)
    norms = [torch.nn.RMSNorm(HEAD_DIM, eps = 1e-6, device = "cuda", dtype = dtype) for _ in range(2)]
    for n in norms:
        n.weight.requires_grad_(False)
        n.weight.uniform_(0.5, 1.5)
    fn = lambda q, k, ids: fast_qk_norm_rope(norms[0], norms[1], q, k, cos, sin, ids)
    _assert_same(fn, dtype, "bshd")
