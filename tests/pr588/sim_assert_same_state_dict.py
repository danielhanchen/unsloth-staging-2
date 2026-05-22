"""Sim test: assert_same_state_dict normalization and tolerance behavior."""
import sys
# sys.path: unsloth_zoo is pip-installed in CI

import pytest
import torch
from unsloth_zoo.vllm_utils import assert_same_state_dict


def test_same_fp32_tensors():
    t = torch.randn(4, 4)
    assert_same_state_dict({"w": t}, {"w": t.clone()})


def test_parameter_wrapper_detached():
    p_old = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    p_new = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    assert_same_state_dict({"w": p_old}, {"w": p_new})


def test_non_tensor_quant_state_skipped():
    w = torch.randn(4, 4)
    old = {"x.weight": w, "x.weight.quant_state": {"some": "metadata"}}
    new = {"x.weight": w, "x.weight.quant_state": {"some": "metadata"}}
    assert_same_state_dict(old, new)


def test_sparse_tensor_densified():
    indices = torch.tensor([[0, 1], [1, 2]])
    values = torch.tensor([1.0, 2.0])
    s_old = torch.sparse_coo_tensor(indices, values, size=(3, 3))
    s_new = torch.sparse_coo_tensor(indices, values, size=(3, 3))
    assert_same_state_dict({"w": s_old}, {"w": s_new})


def test_mismatched_dtype_uses_loose_tolerance():
    # bf16 vs fp16 are both <2 bytes -> upcast and loose tol
    a_bf = torch.randn(4, 4, dtype=torch.bfloat16)
    a_fp = a_bf.to(torch.float16)
    assert_same_state_dict({"w": a_bf}, {"w": a_fp})


def test_strict_tolerance_catches_real_difference():
    a = torch.zeros(4, 4)
    b = torch.zeros(4, 4)
    b[0, 0] = 1.0  # large diff
    with pytest.raises(RuntimeError):
        assert_same_state_dict({"w": a}, {"w": b})


def test_missing_key_raises():
    a = torch.randn(4, 4)
    with pytest.raises(RuntimeError):
        assert_same_state_dict({"w": a, "x": a}, {"w": a})


def test_lm_head_present_on_both_sides_ok():
    """When lm_head.weight is present on both sides with matching tensor, comparison passes."""
    a = torch.randn(4, 4)
    assert_same_state_dict({"lm_head.weight": a, "w": a}, {"lm_head.weight": a.clone(), "w": a.clone()})
