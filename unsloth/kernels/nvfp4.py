# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
# NVFP4 (compressed-tensors nvfp4-pack-quantized) weights run as W4A16: dequantize to 16 bit, activations untouched.
import torch

__all__ = ["NVFP4QuantState", "nvfp4_dequantize", "nvfp4_linear", "is_nvfp4_quant_state"]

NVFP4_GROUP = 16
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class NVFP4QuantState:
    __slots__ = ("scale", "global_scale", "shape", "dtype")

    def __init__(self, scale, global_scale, shape, dtype = torch.bfloat16):
        self.scale = scale
        self.global_scale = global_scale
        self.shape = tuple(shape)
        self.dtype = dtype


def is_nvfp4_quant_state(quant_state):
    return isinstance(quant_state, NVFP4QuantState)


def _is_transposed_view(x):
    return x.dim() == 2 and x.shape[0] > 1 and x.shape[1] > 1 and x.stride(0) == 1 and x.stride(1) != 1


def nvfp4_dequantize(packed, scale, global_scale, dtype = torch.bfloat16):
    """(out, in/2) uint8, low nibble = even column, E2M1 codes with bit 3 = sign -> (out, in) `dtype`."""
    # A transposed view (LoRA backward's W.t()) keeps scales in storage order.
    if _is_transposed_view(packed):
        return nvfp4_dequantize(packed.t(), scale, global_scale, dtype).t()
    rows, half = packed.shape
    codes = torch.stack((packed & 0x0F, packed >> 4), dim = -1).reshape(rows, half * 2)
    lut = torch.tensor(_E2M1, dtype = torch.float32, device = packed.device)
    values = lut[(codes & 0x07).long()]
    values = torch.where((codes & 0x08).bool(), -values, values)
    group_scale = scale.to(torch.float32)
    if global_scale is not None:
        group_scale = group_scale / global_scale.to(torch.float32)
    values = values.view(rows, -1, NVFP4_GROUP) * group_scale.view(rows, -1, 1)
    return values.view(rows, half * 2).to(dtype)


class NVFP4Linear_matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, packed, scale, global_scale, bias = None):
        W = nvfp4_dequantize(packed, scale, global_scale, X.dtype)
        out = torch.matmul(X, W.t())
        del W
        if bias is not None:
            out = out + bias
            if out.dtype != X.dtype:
                out = out.to(X.dtype)
        ctx.save_for_backward(packed, scale, global_scale)
        return out

    @staticmethod
    def backward(ctx, dY):
        packed, scale, global_scale = ctx.saved_tensors
        W = nvfp4_dequantize(packed, scale, global_scale, dY.dtype)
        return torch.matmul(dY, W), None, None, None, None


def nvfp4_linear(X, packed, scale, global_scale, bias = None):
    return NVFP4Linear_matmul.apply(X, packed, scale, global_scale, bias)
