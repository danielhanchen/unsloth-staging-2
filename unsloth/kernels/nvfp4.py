# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
# Stand-in torch reference for the NVFP4 W4A16 kernels (the Triton version replaces this module, same interface).
import torch

__all__ = ["NVFP4QuantState", "nvfp4_dequantize", "nvfp4_linear"]

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class NVFP4QuantState:
    __slots__ = ("scale", "global_scale", "shape", "dtype")

    def __init__(self, scale, global_scale, shape, dtype = torch.bfloat16):
        self.scale = scale
        self.global_scale = global_scale
        self.shape = tuple(shape)
        self.dtype = dtype


def nvfp4_dequantize(packed, scale, global_scale, dtype = torch.bfloat16):
    if packed.dim() == 2 and packed.stride(0) == 1 and packed.stride(1) != 1:
        return nvfp4_dequantize(packed.t(), scale, global_scale, dtype).t()
    out_f, half = packed.shape
    lut = torch.tensor(_E2M1, dtype = torch.float32, device = packed.device)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim = -1).reshape(out_f, half * 2).long()
    values = lut[codes & 0x7] * torch.where((codes & 0x8) != 0, -1.0, 1.0)
    group = scale.float() / global_scale.float()
    return (values.reshape(out_f, -1, 16) * group.unsqueeze(-1)).reshape(out_f, half * 2).to(dtype)


class _NVFP4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, packed, scale, global_scale, bias):
        W = nvfp4_dequantize(packed, scale, global_scale, X.dtype)
        out = torch.matmul(X, W.t())
        ctx.save_for_backward(packed, scale, global_scale)
        if bias is not None:
            out = out + bias
        return out if out.dtype == X.dtype else out.to(X.dtype)

    @staticmethod
    def backward(ctx, dY):
        packed, scale, global_scale = ctx.saved_tensors
        W = nvfp4_dequantize(packed, scale, global_scale, dY.dtype)
        return torch.matmul(dY, W), None, None, None, None


def nvfp4_linear(X, packed, scale, global_scale, bias = None):
    return _NVFP4Linear.apply(X, packed, scale, global_scale, bias)
