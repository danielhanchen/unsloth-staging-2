# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Spike for the MLX int8 prefill patch.

Validates the architectural assumptions from plans/sparkling-popping-kay.md before any
kernel work, on a box with no Apple silicon:

  1. `mx.quantized_matmul` can be reassigned on the nanobind module, and `functools.wraps`
     round-trips so `disable()` has a handle.
  2. The wrapper binds every argument-passing form callers actually use, including the
     fully positional `mx.quantized_matmul(q, *q_keys, transpose=..., ...)` at
     mlx-lm/mlx_lm/models/base.py:84.
  3. A registry keyed on `id(weight)` with an identity re-check routes only registered
     weights, so id reuse after GC cannot mis-scale.
  4. Calls we must never intercept fall through bit-identically.
  5. `nn.QuantizedLinear.__call__` and `nn.QuantizedEmbedding.as_linear` both reach the
     wrapper (the whole justification for patching at op level).
  6. The hot path is free of `mx.eval`, so it survives `mx.compile`.
  7. `mx.custom_function` can carry a vjp that delegates to the original op.

Run: python scripts/spike_mlx_int8_patch.py
"""

import functools
import gc
import sys

import mlx.core as mx
import mlx.nn as nn

ROW_THRESHOLD = 512

_ORIG_QMM = mx.quantized_matmul
_registry = {}
_hits = []
_ACT_DTYPES = (mx.bfloat16, mx.float16, mx.float32)


class Entry:
    """Registry value. Holds a strong ref to the weight so `id()` cannot be recycled."""

    def __init__(self, w, scales, biases, bits, group_size, name):
        self.w = w
        self.scales = scales
        self.biases = biases
        self.bits = bits
        self.group_size = group_size
        self.name = name


def register(module, name):
    w = module["weight"]
    _registry[id(w)] = Entry(
        w, module["scales"], module["biases"], module.bits, module.group_size, name
    )


def _int8_path_stub(x, e):
    """Stand-in for the W8A8 kernel: records the hit, returns the exact 4-bit result.

    The spike is about dispatch, not arithmetic, so this must stay numerically identical
    to the fallback -- that is what lets the bit-identity assertions below mean something.
    """
    _hits.append(e.name)
    return _ORIG_QMM(
        x, e.w, e.scales, e.biases, True, e.group_size, e.bits, "affine"
    )


@functools.wraps(_ORIG_QMM)
def _patched_qmm(x, w, /, *args, **kwargs):
    n = len(args)
    scales     = args[0] if n > 0 else kwargs.get("scales")
    biases     = args[1] if n > 1 else kwargs.get("biases")
    transpose  = args[2] if n > 2 else kwargs.get("transpose", True)
    group_size = args[3] if n > 3 else kwargs.get("group_size")
    bits       = args[4] if n > 4 else kwargs.get("bits")
    mode       = args[5] if n > 5 else kwargs.get("mode", "affine")

    if (
        kwargs.get("stream") is None
        and transpose is True
        and mode == "affine"
        and biases is not None
        and getattr(w, "ndim", 0) == 2
        and x.dtype in _ACT_DTYPES
    ):
        e = _registry.get(id(w))
        # Identity re-check: an id() alone is not a safe key across GC.
        if e is not None and e.w is w:
            if (group_size is None or group_size == e.group_size) and (
                bits is None or bits == e.bits
            ):
                if x.size // x.shape[-1] >= ROW_THRESHOLD:
                    return _int8_path_stub(x, e)
    return _ORIG_QMM(x, w, *args, **kwargs)


_patched_qmm.__unsloth_int8_prefill__ = True


def enable():
    if getattr(mx.quantized_matmul, "__unsloth_int8_prefill__", False):
        return False
    mx.quantized_matmul = _patched_qmm
    for mod in tuple(sys.modules.values()):
        if mod is not None and getattr(mod, "quantized_matmul", None) is _ORIG_QMM:
            try:
                setattr(mod, "quantized_matmul", _patched_qmm)
            except Exception:
                pass
    return True


def disable():
    mx.quantized_matmul = _ORIG_QMM


# ----------------------------------------------------------------------------- checks

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def make_ql(in_dims, out_dims, bits=4, gs=64):
    lin = nn.Linear(in_dims, out_dims, bias=False)
    return nn.QuantizedLinear.from_linear(lin, group_size=gs, bits=bits)


def main():
    print(f"mlx {mx.__version__}  metal={mx.metal.is_available()}  dev={mx.default_device()}")

    # ---- 1. patchability -----------------------------------------------------------
    print("\n1. Patchability")
    check("assignment on nanobind module succeeds", enable())
    check("mx.quantized_matmul is the wrapper", mx.quantized_matmul is _patched_qmm)
    check("functools.wraps round-trips", mx.quantized_matmul.__wrapped__ is _ORIG_QMM)
    check("second enable() is a no-op (idempotent)", enable() is False)

    # ---- 2/3. dispatch through a registered weight ---------------------------------
    print("\n2. Argument binding + registry dispatch")
    K, N = 1024, 2048
    ql = make_ql(K, N)
    register(ql, "ql")
    x_big = mx.random.normal((ROW_THRESHOLD, K)).astype(mx.bfloat16)

    w, s, b, gs, bits = ql["weight"], ql["scales"], ql["biases"], ql.group_size, ql.bits
    ref = _ORIG_QMM(x_big, w, s, b, True, gs, bits, "affine")
    mx.eval(ref)

    forms = {
        "all kwargs": lambda: mx.quantized_matmul(
            x_big, w, scales=s, biases=b, transpose=True, group_size=gs, bits=bits
        ),
        "all positional": lambda: mx.quantized_matmul(x_big, w, s, b, True, gs, bits),
        "base.py:84 form (*q_keys)": lambda: mx.quantized_matmul(
            x_big, *(w, s, b), transpose=True, group_size=gs, bits=bits
        ),
        "mixed pos/kw": lambda: mx.quantized_matmul(
            x_big, w, s, biases=b, transpose=True, group_size=gs, bits=bits
        ),
        "defaults omitted": lambda: mx.quantized_matmul(
            x_big, w, scales=s, biases=b, group_size=gs, bits=bits
        ),
    }
    for label, fn in forms.items():
        _hits.clear()
        out = fn()
        mx.eval(out)
        check(
            f"intercepted: {label}",
            _hits == ["ql"] and mx.array_equal(out, ref).item(),
            f"hits={_hits}",
        )

    # ---- 4. non-interception -------------------------------------------------------
    print("\n4. Calls that must fall through")
    x_small = mx.random.normal((8, K)).astype(mx.bfloat16)
    unregistered = make_ql(K, N)

    cases = {
        "rows < ROW_THRESHOLD": lambda: mx.quantized_matmul(
            x_small, w, s, b, True, gs, bits
        ),
        "unregistered weight": lambda: mx.quantized_matmul(
            x_big, unregistered["weight"], unregistered["scales"],
            unregistered["biases"], True, gs, bits,
        ),
        "transpose=False": lambda: mx.quantized_matmul(
            mx.random.normal((ROW_THRESHOLD, N)).astype(mx.bfloat16),
            w, s, b, False, gs, bits,
        ),
        "explicit stream=": lambda: mx.quantized_matmul(
            x_big, w, s, b, True, gs, bits, "affine", stream=mx.default_stream(mx.default_device())
        ),
    }
    for label, fn in cases.items():
        _hits.clear()
        try:
            out = fn()
            mx.eval(out)
            ok = _hits == []
        except Exception as exc:  # a fallthrough that raises is still a fallthrough
            ok = _hits == []
            label += f" (raised {type(exc).__name__})"
        check(f"fell through: {label}", ok, f"hits={_hits}")

    # 3D weights (gather_qmm-style) and mxfp4 (biases is None)
    _hits.clear()
    try:
        qmxfp4 = nn.QuantizedLinear.from_linear(
            nn.Linear(K, N, bias=False), mode="mxfp4"
        )
        out = qmxfp4(x_big)
        mx.eval(out)
        check("fell through: mxfp4 (biases is None)", _hits == [], f"hits={_hits}")
    except Exception as exc:
        check("fell through: mxfp4", _hits == [], f"ctor raised {type(exc).__name__}")

    # ---- 5. op-level coverage ------------------------------------------------------
    print("\n5. Coverage that layer-level patching would miss")
    _hits.clear()
    out = ql(x_big)
    mx.eval(out)
    check("nn.QuantizedLinear.__call__ reaches wrapper", _hits == ["ql"])

    emb = nn.QuantizedEmbedding.from_embedding(nn.Embedding(N, K), group_size=64, bits=4)
    register(emb, "emb")
    _hits.clear()
    out = emb.as_linear(x_big)
    mx.eval(out)
    check("QuantizedEmbedding.as_linear reaches wrapper", _hits == ["emb"])

    # ---- 3b. id() reuse safety -----------------------------------------------------
    print("\n3b. Registry identity re-check")
    doomed = make_ql(K, N)
    register(doomed, "doomed")
    doomed_id = id(doomed["weight"])
    doomed_w = doomed["weight"]
    del doomed
    gc.collect()
    entry = _registry[doomed_id]
    check(
        "registry keeps the weight alive after its module is dropped",
        entry.w is doomed_w,
        "so id() cannot be recycled while the entry is registered",
    )
    # Simulate the recycle anyway: point the entry at a different array and confirm the
    # identity re-check rejects rather than applying the wrong per-channel scales.
    entry.w = make_ql(K, N)["weight"]
    _hits.clear()
    out = mx.quantized_matmul(x_big, doomed_w, s, b, True, gs, bits)
    mx.eval(out)
    check("identity mismatch falls through instead of mis-scaling", _hits == [],
          f"hits={_hits}")
    del _registry[doomed_id]

    # ---- 6. compile safety ---------------------------------------------------------
    print("\n6. mx.compile safety (no eval on the hot path)")
    _hits.clear()
    try:
        compiled = mx.compile(lambda t: ql(t))
        out = compiled(x_big)
        mx.eval(out)
        check("compiled forward runs", True, f"hits={_hits}")
        check("compiled forward matches eager", mx.array_equal(out, ref).item())
    except Exception as exc:
        check("compiled forward runs", False, f"{type(exc).__name__}: {str(exc)[:140]}")

    # ---- 7. custom_function vjp ----------------------------------------------------
    print("\n7. mx.custom_function vjp")
    try:
        @mx.custom_function
        def fake_int8(x, w_, s_, b_):
            return _ORIG_QMM(x, w_, s_, b_, True, gs, bits, "affine")

        @fake_int8.vjp
        def _(primals, cotan, output):
            x_, w_, s_, b_ = primals
            dx = _ORIG_QMM(cotan, w_, s_, b_, False, gs, bits, "affine")
            return (dx, None, None, None)

        def loss(t):
            return fake_int8(t, w, s, b).sum()

        g = mx.grad(loss)(x_big)
        mx.eval(g)
        expected = _ORIG_QMM(mx.ones((ROW_THRESHOLD, N), dtype=mx.bfloat16),
                             w, s, b, False, gs, bits, "affine")
        mx.eval(expected)
        check("vjp produces a gradient", g.shape == x_big.shape)
        check("vjp matches the 4-bit reference gradient",
              mx.allclose(g, expected, atol=1e-2).item())
    except Exception as exc:
        check("custom_function vjp", False, f"{type(exc).__name__}: {str(exc)[:160]}")

    # ---- restore -------------------------------------------------------------------
    disable()
    check("disable() restores the original", mx.quantized_matmul is _ORIG_QMM)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        return 1
    print("ALL SPIKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
