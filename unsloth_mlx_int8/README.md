# unsloth_mlx_int8

int8 W8A8 prefill acceleration for MLX quantized models on Apple M5.

## What this is

MLX computes quantized matmuls in 16-bit regardless of how the weights are stored, so a
4-bit model prefills no faster than an 8-bit one. On M5, whose neural accelerators run
int8 at roughly twice the fp16 rate, routing prefill-sized matmuls through an
int8 x int8 -> int32 GEMM recovers that. Decode is memory-bound and stays on MLX's
kernels, untouched.

Derived from [JetBrains' int8 NAX prefill patch for
mlx-vlm](https://github.com/JetBrains/mlx-vlm/tree/feature/int8-prefill) (MIT), which
measured +29.5% at 6.4k tokens and +47.3% at 11.8k on Qwen3.6-27B-4bit. This version
patches at op level rather than layer level, covers group sizes 32/64/128, is safe under
`mx.compile` and LoRA backward, and self-tests on the device before it will dispatch.

## Status: the speedup is unverified

**Nobody has run this on an M5.** There is no M5 in any CI runner tier, and none was
available while building it. What *is* verified, on Linux and in principle on a macOS
M1 runner:

| | Verified | How |
| --- | --- | --- |
| Dispatch: what gets intercepted, what falls through bit-identically | yes | Linux, 69 tests |
| The W8A8 arithmetic and its error vs MLX's 4-bit op | yes | Linux, via the portable backend |
| `mx.compile` safety, LoRA backward, idempotency, no-op path | yes | Linux |
| Packed-weight unpacking in real Metal | pending | macOS M1 CI (kernels are MPP-free by design) |
| The GEMM kernel's tile conventions | **no** | needs M5 |
| Any performance claim | **no** | needs M5 |

If you have an M5, `python bench/mlx_int8_bench.py --model mlx-community/Qwen3.6-27B-4bit`
and please report what it says.

## Usage

```python
import unsloth_mlx_int8

if unsloth_mlx_int8.enable():        # False, and a no-op, off M5
    unsloth_mlx_int8.warmup(model)   # required: builds the allow-list
    ok, detail = unsloth_mlx_int8.self_test()
    if not ok:
        print("int8 path disabled:", detail)
```

`warmup()` is not optional. Nothing is accelerated until a weight is registered, and
registration is also where every `mx.eval` happens -- which is what leaves the hot path
safe inside `mx.compile`.

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `UNSLOTH_MLX_INT8_PREFILL` | unset | `0` disables entirely; `force` skips the capability probe |
| `UNSLOTH_MLX_INT8_BACKEND` | `metal_mpp` | `portable` runs the same arithmetic in plain MLX ops, anywhere |
| `UNSLOTH_MLX_INT8_ROW_THRESHOLD` | `512` | below this, calls keep MLX's kernels |
| `UNSLOTH_MLX_INT8_ALLOW_8BIT` | `0` | affine 8-bit, off for the reason below |
| `UNSLOTH_MLX_INT8_EXACT_SCALES` | `0` | exact absmax instead of the analytic bound |
| `UNSLOTH_MLX_INT8_VERIFY` | `0` | shadow mode: log max relative error per intercepted call |

## Two design notes worth knowing

**Affine 8-bit is off by default, and that is arithmetic rather than caution.**
Requantizing affine-8 to symmetric int8 has zero bits of headroom. Let `R` be a channel's
inter-group range ratio: the requant step is `max_g range_g / 254` against a native step
of `range_g / (2**bits - 1)`. At 4 bits that absorbs `R` up to about 17, which is the
only reason this technique works at all. At 8 bits any channel with `R > ~1.2` comes out
strictly worse than the weights it started from. Turn it on only with a KLD comparison in
hand.

**MoE models get nothing.** Expert FFNs go through `mx.gather_qmm`
(`mlx_lm/models/switch_layers.py:76`), not `mx.quantized_matmul`, and on a Qwen3-MoE-class
model those are the dominant prefill FLOPs. A grouped GEMM over ragged per-expert row
counts is a separate kernel, deferred. The registry already carries an optional expert
dimension so adding it is a GEMM rather than a rework.

## Testing

```bash
python -m pytest tests/mlx_int8/ -q          # 69 tests, runs on any MLX backend
python scripts/spike_mlx_int8_patch.py       # architectural assumptions, standalone
```

One test skips on Linux: MLX 0.32.1's CUDA backend raises `cuGraphAddKernelNode
"invalid argument"` for `mx.quantized_matmul` at `group_size=128`. That is stock MLX,
reproducible with none of this module loaded, and absent on Metal. Our own path
evaluates fine at that group size; it is the reference we cannot build on Linux.
