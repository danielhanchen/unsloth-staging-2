"""Phase 1 probe of a GitHub macos-15 runner for torch MPS diffusion work.

Answers: is MPS usable at all, matmul / SDPA throughput per dtype, does torch._int_mm / torchao int8
run on MPS, does torch.compile on MPS help a DiT-like block, and MLX quantized matmul speed plus the
torch <-> MLX hand-off cost. Every timing: 2 warm-ups, median of 5 with spread.
"""

import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback

OUT = sys.argv[1] if len(sys.argv) > 1 else "probe.json"
rec = {"python": sys.version.split()[0], "machine": platform.machine(), "mac_ver": platform.mac_ver()[0]}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell = True, capture_output = True, text = True, timeout = 60).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return repr(exc)


rec["cpu"] = sh("sysctl -n machdep.cpu.brand_string")
rec["ncpu"] = sh("sysctl -n hw.ncpu")
rec["memsize_gib"] = round(int(sh("sysctl -n hw.memsize") or 0) / 2**30, 2)
rec["gpu"] = sh("system_profiler SPDisplaysDataType | head -20")
print(json.dumps(rec, indent = 1), flush = True)


def save():
    with open(OUT, "w") as f:
        json.dump(rec, f, indent = 1)


def log(key, val):
    rec[key] = val
    print(key, json.dumps(val), flush = True)
    save()


import torch
import torch.nn.functional as F

log("torch", torch.__version__)
mps = torch.backends.mps.is_available()
log("mps_available", mps)
log("mps_built", torch.backends.mps.is_built())
if not mps:
    save()
    sys.exit(0)
try:
    log("recommended_max_memory_gib", round(torch.mps.recommended_max_memory() / 2**30, 2))
except Exception as exc:  # noqa: BLE001
    log("recommended_max_memory_gib", repr(exc))


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def timed(fn, dev = "mps", reps = 5, warm = 2):
    for _ in range(warm):
        fn()
    sync(dev)
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync(dev)
        out.append(time.perf_counter() - t)
    return {"med_ms": round(statistics.median(out) * 1e3, 3), "spread_ms": round((max(out) - min(out)) * 1e3, 3)}


def guarded(name, fn):
    try:
        log(name, fn())
    except Exception as exc:  # noqa: BLE001
        log(name, {"error": repr(exc)[:400], "tb": traceback.format_exc()[-600:]})


# bf16 probe as Studio does it
guarded("bf16_probe", lambda: bool(torch.isfinite((torch.ones(2, dtype = torch.bfloat16, device = "mps") * 2).float()).all().item()))

# GEMM throughput at Z-Image-like shapes: tokens 1024 (512px) + text, dim 3840, ffn 10240
M, K, N = 1280, 3840, 3840


def gemm(dev, dt, m = M, k = K, n = N):
    a = torch.randn(m, k, device = dev, dtype = dt)
    b = torch.randn(n, k, device = dev, dtype = dt)
    r = timed(lambda: F.linear(a, b), dev)
    r["tflops"] = round(2 * m * k * n / (r["med_ms"] / 1e3) / 1e12, 3)
    return r


for dt in (torch.float32, torch.float16, torch.bfloat16):
    guarded(f"gemm_mps_{str(dt)[6:]}", lambda dt = dt: gemm("mps", dt))
for dt in (torch.float32, torch.bfloat16):
    guarded(f"gemm_cpu_{str(dt)[6:]}", lambda dt = dt: gemm("cpu", dt))

# SDPA at Z-Image 512 px (30 heads x 128, 1024 image + ~256 text tokens) and 1024 px (4096 + 256)
for side in (512, 1024):
    S = (side // 16) ** 2 + 256
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        def sdpa(dt = dt, S = S):
            q, k, v = (torch.randn(1, 30, S, 128, device = "mps", dtype = dt) for _ in range(3))
            r = timed(lambda: F.scaled_dot_product_attention(q, k, v))
            r["tflops"] = round(4 * 30 * S * S * 128 / (r["med_ms"] / 1e3) / 1e12, 3)
            return r
        guarded(f"sdpa_{side}_{str(dt)[6:]}", sdpa)

# int8 on MPS: torch._int_mm and torchao
def int_mm():
    a = torch.randint(-127, 127, (M, K), dtype = torch.int8, device = "mps")
    b = torch.randint(-127, 127, (K, N), dtype = torch.int8, device = "mps")
    r = timed(lambda: torch._int_mm(a, b))
    r["tops"] = round(2 * M * K * N / (r["med_ms"] / 1e3) / 1e12, 3)
    return r


guarded("int_mm_mps", int_mm)


def int8_dequant_linear():
    # weight-only int8 the simple way: int8 weight + per-row scale, dequant then bf16 linear
    a = torch.randn(M, K, device = "mps", dtype = torch.bfloat16)
    w = torch.randint(-127, 127, (N, K), dtype = torch.int8, device = "mps")
    s = torch.rand(N, 1, device = "mps", dtype = torch.bfloat16)
    return timed(lambda: F.linear(a, w.to(torch.bfloat16) * s))


guarded("int8_dequant_linear_mps_bf16", int8_dequant_linear)


def mixed_mm():
    # aten _weight_int8pack_mm exists for MPS (used by torchao int8 weight-only on MPS / gpt-fast)
    a = torch.randn(M, K, device = "mps", dtype = torch.bfloat16)
    w = torch.randint(-127, 127, (N, K), dtype = torch.int8, device = "mps")
    s = torch.rand(N, device = "mps", dtype = torch.bfloat16)
    ref = F.linear(a.float(), w.float() * s.float()[:, None])
    out = torch._weight_int8pack_mm(a, w, s)
    r = timed(lambda: torch._weight_int8pack_mm(a, w, s))
    r["rel_err"] = float((out.float() - ref).norm() / ref.norm())
    return r


guarded("weight_int8pack_mm_mps_bf16", mixed_mm)


def torchao_int8():
    import torchao
    from torchao.quantization import quantize_
    out = {"torchao": torchao.__version__}
    for cfg_name in ("Int8WeightOnlyConfig", "Int8DynamicActivationInt8WeightConfig"):
        try:
            import torchao.quantization as tq
            cfg = getattr(tq, cfg_name)()
            lin = torch.nn.Sequential(torch.nn.Linear(K, N, bias = False)).to("mps", torch.bfloat16)
            x = torch.randn(M, K, device = "mps", dtype = torch.bfloat16)
            ref = lin(x).float()
            quantize_(lin, cfg)
            y = lin(x).float()
            r = timed(lambda: lin(x))
            r["rel_err"] = float((y - ref).norm() / ref.norm())
            r["weight_type"] = type(lin[0].weight).__name__
            out[cfg_name] = r
        except Exception as exc:  # noqa: BLE001
            out[cfg_name] = {"error": repr(exc)[:300]}
    return out


guarded("torchao_mps", torchao_int8)


# torch.compile on MPS: DiT-ish block (norm, qkv, sdpa, out, mlp) eager vs compiled
class Block(torch.nn.Module):
    def __init__(self, d = 1536, h = 12):
        super().__init__()
        self.h = h
        self.n1 = torch.nn.LayerNorm(d)
        self.qkv = torch.nn.Linear(d, 3 * d)
        self.o = torch.nn.Linear(d, d)
        self.n2 = torch.nn.LayerNorm(d)
        self.f1 = torch.nn.Linear(d, 4 * d)
        self.f2 = torch.nn.Linear(4 * d, d)

    def forward(self, x):
        B, S, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, S, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, S, D)
        x = x + self.o(a)
        return x + self.f2(F.gelu(self.f1(self.n2(x)), approximate = "tanh"))


def compile_block():
    out = {}
    blk = torch.nn.Sequential(*[Block() for _ in range(4)]).to("mps", torch.bfloat16).eval()
    x = torch.randn(1, 1280, 1536, device = "mps", dtype = torch.bfloat16)
    with torch.no_grad():
        ref = blk(x).float()
        out["eager"] = timed(lambda: blk(x))
        t = time.perf_counter()
        c = torch.compile(blk)
        y = c(x).float()
        torch.mps.synchronize()
        out["compile_s"] = round(time.perf_counter() - t, 2)
        out["compiled"] = timed(lambda: c(x))
        out["max_diff"] = float((y - ref).abs().max())
    return out


os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.abspath("inductor_cache"))
guarded("compile_mps", compile_block)

# MLX
def mlx_probe():
    import mlx.core as mx
    import numpy as np
    out = {"mlx": getattr(mx, "__version__", "?"), "metal": bool(mx.metal.is_available()), "device": str(mx.default_device())}

    def mt(fn):
        for _ in range(2):
            mx.eval(fn())
        ts = []
        for _ in range(5):
            t = time.perf_counter()
            mx.eval(fn())
            ts.append(time.perf_counter() - t)
        return {"med_ms": round(statistics.median(ts) * 1e3, 3), "spread_ms": round((max(ts) - min(ts)) * 1e3, 3)}

    a = mx.random.normal((M, K)).astype(mx.bfloat16)
    w = mx.random.normal((N, K)).astype(mx.bfloat16)
    r = mt(lambda: a @ w.T)
    r["tflops"] = round(2 * M * K * N / (r["med_ms"] / 1e3) / 1e12, 3)
    out["mlx_gemm_bf16"] = r
    a16 = a.astype(mx.float16)
    w16 = w.astype(mx.float16)
    r = mt(lambda: a16 @ w16.T)
    r["tflops"] = round(2 * M * K * N / (r["med_ms"] / 1e3) / 1e12, 3)
    out["mlx_gemm_fp16"] = r
    ref = (a.astype(mx.float32) @ w.astype(mx.float32).T)
    for bits in (8, 4):
        wq, sc, bi = mx.quantize(w, group_size = 64, bits = bits)
        f = lambda: mx.quantized_matmul(a, wq, sc, bi, transpose = True, group_size = 64, bits = bits)
        r = mt(f)
        y = f().astype(mx.float32)
        r["rel_err"] = float(mx.sqrt(((y - ref) ** 2).sum() / (ref ** 2).sum()).item())
        r["tflops_equiv"] = round(2 * M * K * N / (r["med_ms"] / 1e3) / 1e12, 3)
        out[f"mlx_qmm_{bits}bit_g64"] = r
    q = mx.random.normal((1, 30, 1280, 128)).astype(mx.bfloat16)
    out["mlx_sdpa_512_bf16"] = mt(lambda: mx.fast.scaled_dot_product_attention(q, q, q, scale = 128 ** -0.5))
    # torch MPS -> MLX -> torch MPS hand-off for one activation (1280 x 3840 bf16)
    x = torch.randn(M, K, device = "mps", dtype = torch.float16)

    def handoff():
        arr = mx.array(x.cpu().numpy())
        mx.eval(arr)
        back = torch.from_numpy(np.array(arr)).to("mps")
        return back

    out["torch_mlx_roundtrip_ms"] = timed(handoff)
    return out


guarded("mlx", mlx_probe)
save()
print("DONE", flush = True)
