"""One arm, fresh process: real-shape Qwen-Image-2.1 blocks (dim 4096, 32 x 128 heads, mlp x3, bf16, random weights)
on MPS through Studio's bounded MPS attention processor, the fast step's cached-decode call (target rows only, prefix
K/V cached, all-True tail modulation mask). Head trees also install diffusion_qwenimage21_mps."""

import argparse
import ctypes
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

ap = argparse.ArgumentParser()
ap.add_argument("--backend-dir", required = True)
ap.add_argument("--out", required = True)
ap.add_argument("--tag", required = True)
ap.add_argument("--blocks", type = int, default = 2)
ap.add_argument("--seq", type = int, default = 4096)
ap.add_argument("--prefix", type = int, default = 512)
ap.add_argument("--warmup", type = int, default = 3)
ap.add_argument("--iters", type = int, default = 8)
ap.add_argument("--smoke", action = "store_true", help = "CPU, dim 256: harness check only")
ap.add_argument("--ops", action = "store_true", help = "also time each op stock vs fused (head only)")
args = ap.parse_args()
sys.path.insert(0, os.path.abspath(args.backend_dir))

import torch  # noqa: E402
from diffusers.models.transformers import transformer_qwenimage21 as qmod  # noqa: E402

from core.inference import diffusion_qwenimage21_math as bounded  # noqa: E402

_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib") if sys.platform == "darwin" else None


def lifetime_peak_gib():
    if _libc is None:
        return 0.0
    buf = (ctypes.c_uint64 * 64)()
    if _libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) != 0:
        return None
    return int(buf[2 + 28]) / 2**30


dev = torch.device("cpu" if args.smoke else "mps")
dim, heads, hd = (256, 2, 128) if args.smoke else (4096, 32, 128)
if args.smoke:
    torch.mps.synchronize = lambda: None
    torch.mps.driver_allocated_memory = lambda: 0
    bounded.mps_score_budget = lambda: 2**30
torch.manual_seed(0)
blocks = []
for i in range(args.blocks):
    b = qmod.QwenImage21TransformerBlock(dim, heads, hd, mlp_ratio = 3, eps = 1e-6)
    with torch.no_grad():
        for n, p in b.named_parameters():
            if "norm" in n:
                p.copy_(1 + 0.1 * torch.randn_like(p))
    blocks.append(b.to(dev, torch.bfloat16).eval())
cls = bounded._processor_class()
for b in blocks:
    proc = cls()
    proc.__dict__.update(b.attn.processor.__dict__)
    proc._unsloth_score_budget = bounded.mps_score_budget()
    b.attn.set_processor(proc)


class QwenImage21Transformer2DModel:
    def __init__(self, blocks):
        self.transformer_blocks = blocks

    def parameters(self):
        return self.transformer_blocks[0].parameters()


pipe = SimpleNamespace(transformer = QwenImage21Transformer2DModel(blocks))
head = False
try:
    from core.inference import diffusion_qwenimage21_mps as fused
    head = fused.install_for_pipe(pipe)
except ImportError:
    fused = None

g = torch.Generator().manual_seed(1)
S, P = args.seq, args.prefix
x0 = torch.randn(1, S, dim, generator = g).to(dev, torch.bfloat16)
modulation = (0.5 * torch.randn(2, 4 * dim, generator = g)).to(dev, torch.bfloat16)
mask_full = torch.arange(P + S) >= P
mask = mask_full.to(dev)[P:]
angles = torch.randn(P + S, hd // 2, generator = g) * 30
rot_full = torch.polar(torch.ones_like(angles), angles).to(dev)
rot = rot_full[P:]
caches = []
for b in blocks:
    c = qmod.QwenImage21KVLayerCache()
    c.store(
        torch.randn(1, P, heads, hd, generator = g).to(dev, torch.bfloat16),
        torch.randn(1, P, heads, hd, generator = g).to(dev, torch.bfloat16),
    )
    caches.append(c)


def step():
    x = x0
    with torch.no_grad():
        for b, c in zip(blocks, caches):
            x = b(
                hidden_states = x, modulation = modulation, rotary_emb = rot, target_token_mask = mask,
                layer_cache = c, kv_cache_mode = "cached",
            )
    return x


def prefill_case():
    """Small masked prefill (prefix rows + target rows, segments) for parity only."""
    gg = torch.Generator().manual_seed(2)
    s = min(640, P + S)
    p = s // 5
    x = torch.randn(1, s, dim, generator = gg).to(dev, torch.bfloat16)
    m = (torch.arange(s) >= p).to(dev)
    r = rot_full[:s]
    with torch.no_grad():
        return blocks[0](
            hidden_states = x, modulation = modulation, rotary_emb = r, target_token_mask = m,
            segments = [(0, p, True)],
        )


times = []
for i in range(args.warmup + args.iters):
    torch.mps.synchronize()
    t0 = time.perf_counter()
    out = step()
    torch.mps.synchronize()
    dt = time.perf_counter() - t0
    if i >= args.warmup:
        times.append(dt)
pre = prefill_case()
torch.mps.synchronize()
os.makedirs(args.out, exist_ok = True)
torch.save({"decode": out.float().cpu(), "prefill": pre.float().cpu()}, os.path.join(args.out, f"{args.tag}.pt"))
res = {
    "tag": args.tag,
    "head_installed": bool(head),
    "torch": torch.__version__,
    "blocks": args.blocks,
    "seq": S,
    "prefix": P,
    "times": times,
    "median_s_per_call": statistics.median(times),
    "median_s_per_block": statistics.median(times) / args.blocks,
    "est_s_per_step_32_blocks": statistics.median(times) / args.blocks * 32,
    "lifetime_peak_phys_gib": lifetime_peak_gib(),
    "driver_alloc_gib": torch.mps.driver_allocated_memory() / 2**30,
    "env": {k: v for k, v in os.environ.items() if k.startswith("UNSLOTH_DIFFUSION_Q21_MPS")},
}
if fused is not None:
    res["stats"] = dict(fused.STATS)
    res["probes"] = dict(fused._PROBED)
    res["lib_failed"] = bool(fused._LIB[1])

if args.ops and fused is not None and head:
    def timeit(fn, n = 20):
        for _ in range(3):
            fn()
        torch.mps.synchronize()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            torch.mps.synchronize()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts), min(ts), max(ts)

    b = blocks[0]
    xn = b.img_norm1(x0)
    mod1, _ = modulation.chunk(2, dim = -1)
    scale, gate = mod1.chunk(2, dim = -1)
    y = torch.randn(1, S, dim, generator = g).to(dev, torch.bfloat16)
    q = torch.randn(1, S, heads, hd, generator = g).to(dev, torch.bfloat16)
    stock_mod = lambda: qmod.QwenImage21TransformerBlock._modulate(b, xn, mod1, mask)[0]  # noqa: E731
    fused_mod = lambda: fused.modulate(xn, 1 + scale, True, 0)  # noqa: E731
    stock_res = lambda: x0 + qmod._select_modulation_rows(gate, mask).tanh() * y  # noqa: E731
    fused_res = lambda: fused.gated_residual(x0, y, gate.tanh().contiguous(), True, 0)  # noqa: E731
    stock_rope = lambda: qmod.apply_rotary_emb_qwen(b.attn.norm_q(q).to(q.dtype), rot, use_real = False)  # noqa: E731
    fused_rope = lambda: fused.norm_rope(q, b.attn.norm_q, rot)  # noqa: E731
    ops = {}
    with torch.no_grad():
        for name, (s_fn, f_fn) in {
            "modulate": (stock_mod, fused_mod),
            "residual_metal": (stock_res, fused_res),
            "rope_metal": (stock_rope, fused_rope),
        }.items():
            a, bb = s_fn(), f_fn()
            if bb is None:
                ops[name] = {"fused": None}
                continue
            d = (a.float() - bb.float()).abs()
            # A/A: stock timed twice around the fused one
            s1, f1, s2 = timeit(s_fn), timeit(f_fn), timeit(s_fn)
            ops[name] = {
                "stock_ms": [1e3 * s1[0], 1e3 * s2[0]],
                "stock_minmax_ms": [1e3 * min(s1[1], s2[1]), 1e3 * max(s1[2], s2[2])],
                "fused_ms": 1e3 * f1[0],
                "fused_minmax_ms": [1e3 * f1[1], 1e3 * f1[2]],
                "bit_equal_frac": float((d == 0).float().mean()),
                "max_abs": float(d.max()),
            }
    res["ops"] = ops

with open(os.path.join(args.out, f"{args.tag}.json"), "w") as f:
    json.dump(res, f, indent = 1)
print(json.dumps({k: v for k, v in res.items() if k != "times"}, indent = 1))
