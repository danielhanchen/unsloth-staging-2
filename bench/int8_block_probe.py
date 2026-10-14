"""Whole-block vs single-layer Metal int8 probe on synthetic DiT blocks (random bf16 weights, real shapes).
For each block: PR #13344's single-layer verdict (largest layer, kernel vs dense bf16 GEMM), the whole-block verdict
(every layer through its own forward, kernel on vs off), and the dense bf16 block for reference. Usage: <pr_root> <head_root> <out.json>"""
import importlib, json, sys, time
import torch

pr_root, head_root, out = sys.argv[1:4]
SHAPES = {  # one transformer block's linears (in, out)
    "zimage_like_d3840": [(3840, 3840)] * 4 + [(3840, 10240), (10240, 3840), (3840, 10240), (256, 15360)],
    "flux_like_d3072": [(3072, 3072)] * 4 + [(3072, 12288), (12288, 3072), (3072, 18432)],
    "small_d1536": [(1536, 1536)] * 4 + [(1536, 6144), (6144, 1536)],
}


class Block(torch.nn.Module):
    def __init__(self, shapes):
        super().__init__()
        self.lins = torch.nn.ModuleList([torch.nn.Linear(i, o, bias = False) for i, o in shapes])


class Model(torch.nn.Module):
    def __init__(self, shapes):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block(shapes), Block(shapes)])


def load(root):
    for k in [k for k in sys.modules if k == "core" or k.startswith("core.")]:
        del sys.modules[k]
    sys.path.insert(0, root + "/studio/backend")
    nq = importlib.import_module("core.inference.diffusion_native_quant")
    sys.path.pop(0)
    return nq


def median_s(fn, reps = 7):
    ts = []
    for _ in range(reps):
        torch.mps.synchronize(); t0 = time.perf_counter(); fn(); torch.mps.synchronize(); ts.append(time.perf_counter() - t0)
    return sorted(ts)[reps // 2]


res = {"torch": torch.__version__}
for name, shapes in SHAPES.items():
    r = {}
    for arm, root in (("pr13344", pr_root), ("block", head_root)):
        nq = load(root)
        torch.manual_seed(0)
        model = Model(shapes).to(torch.bfloat16)
        dense_layers = [l for l in model.blocks[0].lins]
        xs = {i: torch.randn(nq._INT8PACK_PROBE_ROWS, i, dtype = torch.bfloat16, device = "mps") for i, _ in shapes}
        if arm == "pr13344":
            dm = Model(shapes).to(torch.bfloat16).to("mps")
            dense_s = median_s(lambda: [l(xs[l.in_features]) for l in dm.blocks[0].lins])
            r["dense_bf16_block_ms"] = dense_s * 1e3
            del dm
        nq.apply_native_weight_quant(model, "int8", filter_fn = lambda m, n: isinstance(m, torch.nn.Linear),
                                     act_int8 = False, scale_dtype = torch.bfloat16)
        model.to("mps")
        nq._INT8PACK_PROBED.clear()
        t0 = time.perf_counter()
        verdict = nq.probe_mps_int8pack(model, torch.bfloat16)
        r[f"{arm}_probe_s"] = time.perf_counter() - t0
        r[f"{arm}_verdict_kernel"] = verdict
        layers = list(model.blocks[0].lins)
        with torch.no_grad():
            for on in (True, False):
                nq._INT8PACK_PROBED[torch.bfloat16] = on
                r[f"{arm}_block_{'kernel' if on else 'fallback'}_ms"] = median_s(
                    lambda: [l(xs[l.in_features]) for l in layers]) * 1e3
        nq._INT8PACK_PROBED.clear()
        if arm == "pr13344":
            big = max(layers, key = lambda m: m.in_features * m.out_features)
            k, d = nq._bench_int8pack_vs_dense(big, torch.bfloat16, torch.device("mps"))
            r["pr13344_single_layer"] = {"shape": [big.in_features, big.out_features], "kernel_ms": k * 1e3, "dense_ms": d * 1e3, "speedup": d / k}
        del model
        torch.mps.empty_cache()
    r["block_kernel_vs_fallback"] = r["block_block_fallback_ms"] / r["block_block_kernel_ms"]
    r["block_kernel_vs_dense_bf16"] = r["dense_bf16_block_ms"] / r["block_block_kernel_ms"]
    res[name] = r
    print(name, json.dumps(r, indent = 1), flush = True)
json.dump(res, open(out, "w"), indent = 1)
assert all(isinstance(res[n]["block_verdict_kernel"], bool) for n in SHAPES)
