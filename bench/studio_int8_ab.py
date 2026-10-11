"""Studio's own NativeWeightOnlyLinear on real Z-Image-Turbo blocks, MPS, one arm per process.

    python bench/studio_int8_ab.py fetch                      # truncated real weights -> zi_trunc.safetensors (bf16)
    python bench/studio_int8_ab.py run ARM SIDE OUT.json      # ARM: dense | base_native | head_native

dense = Studio on MPS today (bf16 transformer). base_native / head_native = Studio's int8 weight-only Linear from
origin/main (dequantise per forward) / from the branch (``_weight_int8pack_mm``), swapped in by
``apply_native_weight_quant`` with Studio's own filter (``make_filter_fn``, adaLN kept dense like Studio's int8 rule).
"""

import importlib.util
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "head_backend"))
CFG = dict(all_patch_size = (2,), all_f_patch_size = (1,), in_channels = 16, dim = 3840, n_layers = 1,
           n_refiner_layers = 1, n_heads = 30, n_kv_heads = 30, norm_eps = 1e-5, qk_norm = True,
           cap_feat_dim = 2560, rope_theta = 256.0, t_scale = 1000.0, axes_dims = [32, 48, 48],
           axes_lens = [1536, 512, 512])
WEIGHTS = "zi_trunc.safetensors"
DEV = os.environ.get("AB_DEV", "mps")


def sync():
    if DEV == "mps":
        torch.mps.synchronize()


def fetch():
    os.environ.update(ZI_KEEP_SD = "1", ZI_DEV = "cpu", ZI_LAYERS = "1", ZI_REFINERS = "1", ZI_GEMM = "0", ZI_ARMS = "", ZI_VAE = "0")
    sys.argv = ["zimage_mps.py", "out/_fetch.json"]
    import runpy
    g = runpy.run_path(os.path.join(HERE, "zimage_mps.py"), run_name = "__fetch__")
    from safetensors.torch import save_file
    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in g["SD"].items()} if "SD" in g else None
    if sd is None:
        raise SystemExit("zimage_mps.py dropped SD")
    save_file(sd, WEIGHTS)
    print("saved", len(sd))


def load_native(which):
    if which == "head":
        import core.inference.diffusion_native_quant as m
        return m
    path = os.path.join(HERE, "base_native_quant.py")
    spec = importlib.util.spec_from_file_location("core.inference._base_native_quant", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run(arm, side, out):
    from diffusers import ZImageTransformer2DModel
    from safetensors.torch import load_file

    with torch.device("meta"):
        model = ZImageTransformer2DModel(**CFG)
    model.load_state_dict(load_file(WEIGHTS), strict = False, assign = True)
    model = model.to(DEV).eval()
    rec = {"arm": arm, "side": side, "torch": torch.__version__}
    if arm != "dense":
        import types
        import core.inference.diffusion_transformer_quant as tq
        if arm == "head_native":
            # Studio's own entry point, as the loader calls it for a pinned transformer_quant=int8 on MPS.
            got = tq.quantize_transformer(types.SimpleNamespace(transformer = model),
                                          types.SimpleNamespace(device = "mps", dtype = torch.bfloat16),
                                          mode = "int8", family = "z-image")
            assert got == "int8", got
        else:
            nq = load_native("base")
            filt = tq.make_filter_fn(tq.DEFAULT_MIN_LINEAR_FEATURES,
                                     exclude_name_tokens = tq.exclude_tokens_for_scheme("int8", "z-image") + ("lora_",),
                                     require_bf16 = True)
            nq.apply_native_weight_quant(model, "int8", filter_fn = filt, act_int8 = False)
        rec["swapped"] = sum(1 for m in model.modules() if type(m).__name__ == "NativeWeightOnlyLinear")
        if arm == "head_native":
            calls = [0]
            real = torch._weight_int8pack_mm

            def counted(*a):
                calls[0] += 1
                return real(*a)

            torch._weight_int8pack_mm = counted
    if DEV == "mps":
        torch.mps.empty_cache()
        rec["alloc_gib"] = round(torch.mps.current_allocated_memory() / 2**30, 3)
    g = torch.Generator().manual_seed(0)
    lat = side // 8
    x = torch.randn(16, 1, lat, lat, generator = g).to(DEV, torch.bfloat16)
    cap = (torch.randn(128, 2560, generator = g) * 0.5).to(DEV, torch.bfloat16)
    t = torch.tensor([0.5]).to(DEV, torch.bfloat16)

    def step():
        with torch.no_grad():
            o = model([x], t, [cap], return_dict = False)[0]
        return o[0] if isinstance(o, (list, tuple)) else o

    for _ in range(2):
        step()
    sync()
    if arm == "head_native":
        calls[0] = 0
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        step()
        sync()
        ts.append(time.perf_counter() - t0)
    rec["step_ms"] = [round(v * 1e3, 1) for v in ts]
    rec["med_ms"] = round(statistics.median(ts) * 1e3, 1)
    if arm == "head_native":
        rec["int8pack_calls_per_step"] = calls[0] / 5
    torch.save(step().float().cpu(), out.replace(".json", ".pt"))
    json.dump(rec, open(out, "w"), indent = 1)
    print(json.dumps(rec), flush = True)


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        fetch()
    else:
        run(sys.argv[2], int(sys.argv[3]), sys.argv[4])
