"""Phase 2: real Z-Image-Turbo weights on torch MPS (truncated transformer: REAL first main layers + refiners).

Arms (fresh weights per arm, same inputs): fp32 (reference), bf16 (Studio today), fp16, bf16+int8wo, fp16+int8wo,
bf16+compile (regional, blocks). int8wo = per-output-channel symmetric int8 weights through aten
``_weight_int8pack_mm`` (Metal kernel), applied to every Linear inside the transformer blocks except adaLN modulation.
Also: GEMM sweep at Z-Image shapes on real layer-0 weights, per-category time split (sync hooks), MPS CPU-fallback
warnings, VAE decode (dtype / tiling / channels_last). Writes out/zimage.json.
"""


import gc
import json
import os
import statistics
import struct
import sys
import time
import urllib.request
import warnings

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = sys.argv[1] if len(sys.argv) > 1 else "out/zimage.json"
N_LAYERS = int(os.environ.get("ZI_LAYERS", "2"))
N_REF = int(os.environ.get("ZI_REFINERS", "1"))
SIDE = int(os.environ.get("ZI_SIDE", "512"))
CAP = int(os.environ.get("ZI_CAP", "128"))
REPO = "Tongyi-MAI/Z-Image-Turbo"
SHARD = "transformer/diffusion_pytorch_model-00001-of-00003.safetensors"
rec = {"torch": torch.__version__, "layers": N_LAYERS, "refiners": N_REF, "side": SIDE, "cap": CAP}
try:
    import diffusers
    rec["diffusers"] = diffusers.__version__
except Exception:  # noqa: BLE001
    pass


def save():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok = True)
    with open(OUT, "w") as f:
        json.dump(rec, f, indent = 1)


def log(k, v):
    rec[k] = v
    print(k, json.dumps(v), flush = True)
    save()


DEV = os.environ.get("ZI_DEV", "mps")


def sync():
    if DEV == "mps":
        torch.mps.synchronize()


def empty_cache():
    if DEV == "mps":
        torch.mps.empty_cache()


def alloc_gib():
    return round(torch.mps.current_allocated_memory() / 2**30, 2) if DEV == "mps" else None


def timed(fn, reps = 5, warm = 2):
    for _ in range(warm):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    return {"med_ms": round(statistics.median(ts) * 1e3, 2), "spread_ms": round((max(ts) - min(ts)) * 1e3, 2),
            "n": reps}


# ---------- range-read only the needed tensors from the 9.3 GB fp32 shard ----------
def url(path):
    return f"https://huggingface.co/{REPO}/resolve/main/{path}"


def get_range(path, a, b):
    req = urllib.request.Request(url(path), headers = {"Range": f"bytes={a}-{b - 1}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout = 120) as r:
                data = r.read()
            if len(data) == b - a:
                return data
        except Exception as exc:  # noqa: BLE001
            print("range retry", attempt, exc, flush = True)
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"range read failed {path} {a}-{b}")


def load_needed(keep):
    n = struct.unpack("<Q", get_range(SHARD, 0, 8))[0]
    header = json.loads(get_range(SHARD, 8, 8 + n))
    base = 8 + n
    want = sorted(((k, v) for k, v in header.items() if k != "__metadata__" and keep(k)),
                  key = lambda kv: kv[1]["data_offsets"][0])
    # coalesce adjacent tensors into big range reads
    groups, cur = [], []
    for k, v in want:
        if cur and v["data_offsets"][0] != cur[-1][1]["data_offsets"][1]:
            groups.append(cur)
            cur = []
        cur.append((k, v))
    if cur:
        groups.append(cur)
    sd = {}
    dmap = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
    total = 0
    for g in groups:
        a, b = g[0][1]["data_offsets"][0], g[-1][1]["data_offsets"][1]
        # chunk at 512 MB
        buf = bytearray()
        s = a
        while s < b:
            e = min(b, s + (512 << 20))
            buf += get_range(SHARD, base + s, base + e)
            s = e
        total += b - a
        for k, v in g:
            o0, o1 = v["data_offsets"][0] - a, v["data_offsets"][1] - a
            t = torch.frombuffer(bytes(buf[o0:o1]), dtype = dmap[v["dtype"]]).reshape(v["shape"]).clone()
            sd[k] = t
    return sd, total


def keep_key(k):
    p = k.split(".")
    if p[0] == "layers":
        return int(p[1]) < N_LAYERS
    if p[0] in ("noise_refiner", "context_refiner"):
        return int(p[1]) < N_REF
    return True


from diffusers import ZImageTransformer2DModel  # noqa: E402

t0 = time.perf_counter()
SD, nbytes = load_needed(keep_key)
log("download", {"gib": round(nbytes / 2**30, 2), "s": round(time.perf_counter() - t0, 1), "tensors": len(SD),
                 "dtypes": sorted({str(v.dtype) for v in SD.values()})})
CFG = dict(all_patch_size = (2,), all_f_patch_size = (1,), in_channels = 16, dim = 3840, n_layers = N_LAYERS,
           n_refiner_layers = N_REF, n_heads = 30, n_kv_heads = 30, norm_eps = 1e-5, qk_norm = True,
           cap_feat_dim = 2560, rope_theta = 256.0, t_scale = 1000.0, axes_dims = [32, 48, 48],
           axes_lens = [1536, 512, 512])


def build(dtype):
    with torch.device("meta"):
        m = ZImageTransformer2DModel(**CFG)
    missing, unexpected = m.load_state_dict({k: v.to(dtype) for k, v in SD.items()}, strict = False, assign = True)
    assert not unexpected, unexpected[:5]
    # pad tokens etc may be buffers created at init; anything still meta is an error
    bad = [n for n, p in list(m.named_parameters()) + list(m.named_buffers()) if p.device.type == "meta"]
    assert not bad, bad[:5]
    return m.to(DEV).eval()


# ---------- int8 weight-only via _weight_int8pack_mm ----------
class I8Linear(nn.Module):
    def __init__(self, lin, act_dtype):
        super().__init__()
        w = lin.weight.detach().float()
        amax = w.abs().amax(dim = 1).clamp_min(1e-12)
        # round the scale to the activation dtype FIRST (upwards), then quantise against the rounded scale: no
        # systematic per-channel scale error and no clipping.
        s = (amax / 127.0).to(act_dtype)
        s = torch.where(s.float() * 127.0 < amax, torch.nextafter(s.float(), torch.full_like(s.float(), 1e30)).to(act_dtype), s)
        q = torch.round(w / s.float()[:, None]).clamp(-127, 127).to(torch.int8)
        self.register_buffer("w", q.contiguous())
        self.register_buffer("s", s.contiguous())
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().to(act_dtype), requires_grad = False)
        self.out_features, self.in_features = lin.out_features, lin.in_features

    def forward(self, x):
        shp = x.shape
        y = torch._weight_int8pack_mm(x.reshape(-1, shp[-1]).contiguous(), self.w, self.s)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shp[:-1], self.out_features)


def int8ify(model, act_dtype):
    n = 0
    for blk_list in (model.layers, model.noise_refiner, model.context_refiner):
        for blk in blk_list:
            for name, mod in list(blk.named_modules()):
                for cname, child in list(mod.named_children()):
                    full = f"{name}.{cname}" if name else cname
                    if isinstance(child, nn.Linear) and "adaLN" not in full:
                        setattr(mod, cname, I8Linear(child, act_dtype).to(DEV))
                        n += 1
    return n


# ---------- inputs ----------
g = torch.Generator().manual_seed(0)
LAT = SIDE // 8
x_lat = torch.randn(16, 1, LAT, LAT, generator = g)
cap = torch.randn(CAP, 2560, generator = g) * 0.5
t_in = torch.tensor([0.5])


def run(model, dtype):
    with torch.no_grad():
        out = model([x_lat.to(DEV, dtype)], t_in.to(DEV, dtype), [cap.to(DEV, dtype)], return_dict = False)[0]
    return out[0] if isinstance(out, (list, tuple)) else out


# ---------- category split via sync hooks ----------
def category_split(model, dtype):
    acc = {"linear": 0.0, "sdpa": 0.0}
    hooks = []

    def pre(m, a):
        sync()
        m._t = time.perf_counter()

    def post(m, a, o):
        sync()
        acc["linear"] += time.perf_counter() - m._t

    for m in model.modules():
        if isinstance(m, (nn.Linear, I8Linear)):
            hooks += [m.register_forward_pre_hook(pre), m.register_forward_hook(post)]
    orig = F.scaled_dot_product_attention

    def sdpa(*a, **k):
        sync()
        t = time.perf_counter()
        r = orig(*a, **k)
        sync()
        acc["sdpa"] += time.perf_counter() - t
        return r

    F.scaled_dot_product_attention = sdpa
    torch.nn.functional.scaled_dot_product_attention = sdpa
    try:
        run(model, dtype)
        sync()
        acc = {"linear": 0.0, "sdpa": 0.0}
        t = time.perf_counter()
        run(model, dtype)
        sync()
        tot = time.perf_counter() - t
    finally:
        F.scaled_dot_product_attention = orig
        for h in hooks:
            h.remove()
    return {"total_ms_synced": round(tot * 1e3, 1), "linear_ms": round(acc["linear"] * 1e3, 1),
            "sdpa_ms": round(acc["sdpa"] * 1e3, 1), "other_ms": round((tot - acc["linear"] - acc["sdpa"]) * 1e3, 1)}


def block_amax(model, dtype):
    vals = {}
    hooks = []
    for name, m in model.named_modules():
        if name.count(".") == 1 and name.split(".")[0] in ("layers", "noise_refiner", "context_refiner"):
            hooks.append(m.register_forward_hook(lambda mod, a, o, name = name: vals.__setitem__(name, float(o.float().abs().max()))))
        if name.endswith("feed_forward"):
            hooks.append(m.register_forward_hook(lambda mod, a, o, name = name: vals.__setitem__(name, float(o.float().abs().max()))))
    run(model, dtype)
    for h in hooks:
        h.remove()
    return vals


def metrics(out, ref):
    o, r = out.float().cpu(), ref.float().cpu()
    err = (o - r)
    mse = float((err ** 2).mean())
    peak = float(r.abs().max())
    return {"rel_l2": round(float(err.norm() / r.norm()), 5), "psnr_db": round(10 * torch.log10(torch.tensor(peak ** 2 / max(mse, 1e-20))).item(), 2),
            "cos": round(float(F.cosine_similarity(o.flatten(), r.flatten(), dim = 0)), 6),
            "nonfinite": int((~torch.isfinite(o)).sum())}


fallbacks = []
warnings.simplefilter("always")
_showwarning = warnings.showwarning


def _sw(message, category, filename, lineno, file = None, line = None):
    s = str(message)
    if "fall back" in s or "MPS" in s:
        fallbacks.append(s[:200])
    return _showwarning(message, category, filename, lineno, file, line)


warnings.showwarning = _sw

# ---------- GEMM sweep at Z-Image shapes, real layer-0 weights ----------
def gemm_sweep():
    res = []
    keys = [("attention.to_q.weight", "qkv/o 3840x3840"), ("feed_forward.w1.weight", "w1/w3 3840->10240"),
            ("feed_forward.w2.weight", "w2 10240->3840")]
    for key, label in keys:
        W = SD[f"layers.0.{key}"].float()
        N, K = W.shape
        for M in (256, (SIDE // 16) ** 2 + CAP, 4096 + CAP):
            X = torch.randn(M, K, generator = g)
            ref = (X @ W.t())
            row = {"shape": label, "M": M}
            for dt in (torch.bfloat16, torch.float16):
                lin = nn.Linear(K, N, bias = False)
                lin.weight.data = W.clone()
                i8 = I8Linear(lin, dt).to(DEV)
                wd = W.to(DEV, dt)
                xd = X.to(DEV, dt)
                tag = str(dt)[6:]
                row[f"linear_{tag}"] = timed(lambda: F.linear(xd, wd))
                row[f"int8wo_{tag}"] = timed(lambda: i8(xd))
                row[f"err_linear_{tag}"] = round(float((F.linear(xd, wd).float().cpu() - ref).norm() / ref.norm()), 5)
                row[f"err_int8wo_{tag}"] = round(float((i8(xd).float().cpu() - ref).norm() / ref.norm()), 5)
                del i8, wd, xd
            res.append(row)
            print(json.dumps(row), flush = True)
            empty_cache()
    return res


try:
    if os.environ.get("ZI_GEMM", "1") == "1":
        log("gemm_sweep", gemm_sweep())
except Exception as exc:  # noqa: BLE001
    log("gemm_sweep", {"error": repr(exc)[:400]})

# ---------- model arms ----------
arms = {}
ref_out = None
order = os.environ.get("ZI_ARMS", "bf16,fp16,bf16_int8wo,fp16_int8wo,bf16_compile,fp32").split(",")
order = [a for a in order if a]
DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
outs = {}
hidden = {}
for arm in order:
    dt = DT[arm.split("_")[0]]
    r = {}
    try:
        m = build(dt)
        if "int8wo" in arm:
            r["int8_linears"] = int8ify(m, dt)
            gc.collect()
        if "compile" in arm:
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.abspath("inductor_cache"))
            t = time.perf_counter()
            m.compile_repeated_blocks(fullgraph = False) if hasattr(m, "compile_repeated_blocks") else None
            run(m, dt)
            sync()
            r["compile_first_call_s"] = round(time.perf_counter() - t, 1)
        r["mps_alloc_gib"] = alloc_gib()
        r["step"] = timed(lambda: run(m, dt), reps = 5, warm = 2)
        if "compile" not in arm:
            r["split"] = category_split(m, dt)
        if arm in ("bf16", "fp16", "fp32"):
            r["block_amax"] = block_amax(m, dt)
        cap_h = {}
        hk = m.layers[-1].register_forward_hook(lambda mod, a, o: cap_h.__setitem__("h", o.float().cpu()))
        outs[arm] = run(m, dt).float().cpu()
        hk.remove()
        hidden[arm] = cap_h["h"]
        del m
    except Exception as exc:  # noqa: BLE001
        import traceback
        r["error"] = repr(exc)[:400]
        r["tb"] = traceback.format_exc()[-800:]
    gc.collect()
    empty_cache()
    arms[arm] = r
    log("arms", arms)

if "fp32" in outs:
    log("accuracy_vs_fp32", {a: metrics(o, outs["fp32"]) for a, o in outs.items() if a != "fp32"})
    log("hidden_accuracy_vs_fp32", {a: metrics(o, hidden["fp32"]) for a, o in hidden.items() if a != "fp32"})
if "bf16" in outs and "bf16_int8wo" in outs:
    log("int8wo_vs_bf16", {"out": metrics(outs["bf16_int8wo"], outs["bf16"]), "hidden": metrics(hidden["bf16_int8wo"], hidden["bf16"])})
log("mps_fallback_warnings", sorted(set(fallbacks))[:20])
del SD
gc.collect()

# ---------- VAE decode ----------
def vae_bench():
    from diffusers import AutoencoderKL
    from huggingface_hub import snapshot_download
    p = snapshot_download(REPO, allow_patterns = ["vae/*"], local_dir = "zimg")
    res = {}
    lat_dec = {}
    for side in (512, 1024):
        z = torch.randn(1, 16, side // 8, side // 8, generator = g)
        for arm in ("fp32", "bf16", "fp16", "bf16_tiled", "bf16_cl", "fp32_tiled"):
            dt = DT[arm.split("_")[0]]
            try:
                vae = AutoencoderKL.from_pretrained(f"{p}/vae", torch_dtype = dt).to(DEV).eval()
                if "tiled" in arm:
                    vae.enable_tiling()
                if arm.endswith("_cl"):
                    vae = vae.to(memory_format = torch.channels_last)
                zz = z.to(DEV, dt)
                with torch.no_grad():
                    fn = lambda: vae.decode(zz, return_dict = False)[0]
                    r = timed(fn, reps = 3, warm = 1)
                    img = fn().float().cpu().clamp(-1, 1)
                lat_dec[(side, arm)] = img
                if arm != "fp32":
                    ref = lat_dec[(side, "fp32")]
                    mse = float(((img - ref) / 2).pow(2).mean())
                    r["psnr_vs_fp32_db"] = round(10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-20))).item(), 2)
                    r["nonfinite"] = int((~torch.isfinite(img)).sum())
                res[f"{side}_{arm}"] = r
                del vae, zz
            except Exception as exc:  # noqa: BLE001
                res[f"{side}_{arm}"] = {"error": repr(exc)[:300]}
            gc.collect()
            empty_cache()
            print(f"vae {side} {arm}", json.dumps(res[f'{side}_{arm}']), flush = True)
    return res


if os.environ.get("ZI_VAE", "1") == "1":
    log("vae", vae_bench())
log("mps_fallback_warnings_all", sorted(set(fallbacks))[:20])
print("DONE", flush = True)
