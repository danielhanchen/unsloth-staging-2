"""One Z-Image base bisect arm in a fresh process.

--mode studio: Studio's MLX engine (env UNSLOTH_DIFFUSION_MLX_PRECISION / _DENSE_BITS / _TE_BITS pick the variant).
--mode mflux: mflux alone on the Hub bf16 weights (ZImage(quantize=Q)), the reference.
Both record per-step latent stats, a t=0 cond / uncond predict probe and the text-encoding stats.
"""

import argparse
import faulthandler
import json
import os
import sys
import time

faulthandler.enable(all_threads = True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices = ("studio", "mflux"), default = "studio")
ap.add_argument("--backend-dir")
ap.add_argument("--out", required = True)
ap.add_argument("--tag", required = True)
ap.add_argument("--repo", default = "unsloth/Z-Image-GGUF")
ap.add_argument("--gguf", default = "z-image-Q4_0.gguf")
ap.add_argument("--base-repo", default = "Tongyi-MAI/Z-Image")
ap.add_argument("--size", type = int, default = 512)
ap.add_argument("--steps", type = int, default = 12)
ap.add_argument("--guidance", type = float, default = 3.0)
ap.add_argument("--seed", type = int, default = 42)
ap.add_argument("--scheduler", default = None)
ap.add_argument("--quantize", type = int, default = 4)
ap.add_argument("--prompt", default = "A red apple on a wooden table, soft window light, shallow depth of field, photograph")
ap.add_argument("--no-probe", action = "store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok = True)
if args.backend_dir:
    sys.path.insert(0, os.path.abspath(args.backend_dir))
os.environ["UNSLOTH_DIFFUSION_MLX"] = "1"
sampler = common.Sampler()
rec = {"tag": args.tag, "args": vars(args), "env": {k: v for k, v in os.environ.items() if k.startswith("UNSLOTH_DIFFUSION_MLX")}}
path = os.path.join(args.out, f"{args.tag}.json")


def dump():
    with open(path, "w") as f:
        json.dump(rec, f, indent = 1, default = str)


def log(*a):
    print(time.strftime("%H:%M:%S"), args.tag, *a, flush = True)


import importlib.metadata as md  # noqa: E402

import mlx.core as mx  # noqa: E402

rec["versions"] = {"mlx": mx.__version__, "mflux": md.version("mflux")}


def stats(a):
    if a is None:
        return None
    f = a.astype(mx.float32)
    finite = mx.isfinite(f)
    nf = int((~finite).sum().item())
    g = mx.where(finite, f, 0.0)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "nonfinite": nf, "max_abs": float(mx.max(mx.abs(g)).item()),
            "mean": float(mx.mean(g).item()), "std": float(mx.std(g).item())}


steps_log = []


class _Stats:
    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):  # noqa: ARG002
        s = stats(latents)
        s["t"] = int(t)
        try:
            s["sigma"] = float(config.scheduler.sigmas[int(t)].item())
        except Exception:  # noqa: BLE001
            pass
        steps_log.append(s)


t0 = time.perf_counter()
if args.mode == "studio":
    from core.inference import mlx_diffusion_backend as mb

    backend = mb.get_mlx_diffusion_backend()
    p = common.load(backend, args.repo, args.gguf, base_repo = args.base_repo, log = log)
    rec["load"] = {"wall_s": round(time.perf_counter() - t0, 1), "phase": p["phase"], "error": p.get("error"),
                   "status_mlx": backend.status().get("mlx"), **sampler.phase()}
    dump()
    if p["phase"] != "ready":
        log("load failed", p)
        raise SystemExit(3)
    model = backend._state.model
    rec["convert_dir"] = backend._state.convert_dir
    rec["precision"] = backend._state.precision
else:
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.z_image import ZImage

    model = ZImage(model_config = ModelConfig.z_image(), quantize = args.quantize)
    mx.eval(model.parameters())
    rec["load"] = {"wall_s": round(time.perf_counter() - t0, 1), "bits": getattr(model, "bits", None), **sampler.phase()}
    dump()
log("load", rec["load"])
rec["model_config"] = {"name": getattr(model.model_config, "model_name", None),
                       "supports_guidance": bool(model.model_config.supports_guidance)}
rec["transformer_dtypes"] = sorted({str(v.dtype) for _k, v in __import__("mlx.utils", fromlist = ["tree_flatten"]).tree_flatten(model.transformer.parameters())})

pos, neg = model._encode_prompts(prompt = args.prompt, negative_prompt = None, guidance = args.guidance)
mx.eval(pos) if neg is None else mx.eval(pos, neg)
rec["text"] = {"pos": stats(pos), "neg": stats(neg)}
log("text", rec["text"])
dump()

if not args.no_probe:
    try:
        mx.random.seed(0)
        lat = mx.random.normal((16, 1, args.size // 8, args.size // 8)).astype(mx.bfloat16)
        sig = mx.linspace(1, 0, args.steps + 1)
        for name, enc, sigma in (("pos_t0", pos, 1.0), ("neg_t0", neg, 1.0), ("pos_mid", pos, 0.5)):
            if enc is None:
                continue
            ts = mx.array([1.0 - sigma])
            out = model.transformer(timestep = ts, x = lat, cap_feats = enc, sigmas = sig)
            mx.eval(out)
            rec.setdefault("probe", {})[name] = stats(out)
            del out
        del lat
    except Exception as exc:  # noqa: BLE001
        rec["probe"] = {"error": repr(exc)}
    log("probe", rec.get("probe"))
    dump()

model.callbacks.register(_Stats())
if args.scheduler:
    import functools

    model.generate_image = functools.partial(model.generate_image, scheduler = args.scheduler)
mx.reset_peak_memory()
sampler.phase()
t = time.perf_counter()
try:
    if args.mode == "studio":
        res = backend.generate(prompt = args.prompt, width = args.size, height = args.size, steps = args.steps,
                               guidance = args.guidance, seed = args.seed)
        img = res["images"][0]
        rec["timings"] = backend.last_timings
    else:
        img = model.generate_image(seed = args.seed, prompt = args.prompt, num_inference_steps = args.steps,
                                   width = args.size, height = args.size, guidance = args.guidance,
                                   negative_prompt = None).image
except Exception as exc:  # noqa: BLE001
    rec["gen_error"] = repr(exc)
    rec["steps"] = steps_log
    dump()
    raise
wall = time.perf_counter() - t
name = f"{args.tag}.png"
img.save(os.path.join(args.out, name))
rec["gen"] = {"wall_s": round(wall, 1), "mlx_peak_gib": mx.get_peak_memory() / common.GIB, **sampler.phase(),
              "pixel_sha256": common.pixel_sha(img), "image": name}
rec["steps"] = steps_log
log("gen", rec["gen"])
log("steps", [(s["t"], round(s["max_abs"], 3), round(s["std"], 3), s["nonfinite"]) for s in steps_log])
dump()
log("done")
