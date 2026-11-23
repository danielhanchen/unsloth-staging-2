"""TeaCache on Z-Image base (Studio MLX engine, post-#18964 main + the opt-in patch), one prompt per process.

Rounds interleave the uncached render with each threshold so drift hits every arm alike. Per cell: wall s, s/step,
skips, per-step gate rel-L1, MLX peak + phys peak, pixel sha. Then an uncached render at each cached arm's computed
step count (equal-compute comparator). --tree-label main renders only the uncached reference (proves off == main).
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
ap.add_argument("--backend-dir", required = True)
ap.add_argument("--out", required = True)
ap.add_argument("--tag", required = True)
ap.add_argument("--prompt", required = True)
ap.add_argument("--repo", default = "unsloth/Z-Image-GGUF")
ap.add_argument("--gguf", default = "z-image-Q4_0.gguf")
ap.add_argument("--base-repo", default = "Tongyi-MAI/Z-Image")
ap.add_argument("--size", type = int, default = 512)
ap.add_argument("--steps", type = int, default = 25)
ap.add_argument("--guidance", type = float, default = 3.0)
ap.add_argument("--seed", type = int, default = 42)
ap.add_argument("--thresholds", default = "0.12,0.3,0.6")
ap.add_argument("--rounds", type = int, default = 3)
ap.add_argument("--ref-only", action = "store_true")
ap.add_argument("--deadline-min", type = float, default = 0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok = True)
sys.path.insert(0, os.path.abspath(args.backend_dir))
os.environ["UNSLOTH_DIFFUSION_MLX"] = "1"
os.environ.pop("UNSLOTH_DIFFUSION_MLX_TEACACHE", None)
T0 = time.time()
sampler = common.Sampler()
rec = {"tag": args.tag, "args": vars(args), "cells": []}
path = os.path.join(args.out, f"{args.tag}.json")


def dump():
    with open(path, "w") as f:
        json.dump(rec, f, indent = 1, default = str)


def log(*a):
    print(time.strftime("%H:%M:%S"), args.tag, *a, flush = True)


import importlib.metadata as md  # noqa: E402

import mlx.core as mx  # noqa: E402
from core.inference import mlx_diffusion_backend as mb  # noqa: E402

tc = None
if not args.ref_only:
    from core.inference import mlx_diffusion_teacache as tc  # noqa: E402

    # Record every gate rel-L1 (decide() passes rel through math.isfinite exactly once) and every decision.
    GATE = []

    class _MathShim:
        def __getattr__(self, name):
            import math

            return getattr(math, name)

        def isfinite(self, x):
            import math

            GATE.append({"rel": float(x)})
            return math.isfinite(x)

    tc.math = _MathShim()
    _decide = tc._Gate.decide

    def decide(self, mx_, signal):
        n = len(GATE)
        compute = _decide(self, mx_, signal)
        if len(GATE) > n:
            GATE[-1].update(step = self.step - 1, compute = bool(compute), acc = round(self.accumulated, 5))
        return compute

    tc._Gate.decide = decide

rec["versions"] = {"mlx": mx.__version__, "mflux": md.version("mflux")}
rec["env"] = {k: v for k, v in os.environ.items() if k.startswith("UNSLOTH_DIFFUSION_MLX")}
backend = mb.get_mlx_diffusion_backend()
t = time.perf_counter()
kw = {} if args.ref_only else {"transformer_cache": "fbcache"}
p = common.load(backend, args.repo, args.gguf, base_repo = args.base_repo, log = log, **kw)
st = backend.status()
rec["load"] = {"wall_s": round(time.perf_counter() - t, 1), "phase": p["phase"], "error": p.get("error"),
               "status_mlx": st.get("mlx"), **sampler.phase()}
log("load", rec["load"])
dump()
if p["phase"] != "ready":
    raise SystemExit("load failed")
state = backend._state


def render(label, steps, thr):
    if tc is not None:
        state.teacache_threshold = thr
        GATE.clear()
    mx.reset_peak_memory()
    sampler.phase()
    t = time.perf_counter()
    res = backend.generate(prompt = args.prompt, width = args.size, height = args.size, steps = int(steps),
                           guidance = args.guidance, seed = args.seed)
    wall = time.perf_counter() - t
    img = res["images"][0]
    name = f"{args.tag}_{label}.png"
    img.save(os.path.join(args.out, name))
    stats = getattr(backend._runtime, "last_teacache", None)
    r = {"label": label, "steps": int(steps), "threshold": thr, "wall_s": round(wall, 2),
         "timings": backend.last_timings["images"][0], "teacache": stats, "gate": list(GATE) if tc else None,
         "mlx_peak_gib": mx.get_peak_memory() / common.GIB, **sampler.phase(),
         "pixel_sha256": common.pixel_sha(img), "image": name, "t_min": round((time.time() - T0) / 60, 1)}
    rec["cells"].append(r)
    log({k: v for k, v in r.items() if k != "gate"})
    if r["gate"]:
        log("gate", [(g.get("step"), round(g["rel"], 4), g.get("compute"), g.get("acc")) for g in r["gate"]])
    dump()
    return r


# Warmup (shader compile, page-in): 2 steps, below MIN_STEPS so nothing is cached.
render("warmup", 2, None)
thresholds = [] if args.ref_only else [float(x) for x in args.thresholds.split(",") if x]
for rnd in range(1, args.rounds + 1):
    for thr in [None, *thresholds]:
        if args.deadline_min and (time.time() - T0) / 60 > args.deadline_min:
            log("deadline, skipping", rnd, thr)
            continue
        render(f"r{rnd}_{'off' if thr is None else thr}", args.steps, thr)

# Equal-compute comparator: uncached at each cached arm's computed step count.
done = set()
for c in list(rec["cells"]):
    tcs = c.get("teacache") or {}
    n = tcs.get("computed")
    if not tcs.get("skipped") or n in done or n is None:
        continue
    if args.deadline_min and (time.time() - T0) / 60 > args.deadline_min:
        break
    done.add(n)
    render(f"eq_{n}steps_off", n, None)
backend.unload()
log("done")
