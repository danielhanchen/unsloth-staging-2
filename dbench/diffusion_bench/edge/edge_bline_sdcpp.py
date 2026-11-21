#!/usr/bin/env python3
"""Fast edge checks for the BlineUI, stable-diffusion.cpp and plain-diffusers backends: the breakage a user would
hit, not the benchmark numbers. Small models, 512 px, turbo step counts; about 2-4 minutes on one GPU.

Per backend suite (Z-Image-Turbo on blineui / sdcpp / diffusers, SDXL-Turbo on blineui / diffusers):
  <suite>.load                  loads, and says what engaged
  <suite>.output_sanity         three prompts: right size, not black / constant / NaN-like
  <suite>.same_seed_repeat      the same prompt + seed again gives the same pixels (BlineUI: proves the cache-bust
                                nonce really re-executes, and that a re-execution is deterministic)
  <suite>.seed_changes_output   seed + 1 gives a different image (a backend that drops the seed is caught here)
  <suite>.odd_size_WxH          520x392 and 333x257: renders at the asked size, snaps (info, with the size it
                                returned), or rejects with a clear error; the backend must still render afterwards
  <suite>.bad_sampler           an unknown sampler name is rejected clearly (pass), ignored silently (info), or kills
                                the backend (fail)
  <suite>.cache_hit_detected    BlineUI only: with cache_bust none a repeated graph IS a cache hit and the backend
                                flags it (sampler_cached), i.e. the default nonce is what keeps timings honest
  <suite>.close_frees           close() leaves no server process behind
Failure paths (no GPU work):
  blineui.missing_model_file, blineui.server_startup_failure, blineui.unknown_family,
  sdcpp.binary_missing, sdcpp.wrong_backend_binary, sdcpp.missing_model_file, diffusers.missing_model
Cross-framework:
  cross.z-image / cross.sdxl-turbo  every pair of frameworks, same prompt + seed, mean LPIPS over the prompts. The
                                frameworks draw their initial noise differently, so this is NOT a quality number;
                                it passes when each pair is closer than two DIFFERENT prompts are within one
                                framework (the unrelated-content floor, recorded) and below an absolute 0.75.
                                Studio joins when --studio-src (or DIFFUSION_BENCH_STUDIO_SRC) is given.

Tiny tier (--tier tiny | all): the hf-internal-testing tiny random pipelines under $WORKSPACE/hf_tiny, diffusers
only (BlineUI / sd.cpp cannot load diffusers-format repos), each in its own process, output is noise:
  tiny.<pipe>.diffusers         loads and renders at the size the tiny model allows, right output shape, not black
  tiny.<pipe>.repeat            same seed identical, seed + 1 different
  tiny.<pipe>.studio_vs_diffusers  Studio (with --studio-src) on the same pipe, seed, size: near-identical expected,
                                both being diffusers with a CUDA generator (mean abs <= 2 pass, <= 8 info, else fail)

Writes <out>/results.json ({"checks": [{name, status, seconds, evidence}], ...}) and <out>/summary.md.

Usage (from any cwd; BlineUI / sd.cpp are reused through the env vars, or set up fresh by setup_*.py):
  DIFFUSION_BENCH_BLINE_PYTHON=... DIFFUSION_BENCH_BLINE_DIR=... CUDA_VISIBLE_DEVICES=5 \\
    python diffusion_bench/edge/edge_bline_sdcpp.py --out outputs/dbench/edge_cs [--only 'blineui*'] \\
    [--models models.json] [--studio-src $WORKSPACE/unsloth] [--tier small|tiny|all]
"""

from __future__ import annotations

import argparse
import fnmatch
import itertools
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

import common as C  # noqa: E402
from backends.base import get_backend  # noqa: E402

W = "$WORKSPACE"
# Defaults are this workspace's copies; --models JSON overrides any key.
MODELS = {
    "zimage_diffusers": f"{W}/hf_local_bases/Tongyi-MAI/Z-Image-Turbo",
    "zimage_bline": {"dit": f"{W}/bline_models_agent/split_files/diffusion_models/z_image_turbo_bf16.safetensors",
                     "te": f"{W}/bline_models_agent/split_files/text_encoders/qwen_3_4b.safetensors",
                     "vae": f"{W}/bline_models_agent/split_files/vae/ae.safetensors"},
    "zimage_gguf": {"dit": f"{W}/dyn_gguf/zimage_ext/jayn7/z_image_turbo-Q4_K_S.gguf",
                    "te": f"{W}/sdcpp_assets/qwen3_4b_te/Qwen3-4B-Q8_0.gguf",
                    "vae": f"{W}/sdcpp_assets/flux_vae/ae.safetensors"},
    "sdxl_turbo_diffusers": f"{W}/sdxl_turbo_base",
    # a build with no GPU device, for the wrong-backend check (setup_sdcpp.py --backend cpu makes one)
    "sdcpp_cpu_bin": f"{W}/temp/diffusion_bench/src/sdcpp-7bbffa35fc-cpu/build/bin/sd-cli",
}
ROWS = [
    {"id": "e0", "prompt": "a red vintage pickup truck parked beside a wheat field, clear blue sky", "seed": 1234},
    {"id": "e1", "prompt": "a bowl of ramen with a soft boiled egg, top-down food photography", "seed": 777},
    {"id": "e2", "prompt": "a lighthouse on a rocky coast at sunset, dramatic clouds", "seed": 4242},
]
ODD_SIZES = [(520, 392), (333, 257)]
LPIPS_ABS_BOUND = 0.75


class Runner:
    def __init__(self, out: Path, only: list):
        self.out, self.only = out, only
        self.results: list = []

    def wanted(self, name: str) -> bool:
        return not self.only or any(fnmatch.fnmatch(name, p) for p in self.only)

    def record(self, name: str, status: str, seconds: float, evidence) -> dict:
        row = {"name": name, "status": status, "seconds": round(seconds, 2), "evidence": evidence}
        self.results.append(row)
        C.log(f"[edge] {status.upper():4s} {name} ({row['seconds']}s): {json.dumps(evidence, default = str)[:300]}")
        return row

    def check(self, name: str, fn):
        """Run ``fn() -> (status, evidence)``; an exception is a FAIL with the error as evidence."""
        if not self.wanted(name):
            return None
        t = time.perf_counter()
        try:
            status, evidence = fn()
        except Exception as exc:  # noqa: BLE001
            status, evidence = "fail", {"error": f"{type(exc).__name__}: {str(exc)[:800]}",
                                        "traceback": traceback.format_exc()[-1500:]}
        return self.record(name, status, time.perf_counter() - t, evidence)


# ---------------------------------------------------------------------------------------------- helpers
def arr(img):
    import numpy as np

    return np.asarray(C.to_pil(img).convert("RGB"))


def image_facts(a) -> dict:
    import score

    return {"size": [int(a.shape[1]), int(a.shape[0])], "mean": round(float(a.mean()), 2),
            "std": round(float(a.std()), 2), "flags": score.sanity(a[None])}


def make(backend: str, out: Path, tag: str, **cell) -> object:
    merged = C.expand_env(C.merge_cell({}, {"tag": tag, "backend": backend, "width": 512, "height": 512, **cell}))
    d = out / "cells" / tag
    d.mkdir(parents = True, exist_ok = True)
    return get_backend(backend)(merged, d)


def expect_error(fn, *types, contains: tuple = ()) -> tuple:
    """pass when ``fn`` raises one of ``types`` whose message has every ``contains`` fragment."""
    try:
        fn()
    except types as exc:
        msg = str(exc)
        missing = [c for c in contains if c.lower() not in msg.lower()]
        ev = {"raised": type(exc).__name__, "message": msg[:600]}
        return ("pass", ev) if not missing else ("fail", {**ev, "message_lacks": missing})
    except Exception as exc:  # noqa: BLE001
        return "fail", {"raised_unexpected": f"{type(exc).__name__}: {str(exc)[:600]}"}
    return "fail", {"raised": None, "note": "no error: the bad input was accepted"}


# ---------------------------------------------------------------------------------------------- suites
def suite(r: Runner, label: str, backend: str, cell: dict, steps: int) -> dict:
    """The per-backend checks on one loaded model. Returns {row id: image array} for the cross-framework check."""
    names = [f"{label}.{n}" for n in ("load", "output_sanity", "same_seed_repeat", "seed_changes_output",
                                      "bad_sampler", "cache_hit_detected", "close_frees")] + \
            [f"{label}.odd_size_{w}x{h}" for w, h in ODD_SIZES]
    if not any(r.wanted(n) for n in names) and not (r.wanted("cross.z-image") or r.wanted("cross.sdxl-turbo")):
        return {}
    b = make(backend, r.out, label, **cell)
    images: dict = {}
    state = {"loaded": False}

    def load():
        t = time.perf_counter()
        st = b.load() or {}
        state["loaded"] = True
        keep = ("family", "mode", "bline_commit", "commit", "devices", "engaged", "pipeline", "dtype", "offload",
                "port", "facts")
        return "pass", {"load_s": round(time.perf_counter() - t, 2), **{k: st[k] for k in keep if k in st}}

    try:
        if r.check(f"{label}.load", load) is None:
            load()
        if not state["loaded"]:
            return images

        def render(row, n_steps = steps):
            res = b.render(row, n_steps)
            return arr(res.image), res

        def sanity():
            facts, walls, first = [], [], None
            for i, row in enumerate(ROWS):
                t = time.perf_counter()
                a, res = render(row)
                walls.append(round(time.perf_counter() - t, 3))
                images[row["id"]] = a
                f = image_facts(a)
                f["step_s"] = res.step_s
                facts.append(f)
                if i == 0:
                    first = walls[0]
            want = [b.cell["width"], b.cell["height"]]
            bad = [f for f in facts if f["flags"] or f["size"] != want]
            return ("fail" if bad else "pass"), {"wall_s": walls, "first_includes_lazy_load_s": first,
                                                  "images": facts, "want_size": want}

        r.check(f"{label}.output_sanity", sanity)
        if not images:  # --only skipped sanity: the other checks still need a first render
            images[ROWS[0]["id"]] = render(ROWS[0])[0]

        def repeat():
            import numpy as np

            a, res = render(ROWS[0])
            d = np.abs(a.astype(np.int16) - images[ROWS[0]["id"]].astype(np.int16))
            ev = {"max_abs": int(d.max()), "mean_abs": round(float(d.mean()), 4), "extra": res.extra or None}
            if res.extra and res.extra.get("sampler_cached"):
                return "fail", {**ev, "note": "BlineUI served the sampler from cache: the timing would be a cache hit"}
            if d.max() == 0:
                return "pass", ev
            return ("info" if d.mean() < 1.0 else "fail"), {**ev, "note": "not bitwise repeatable"}

        r.check(f"{label}.same_seed_repeat", repeat)

        def seed_changes():
            import numpy as np

            a, _ = render({**ROWS[0], "seed": ROWS[0]["seed"] + 1})
            d = float(np.abs(a.astype(np.int16) - images[ROWS[0]["id"]].astype(np.int16)).mean())
            return ("pass" if d > 2.0 else "fail"), {"mean_abs_vs_seed": round(d, 3)}

        r.check(f"{label}.seed_changes_output", seed_changes)

        for w, h in ODD_SIZES:
            def odd(w = w, h = h):
                saved = (b.cell["width"], b.cell["height"])
                b.cell["width"], b.cell["height"] = w, h
                ev: dict = {"asked": [w, h]}
                try:
                    a, _ = render(ROWS[1])
                    f = image_facts(a)
                    ev.update(f)
                    outcome = "rendered"
                except Exception as exc:  # noqa: BLE001
                    ev["rejected"] = f"{type(exc).__name__}: {str(exc)[:400]}"
                    outcome = "rejected"
                finally:
                    b.cell["width"], b.cell["height"] = saved
                try:
                    a2, _ = render(ROWS[2])
                    ev["after"] = "backend still renders"
                except Exception as exc:  # noqa: BLE001
                    return "fail", {**ev, "after": f"backend broken afterwards: {type(exc).__name__}: {str(exc)[:300]}"}
                if outcome == "rejected":
                    return "pass", ev
                if ev["flags"]:
                    return "fail", ev
                return ("pass" if ev["size"] == [w, h] else "info"), ev

            r.check(f"{label}.odd_size_{w}x{h}", odd)

        def bad_sampler():
            key = "sampler"
            saved = b.opts.get(key)
            b.opts[key] = "not_a_sampler"
            try:
                try:
                    a, res = render(ROWS[1])
                    outcome = ("info", {"note": "unknown sampler accepted silently; rendered with some default",
                                        **image_facts(a)})
                except Exception as exc:  # noqa: BLE001
                    outcome = ("pass", {"rejected": f"{type(exc).__name__}: {str(exc)[:400]}"})
            finally:
                if saved is None:
                    b.opts.pop(key, None)
                else:
                    b.opts[key] = saved
            try:
                render(ROWS[2])
            except Exception as exc:  # noqa: BLE001
                return "fail", {**outcome[1], "after": f"backend broken afterwards: {exc}"}
            return outcome

        if backend in ("blineui", "sdcpp"):
            r.check(f"{label}.bad_sampler", bad_sampler)

        def cache_probe():
            """Why the nonce exists: with cache_bust none, the same graph twice is a BlineUI cache hit."""
            saved = b.opts.get("cache_bust")
            b.opts["cache_bust"] = "none"
            try:
                t = time.perf_counter()
                _, r1 = render(ROWS[2])
                t1 = time.perf_counter() - t
                t = time.perf_counter()
                _, r2 = render(ROWS[2])
                t2 = time.perf_counter() - t
            finally:
                if saved is None:
                    b.opts.pop("cache_bust", None)
                else:
                    b.opts["cache_bust"] = saved
            ev = {"first_s": round(t1, 3), "repeat_s": round(t2, 3), "repeat_sampler_cached": r2.extra.get("sampler_cached"),
                  "repeat_cached_nodes": r2.extra.get("cached_nodes")}
            return ("pass" if r2.extra.get("sampler_cached") and not r1.extra.get("sampler_cached") else "fail"), ev

        if backend == "blineui":
            r.check(f"{label}.cache_hit_detected", cache_probe)
    finally:
        srv = getattr(b, "srv", None)
        t = time.perf_counter()
        b.close()
        if srv is not None:
            gone = srv.poll() is not None
            r.check(f"{label}.close_frees", lambda: (("pass" if gone else "fail"),
                                                      {"server_pid": srv.pid, "exit": srv.returncode,
                                                       "close_s": round(time.perf_counter() - t, 2)}))
    return images


def failure_paths(r: Runner, models: dict) -> None:
    from backends.blineui import BlineUIError
    from backends.sdcpp import SdCppError

    zc = models["zimage_bline"]

    def bline_missing():
        b = make("blineui", r.out, "bline_missing", options = {"family": "z-image",
                                                                  "files": {**zc, "dit": f"{W}/nope/missing_dit.safetensors"}})
        res = expect_error(b.load, FileNotFoundError, contains = ("missing_dit.safetensors",))
        started = b.srv is not None
        b.close()
        return ("fail", {**res[1], "note": "a server was started anyway"}) if started else res

    r.check("blineui.missing_model_file", bline_missing)

    def bline_startup():
        b = make("blineui", r.out, "bline_badflag", options = {"family": "z-image", "files": zc,
                                                                  "bline_args": "--definitely-not-a-bline-flag"})
        t = time.perf_counter()
        try:
            res = expect_error(b.load, BlineUIError, contains = ("exited", "unrecognized"))
        finally:
            b.close()
        res[1]["fail_after_s"] = round(time.perf_counter() - t, 2)
        return res

    r.check("blineui.server_startup_failure", bline_startup)
    r.check("blineui.unknown_family", lambda: expect_error(
        make("blineui", r.out, "bline_fam", options = {"family": "not-a-family", "files": zc}).load, KeyError,
        contains = ("z-image",)))

    def sd_missing_bin():
        old = os.environ.get("DIFFUSION_BENCH_SDCPP_BIN")
        os.environ["DIFFUSION_BENCH_SDCPP_BIN"] = str(C.WS / "nope" / "sd-cli")
        try:
            b = make("sdcpp", r.out, "sd_nobin", options = {"family": "z-image", "files": models["zimage_gguf"]})
            return expect_error(b.load, FileNotFoundError, contains = ("sd-cli",))
        finally:
            if old is None:
                os.environ.pop("DIFFUSION_BENCH_SDCPP_BIN", None)
            else:
                os.environ["DIFFUSION_BENCH_SDCPP_BIN"] = old

    r.check("sdcpp.binary_missing", sd_missing_bin)

    def sd_wrong_backend():
        cpu = Path(C.expand_env(models["sdcpp_cpu_bin"]))
        if not cpu.exists():
            return "skip", {"note": f"no CPU-only sd-cli at {cpu} (setup_sdcpp.py --backend cpu builds one)"}
        old = os.environ.get("DIFFUSION_BENCH_SDCPP_BIN")
        os.environ["DIFFUSION_BENCH_SDCPP_BIN"] = str(cpu)
        try:
            b = make("sdcpp", r.out, "sd_cpu", options = {"family": "z-image", "files": models["zimage_gguf"],
                                                          "expect_device": "CUDA"})
            res = expect_error(b.load, SdCppError, contains = ("no CUDA device",))
            b.close()
            return res
        finally:
            if old is None:
                os.environ.pop("DIFFUSION_BENCH_SDCPP_BIN", None)
            else:
                os.environ["DIFFUSION_BENCH_SDCPP_BIN"] = old

    r.check("sdcpp.wrong_backend_binary", sd_wrong_backend)
    r.check("sdcpp.missing_model_file", lambda: expect_error(
        make("sdcpp", r.out, "sd_missing", options = {"family": "z-image",
                                                      "files": {**models["zimage_gguf"], "dit": f"{W}/nope/x.gguf"}}).load,
        FileNotFoundError, contains = ("x.gguf",)))
    r.check("diffusers.missing_model", lambda: expect_error(
        make("diffusers", r.out, "d_missing", model = f"{W}/nope/not_a_pipeline").load, FileNotFoundError,
        contains = ("not_a_pipeline",)))


def cross(r: Runner, name: str, per_fw: dict) -> None:
    """Pairwise LPIPS between frameworks on the same prompt + seed, against the different-prompt floor."""
    def run():
        import score

        fws = {k: v for k, v in per_fw.items() if len(v) == len(ROWS)}
        if len(fws) < 2:
            return "skip", {"note": f"need two frameworks with all renders, have {sorted(per_fw)}"}
        m = score.Metrics()
        if m.net is None:
            return "skip", {"note": "LPIPS unavailable"}

        def lp(a, b):
            if a.shape != b.shape:
                return None
            return m.lpips(a[None], b[None])

        pairs = {}
        for fa, fb in itertools.combinations(sorted(fws), 2):
            vals = [lp(fws[fa][r_["id"]], fws[fb][r_["id"]]) for r_ in ROWS]
            vals = [v for v in vals if v is not None]
            pairs[f"{fa}~{fb}"] = {"mean": round(statistics.mean(vals), 4), "max": round(max(vals), 4)} if vals else None
        floors = {}
        for fw, imgs in fws.items():
            vals = [lp(imgs[a["id"]], imgs[b_["id"]]) for a, b_ in itertools.combinations(ROWS, 2)]
            floors[fw] = round(statistics.mean(v for v in vals if v is not None), 4)
        unrelated = min(floors.values())
        bad = {k: v for k, v in pairs.items() if v is None or v["max"] >= LPIPS_ABS_BOUND or v["mean"] >= unrelated}
        return ("fail" if bad else "pass"), {"pairs": pairs, "different_prompt_floor": floors,
                                              "bound": {"abs_max": LPIPS_ABS_BOUND, "mean_below": unrelated},
                                              "failing": sorted(bad), "note": "different noise per framework: a "
                                              "content-agreement check, not a quality score"}

    r.check(name, run)



# ---------------------------------------------------------------------------------------------- tiny tier
# hf-internal-testing tiny random pipelines (output is noise: plumbing checks only). Each runs in its own process,
# because a device-side assert (a real prompt's token ids past a tiny text encoder's vocab, or a size past a tiny
# rope table) poisons the CUDA context for everything after it. Per-model settings were found by probing:
#   size      what the tiny rope / VAE accept (Z-Image: 256 overflows its rope)
#   prompt    "" where the tiny text encoder's vocab is smaller than the real tokenizer's ids (Sana 8, Hunyuan 1000)
#   out_size  what the pipeline returns for size x size (tiny Lumina2's VAE does not upsample: 1/8 of the ask)
#   studio    the Studio family_override for the diffusers-vs-Studio cross-check (image families Studio serves)
#   cfg_kw / guidance  the no-CFG spelling on both sides (Studio's family cfg_kwarg)
TINY = {
    # Studio refuses anything under 256 px, so its cross-check runs at studio_size on both sides; tiny Z-Image's rope
    # table overflows (device-side assert) at 256, and Studio's flux.1-kontext insists on an input image.
    "tiny-zimage-pipe": {"size": 32, "studio": "z-image", "guidance": 0.0,
                         "studio_skip": "Studio's 256 px minimum is past the tiny Z-Image rope limit (asserts at 256)"},
    "tiny-flux-pipe": {"size": 32, "studio": "flux.1", "studio_size": 256, "guidance": 1.0},
    "tiny-flux-kontext-pipe": {"size": 32, "studio": "flux.1-kontext", "guidance": 1.0,
                               "studio_skip": "Studio's flux.1-kontext is image-editing only (needs an input image)",
                               # FluxKontextPipeline rescales to max_area (1024^2 by default) whatever the size asked
                               "call_kwargs": {"max_area": 32 * 32}},
    "tiny-qwenimage-pipe": {"size": 32, "studio": "qwen-image", "studio_size": 256, "guidance": 1.0,
                            "cfg_kw": "true_cfg_scale"},
    "tiny-qwenimage21-pipe": {"size": 32, "studio": "qwen-image-2.1", "studio_size": 256, "guidance": 1.0,
                              "cfg_kw": "true_cfg_scale"},
    "tiny-stable-diffusion-xl-pipe": {"size": 32, "studio": "sdxl", "studio_size": 256, "guidance": 1.0},
    "tiny-sd3-pipe": {"size": 32, "guidance": 1.0},
    "tiny-lumina2-pipe": {"size": 32, "out_size": 4, "pipeline": "Lumina2Pipeline", "studio": "lumina-2",
                          "studio_size": 256, "guidance": 1.0},
    "tiny-sana-pipe": {"size": 32, "prompt": "", "guidance": 1.0,
                       "call_kwargs": {"max_sequence_length": 16, "complex_human_instruction": None}},
    "tiny-wan-pipe": {"size": 32, "kind": "video", "frames": 5, "guidance": 1.0},
    "tiny-random-hunyuanvideo": {"size": 16, "kind": "video", "frames": 5, "prompt": "", "guidance": 1.0,
                                 "call_kwargs": {"max_sequence_length": 16,
                                                 "prompt_template": {"template": "{}", "crop_start": 0}}},
    "tiny-cogvideox-pipe": {"size": 16, "kind": "video", "frames": 8, "guidance": 1.0,
                            "call_kwargs": {"max_sequence_length": 16}},
}
TINY_PROMPT = "a red fox in snow"


def tiny_child(name: str, backend: str, tiny_dir: Path, out: Path, studio_src: Optional[str],
               size: Optional[int] = None) -> int:
    """One tiny pipe, one backend, in this process: three renders (seed, same seed, seed + 1), the first saved as
    <out>/tiny/<name>/<backend>_<size>.npz, a TINY_RESULT json line on stdout."""
    import numpy as np

    t = TINY[name]
    size = int(size or t["size"])
    res: dict = {"name": name, "backend": backend, "size": size}
    kind = t.get("kind", "image")
    cell = {"kind": kind, "width": size, "height": size, "frames": t.get("frames"), "steps": 2,
            "guidance": t.get("guidance"), "model": str(tiny_dir / name)}
    if backend == "diffusers":
        # Studio seeds a torch.Generator on its device; draw the noise the same way so the two are comparable.
        cell["options"] = {"pipeline": t.get("pipeline"), "call_kwargs": t.get("call_kwargs") or {},
                           "cfg_kw": t.get("cfg_kw", "guidance_scale"), "generator_device": "cuda"}
    else:
        cell["options"] = {"studio_src": studio_src, "model_kind": "pipeline", "family_override": t["studio"],
                           "local_files_only": True, "speed_mode": "off"}
    try:
        b = make(backend, out, f"tiny_{name}_{backend}_{size}", **cell)
        t0 = time.perf_counter()
        st = b.load() or {}
        res["load_s"] = round(time.perf_counter() - t0, 2)
        res["status"] = {k: st.get(k) for k in ("pipeline", "family", "speed_mode", "offload_policy", "dtype") if k in st}
        row = {"id": "t0", "prompt": t.get("prompt", TINY_PROMPT), "seed": 7}
        arrs = []
        for rr in (row, row, {**row, "seed": 8}):
            t0 = time.perf_counter()
            r = b.render(rr, 2)
            a = C.frames_array(r.frames) if kind == "video" else arr(r.image)[None]
            arrs.append(a)
            res.setdefault("render_s", []).append(round(time.perf_counter() - t0, 3))
        b.close()
        d = out / "tiny" / name
        d.mkdir(parents = True, exist_ok = True)
        np.savez_compressed(d / f"{backend}_{size}.npz", frames = arrs[0])
        res.update({"shape": list(arrs[0].shape), "mean": round(float(arrs[0].mean()), 2),
                    "std": round(float(arrs[0].std()), 2),
                    "repeat_max_abs": int(np.abs(arrs[0].astype(np.int16) - arrs[1].astype(np.int16)).max()),
                    "seed_mean_abs": round(float(np.abs(arrs[0].astype(np.int16) - arrs[2].astype(np.int16)).mean()), 3)})
        res["ok"] = True
    except Exception as exc:  # noqa: BLE001
        res.update({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:600]}"})
    print("TINY_RESULT " + json.dumps(res, default = str), flush = True)
    return 0


def tiny_tier(r: Runner, tiny_dir: Path, studio_src: Optional[str], workers: int = 4) -> None:
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    names = [n for n in TINY if (tiny_dir / n / "model_index.json").exists()]
    missing = [n for n in TINY if n not in names]
    if missing:
        r.record("tiny.available", "info" if names else "skip", 0, {"missing": missing, "dir": str(tiny_dir)})
    jobs = [(n, "diffusers", TINY[n]["size"]) for n in names
            if r.wanted(f"tiny.{n}.diffusers") or r.wanted(f"tiny.{n}.repeat")]
    if studio_src:
        for n in names:
            if TINY[n].get("studio_size") and r.wanted(f"tiny.{n}.studio_vs_diffusers"):
                jobs += [(n, "diffusers", TINY[n]["studio_size"]), (n, "studio", TINY[n]["studio_size"])]

    def child(job):
        name, backend, size = job
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--out", str(r.out), "--tiny-child", name,
               "--tiny-backend", backend, "--tiny-size", str(size), "--tiny-dir", str(tiny_dir)] + \
              (["--studio-src", studio_src] if studio_src else [])
        t = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output = True, text = True, timeout = 300)
            line = next((ln for ln in p.stdout.splitlines() if ln.startswith("TINY_RESULT ")), None)
            res = json.loads(line[len("TINY_RESULT "):]) if line else {
                "ok": False, "error": f"child exit {p.returncode}, no result: {(p.stderr or p.stdout)[-600:]}"}
        except subprocess.TimeoutExpired:
            res = {"ok": False, "error": "child timed out after 300 s"}
        res["wall_s"] = round(time.perf_counter() - t, 2)
        return job, res

    with ThreadPoolExecutor(max_workers = workers) as ex:
        results = dict(ex.map(child, jobs))
    for name in names:
        t = TINY[name]
        res = results.get((name, "diffusers", t["size"]))
        want = t.get("out_size", t["size"])
        ev = {k: res.get(k) for k in ("load_s", "render_s", "shape", "mean", "std", "error", "status")}
        if res is not None and not res.get("ok"):
            r.record(f"tiny.{name}.diffusers", "fail", res["wall_s"], ev)
        elif res is not None:
            size_ok = res["shape"][1:3] == [want, want]
            black = res["mean"] < 4 and res["std"] < 4
            r.record(f"tiny.{name}.diffusers", "pass" if size_ok and not black else "fail", res["wall_s"],
                     {**ev, "want_hw": [want, want], **({"flag": "black / NaN"} if black else {})})
            rep_ok = res["repeat_max_abs"] == 0 and res["seed_mean_abs"] > 0
            r.record(f"tiny.{name}.repeat", "pass" if rep_ok else "fail", 0,
                     {"same_seed_max_abs": res["repeat_max_abs"], "seed_plus_1_mean_abs": res["seed_mean_abs"]})
        if not studio_src or not t.get("studio") or not r.wanted(f"tiny.{name}.studio_vs_diffusers"):
            continue
        if t.get("studio_skip"):
            r.record(f"tiny.{name}.studio_vs_diffusers", "skip", 0, {"note": t["studio_skip"]})
            continue
        ss = t["studio_size"]
        dres, sres = results.get((name, "diffusers", ss)) or {}, results.get((name, "studio", ss)) or {}
        if not (dres.get("ok") and sres.get("ok")):
            r.record(f"tiny.{name}.studio_vs_diffusers", "fail", sres.get("wall_s", 0),
                     {"studio_error": sres.get("error"), "diffusers_error": dres.get("error"), "family": t["studio"],
                      "size": ss})
            continue
        a = np.load(r.out / "tiny" / name / f"diffusers_{ss}.npz")["frames"]
        b_ = np.load(r.out / "tiny" / name / f"studio_{ss}.npz")["frames"]
        if a.shape != b_.shape:
            r.record(f"tiny.{name}.studio_vs_diffusers", "fail", sres["wall_s"],
                     {"diffusers_shape": list(a.shape), "studio_shape": list(b_.shape)})
            continue
        dd = np.abs(a.astype(np.int16) - b_.astype(np.int16))
        mean_abs = round(float(dd.mean()), 3)
        status = "pass" if mean_abs <= 2.0 else ("info" if mean_abs <= 8.0 else "fail")
        r.record(f"tiny.{name}.studio_vs_diffusers", status, sres["wall_s"],
                 {"mean_abs": mean_abs, "max_abs": int(dd.max()), "identical": bool(dd.max() == 0),
                  "family": t["studio"], "size": ss, "studio_load_s": sres.get("load_s"),
                  "studio_status": sres.get("status"),
                  "note": "same diffusers pipeline, seed and CUDA generator on both sides: expected near-identical"})

# ---------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required = True)
    ap.add_argument("--only", action = "append", default = [], help = "fnmatch on check names (repeatable)")
    ap.add_argument("--models", default = None, help = "JSON file overriding MODELS keys")
    ap.add_argument("--studio-src", default = os.environ.get("DIFFUSION_BENCH_STUDIO_SRC"),
                    help = "add Studio to the Z-Image cross-framework check (path / git ref / pypi)")
    ap.add_argument("--sdcpp-mode", default = "server", choices = ["server", "cli"])
    ap.add_argument("--tier", default = "small", choices = ["small", "tiny", "all"],
                    help = "small: real small models on every backend; tiny: hf-internal-testing random pipes on "
                           "diffusers (+ Studio cross-check); all: both")
    ap.add_argument("--tiny-dir", default = str(C.WS / "hf_tiny"))
    ap.add_argument("--tiny-workers", type = int, default = 4)
    ap.add_argument("--tiny-child", default = None, help = argparse.SUPPRESS)
    ap.add_argument("--tiny-backend", default = "diffusers", help = argparse.SUPPRESS)
    ap.add_argument("--tiny-size", type = int, default = None, help = argparse.SUPPRESS)
    args = ap.parse_args()
    if args.tiny_child:
        C.scrub_tokens()
        return tiny_child(args.tiny_child, args.tiny_backend, Path(args.tiny_dir), Path(args.out), args.studio_src,
                          args.tiny_size)
    if not os.environ.get("TORCH_HOME"):
        os.environ["TORCH_HOME"] = str(C.WS / "temp" / "torch_home")  # LPIPS' alexnet stays in the workspace
    C.scrub_tokens()
    models = {**MODELS, **(json.loads(Path(args.models).read_text()) if args.models else {})}
    models = C.expand_env(models)
    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)
    r = Runner(out, args.only)
    t0 = time.perf_counter()
    gpu = C.gpu_query()

    if args.tier in ("tiny", "all"):
        tiny_tier(r, Path(C.expand_env(args.tiny_dir)), args.studio_src, args.tiny_workers)
    if args.tier == "tiny":
        return finish(r, out, t0, models)
    failure_paths(r, models)

    zimg: dict = {}
    zimg["blineui"] = suite(r, "blineui.z-image", "blineui", {"steps": 8, "guidance": 1.0, "options": {
        "family": "z-image", "files": models["zimage_bline"]}}, 8)
    zimg["sdcpp_q4k"] = suite(r, "sdcpp.z-image", "sdcpp", {"steps": 8, "guidance": 1.0, "options": {
        "family": "z-image", "mode": args.sdcpp_mode, "expect_device": "CUDA" if gpu else None,
        "files": models["zimage_gguf"]}}, 8)
    # ZImagePipeline's CFG is pos + g * (pos - neg) and runs whenever g > 0, so diffusers' "no CFG" is 0.0 where
    # BlineUI / sd.cpp spell it cfg 1.0.
    zimg["diffusers"] = suite(r, "diffusers.z-image", "diffusers", {"steps": 8, "guidance": 0.0,
                                                                    "model": models["zimage_diffusers"]}, 8)
    if args.studio_src:
        zimg["studio"] = suite(r, "studio.z-image", "studio", {"steps": 8, "guidance": 0.0,
                                                               "model": models["zimage_diffusers"], "options": {
            "studio_src": args.studio_src, "model_kind": "pipeline", "family_override": "z-image"}}, 8)
    cross(r, "cross.z-image", zimg)

    sdxl: dict = {}
    sdxl["blineui"] = suite(r, "blineui.sdxl-turbo", "blineui", {"steps": 4, "guidance": 1.0, "options": {
        "family": "sdxl-turbo", "files": {"diffusers": models["sdxl_turbo_diffusers"]}}}, 4)
    sdxl["diffusers"] = suite(r, "diffusers.sdxl-turbo", "diffusers", {"steps": 4, "guidance": 0.0,
                                                                       "model": models["sdxl_turbo_diffusers"],
                                                                       "options": {"dtype": "fp16"}}, 4)
    cross(r, "cross.sdxl-turbo", sdxl)

    return finish(r, out, t0, models)


def finish(r: Runner, out: Path, t0: float, models: dict) -> int:
    counts = {s: sum(1 for x in r.results if x["status"] == s) for s in ("pass", "fail", "skip", "info")}
    doc = {"suite": "edge_bline_sdcpp", "seconds": round(time.perf_counter() - t0, 1), "counts": counts,
           "checks": r.results, "env": C.env_fingerprint(with_torch = True), "models": models}
    (out / "results.json").write_text(json.dumps(doc, indent = 1, default = str))
    lines = [f"# edge_bline_sdcpp: {counts['pass']} pass, {counts['fail']} fail, {counts['info']} info, "
             f"{counts['skip']} skip in {doc['seconds']} s", "", "| check | status | s | evidence |", "|---|---|---|---|"]
    for x in r.results:
        ev = json.dumps(x["evidence"], default = str)
        lines.append(f"| {x['name']} | {x['status'].upper()} | {x['seconds']} | {ev[:220].replace('|', '/')} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    C.log(f"[edge] {counts} in {doc['seconds']} s -> {out / 'results.json'}")
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
