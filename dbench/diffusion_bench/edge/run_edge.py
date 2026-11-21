#!/usr/bin/env python3
"""Studio edge-case suite: pokes every user-facing image / video surface of Unsloth Studio, in process and over
HTTP, and reports PASS / FAIL / SKIP / INFO per check with evidence. Built to be re-run on every change.

  python diffusion_bench/edge/run_edge.py --surface both --tier fast --out outputs/edge/$(date +%Y%m%d_%H%M%S)
  python diffusion_bench/edge/run_edge.py --surface http --only 'image.*' --out DIR
  python diffusion_bench/edge/run_edge.py --list

Tiers:
  tiny  hf-internal-testing random-weight pipelines ($WORKSPACE/hf_tiny/<repo name>, fetch them once with
        diffusion_bench/fetch_tiny.py): every fast + full check on
        tiny SDXL / tiny FLUX / tiny Wan (plumbing only: sizes, determinism, leaks, validation, offload, compile,
        quant), plus a per-family load/render matrix. Output is noise, so no quality assertion runs.
  fast  (default) SDXL-Turbo 512px 1-4 steps, Z-Image-Turbo 512/768 8 steps, Wan2.2-TI2V-5B 17 frames 4 steps;
        target < 10 min for both surfaces on one GPU (measured 5.4 min on a B200)
  full  fast + compile / CUDA-graph resize, quant switching, FLUX.1-schnell, 2048px, video memory modes, unload
        mid-clip, a third leak cycle

Environment:
  DIFFUSION_BENCH_STUDIO_SRC      Studio source (path, git ref, pr/N, pypi); --studio-src overrides
  DIFFUSION_BENCH_STUDIO_PYTHON   interpreter for the in-process worker and the launched server (a Studio install's
                                  unsloth_studio/bin/python runs both); --python / --inproc-python override
  DIFFUSION_BENCH_STUDIO_URL (+ _USER / _PASSWORD / _TOKEN / _PID)   --attach: test a running Studio / Desktop
  CUDA_VISIBLE_DEVICES            the first device is used (--gpu overrides)

Writes <out>/results.json and <out>/summary.md; exits 1 when any check FAILs, 2 on a harness error, 75 when
nothing FAILed but a check was VOID (a worker that never started: its stack dump is in the check's evidence; or an
identity check whose reference and compared render ran under different load plans, both plans in its evidence).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import common as C  # noqa: E402
import setup_studio  # noqa: E402
from framework import CHECKS, FAIL, INFO, PASS, SKIP, VOID, Runner  # noqa: E402
from surfaces import HarnessHang  # noqa: E402
import checks  # noqa: E402,F401 - registers the checks
from surfaces import HttpSurface, InprocSurface  # noqa: E402


def model_cfgs(args) -> dict:
    """Model configs by role: img_a (the workhorse image model), img_b (the switch target), vid (video)."""
    fast = {"speed_mode": "off", "memory_mode": "fast"}

    def cfg(kind, model, family, gen, long = None):
        return {"kind": kind, "model": model, "family": family, "opts": {"family_override": family, **fast},
                "gen": gen, **({"long": long} if long else {})}

    if args.tier == "tiny":
        t = Path(args.tiny_root)
        # Random-weight hf-internal-testing pipelines: output is noise, so only plumbing is judged. Sizes stay at
        # Studio's 256px minimum where the tiny rope tables fit. HTTP video only accepts the family presets
        # (1280x704), past the stock tiny Wan rope (32); fetch_tiny.py derives tiny-wan-pipe-rope128 (same weights,
        # rope_max_seq_len 128), which renders them. Without it the video checks SKIP over HTTP.
        wan = t / "tiny-wan-pipe-rope128"
        vid_gen = {"inproc": {"width": 256, "height": 256, "num_frames": 9, "steps": 2, "guidance": 1.0}}
        if wan.exists():
            vid_gen["http"] = {"width": 1280, "height": 704, "num_frames": 9, "steps": 2, "guidance": 1.0}
        else:
            wan = t / "tiny-wan-pipe"
        return {
            "img_a": cfg("image", str(t / "tiny-stable-diffusion-xl-pipe"), "sdxl",
                         {"*": {"width": 256, "height": 256, "steps": 2, "guidance": 0.0}},
                         {"width": 1024, "height": 1024, "steps": 100}),
            "img_b": cfg("image", str(t / "tiny-flux-pipe"), "flux.1",
                         {"*": {"width": 256, "height": 256, "steps": 2, "guidance": 0.0}}),
            "vid": cfg("video", str(wan), "wan2.2-ti2v-5b", vid_gen, {"steps": 100}),
        }
    cfgs = {
        "img_a": cfg("image", args.sdxl, "sdxl", {"*": {"width": 512, "height": 512, "steps": 2, "guidance": 0.0}},
                     {"width": 1024, "height": 1024, "steps": 80}),
        "img_b": cfg("image", args.zimage, "z-image", {"*": {"width": 512, "height": 512, "steps": 8, "guidance": 0.0}}),
        # HTTP only accepts the family's resolution presets (1280x704 / 704x1280) and 4k+1 frame counts; in process
        # the same backend takes any size on its 32px grid, so the fast tier renders smaller there.
        "vid": cfg("video", args.wan, "wan2.2-ti2v-5b",
                   {"inproc": {"width": 480, "height": 320, "num_frames": 17, "steps": 4, "guidance": 1.0},
                    "http": {"width": 1280, "height": 704, "num_frames": 17, "steps": 4, "guidance": 1.0}},
                   {"steps": 40}),
    }
    if args.flux and Path(args.flux).exists():
        cfgs["flux"] = cfg("image", args.flux, "flux.1", {"*": {"width": 512, "height": 512, "steps": 4,
                                                                "guidance": 0.0}})
    return cfgs


# Checks that fail on current Studio main for a Studio reason, not a harness one (seen on be4598e16, B200, fast tier,
# both surfaces). They stay FAIL and keep the exit code at 1: the annotation only tells a reader which failures are
# already understood, so a NEW failure stands out. Delete an entry once Studio fixes it.
KNOWN_STUDIO_FAILS = {
    "image.leak_cycles": "Studio bug: host anon RSS is not returned after an image unload (SDXL-Turbo: +5 to +6 GiB "
                         "per load / unload cycle, 8.4 -> 14.4 -> 19.5 GiB in process); same family as the RAM growth "
                         "in unslothai/unsloth#10156",
    "image.unload_during_generate": "Studio bug: unloading during a render leaves the pipeline's VRAM resident "
                                    "(~7 GiB for SDXL-Turbo, flat for 20 s+) until a later load or unload",
    "switch.video_to_image": "Studio bug: host anon RSS is not returned after the video model is unloaded (+7 to "
                             "+14 GiB over the first image unload); same root as image.leak_cycles",
}


# Exit when nothing FAILed but a check could not run (a worker that never started): EX_TEMPFAIL, which the
# switchboard reads as a harness VOID for that side (studio_regress/external.py), never as a pass.
EXIT_HARNESS_VOID = 75


def write_reports(out: Path, meta: dict, results: list) -> tuple[int, dict]:
    for r in results:
        if r["status"] == FAIL and r["check"] in KNOWN_STUDIO_FAILS:
            r["known"] = KNOWN_STUDIO_FAILS[r["check"]]
    counts = {s: sum(1 for r in results if r["status"] == s) for s in (PASS, FAIL, SKIP, INFO, VOID)}
    counts["FAIL_known"] = sum(1 for r in results if r["status"] == FAIL and r.get("known"))
    (out / "results.json").write_text(json.dumps({"meta": meta, "counts": counts, "results": results}, indent = 1,
                                                 default = str))
    lines = [f"# Studio edge suite ({meta['tier']})", "",
             f"- tree: `{meta.get('tree')}` @ `{meta.get('rev')}`", f"- python: `{meta.get('python')}`",
             f"- surfaces: {', '.join(meta['surfaces'])}; GPU {meta['gpu']} ({meta.get('gpu_name')})",
             f"- wall: {meta.get('wall_s')} s " + " ".join(f"({k} {v} s)" for k, v in meta.get("surface_s", {}).items()),
             f"- **{counts[PASS]} pass, {counts[FAIL]} fail ({counts['FAIL_known']} known Studio issues, "
             f"{counts[FAIL] - counts['FAIL_known']} new), {counts[SKIP]} skip, {counts[INFO]} info"
             f"{f', {counts[VOID]} VOID (harness)' if counts[VOID] else ''}**", "",
             "| surface | check | status | s | detail |", "|---|---|---|---|---|"]
    for r in results:
        detail = r.get("error") or "; ".join(r.get("failures") or [])
        if not detail and r.get("infos"):
            detail = json.dumps(r["infos"], default = str)
        status = f"{r['status']} (known)" if r.get("known") else r["status"]
        lines.append(f"| {r['surface']} | {r['check']} | {status} | {r['wall_s']} | "
                     f"{detail.replace('|', '/')[:300]} |")
    fails = [r for r in results if r["status"] == FAIL]
    if fails:
        lines += ["", "## Failures", ""]
        for r in fails:
            lines.append(f"### {r['surface']} / {r['check']}")
            lines.append(f"{r.get('doc', '')}")
            if r.get("known"):
                lines += ["", f"Known: {r['known']}"]
            lines.append("")
            lines.append("```")
            lines.append(json.dumps({"error": r.get("error"), "failures": r.get("failures"),
                                     "evidence": {k: v for k, v in (r.get("evidence") or {}).items()
                                                  if k in (r.get("failures") or []) or k == "traceback"}},
                                    indent = 1, default = str)[:4000])
            lines.append("```")
    infos = [r for r in results if r.get("infos")]
    if infos:
        lines += ["", "## Observations (INFO)", ""]
        for r in infos:
            lines.append(f"- {r['surface']} / {r['check']}: `{json.dumps(r['infos'], default = str)[:600]}`")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    return counts[FAIL], counts


def main() -> int:
    ws = C.WS
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface", choices = ["inproc", "http", "both"], default = "both")
    ap.add_argument("--tier", choices = ["tiny", "fast", "full"], default = "fast")
    ap.add_argument("--out", default = None)
    ap.add_argument("--only", action = "append", default = [], help = "glob or substring of check names (repeatable)")
    ap.add_argument("--list", action = "store_true")
    ap.add_argument("--studio-src", default = os.environ.get("DIFFUSION_BENCH_STUDIO_SRC"))
    ap.add_argument("--python", default = os.environ.get("DIFFUSION_BENCH_STUDIO_PYTHON"),
                    help = "interpreter for the server and (default) the in-process worker")
    ap.add_argument("--inproc-python", default = None)
    ap.add_argument("--gpu", default = None)
    ap.add_argument("--attach", action = "store_true", help = "HTTP against DIFFUSION_BENCH_STUDIO_URL instead of launching")
    ap.add_argument("--sdxl", default = str(ws / "sdxl_turbo_base"))
    ap.add_argument("--zimage", default = str(ws / "hf_local_bases/Tongyi-MAI/Z-Image-Turbo"))
    ap.add_argument("--wan", default = str(ws / "hf_local_bases/Wan-AI/Wan2.2-TI2V-5B-Diffusers"))
    ap.add_argument("--flux", default = str(ws / "hf_local_bases/black-forest-labs/FLUX.1-schnell"))
    ap.add_argument("--tiny-root", default = str(ws / "hf_tiny"),
                    help = "hf-internal-testing tiny pipelines (tiny tier), one directory per repo name")
    args = ap.parse_args()

    if args.list:
        for c in CHECKS:
            print(f"{c.name:<34} {c.tier:<5} {','.join(c.surfaces):<12} {c.needs or '-':<5} {c.doc}")
        return 0
    if not args.out:
        ap.error("--out is required")
    C.scrub_tokens()
    import signal

    def _exit_on(sig, _frame):  # SystemExit unwinds through the finally that stops each surface's Studio
        raise SystemExit(128 + sig)
    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):  # no SIGHUP on Windows
        if sig is not None:
            signal.signal(sig, _exit_on)
    out = Path(args.out).resolve()
    out.mkdir(parents = True, exist_ok = True)
    gpu = args.gpu or C.visible_gpu()
    cfgs = model_cfgs(args)
    missing = [n for n, c in cfgs.items() if not Path(c["model"]).exists()]
    if missing:
        print(f"missing local models: {missing}", file = sys.stderr)
        if args.tier == "tiny":
            print(f"fetch them with: python {HERE.parent / 'fetch_tiny.py'} --root {args.tiny_root}", file = sys.stderr)
        return 2
    if args.surface in ("http", "both") and "vid" in cfgs:
        import importlib.util

        if not any(importlib.util.find_spec(m) for m in ("av", "imageio")):
            # HTTP video comes back as MP4 and is decoded in THIS interpreter (studio_client.decode_mp4)
            print("the http surface decodes MP4 in this interpreter: install av (or imageio + imageio-ffmpeg), or "
                  "run run_edge.py with the Studio install's python", file = sys.stderr)
            return 2
    if args.python:
        tree = setup_studio.resolve_tree(args.studio_src, python = args.python)
        python = args.python
    else:
        info = setup_studio.ensure(args.studio_src)
        tree, python = info, info["python"]
    surfaces = ["inproc", "http"] if args.surface == "both" else [args.surface]
    meta = {"tier": args.tier, "surfaces": surfaces, "gpu": gpu, "gpu_name": C.gpu_query(gpu).get("name"),
            "tree": tree.get("tree"), "rev": tree.get("rev"), "python": python, "only": args.only,
            "models": {k: v["model"] for k, v in cfgs.items()}, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    results: list = []
    t_all = time.perf_counter()
    meta["surface_s"] = {}
    for name in surfaces:
        t0 = time.perf_counter()
        C.log(f"=== surface {name} (tier {args.tier}) -> {out / name}")
        (out / name).mkdir(parents = True, exist_ok = True)
        if name == "inproc":
            surf = InprocSurface(args.inproc_python or python, tree["tree"] if tree.get("mode") != "pypi" else "pypi",
                                 out / name, gpu, out / name / "worker.log",
                                 studio_home = str(out / name / "studio_home"))
        else:
            surf = HttpSurface(out / name, gpu, studio_src = tree["tree"] if tree.get("mode") != "pypi" else "pypi",
                               python = python, attach = {} if args.attach else None)
        try:
            start = surf.start()
            meta.setdefault("surface_info", {})[name] = {k: start.get(k) for k in ("base_url", "home", "log", "pid",
                                                                                     "server_pid")} if isinstance(start, dict) else start
            runner = Runner(surf, cfgs, out, args.tier, args.only, log = C.log)
            runner.tiny_root = args.tiny_root
            for r in runner.run():
                results.append(r.__dict__)
                write_reports(out, meta, results)
        except HarnessHang as e:   # the worker never came up: nothing was tested (VOID), dump attached
            C.log(f"surface {name} worker never started: {e.detail[:300]}")
            results.append({"surface": name, "check": "_surface_start", "tier": args.tier, "status": VOID,
                            "wall_s": round(time.perf_counter() - t0, 1), "error": f"harness hang: {e.detail[:800]}",
                            "failures": [], "infos": {},
                            "evidence": {"startup_dump": e.extra.get("dump"), "startup_dump_path": e.extra.get("dump_path")}})
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            C.log(f"surface {name} harness error:\n{tb}")
            results.append({"surface": name, "check": "_surface_start", "tier": args.tier, "status": FAIL,
                            "wall_s": round(time.perf_counter() - t0, 1), "error": tb[-1500:], "failures": [],
                            "infos": {}, "evidence": {}})
        finally:
            try:
                surf.stop()
            except Exception as e:  # noqa: BLE001
                C.log(f"stop {name} failed: {e}")
        meta["surface_s"][name] = round(time.perf_counter() - t0, 1)
    meta["wall_s"] = round(time.perf_counter() - t_all, 1)
    n_fail, counts = write_reports(out, meta, results)
    C.log(f"done in {meta['wall_s']} s: {counts} -> {out / 'summary.md'}")
    return 1 if n_fail else (EXIT_HARNESS_VOID if counts[VOID] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
