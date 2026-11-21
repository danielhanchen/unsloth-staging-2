#!/usr/bin/env python3
"""Run ONE benchmark cell in this process and write <out>/<tag>/record.json (schema v2, schema.py) plus media.

The timing protocol is the same for every backend, so numbers are comparable across Studio (in process or HTTP),
plain diffusers, BlineUI and stable-diffusion.cpp:

  gates          capability (min_compute_cap: an NVIDIA requirement on AMD / Apple / CPU is NOT_RUN unsupported),
                 cooldown (options.cool_to_c), foreign-VRAM idle wait (default on for remote timing targets)
  budget         options.budget / target.simulated_budget_gib: VRAM ballast + simulated total (labelled simulated),
                 host-RAM cap over the whole tree (budget.py)
  load           timing.load_s
  cold           request kind "cold": the warm-up prompt (never in the timed set) with its own seed.
                 timing.first_render_s = its total, timing.cold_s = load_s + first_render_s, timing.first_step_s =
                 submit -> first denoise step of that request; compile_s from Dynamo in process, else derived and
                 named as such
  warmup x W-1   more warm-up renders (kind "warmup"), new seeds
  N new          kind "new": each prompt of the set once with its seed (paired with other cells for quality), media
                 saved; total_s, polled step_s[] (s/step median from step 3), first_step_s, TE / denoise / VAE phases
  short_n pairs  kind "same": a seen prompt with a NEW seed at full steps and at short_steps; timing.step_s_derived =
                 (median long - median short) / (steps - short_steps), flagged when a step cache is on
  telemetry      20 ms NVML sampler: device / own-tree VRAM, process-tree RSS (servers and short-lived sd-cli
                 children included), SM clock, temperature, power, PCIe link (telemetry.py)
  engagement     options.levers {name: expected}: each must show evidence of engaging, else the cell is VOID
  parity         --parity (or options.parity_probe): one render with the parity hooks on (sigmas, denoiser
                 evaluations, initial noise), plus resolved settings and component precision, for parity.py

One process per cell (``matrix.py`` does it): compile caches, allocator state and cuDNN autotuning leak across
cells otherwise. Caches are fresh per cell by default (Inductor, Triton, Studio home under the cell dir);
options.cache = "returning" reuses one shared set per stack, the explicit "returning user" arm. A failing cell
writes verdict BROKEN with the error, never a partial "ok".

Usage:
  python diffusion_bench/run_cell.py --cell '{"tag":"zimg_bf16","backend":"studio","model":"..."}' --out DIR
  python diffusion_bench/run_cell.py --spec diffusion_bench/specs/smoke.json --tag zimg_bf16 --out DIR [--parity]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import budget as B  # noqa: E402
import common as C  # noqa: E402
import schema  # noqa: E402
import telemetry as T  # noqa: E402
from backends.base import get_backend  # noqa: E402

# Outside every prompt set, so the cold / warm-up renders never share a prompt (or a text-encoder cache entry)
# with a timed render.
WARMUP_PROMPT = ("A plain ceramic mug on a wooden table next to a window, soft morning light, a small plant in the "
                 "background, photographed with a 50mm lens")
WARMUP_SEED = 917_000_003
SAME_SEED_STRIDE = 7919


def _cache_env(cell: dict, out: Path) -> dict:
    """Fresh Inductor / Triton / Studio-home dirs per cell, or the shared "returning" set. Applied to os.environ
    (inherited by every server the backend starts) and options.studio_home (in-process Studio)."""
    opts = cell.setdefault("options", {})
    mode = opts.get("cache") or "fresh"
    stack = schema.stack_of(cell.get("backend")) or "x"
    # absolute (a server runs with its own cwd); a launcher's per-cell fresh root (DBENCH_FRESH_CACHE_ROOT,
    # vm_driver) wins over the cell dir; the fresh root is removed after the cell (_drop_fresh_cache)
    fresh = Path(os.environ.get("DBENCH_FRESH_CACHE_ROOT") or (out / "_cache")).resolve()
    root = (C.WS / "temp" / "diffusion_bench" / "returning" / stack) if mode == "returning" else fresh
    dirs = {"TORCHINDUCTOR_CACHE_DIR": str(root / "inductor"), "TRITON_CACHE_DIR": str(root / "triton")}
    for k, v in dirs.items():
        if not (cell.get("env") or {}).get(k):
            os.environ[k] = v
            Path(v).mkdir(parents = True, exist_ok = True)
    if not opts.get("studio_home"):
        opts["studio_home"] = str(root / "studio_home")
    return {"mode": mode, "inductor": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "triton": os.environ.get("TRITON_CACHE_DIR"), "studio_home": opts["studio_home"],
            "fresh_root": str(root) if mode != "returning" else None}


def vram_source(rec: dict, sampler, backend) -> None:
    """memory.vram_source: the backend's own reading (e.g. "mps_driver") when the sampler used it, else the device
    sampler's source; on the Metal target with no reading at all, an explicit UNAVAILABLE with the reason (the stack
    exposes no Metal device memory: a server's /system_stats reports system RAM for MPS). Elsewhere a missing metric
    stays missing, and schema.problems VOIDs an ok record without it."""
    mem = rec["memory"]
    has = any(mem.get(k) is not None for k in ("peak_vram_own_gib", "peak_vram_over_base_gib", "peak_vram_device_gib"))
    if getattr(sampler, "fallback_hits", 0) and mem.get("peak_vram_own_gib") is not None:
        mem["vram_source"] = getattr(backend, "vram_source", None) or "backend"
    elif has:
        mem["vram_source"] = sampler.source
    elif schema.metal_target(rec):
        mem["vram_source"] = "unavailable"
        mem["vram_unavailable_reason"] = (f"{schema.stack_of((rec.get('cell') or {}).get('backend')) or 'stack'} "
                                          "reports no Metal device memory on this target (no NVML / SMI on macOS)")


def _drop_fresh_cache(rec: dict) -> None:
    """Remove a fresh cell's own cache root (only the path this run created / was given for this cell)."""
    root = (rec.get("cache") or {}).get("fresh_root")
    if not root:
        return
    given = os.environ.get("DBENCH_FRESH_CACHE_ROOT")
    if Path(root).name == "_cache" or (given and Path(given).resolve() == Path(root)):
        shutil.rmtree(root, ignore_errors = True)


# Environment a launcher (targets/vm_driver.py, the tester bundle's run_tester.py, the AMD workflow) sets for every
# cell: read here, recorded in the record (``launcher``), never needed when run by hand.
LAUNCHER_ENV = ("DBENCH_START_TEMP_C", "DBENCH_GPU_LABEL", "DBENCH_SCRIPTS_SHA", "DBENCH_PLAN_SHA256",
                "DBENCH_NO_PROFILER")


def launcher_env() -> dict:
    return {k: os.environ.get(k) for k in LAUNCHER_ENV if os.environ.get(k) not in (None, "")}


def _gates(cell: dict, device, log) -> dict:
    opts = cell.get("options") or {}
    out: dict = {}
    # options.cool_to_c wins; the tester / VM launcher's start-temperature gate (DBENCH_START_TEMP_C) otherwise
    cool = opts.get("cool_to_c") or os.environ.get("DBENCH_START_TEMP_C")
    if cool:
        out["cooldown"] = T.cooldown(device, float(cool), log = log)
    target = (opts.get("target") or {}).get("name") or ""
    fw = opts.get("foreign_wait")
    if fw is None:  # on for remote timing targets; the fake backend never touches a GPU
        fw = {} if target and not target.startswith(("local", "cpu")) and cell.get("backend") != "fake" else False
    if fw is not False and device is not None:
        fw = dict(fw or {})
        out["foreign"] = T.foreign_idle_wait(device, T.tree_pids([os.getpid()]),
                                             max_foreign_mib = int(fw.get("max_foreign_mib", 1024)),
                                             hold_s = float(fw.get("hold_s", 5)),
                                             timeout_s = float(fw.get("timeout_s", 900)), log = log)
    gates = [g.get("gate") for g in out.values()]
    # a gate that could not run (no NVML / smi reading) is "unavailable", never summarised as ok
    out["gate"] = "timeout" if "timeout" in gates else ("unavailable" if "unavailable" in gates
                                                        else ("ok" if gates else "none"))
    return out


def _phases(t0: float, t1: float, steps: list, phase: dict, step_med) -> dict:
    """TE / denoise / VAE split of one request. Phase marks (sampler start / end) win; else the polled steps, with
    the first step's start estimated as its timestamp minus one median step."""
    if "denoise_start" in phase:
        ds = phase["denoise_start"]
        de = phase.get("denoise_end") or (steps[-1][2] if steps else None)
        if de is not None:
            return {"te_s": round(ds - t0, 4), "denoise_s": round(de - ds, 4), "vae_s": round(t1 - de, 4),
                    "source": "phase_marks"}
    if len(steps) >= 2 and step_med:
        ds = steps[0][2] - step_med
        return {"te_s": round(max(0.0, ds - t0), 4), "denoise_s": round(steps[-1][2] - ds, 4),
                "vae_s": round(t1 - steps[-1][2], 4), "source": "poll"}
    return {"te_s": None, "denoise_s": None, "vae_s": None, "source": None}


def _parity_from(backend, dump: Path | None, t0: float, t1: float) -> dict:
    ev = {}
    if hasattr(backend, "parity_evidence"):
        ev = backend.parity_evidence() or {}
    if not ev and dump is not None:
        sys.path.insert(0, str(B.SHIM))
        import dbench_hooks

        ev = dbench_hooks.summarize(dbench_hooks.read_events(dump, t0, t1))
        ev["source"] = "hooks"
    elif ev:
        ev["source"] = "backend"
    return ev


def _compile_s(backend=None) -> tuple:
    """(seconds, source). In process: torch._dynamo's own compile metrics. A server backend (BlineUI, Studio HTTP,
    sd.cpp) compiles in another process: its own ``compile_s()`` figure when it has one (server log), else
    (None, None), so the report falls back to the cell's first-vs-steady delta (compile_s_derived) or prints
    UNAVAILABLE, never this driver's 0.0."""
    if backend is not None and not getattr(backend, "in_process", True):
        fn = getattr(backend, "compile_s", None)
        try:
            v = fn() if callable(fn) else None
        except Exception:  # noqa: BLE001
            v = None
        return (round(float(v), 3), "server log") if v is not None else (None, None)
    torch = sys.modules.get("torch")
    if torch is None:
        return None, None
    try:
        from torch._dynamo.utils import compilation_time_metrics as m
    except Exception:  # noqa: BLE001
        return None, None
    keys = [k for k in m if "compile_inner" in k or k == "entire_frame_compile"]
    if not keys:
        return (0.0, "dynamo") if "torch._dynamo" in sys.modules else (None, None)
    return round(sum(sum(m[k]) for k in keys[:1]), 3), "dynamo"


def _settings(cell: dict, backend, rec: dict) -> list:
    """The planner's cell settings (plan.card_to_cell) as each backend honoured them: {key: {requested, applied,
    where}}; returns contradictions (a setting the stack reports differently = VOID: it measured something else)."""
    o = cell.get("options") or {}
    st = rec.get("status_after") or rec.get("status") or {}
    st = st if isinstance(st, dict) else {}
    out, bad = {}, []

    def put(key, applied, where):
        out[key] = {"requested": o.get(key), "applied": applied, "where": where}

    stack = (rec.get("arm") or {}).get("stack")
    if stack == "studio":
        put("studio_src", (rec.get("revisions") or {}).get("studio") or o.get("studio_src"), "backend setup_studio tree")
        put("speed_mode", st.get("speed_mode"), "Studio status.speed_mode")
        if str(o.get("speed_mode")) == "default" and st.get("speed_mode") not in (None, "", "default"):
            # "default" asks for what a user gets with no choice; Studio resolving it to another tier on this card
            # (a family outside the tier system, an offloaded load that declines the compiled profile) IS that, so
            # it is measured and labelled. An explicit tier Studio did not run stays a contradiction (below).
            out["speed_mode"]["flag"] = f"Studio's default resolved to {st['speed_mode']!r} on this load"
        elif o.get("speed_mode") not in (None, "") and st.get("speed_mode") not in (None, "") and \
                str(st["speed_mode"]) != str(o["speed_mode"]):
            bad.append(f"speed_mode requested {o['speed_mode']!r}, Studio ran {st['speed_mode']!r}")
        elif o.get("speed_mode") not in (None, "") and st and st.get("speed_mode") in (None, ""):
            out["speed_mode"]["flag"] = "not reported by Studio status: the tier that ran is unproven"
        put("generate", sorted((o.get("generate") or {}).keys()), "generate() keywords")
        # the tier Studio RESOLVED (compiled or eager, CUDA graph, INT8 attention, quant, cache): what the report
        # labels, splits and compares, whatever was requested
        tier = C.resolved_tier(st)
        rec["resolved_tier"] = None if tier is None else {**tier, "requested": o.get("speed_mode")}
    elif stack in ("blineui", "sdcpp"):
        put("family", st.get("family") or o.get("family"), f"{stack} graph / flag family")
        if o.get("staged_files"):
            # mirror copies the driver verified against the official checkpoint (mirror.py): which and how
            rec["staged_files"] = o["staged_files"]
        names = getattr(backend, "names", None) or {}
        put("files", {k: (Path(v).name if isinstance(v, str) else v) for k, v in (names or o.get("files") or {}).items()},
            "files linked for the server")
        if stack == "blineui":
            graph = getattr(backend, "graph", None) or {}
            wd = sorted({n["inputs"]["weight_dtype"] for n in graph.values() if isinstance(n, dict)
                         and isinstance((n.get("inputs") or {}).get("weight_dtype"), str)})
            put("weight_dtype", wd or None, "UNETLoader weight_dtype in the submitted graph")
            if o.get("weight_dtype") not in (None, "default") and wd and o["weight_dtype"] not in wd:
                bad.append(f"weight_dtype requested {o['weight_dtype']!r}, graph has {wd}")
            put("cache_bust", st.get("cache_bust") or o.get("cache_bust", "all"), "bline_graphs.apply_nonce nodes")
            put("template", o.get("template"), "recorded only: the registered graph is the template's recipe "
                                               "(models card bline.template / template_sha)")
    if cell.get("budget_gib") is not None:
        put("budget_gib", ((rec.get("memory") or {}).get("budget") or {}).get("gib"), "budget.py (simulated)")
        out["budget_gib"]["requested"] = cell.get("budget_gib")
    rec["settings"] = out
    return bad


def _engagement(cell: dict, backend, rec: dict) -> None:
    levers = dict((cell.get("options") or {}).get("levers") or {})
    eng = rec["engagement"]
    eng["status_after"] = rec.get("status_after")
    for name, expected in levers.items():
        try:
            engaged, evidence = backend.lever_evidence(name, expected)
        except Exception as exc:  # noqa: BLE001
            engaged, evidence = None, f"evidence lookup failed: {exc}"
        eng["levers"][name] = {"requested": expected, "engaged": engaged, "evidence": evidence}
    lines = getattr(backend, "engaged_lines", None)
    if callable(lines):
        eng["bline_log_lines"] = lines()[:50]


def run(cell: dict, out_root: Path, quiet: bool = False, parity: bool = False, mem_block: dict | None = None) -> dict:
    cell = C.expand_env(cell)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # torch indices == NVML / nvidia-smi indices
    import extnames  # noqa: PLC0415

    for key, value in extnames.ext_env({str(k): v for k, v in (cell.get("env") or {}).items()}).items():
        os.environ[key] = str(value)  # a variable the real package reads keeps its real spelling (public copy)
    opts = cell.setdefault("options", {})
    ci_dropped = None
    if os.environ.get("CI") and not opts.get("keep_ci_env") and "CI" not in (cell.get("env") or {}):
        ci_dropped = os.environ.pop("CI")  # CI=true flips torch's functorch aliasing checks on AMD runners
    if not cell.get("keep_tokens"):
        C.scrub_tokens()
    if parity or opts.get("parity_probe"):
        parity = True
        cell.update({"n": 1, "short_n": 0, "warmup": 1})
    out = out_root / cell["tag"]
    out.mkdir(parents = True, exist_ok = True)
    rec = schema.new_record(cell)
    rec.update({"backend": cell["backend"], "renders": [], "long": [], "short": []})
    rec["mode"] = "parity" if parity else "timed"
    say = (lambda m: None) if quiet else (lambda m: C.log(f"[{cell['tag']}] {m}"))
    device = C.visible_gpu()
    rec["cache"] = _cache_env(cell, out)
    rec["env_flags"] = {"CI_dropped": ci_dropped, "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
                        "no_profiler": bool(os.environ.get("DBENCH_NO_PROFILER"))}
    rec["launcher"] = launcher_env()
    rec["revisions"]["scripts_sha"] = C.scripts_sha()
    rec["revisions"]["plan_sha256"] = os.environ.get("DBENCH_PLAN_SHA256") or None
    if os.environ.get("DBENCH_GPU_LABEL"):
        rec["gpu"]["label"] = os.environ["DBENCH_GPU_LABEL"]
    ident = T.identity(device)
    rec["gpu"].update({k: ident.get(k) for k in ("name", "uuid", "driver", "pcie_gen", "pcie_width", "cc")})
    rec["gpu"].update({k: ident.get(k) for k in ("pcie_gen_max", "pcie_width_max", "total_mib", "source")})
    ok, why = C.capability_gate(cell.get("min_compute_cap"), ident if ident.get("source") != "none" else None, device)
    if not ok:
        rec.update({"verdict": "NOT_RUN", "reason": why, "phase": "gate"})
        C.write_record(out, rec)
        say(f"NOT_RUN: {why}")
        return rec
    b = B.spec(cell)
    ballast = None
    backend = None
    sampler = None
    # absolute (resolved against this driver's cwd): the hooks write it from servers that run with their own cwd
    dump = (out / "_parity.jsonl").resolve() if parity else None
    if dump is not None:
        dump.unlink(missing_ok = True)
    try:
        rec["gate"] = _gates(cell, device, say)
        idle = T.device_used_mib(device)
        target_free = B.ballast_target_mib(b, idle)
        if target_free is not None and device is not None:
            ballast = B.Ballast(target_free, device, python = (opts.get("budget") or {}).get("python")).start()
            say(f"ballast: {ballast.got_free_mib:.0f} MiB free (target {target_free})")
        rec["memory"]["budget"] = B.record_block(b, ballast, mem_block or {"mem_max": b.get("mem_max"),
                                                                           "applied": False}, idle)
        # per mechanism: the simulated mem_get_info total is a Python shim, so a C++ server (sd.cpp) gets only the
        # ballast; "simulated" alone would read as both applied
        rec["memory"]["budget"]["ballast_applied"] = ballast is not None
        rec["memory"]["budget"]["sim_total_applied"] = bool(b.get("sim_total_gib")) and \
            schema.stack_of(cell.get("backend")) != "sdcpp"
        hooks = B.install_in_process(b, str(dump) if dump else None)
        if hooks:
            rec["hooks"] = hooks
        cls = get_backend(cell["backend"])
        backend = cls(cell, out)
        backend.server_env = B.child_env(dict(os.environ), b, str(dump) if dump else None)
        sampler = T.Sampler(device, root_pids = lambda: [os.getpid()] + list(backend.extra_pids() or []),
                            exclude_pids = lambda: [ballast.pid] if ballast else [],
                            own_vram_fallback = getattr(backend, "own_vram_mib", None))
        sampler.start()
        rows = C.load_prompts(cell["prompts"], n = cell.get("n"), ids = cell.get("ids"))
        steps = int(cell["steps"])
        rec["gpu_before_load"] = C.gpu_query() or None  # matrix.pressure_void reads it when the load breaks
        t_load = time.time()
        rec["phase"] = "load"  # where a BROKEN cell stopped (matrix.pressure_void excuses load failures only)
        rec["status"] = backend.load() or {}
        backend.last_status = rec["status"]
        rec["load_s"] = round(time.time() - t_load, 2)
        rec["timing"]["load_s"] = rec["load_s"]
        rec["phase"] = "render"
        rec["host_after_load"] = C.host_memory()
        say(f"loaded in {rec['load_s']}s: {json.dumps(rec['status'], default = str)[:400]}")

        def one(kind: str, row: dict, n_steps: int) -> tuple:
            backend.begin_render()
            backend.reset_peak()
            t0 = time.time()
            result = backend.render(row, n_steps)
            backend.sync()
            t1 = time.time()
            if result.peak_alloc_gib is None and backend.in_process:
                result.peak_alloc_gib = backend.peak_alloc_gib()
            ev = sorted(backend.step_events, key = lambda e: e[2])
            stamps = [(s, t) for s, _, t in ev]
            step_med = C.polled_step_s(stamps)
            seen, deltas, prev = set(), [], None
            for s, _, t in ev:
                if s in seen:
                    continue
                seen.add(s)
                if prev is not None and s > prev[0]:
                    deltas.append(round((t - prev[1]) / (s - prev[0]), 5))
                prev = (s, t)
            win = sampler.window(t0, t1)
            req = {"kind": kind, "id": row.get("id"), "seed": row.get("seed"), "prompt_id": row.get("id"),
                   "total_s": round(t1 - t0, 4), "steps": n_steps, "step_s": deltas,
                   "step_s_polled": round(step_med, 5) if step_med is not None else None,
                   "first_step_s": round(ev[0][2] - t0, 4) if ev else None,
                   "phases": _phases(t0, t1, ev, backend.phase_events, step_med),
                   "peak_vram_over_base_gib": None if win["over_base_mib"] is None else round(win["over_base_mib"] / 1024, 3),
                   "peak_vram_own_gib": None if win["own_mib"] is None else round(win["own_mib"] / 1024, 3),
                   "peak_rss_tree_gib": None if win["rss_tree_mib"] is None else round(win["rss_tree_mib"] / 1024, 3),
                   "clocks": sampler.clocks(t0, t1), "t0": round(t0, 3), "t1": round(t1, 3)}
            if result.step_s is not None:
                req["step_s_reported"] = result.step_s
            req.update(backend.request_facts or {})
            if parity and kind == "new":
                req["parity"] = _parity_from(backend, dump, t0, t1)
            rec["requests"].append(req)
            return result, t1 - t0, req

        warm = {"id": "warmup", "prompt": WARMUP_PROMPT, "seed": WARMUP_SEED, "split": "warmup"}
        _, wall, req = one("cold", warm, steps)
        rec["cold_s"] = round(wall, 3)
        rec["timing"].update({"first_render_s": round(wall, 3), "cold_s": round(rec["load_s"] + wall, 3),
                              "first_step_s": req.get("first_step_s")})
        tw = time.time()
        for i in range(1, int(cell.get("warmup") or 1)):
            one("warmup", {**warm, "seed": WARMUP_SEED + i}, steps)
        rec["warmup_s"] = round(time.time() - tw, 2)
        t_timed = time.time()
        for row in rows:
            result, wall, req = one("new", row, steps)
            entry = {**row, "wall_s": round(wall, 3), "step_s": result.step_s, "peak_alloc_gib": result.peak_alloc_gib,
                     **({"extra": result.extra} if result.extra else {})}
            if cell["kind"] == "video":
                entry.update(C.save_video(result.frames, out / row["id"], fps = result.fps or cell.get("fps") or 16,
                                          every = int(cell.get("save_frames_every") or 1)))
                media = {"id": row["id"], "file": entry["file"], "frames": entry.get("frames"),
                         "height": (entry.get("shape") or [None])[0], "width": (entry.get("shape") or [None, None])[1]}
            else:
                img = C.to_pil(result.image)
                C.save_image(img, out / f"{row['id']}.png")
                entry["file"] = f"{row['id']}.png"
                media = {"id": row["id"], "file": entry["file"], "width": img.size[0], "height": img.size[1],
                         "frames": None}
            rec["renders"].append(entry)
            rec["media"].append(media)
            say(f"{row['id']} {wall:.2f}s" + (f" step {req['step_s_polled']:.4f}s" if req.get("step_s_polled") else ""))
        short_steps = int(cell.get("short_steps") or 0)
        if short_steps and short_steps < steps:
            for j, row in enumerate(rows[: int(cell.get("short_n") or 0)]):
                seed = int(row["seed"]) + SAME_SEED_STRIDE * (j + 1)
                rec["long"].append(round(one("same", {**row, "seed": seed}, steps)[1], 3))
                rec["short"].append(round(one("same", {**row, "seed": seed + 1}, short_steps)[1], 3))
        t_end = time.time()
        rec["status_after"] = backend.status() or {}
        backend.last_status = rec["status_after"] or rec["status"]
        comp, src = _compile_s(backend)
        rec["timing"]["compile_s"], rec["timing"]["compile_s_source"] = comp, src
        new_med = C.stat_median([r["total_s"] for r in rec["requests"] if r["kind"] == "new"])
        if new_med is not None:
            rec["timing"]["compile_s_derived"] = round(rec["timing"]["first_render_s"] - new_med, 3)
        clocks = sampler.clocks(t_timed, t_end)
        if not clocks.get("samples"):  # a timed window shorter than the clock sampling period
            clocks = {**sampler.clocks(), "window": "whole cell"}
        rec["telemetry"].update({**clocks, "source": sampler.source})
        _engagement(cell, backend, rec)
        contradictions = _settings(cell, backend, rec)
        if parity:
            new = [r for r in rec["requests"] if r["kind"] == "new"]
            ev = (new[0].get("parity") if new else None) or {}
            resolved = _sampler_from_hooks(backend.resolved_settings() or _resolved_from_cell(cell, steps), ev)
            rec["parity"] = {**ev, "resolved": resolved,
                             "precision": backend.precision() or {}, "output": rec["media"][0] if rec["media"] else None,
                             "bline": new[0].get("bline") if new else None}
        problems = schema.engagement_problems(rec) + contradictions
        if problems:
            rec["verdict"], rec["reason"] = "VOID", "; ".join(problems)
        else:
            rec["verdict"] = "ok"
        rec["phase"] = "done"
    except Exception as exc:  # noqa: BLE001 - a broken cell is a result, recorded as such
        rec["verdict"] = "BROKEN"
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:2000]}"
        rec["reason"] = rec["error"][:500]
        rec["traceback"] = traceback.format_exc()[-4000:]
        say(f"BROKEN {rec['error'][:300]}")
    finally:
        if sampler is not None:
            sampler.stop()
            w = sampler.window()
            gib = (lambda v: None if v is None else round(v / 1024, 3))  # noqa: E731
            rec["memory"].update({"peak_vram_device_gib": gib(w["device_mib"]),
                                  "peak_vram_over_base_gib": gib(w["over_base_mib"]),
                                  "peak_vram_own_gib": gib(w["own_mib"]), "peak_rss_tree_gib": gib(w["rss_tree_mib"]),
                                  "peak_rss_server_gib": gib(w["rss_server_mib"]),
                                  "baseline_vram_gib": gib(sampler.baseline_mib)})
            rec["telemetry"]["source"] = sampler.source
            vram_source(rec, sampler, backend)
            rec["telemetry"]["n_samples"] = w["n"]
            rec["telemetry"]["timeline"] = sampler.timeline(1.0)
            # v1 fields, kept for older readers (score.py, matrix.py)
            rec["peak_smi_gib"] = rec["memory"]["peak_vram_device_gib"] or 0.0
            rec["smi_baseline_gib"] = rec["memory"]["baseline_vram_gib"] or 0.0
            rec["peak_smi_delta_gib"] = rec["memory"]["peak_vram_over_base_gib"] or 0.0
            rec["peak_own_gib"] = rec["memory"]["peak_vram_own_gib"] or 0.0
        rec["host_end"] = C.host_memory()
        try:
            rec["env"] = C.env_fingerprint(backend.trees() if backend else None)
        except Exception as exc:  # noqa: BLE001
            rec["env"] = {"error": str(exc)}
        _revisions(rec, backend)
        pin_problem = _studio_pin(cell, rec)
        if pin_problem and rec.get("verdict") == "ok":
            rec["verdict"], rec["reason"] = "VOID", pin_problem
        C.finalize_record(rec)
        if rec["verdict"] == "ok":
            bad = schema.validate(rec)
            if bad:
                rec["verdict"], rec["reason"] = "VOID", "schema: " + "; ".join(bad)
        C.write_record(out, rec)
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001
                say(f"close failed: {exc}")
        if ballast is not None:
            ballast.stop()
        _drop_fresh_cache(rec)  # a fresh cache is never reused and must not fill a small VM disk
    say(f"{rec['verdict']}: median {rec['timing'].get('new_s_median')}s/{'clip' if cell['kind'] == 'video' else 'image'}, "
        f"s/step polled {rec['timing'].get('step_s_polled_median')} derived {rec['timing'].get('step_s_derived')}, "
        f"peak own {rec['memory'].get('peak_vram_own_gib')} GiB -> {out / C.RECORD}")
    return rec


def _sampler_from_hooks(resolved: dict, ev: dict) -> dict:
    """A server stack whose client sees no scheduler class (Studio over HTTP) leaves sampler / scheduler unstated;
    the class the parity hook saw set the sigmas inside that server names them instead. A stated value is kept."""
    cls = ev.get("sigma_source") if ev.get("source") == "hooks" else None
    if not cls or (resolved.get("sampler") is not None and resolved.get("scheduler") is not None):
        return resolved
    from backends.studio_inproc import scheduler_names  # noqa: PLC0415

    sampler, sched, name = scheduler_names(str(cls))
    out = dict(resolved)
    for key, val in (("sampler", sampler), ("scheduler", sched)):
        if out.get(key) is None and val is not None:
            out[key] = val
            out[f"{key}_source"] = f"server scheduler class {name} (parity hook)"
    return out


def _resolved_from_cell(cell: dict, steps: int) -> dict:
    o = cell.get("options") or {}
    return {"steps": steps, "cfg": o.get("cfg", cell.get("guidance")), "sampler": o.get("sampler"),
            "scheduler": o.get("scheduler"), "shift": o.get("shift", o.get("flow_shift")), "width": cell.get("width"),
            "height": cell.get("height"), "frames": cell.get("frames"), "source": "cell (not read back from the stack)"}


def _studio_pin(cell: dict, rec: dict) -> str | None:
    """A Studio arm that names the commit it must run (options.expect_studio_sha, the rented VM's UNSLOTH_SHA) is
    VOID when the Studio that ran reports another (or none): the pair would time an unpinned Studio."""
    want = str((cell.get("options") or {}).get("expect_studio_sha") or "")
    if not want or cell.get("backend") not in ("studio", "studio_http"):
        return None
    got = str((rec.get("revisions") or {}).get("studio") or "")
    if len(got) >= 7 and (want.startswith(got) or got.startswith(want)):
        return None
    return f"Studio ran {got or 'an unreported revision'!s}, the VM pinned {want[:12]}"


def _revisions(rec: dict, backend) -> None:
    r = rec["revisions"]
    env = rec.get("env") or {}
    pk = env.get("packages") or {}
    st = rec.get("status") or {}
    facts = (st.get("facts") or {}) if isinstance(st, dict) else {}
    trees = env.get("trees") or {}
    r["studio"] = r.get("studio") or st.get("studio_rev") or trees.get("studio")
    # the git commit zoo was installed from (vm_setup pins it with UNSLOTH_ZOO_REF), else the bare version
    r["zoo"] = r.get("zoo") or (env.get("package_commits") or {}).get("unsloth_zoo") or pk.get("unsloth_zoo")
    r["blineui_sha"] = r.get("blineui_sha") or st.get("bline_commit") or trees.get("blineui")
    r["blineui_tag"] = r.get("blineui_tag") or st.get("bline_tag") or facts.get("blineui_version")
    r["custom_nodes"] = r.get("custom_nodes") or st.get("custom_nodes")
    r["torch"] = r.get("torch") or facts.get("torch") or pk.get("torch")
    r["cuda"] = r.get("cuda") or env.get("torch_cuda")
    r["triton"] = r.get("triton") or C._version("triton")


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", help = "cell JSON (merged over the built-in defaults)")
    ap.add_argument("--spec", help = "spec file; with --tag picks one of its cells")
    ap.add_argument("--tag")
    ap.add_argument("--set", action = "append", default = [], metavar = "KEY=JSON",
                    help = "override a cell field, e.g. --set n=2 --set options.speed_mode='\"off\"'")
    ap.add_argument("--parity", action = "store_true", help = "parity probe: one render with sigma / pass / noise hooks")
    ap.add_argument("--out", required = True)
    args = ap.parse_args()
    if args.spec:
        spec = C.load_spec(args.spec)
        cells = {c["tag"]: c for c in spec["cells"]}
        if args.tag not in cells:
            ap.error(f"--tag must be one of {sorted(cells)}")
        cell = C.merge_cell(spec["defaults"], cells[args.tag])
    elif args.cell:
        cell = C.merge_cell({}, json.loads(args.cell))
    else:
        ap.error("--cell or --spec/--tag is required")
    for item in args.set:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        target = cell
        *path, leaf = key.split(".")
        for part in path:
            target = target.setdefault(part, {})
        target[leaf] = value
    mem_block = B.maybe_reexec_scoped(B.spec(cell), [sys.executable, *sys.argv])
    rec = run(cell, Path(args.out), parity = args.parity, mem_block = mem_block)
    return 0 if rec["verdict"] == "ok" else (4 if rec["verdict"] == "NOT_RUN" else 3)


if __name__ == "__main__":
    raise SystemExit(main())
