#!/usr/bin/env python3
"""Run every cell of a spec, one process per cell, in the cell's own venv, and resume where it stopped.

  python diffusion_bench/matrix.py specs/smoke_small.json --out outputs/dbench/smoke
  python diffusion_bench/matrix.py SPEC --out DIR --only 'zimg_*' --pass quick --gpus 5,6 --dry-run
  python diffusion_bench/matrix.py --plan PLAN.json --vm local-b200-1 --out DIR [--dry-run]   # plan.py output

Spec layout (JSON):
  {
    "name": "...", "description": "...",
    "defaults": {<cell fields>},                    # merged under every cell
    "passes": [{"name": "p1", "n": 24}, {"name": "p2", "n": 8}],   # optional; each pass reruns every cell
                                                                   # with these overrides into <out>/<pass>/
    "gate": {"max_util": 10, "min_free_mib": 60000, "timeout_s": 1800},
    "reference": {"studio": "s_bf16", "blineui": "c_bf16"},        # scorer: reference cell per backend
    "cells": [{"tag": "...", "backend": "studio", "venv": "studio" | {"profile": ...} | path, ...}]
  }

Rules the driver enforces, all learned the hard way:
  - one cell per process, in spec order, so Studio / BlineUI cells can be interleaved and drift over the run
    hits every arm equally rather than whichever ran last;
  - a GPU gate before each cell (a neighbour's job on the same device makes the timing meaningless; a
    timed-out gate still runs, and the record carries gate=timeout);
  - with several --gpus, cells run in parallel, one per device; only compare timings from the same device;
  - a cell with an ok record is skipped unless --force, so an interrupted matrix resumes;
  - a per-cell timeout turns a hang into a BROKEN record rather than a stuck run;
  - a BROKEN cell whose load the auto planner put under an offload the spec did not ask for, because the shared
    device was nearly full, is VOID (pressure_void): it measured the other tenants, not the cell. Exit 75 when
    nothing is BROKEN but a cell is VOID, which studio_regress reads as a retryable harness VOID.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402
import envs as V  # noqa: E402
import schema  # noqa: E402



def is_trial_cell(cell: dict) -> bool:
    """A VM's trial: a trial__ cell, or (rented-VM plans) the lane's first parity probe marked arm.trial."""
    arm = (cell.get("options") or {}).get("arm") or {}
    return arm.get("purpose") == "trial" or bool(arm.get("trial")) or str(cell.get("tag", "")).startswith("trial__")

def plan(spec: dict, only: list, pass_names: list) -> list:
    passes = spec.get("passes") or [{"name": ""}]
    if pass_names:
        passes = [p for p in passes if p["name"] in pass_names]
    jobs = []
    for p in passes:
        overrides = {k: v for k, v in p.items() if k != "name"}
        for cell in spec["cells"]:
            if only and not any(fnmatch.fnmatch(cell["tag"], pat) for pat in only):
                continue
            if cell.get("skip"):
                continue
            merged = C.merge_cell(spec["defaults"], {**cell, **overrides})
            jobs.append((p["name"], merged))
    return jobs


# memory_mode values that ask for an offload: a cell requesting one wanted the plan it got. auto / fast / unset ask
# for resident (auto offloads only when the free memory says so).
OFFLOAD_MODES = {"low_vram", "balanced", "sequential", "streaming", "model", "group"}
# The plan's offload as Studio logs it: the int8 placement line (logged before a quant failure) and diffusion.loaded.
PLAN_OFFLOAD_RE = re.compile(r"under the plan's '(\w+)' offload|diffusion\.loaded: .*?\boffload=(\w+)")
# "Nearly full": under this much free before the load, an offload the spec did not request is the planner reacting
# to other tenants (an idle card leaves a tiny cell resident under auto).
PRESSURE_FREE_MIB, PRESSURE_FREE_FRAC = 16384, 0.10
# Load errors a pressure-picked plan causes (a quant the offload cannot carry, an OOM); any other error stays BROKEN.
PRESSURE_ERROR_RE = re.compile(r"memory plan picked|offload|out of memory|OutOfMemory|Not enough (?:GPU|VRAM|system)",
                               re.IGNORECASE)
EXIT_HARNESS_VOID = 75


def run_log_text(log_path: Path, start: int) -> str:
    """The part of an appended log written after byte offset `start`."""
    if not log_path.exists():
        return ""
    with open(log_path, "rb") as f:
        f.seek(start)
        return f.read().decode(errors = "replace")


def pressure_void(rec: dict, log_text: str) -> str | None:
    """The VOID reason for a BROKEN cell whose load ran under memory pressure the spec did not ask for, else None.
    Only a failure during the load counts (run_cell's phase) whose error a pressure plan explains
    (PRESSURE_ERROR_RE); a render / save failure or any other load error stays BROKEN. Then all three must hold: the cell's memory_mode does not request an offload, the log shows the plan offloaded
    (policy other than none), and the device had under max(PRESSURE_FREE_MIB, PRESSURE_FREE_FRAC of total) free
    before the load (run_cell's gpu_before_load, else the gate reading). A BROKEN cell on a card with room, or one
    whose spec asked for the offload, stays BROKEN."""
    if (rec or {}).get("verdict") != "BROKEN" or rec.get("phase") != "load":
        return None
    if not PRESSURE_ERROR_RE.search(str(rec.get("error") or "")):
        return None
    mode = ((rec.get("cell") or {}).get("options") or {}).get("memory_mode") or "auto"
    if mode in OFFLOAD_MODES:
        return None
    policies = [a or b for a, b in PLAN_OFFLOAD_RE.findall(log_text or "")]
    offloaded = [p for p in policies if p and p != "none"]
    if not offloaded:
        return None
    q = rec.get("gpu_before_load") or {}
    gate = rec.get("gate") or {}
    used = q.get("used_mib", gate.get("used_mib"))
    total = q.get("total_mib", gate.get("total_mib"))
    if used is None or not total:
        return None
    free = int(total) - int(used)
    if free >= max(PRESSURE_FREE_MIB, PRESSURE_FREE_FRAC * int(total)):
        return None
    return (f"load planned under memory pressure the spec did not ask for: plan offload '{offloaded[-1]}' with "
            f"memory_mode '{mode}' requested, {free} MiB free of {total} MiB before the load")


def run_job(pass_name: str, cell: dict, out_root: Path, gpu: str, gate: dict, timeout_s: float, force: bool,
            dry: bool) -> str:
    out = out_root / pass_name if pass_name else out_root
    cell_dir = out / cell["tag"]
    if not force and C.record_ok(C.read_record(cell_dir)):
        C.log(f"skip {pass_name or '-'} / {cell['tag']}: ok record exists")
        return "skipped"
    need = cell.get("min_compute_cap")
    if need and not dry:
        # fp8 / nvfp4 cells on an older card: Studio refusing the scheme there is correct, not a broken cell.
        # common.capability_gate when present: an NVIDIA capability means nothing on AMD / Apple / CPU (gfx1151
        # reports (11, 5) through torch), so such a cell is unsupported there instead of "unknown, run it".
        if hasattr(C, "capability_gate"):
            ok, why = C.capability_gate(need, device = gpu or None)
        else:  # pre-merge common.py
            have = C.gpu_compute_cap(gpu or None)
            ok = not (have is not None and have < tuple(int(x) for x in str(need).split(".")))
            why = None if ok else f"needs compute capability {need}, gpu {gpu} is {have[0]}.{have[1]}"
        if not ok:
            # the same NOT_RUN record run_cell writes for an unmet capability: the case stays accounted for
            rec = schema.new_record(cell)
            rec.update({"verdict": "NOT_RUN", "reason": why or f"needs compute capability {need}",
                        "backend": cell.get("backend"), "phase": "gate"})
            C.write_record(cell_dir, rec)
            C.log(f"skip {pass_name or '-'} / {cell['tag']}: NOT_RUN ({why})")
            return "skipped"
    python = V.resolve_python(cell.get("venv"), **(cell.get("venv_args") or {})) if not dry else str(cell.get("venv"))
    env = {**os.environ}
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["HIP_VISIBLE_DEVICES"] = gpu
    cmd = [python, "-u", str(HERE / "run_cell.py"), "--cell", json.dumps(cell), "--out", str(out)]
    if dry:
        C.log(f"[dry] gpu={gpu or '-'} {pass_name or '-'} / {cell['tag']} ({cell['backend']}) python={python}")
        return "dry"
    g = None
    if gate is not False and C.gpu_vendor() != "none":
        g = C.gpu_gate(device = gpu or None, **(gate or {}))
        C.log(f"gate {gpu or C.visible_gpu()}: {g.get('gate')} util={g.get('util')} used={g.get('used_mib')} MiB")
    cell_dir.mkdir(parents = True, exist_ok = True)
    log_path = cell_dir / "cell.log"
    # cell.log is appended across reruns (--force): judge only what this run wrote. The previous attempt's record
    # goes too, so a run that dies before its final write is never read with a stale cell / gpu_before_load.
    log_start = log_path.stat().st_size if log_path.exists() else 0
    (cell_dir / C.RECORD).unlink(missing_ok = True)
    C.log(f"=== {pass_name or '-'} / {cell['tag']} on gpu {gpu or C.visible_gpu()} -> {log_path}")
    with open(log_path, "a") as logf:
        proc = subprocess.Popen(cmd, stdout = logf, stderr = subprocess.STDOUT, env = env, cwd = str(C.WS),
                                start_new_session = True)
        try:
            code = proc.wait(timeout = timeout_s)
        except subprocess.TimeoutExpired:
            if hasattr(os, "killpg"):
                import signal

                os.killpg(proc.pid, signal.SIGKILL)
            else:  # Windows (the AMD Windows runners): no process groups via setsid, no SIGKILL; kill the tree
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output = True)
                proc.kill()
            proc.wait()
            rec = C.read_record(cell_dir) or {"tag": cell["tag"], "cell": cell}
            rec.update({"verdict": "BROKEN", "error": f"timeout after {timeout_s}s"})
            C.write_record(cell_dir, rec)
            code = 124
    rec = C.read_record(cell_dir) or {}
    if rec and g is not None:
        # merge, never replace: run_cell's own gate block (cooldown, foreign-VRAM wait) stays; matrix's reading
        # goes under "matrix"
        prev = rec.get("gate")
        rec["gate"] = {**(prev if isinstance(prev, dict) else {}), "matrix": g}
        C.write_record(cell_dir, rec)
    why = pressure_void(rec, run_log_text(log_path, log_start))
    if why:
        rec.update({"verdict": "VOID", "void_reason": why, "broken_error": rec.get("error")})
        C.write_record(cell_dir, rec)
        C.log(f"--- {cell['tag']}: BROKEN -> VOID ({why})")
    C.log(f"--- {cell['tag']}: exit {code}, {rec.get('verdict')}, median {rec.get('wall_s_median')}s, "
          f"s/step {rec.get('step_s_derived')}, peak {rec.get('peak_smi_gib')} GiB")
    return rec.get("verdict") or f"exit{code}"


def set_value(sets: list, key: str):
    """The last --set KEY=... value (JSON-decoded when it parses), else None."""
    out = None
    for item in sets:
        k, _, raw = item.partition("=")
        if k == key:
            try:
                out = json.loads(raw)
            except json.JSONDecodeError:
                out = raw
    return out


def plan_spec(plan_path: Path, vm_id: str) -> dict:
    """A spec holding one plan VM's cells in slot order (plan.py output; options.target / options.arm already
    set). Only single-lane VMs: a two-lane Kaggle VM needs targets/vm_driver.py, which keeps lane pairs in step."""
    plan = json.loads(Path(plan_path).read_text(encoding = "utf-8"))
    vm = next((v for v in plan.get("vms") or [] if v["vm_id"] == vm_id), None)
    if vm is None:
        raise SystemExit(f"--vm {vm_id!r} is not in {plan_path}: {[v['vm_id'] for v in plan.get('vms') or []]}")
    if int(vm.get("lanes") or 1) > 1 or any(len(slot) > 1 for slot in vm["slots"]):
        raise SystemExit(f"{vm_id} is a two-lane VM (lane GPUs, a trial per lane); use diffusion_bench/targets/vm_driver.py")
    cells = {c["tag"]: c for c in plan["cells"]}
    order = [it["tag"] for slot in vm["slots"] for it in slot]
    out = []
    for tag in order:
        c = {k: v for k, v in cells[tag].items() if k != "est_min"}
        out.append(c)
    return {"name": f"{plan.get('name', 'plan')}:{vm_id}", "description": f"plan.py VM {vm_id} ({vm['target']})",
            "defaults": {}, "cells": out, "plan_vm": {k: vm[k] for k in ("vm_id", "target", "purpose")},
            **C.plan_carry(plan, vm)}  # downloads, Studio prefetch (fetch_files.py), comparisons, revisions


def ensure_derived(spec_path: Path, sets: list) -> list:
    """Build the spec's derived tiny pipes (fetch_tiny.DERIVED) that are not on disk yet, with the cell venv given
    by --set venv=<python> (a Studio install: it has diffusers) as the build interpreter, before load_spec resolves
    the aliases. A fresh host or a Colab / Kaggle VM has none of them, and every cell would fail on both sides."""
    try:
        models = json.loads(Path(spec_path).read_text(encoding = "utf-8")).get("models") or {}
    except (OSError, ValueError):
        return []
    if not any((m or {}).get("derived") for m in models.values()):
        return []
    venv = set_value(sets, "venv")
    python = str(C.expand_env(venv)) if isinstance(venv, str) and Path(C.expand_env(venv)).is_file() else None
    import fetch_tiny  # noqa: E402 - only when a spec has derived pipes
    errors = fetch_tiny.ensure_aliases(models, python = python, log = C.log)
    for e in errors:
        C.log(f"derived tiny pipe not built: {e}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", nargs = "?", help = "spec file (or --plan + --vm)")
    ap.add_argument("--plan", help = "plan.py output: run one VM's cells in plan order")
    ap.add_argument("--vm", help = "with --plan: the vm_id to run")
    ap.add_argument("--out", required = True)
    ap.add_argument("--only", action = "append", default = [], help = "fnmatch pattern on cell tags (repeatable)")
    ap.add_argument("--pass", dest = "passes", action = "append", default = [], help = "run only these passes")
    ap.add_argument("--gpus", default = "",
                    help = "comma list of devices, one worker each (default: inherit the current mask, one worker)")
    ap.add_argument("--timeout", type = float, default = 5400, help = "per cell, seconds")
    ap.add_argument("--no-gate", action = "store_true")
    ap.add_argument("--force", action = "store_true", help = "rerun cells that already have an ok record")
    ap.add_argument("--dry-run", action = "store_true")
    ap.add_argument("--set", action = "append", default = [], metavar = "KEY=JSON",
                    help = "override a default for every cell, e.g. --set n=2 --set warmup=0")
    args = ap.parse_args()

    if args.plan:
        if not args.vm:
            ap.error("--plan needs --vm")
        spec = plan_spec(Path(args.plan), args.vm)
    else:
        if not args.spec:
            ap.error("a spec, or --plan PLAN --vm ID")
        spec_path = Path(args.spec)
        if not spec_path.exists():
            spec_path = HERE / "specs" / args.spec
        ensure_derived(spec_path, args.set)
        spec = C.load_spec(spec_path)
    for item in args.set:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        # Dotted keys reach into the defaults (options.speed_mode=..., env.X=...), same as run_cell.py --set;
        # setting "options" whole would drop every other default option.
        target = spec["defaults"]
        *path, leaf = key.split(".")
        for part in path:
            target = target.setdefault(part, {})
        target[leaf] = value
    jobs = plan(spec, args.only, args.passes)
    out_root = Path(C.expand_env(args.out))
    out_root.mkdir(parents = True, exist_ok = True)
    (out_root / "spec.json").write_text(json.dumps(spec, indent = 1))
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if args.plan and len(gpus) > 1 and (spec.get("plan_vm") or {}).get("purpose") == "timing":
        # one worker per GPU would run S and C side by side and break the ABBA / parity-before-timing order
        raise SystemExit(f"--plan runs a timed VM in slot order on ONE device; --gpus {args.gpus} refused "
                         "(run one VM per device instead)")
    gate = False if args.no_gate else spec.get("gate", {})
    C.log(f"{spec['name']}: {len(jobs)} cell runs on gpus {gpus or ['(current)']} -> {out_root}")
    results: dict = {}
    work: queue.Queue = queue.Queue()
    for job in jobs:
        work.put(job)

    def worker(gpu: str) -> None:
        while True:
            try:
                pass_name, cell = work.get_nowait()
            except queue.Empty:
                return
            try:
                results[(pass_name, cell["tag"])] = runner(pass_name, cell, out_root, gpu, gate, args.timeout,
                                                           args.force, args.dry_run)
            except Exception as exc:  # noqa: BLE001 - one broken cell never stops the matrix
                C.log(f"{cell['tag']}: driver error {type(exc).__name__}: {exc}")
                results[(pass_name, cell["tag"])] = "driver_error"

    runner = run_job
    # the gate follows the DOCUMENT, not the flag: a plan flattened into a spec (AMD, Mac, tester) carries the same
    # comparisons and trial cells, and gets the same parity gate and trial stop
    gate_doc = {"comparisons": spec.get("comparisons") or []}
    has_trial = any(is_trial_cell(c) for c in spec.get("cells") or [])
    if (args.plan or gate_doc["comparisons"] or has_trial) and not args.dry_run:
        import parity as PG

        # pairs already complete under --out (a resumed run, or one cell per call as the tester runner does); a
        # pair whose probe records are missing blocks its timed cells too (parity.blocked_groups)
        void: list = PG.blocked_groups(PG.gate_plan(gate_doc, out_root)) if gate_doc["comparisons"] else []
        lock = threading.Lock()
        stopped: list = []  # the VM rule: a failed trial cell stops every later cell (NOT_RUN with the reason)
        for c in spec.get("cells") or []:  # one cell per process (AMD probe, tester): a failed trial already on disk
            arm = (c.get("options") or {}).get("arm") or {}
            if is_trial_cell(c):
                prev = C.read_record(out_root / c["tag"])
                if prev and prev.get("verdict") not in ("ok", None):
                    stopped.append(f"trial {c['tag']} not ok ({prev.get('verdict')}): VM stopped before any timed cell")

        def not_run(cell, why):
            rec = schema.new_record(cell)
            rec.update({"verdict": "NOT_RUN", "reason": why, "backend": cell.get("backend")})
            C.write_record(out_root / cell["tag"], rec)
            C.log(f"--- {cell['tag']}: NOT_RUN ({why})")
            return "NOT_RUN"

        def run_gated(pass_name, cell, out_root, gpu, gate, timeout_s, force, dry):
            # plan mode: the trial runs first and gates the VM; the parity probes next (plan.py slots them after the
            # trial); after each one the gate runs on the pairs that are complete, and a VOID group's timed cells
            # become NOT_RUN records
            arm = (cell.get("options") or {}).get("arm") or {}
            is_trial = is_trial_cell(cell)
            if stopped and not is_trial:
                return not_run(cell, stopped[0])
            why = PG.gated_out(gate_doc, cell, void)
            if why:
                return not_run(cell, why)
            v = run_job(pass_name, cell, out_root, gpu, gate, timeout_s, force, dry)
            if is_trial and v not in ("ok", "skipped"):
                stopped.append(f"trial {cell['tag']} not ok ({v}): VM stopped before any timed cell")
                if not dry and not C.read_record(out_root / cell["tag"]):
                    # a trial that died before its record: leave one, so a later one-cell-per-process call (tester
                    # runner, AMD probe) reads the stop from disk instead of running the timed cells
                    rec = schema.new_record(cell)
                    rec.update({"verdict": "BROKEN", "reason": f"trial wrote no record ({v})",
                                "backend": cell.get("backend")})
                    C.write_record(out_root / cell["tag"], rec)
                C.log("TRIAL_FAILED " + stopped[-1])
            elif is_trial:  # re-run ok: an older failure of this trial no longer stops the VM
                stopped[:] = [x for x in stopped if not x.startswith(f"trial {cell['tag']} ")]
            if gate_doc["comparisons"] and (cell.get("options") or {}).get("parity_probe"):
                with lock:
                    g = PG.gate_plan(gate_doc, out_root)
                    void[:] = PG.blocked_groups(g)
                    for pr in g["pairs"]:
                        C.log(f"parity {pr['verdict']} {pr['comparison']} / {pr['format']}"
                              + (f": {'; '.join(pr['reasons'])[:300]}" if pr["reasons"] else ""))
            return v

        runner = run_gated
    threads = [threading.Thread(target = worker, args = (g,)) for g in (gpus or [""])]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    summary = {f"{p}/{t}" if p else t: v for (p, t), v in results.items()}
    (out_root / "matrix_summary.json").write_text(json.dumps({"elapsed_s": round(time.time() - t0, 1),
                                                              "results": summary}, indent = 1))
    bad = {k: v for k, v in summary.items() if v not in ("ok", "skipped", "dry")}
    C.log(f"done in {time.time() - t0:.0f}s: {len(summary) - len(bad)} ok/skipped, {len(bad)} not ok {bad or ''}")
    if bad and all(v == "VOID" for v in bad.values()):
        return EXIT_HARNESS_VOID
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
