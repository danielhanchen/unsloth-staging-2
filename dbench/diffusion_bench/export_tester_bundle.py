#!/usr/bin/env python3
"""Build the self-contained zip an outside tester runs on their own cards (3090 / Ada): the canonical toolkit at a
git SHA (run_cell, backends, BlineUI graphs, model cards, schema), their plan slices, their hardware knobs, a runner
and a README with one command per plan.

  python diffusion_bench/export_tester_bundle.py PLAN.json --out OUT.zip \\
      --gpu-labels '0=Ada 48GB,1=RTX 3090' [--start-temp-c 60] [--start-temp-rel-c 3] [--start-temp-c-per-gpu '0=68'] \\
      [--foreign-vram-mib '0=2400,1=500'] [--budgets 8,12,16,24] [--allow-dirty]

PLAN.json is plan.py output with "tester:<label>" targets (one VM each); <label> must be one of --gpu-labels'
values, which is how a plan VM finds its GPU index on the tester's box. --budgets only documents / checks the
simulated budgets the plan's budget arms use (the cells carry options.target.simulated_budget_gib).

Zip layout: diffusion_bench/ (git archive at scripts_sha), plans/<vm>.json, tester.json, run_tester.py, README.md,
MANIFEST.json (scripts_sha, sha256 per file, what the toolkit lacked at export time).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from targets import common as TC  # noqa: E402

RUNNER = r'''#!/usr/bin/env python3
"""Run one plan of this bundle on this machine, one cell per process, in plan order, with the gates the tester's
own harness used before every cell: foreign VRAM on the card under the limit for 30 s, and the card at or below the
start temperature.

  python run_tester.py plans/<vm>.json --studio-py PY --studio-src UNSLOTH_DIR --bline-py PY --bline-dir BLINE_DIR
      [--out results/<vm>] [--dry-run]
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def smi(gpu, field):
    out = subprocess.run(["nvidia-smi", "-i", str(gpu), f"--query-gpu={field}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out.splitlines()[0])
    except (ValueError, IndexError):
        return None


def gate_limit(start_temp_c, per_gpu, gpu, idle_c, rel_c, cap_c=10.0):
    """The start temperature (C) a card must reach before a cell. A per-card override wins; else the absolute limit,
    raised to the card's own idle baseline + rel_c (capped at limit + cap_c) when it idles warmer: an Ada beside a
    busy RTX 3090 idled at 63-65 C, so a fixed 60 C gate never opened. None = no temperature gate."""
    if str(gpu) in (per_gpu or {}):
        return float(per_gpu[str(gpu)]) or None
    if not start_temp_c:
        return None
    lim = float(start_temp_c)
    if rel_c and idle_c is not None:
        lim = max(lim, min(float(idle_c) + float(rel_c), lim + cap_c))
    return lim


def idle_baseline(gpu, max_foreign_mib, n=3):
    """Median temperature over n readings 2 s apart while no foreign VRAM sits on the card, before any cell of
    this run; None when the card is busy or nvidia-smi gives no reading."""
    temps = []
    for _ in range(n):
        used, temp, util = smi(gpu, "memory.used"), smi(gpu, "temperature.gpu"), smi(gpu, "utilization.gpu")
        if temp is not None and (used is None or used <= max_foreign_mib) and (util is None or util <= 5):
            temps.append(temp)
        time.sleep(2)
    return sorted(temps)[len(temps) // 2] if len(temps) == n else None


def wait_gates(gpu, max_foreign_mib, start_temp_c, timeout_s=3600):
    t0, quiet = time.time(), 0
    while time.time() - t0 < timeout_s:
        used, temp = smi(gpu, "memory.used"), smi(gpu, "temperature.gpu")
        ok = (used is None or used <= max_foreign_mib) and (temp is None or not start_temp_c or temp <= start_temp_c)
        quiet = quiet + 1 if ok else 0
        if quiet >= 6:
            return {"used_mib": used, "temp_c": temp, "waited_s": round(time.time() - t0)}
        print(f"gate gpu{gpu}: used {used} MiB (limit {max_foreign_mib}), {temp} C (limit {start_temp_c}); waiting", flush=True)
        time.sleep(5)
    return {"timeout": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan")
    ap.add_argument("--studio-py", required=True)
    ap.add_argument("--studio-src", required=True, help="unsloth checkout at the plan's unsloth_sha")
    ap.add_argument("--bline-py", required=True)
    ap.add_argument("--bline-dir", required=True)
    ap.add_argument("--models", default=str(HERE / "models_cache"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--start-temp-c", type=float, default=None,
                    help="this card's start temperature, overriding tester.json (0 = no temperature gate)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    knobs = json.loads((HERE / "tester.json").read_text())
    plan = json.loads(Path(a.plan).read_text())
    vm = plan["vms"][0]
    label = vm["target"].split(":", 1)[1]
    gpu = knobs["label_to_gpu"][label]
    out = Path(a.out or HERE / "results" / vm["vm_id"])
    out.mkdir(parents=True, exist_ok=True)
    local = out / "plan_local.json"
    subprocess.run([a.studio_py, str(HERE / "diffusion_bench" / "targets" / "fetch_files.py"), "--spec", a.plan,
                    "--root", a.models, "--write", str(local), "--venv", a.studio_py, "--studio-src", a.studio_src,
                    "--baseline-python", a.bline_py, "--baseline-dir", a.bline_dir]
                   + (["--dry-run"] if a.dry_run else []), check=True)
    # the Studio python / checkout also go to Studio-over-HTTP cells, whose server setup_studio.launch() starts from
    # these (otherwise it builds its own venv from Studio main and times that instead of the plan's revision)
    env = {**os.environ, "WORKSPACE": str(HERE), "DIFFUSION_BENCH_BLINE_PYTHON": a.bline_py,
           "DIFFUSION_BENCH_BLINE_DIR": a.bline_dir, "DIFFUSION_BENCH_STUDIO_PYTHON": a.studio_py,
           "DIFFUSION_BENCH_STUDIO_SRC": a.studio_src, "DBENCH_GPU_LABEL": knobs["gpu_labels"][str(gpu)],
           "DBENCH_START_TEMP_C": str(knobs.get("start_temp_c") or ""), "DBENCH_SCRIPTS_SHA": knobs["scripts_sha"]}
    order = [it["tag"] for slot in vm["slots"] for it in slot]
    lim = knobs["foreign_vram_mib"].get(str(gpu), 1024)
    per_gpu = dict(knobs.get("start_temp_c_per_gpu") or {})
    if a.start_temp_c is not None:
        per_gpu[str(gpu)] = a.start_temp_c
    idle = None if (a.dry_run or str(gpu) in per_gpu) else idle_baseline(gpu, lim)
    temp_c = gate_limit(knobs.get("start_temp_c"), per_gpu, gpu, idle, knobs.get("start_temp_rel_c"))
    print(f"start temperature gate gpu{gpu}: {temp_c} C (absolute {knobs.get('start_temp_c')}, idle baseline {idle}, "
          f"+{knobs.get('start_temp_rel_c')} C, per-card {per_gpu.get(str(gpu))})", flush=True)
    env["DBENCH_START_TEMP_C"] = str(temp_c or "")
    stopped = None
    for tag in order:
        # after a failed trial matrix.py writes each remaining cell's NOT_RUN record (it reads the trial record
        # from --out); no GPU gate wait for those
        g = {} if (a.dry_run or stopped) else wait_gates(gpu, lim, temp_c)
        print(f"=== {tag} on gpu{gpu} ({label}) gate {g}", flush=True)
        cmd = [a.studio_py, str(HERE / "diffusion_bench" / "matrix.py"), "--plan", str(local), "--vm", vm["vm_id"],
               "--only", tag, "--gpus", str(gpu), "--out", str(out)] + (["--dry-run"] if a.dry_run else [])
        subprocess.run(cmd, env=env)
        if tag.startswith("trial__") and not a.dry_run:
            try:
                verdict = json.loads((out / tag / "record.json").read_text()).get("verdict")
            except (OSError, ValueError):
                verdict = None
            if verdict != "ok":
                stopped = f"trial {tag} not ok ({verdict}): VM stopped"
                print("TRIAL_FAILED", stopped, flush=True)
    if not a.dry_run:  # the gate result for the report (matrix already applied it between cells)
        subprocess.run([a.studio_py, str(HERE / "diffusion_bench" / "parity.py"), "--plan", str(local), "--results",
                        str(out)], env=env)
    print("TESTER_PLAN_DONE", vm["vm_id"], "->", out)


if __name__ == "__main__":
    main()
'''


def parse_map(s: str | None) -> dict:
    if not s:
        return {}
    return {k.strip(): v.strip() for k, _, v in (p.partition("=") for p in s.split(",")) if k.strip()}


def readme(meta: dict, vms: list) -> str:
    lines = [f"# diffusion_bench tester bundle ({meta['plan_name']})", "",
             f"Toolkit: unslothai scripts `{meta['scripts_sha']}`{' (dirty working tree)' if meta['dirty'] else ''}. "
             f"Unsloth Studio: `{meta['revisions'].get('unsloth_sha')}` (unsloth-zoo `{meta['revisions'].get('zoo_sha')}`). BlineUI: `{meta['revisions'].get('bline_ref')}`.", "",
             "## Your cards", ""]
    for idx, label in sorted(meta["gpu_labels"].items()):
        lines.append(f"- GPU {idx}: {label} (foreign VRAM limit before each cell: {meta['foreign_vram_mib'].get(idx, 1024)} MiB)")
    lines += [(f"- Start temperature gate: {meta['start_temp_c']} C, raised to the card's idle baseline + "
               f"{meta.get('start_temp_rel_c') or 0} C (measured before the first cell) when it idles warmer"
               if meta["start_temp_c"] else "- No temperature gate"),
              *(f"- GPU {k}: start temperature {v} C (per-card override)"
                for k, v in sorted((meta.get("start_temp_c_per_gpu") or {}).items())),
              "- A card next to a busy one may idle above the gate: `python run_tester.py ... --start-temp-c 68` sets "
              "this run's limit by hand",
              f"- Simulated VRAM budgets in this bundle: {meta['budgets_gib'] or 'none'} GiB (labelled simulated in every record)",
              "", "## Setup (once)", "",
              "Use the Studio and BlineUI installs you already have, at the revisions above, or build both with:", "",
              "```bash",
              "ROOT=$HOME/dbench BENCH=$PWD UNSLOTH_SHA=<unsloth_sha above> ZOO_SHA=<unsloth-zoo sha above> BLINE_REF=<bline ref above> \\",
              "  bash diffusion_bench/targets/vm_setup.sh",
              "# then: studio python = $(cat $HOME/dbench/studio_py.txt), bline python = $(cat $HOME/dbench/bline_py.txt)",
              "#       studio src = $HOME/dbench/src/unsloth, bline dir = $HOME/dbench/src/BlineUI",
              "```", "", "## One command per plan", ""]
    for vm in vms:
        lines += [f"### {vm['vm_id']} ({vm['target']}, {sum(len(s) for s in vm['slots'])} cells, ~{vm['est_min']:.0f} min)", "",
                  "```bash",
                  f"python run_tester.py plans/{vm['vm_id']}.json --studio-py $STUDIO_PY --studio-src $STUDIO_SRC \\",
                  "  --bline-py $BLINE_PY --bline-dir $BLINE_DIR",
                  "```", ""]
    lines += ["Each cell is a fresh process; cells run in the plan's order (Studio and BlineUI interleaved S C C S),",
              "after the card has had no foreign VRAM for 30 s and is at or below the start temperature.",
              "Add `--dry-run` to list without running. The parity probes run first: if the two stacks do not",
              "compute the same thing (settings, sigmas, passes, size, precision), the timed cells of that model are",
              "recorded NOT_RUN instead of being timed.", "",
              "Optional, on your side: `python diffusion_bench/report.py results/<vm> --expect cases.json --no-lpips`.", "",
              "## Send back", "",
              "`results/<vm>/` (every `record.json`, `parity.json`, `cell.log`, the BlineUI server logs). Images can stay",
              "local.", ""]
    return "\n".join(lines)


def kit_files(sha: str | None, dirty: bool) -> dict:
    """studio_test_kit as {arcname: bytes}: targets.common.kit_files (one helper for every bundle builder)."""
    return TC.kit_files(sha, dirty)


def build(plan: dict, out_zip: Path, gpu_labels: dict, start_temp_c, foreign: dict, budgets: list,
          allow_dirty: bool = False, start_temp_rel_c: float | None = 3.0, start_temp_per_gpu: dict | None = None) -> dict:
    vms = [v for v in plan["vms"] if v["target"].startswith("tester:")]
    if not vms:
        raise SystemExit("plan has no tester:<label> VM; add a \"tester:<label>\" target to the request")
    if not gpu_labels:
        raise SystemExit("--gpu-labels is required (GPU index -> card label)")
    label_to_gpu = {v: k for k, v in gpu_labels.items()}
    for vm in vms:
        lab = vm["target"].split(":", 1)[1]
        if lab not in label_to_gpu:
            raise SystemExit(f"{vm['vm_id']}: label {lab!r} is not in --gpu-labels {sorted(label_to_gpu)}")
        if any(len(s) > 1 for s in vm["slots"]):
            raise SystemExit(f"{vm['vm_id']}: one card per plan VM on the tester's box")
    plan_budgets = sorted({c["options"]["target"].get("simulated_budget_gib") for c in plan["cells"]
                           if c["options"]["target"]["name"].startswith("tester:")} - {None})
    if budgets and set(plan_budgets) - set(budgets):
        raise SystemExit(f"plan uses budgets {plan_budgets} not all in --budgets {budgets}")
    stage = out_zip.parent / (out_zip.stem + "_stage")
    bundle = TC.build_bundle(stage, allow_dirty = allow_dirty)
    tk = Path(bundle["path"]).read_bytes()
    missing = []
    with tarfile.open(fileobj = io.BytesIO(tk), mode = "r:gz") as tf:
        names = set(tf.getnames())
    for need in ("diffusion_bench/schema.py", "diffusion_bench/models/__init__.py", "diffusion_bench/run_cell.py",
                 "diffusion_bench/bline_graphs/__init__.py", "diffusion_bench/builders/__init__.py",
                 "diffusion_bench/parity.py"):
        if need not in names:
            missing.append(need)
    rev = dict(plan.get("revisions") or {})
    meta = {"plan_name": plan["name"], "scripts_sha": bundle["scripts_sha"], "dirty": bundle["dirty"],
            "revisions": rev, "gpu_labels": gpu_labels, "label_to_gpu": label_to_gpu, "start_temp_c": start_temp_c,
            "start_temp_rel_c": start_temp_rel_c, "start_temp_c_per_gpu": {k: float(v) for k, v in
                                                                          (start_temp_per_gpu or {}).items()},
            "foreign_vram_mib": foreign, "budgets_gib": plan_budgets, "exported": time.strftime("%FT%TZ", time.gmtime()),
            "toolkit_missing": missing}
    files: dict = {}
    with tarfile.open(fileobj = io.BytesIO(tk), mode = "r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile():
                files[m.name] = tf.extractfile(m).read()
    vm_ids = {vm["vm_id"] for vm in vms}
    if TC.needs_kit(c for c in plan["cells"] if (c.get("options") or {}).get("target", {}).get("vm_id") in vm_ids):
        kit = kit_files(bundle["scripts_sha"], bundle["dirty"])
        if not kit:
            missing.append("studio_test_kit (Studio-over-HTTP cells launch their server through it)")
        files.update(kit)
    for vm in vms:
        sl = TC.vm_slice(plan, vm["vm_id"])
        mini = {"name": plan["name"], **TC.C.plan_carry(plan, vm), "vms": [vm], "cells": list(sl["cells"].values())}
        mini["revisions"] = {**rev, "scripts_sha": bundle["scripts_sha"]}
        files[f"plans/{vm['vm_id']}.json"] = json.dumps(mini, indent = 1).encode()
    # report.py's expected case set for the tester's VMs (report.py results/<vm> --expect cases.json)
    import plan as PL

    doc = PL.cases_doc(plan)
    doc["cases"] = [c for c in doc["cases"] if c["vm_id"] in vm_ids]
    doc["comparisons"] = [c for c in doc["comparisons"] if c["vm_id"] in vm_ids]
    files["cases.json"] = json.dumps(doc, indent = 1).encode()
    files["tester.json"] = json.dumps(meta, indent = 1).encode()
    files["run_tester.py"] = RUNNER.encode()
    files["README.md"] = readme(meta, vms).encode()
    manifest = {**meta, "files": {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())}}
    files["MANIFEST.json"] = json.dumps(manifest, indent = 1).encode()
    out_zip.parent.mkdir(parents = True, exist_ok = True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in sorted(files.items()):
            zi = zipfile.ZipInfo(name, date_time = (2026, 1, 1, 0, 0, 0))
            zi.external_attr = (0o755 if name.endswith((".py", ".sh")) else 0o644) << 16
            zf.writestr(zi, data, zipfile.ZIP_DEFLATED)
    return {"zip": str(out_zip), "bytes": out_zip.stat().st_size, "vms": [v["vm_id"] for v in vms],
            "scripts_sha": bundle["scripts_sha"], "dirty": bundle["dirty"], "toolkit_missing": missing,
            "bundle_tgz": bundle["path"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan")
    ap.add_argument("--out", required = True)
    ap.add_argument("--gpu-labels", required = True, help = "'0=Ada 48GB,1=RTX 3090'")
    ap.add_argument("--start-temp-c", type = float, default = 60.0, help = "0 = no temperature gate")
    ap.add_argument("--start-temp-rel-c", type = float, default = 3.0,
                    help = "raise the gate to the card's idle baseline + this when it idles warmer (0 = absolute only)")
    ap.add_argument("--start-temp-c-per-gpu", default = "", help = "per GPU index, e.g. '0=68' (wins over both)")
    ap.add_argument("--foreign-vram-mib", default = "", help = "per GPU index, e.g. '0=2400,1=500' (default 1024)")
    ap.add_argument("--budgets", default = "", help = "simulated budgets (GiB) the plan may use, e.g. 8,12,16,24")
    ap.add_argument("--allow-dirty", action = "store_true")
    a = ap.parse_args(argv)
    plan = json.loads(Path(a.plan).read_text())
    res = build(plan, Path(a.out), parse_map(a.gpu_labels), a.start_temp_c or None,
                {k: int(v) for k, v in parse_map(a.foreign_vram_mib).items()},
                [float(x) for x in a.budgets.split(",") if x.strip()], a.allow_dirty, a.start_temp_rel_c or None,
                parse_map(a.start_temp_c_per_gpu))
    print(json.dumps(res, indent = 1))
    if res["toolkit_missing"]:
        print(f"WARNING: the toolkit at {res['scripts_sha'][:12]} lacks {res['toolkit_missing']}: export after the "
              "model cards and schema are merged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
