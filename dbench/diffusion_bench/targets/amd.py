"""amd:gfx1151-{linux,windows}: one AMD CI branch per plan VM through amd/scaffold_diffusion.py (the entry; it prints
the push commands and never pushes). Linux: amd/gen_workflow.py jobs (trial, then the cells); Windows: the amd_ci
Windows template with an anchor PR, hardened the same way (AOTriton asserted, CI unset, profiler off, arms killed).

  python -m targets.amd prepare PLAN.json --out DIR [--pr N --merged]   # --pr only for Windows VMs
  python -m targets.amd collect DIR_OF_gh_run_download [--plan PLAN.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import common as TC

SCAFFOLD = TC.TOOLKIT / "amd" / "scaffold_diffusion.py"


def prepare(plan: dict, out_dir, plan_path: str | None = None, pr: int | None = None, merged: bool = False,
            base_sha: str | None = None, bline_ref: str | None = None, **_) -> dict:
    out_dir = Path(out_dir)
    vms = TC.vms_of(plan, "amd")
    if not vms:
        return {"kind": "amd", "vms": []}
    out_dir.mkdir(parents = True, exist_ok = True)
    pp = Path(plan_path) if plan_path else out_dir / "plan.json"
    if not plan_path:
        pp.write_text(json.dumps(plan, indent = 1))
    made, skipped = {}, {}
    for vm in vms:
        win = vm["target"].endswith("windows")
        if win and pr is None:
            skipped[vm["vm_id"]] = "Windows needs an anchor --pr (amd_ci states); not scaffolded"
            continue
        dest = out_dir / vm["vm_id"]
        cmd = [sys.executable, str(SCAFFOLD), "--plan", str(pp), "--vm", vm["vm_id"], "--out", str(dest)]
        if win:
            cmd += ["--windows", "--pr", str(pr)] + (["--merged"] if merged else [])
        if base_sha:
            cmd += ["--base-sha", base_sha]
        if bline_ref:
            cmd += ["--bline-ref", bline_ref]
        r = subprocess.run(cmd, capture_output = True, text = True)
        if r.returncode != 0:
            raise SystemExit(f"scaffold_diffusion failed for {vm['vm_id']}:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
        made[vm["vm_id"]] = {"dir": str(dest), "workflow": [str(p) for p in (dest / ".github" / "workflows").glob("*.yml")],
                             "next": r.stdout[r.stdout.find("next ("):][:1200]}
    meta = {"kind": "amd", "vms": list(made), "scaffolds": made, "skipped": skipped,
            "note": "push is manual (the push triggers the run); read VERDICT.md, not the tick"}
    (out_dir / "prepared_amd.json").write_text(json.dumps(meta, indent = 1))
    return meta


def collect(run_dir, plan=None) -> list:
    """Records from a `gh run download` dir: dbench/states/<state>/runs/**/record.json, the state added."""
    recs = TC.collect_records(Path(run_dir), plan = plan)
    for r in recs:
        parts = Path(r["_path"]).parts
        if "states" in parts:
            r.setdefault("amd_state", parts[parts.index("states") + 1])
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest = "cmd", required = True)
    p = sub.add_parser("prepare")
    p.add_argument("plan")
    p.add_argument("--out", required = True)
    p.add_argument("--pr", type = int)
    p.add_argument("--merged", action = "store_true")
    p.add_argument("--base-sha")
    p.add_argument("--bline-ref")
    c = sub.add_parser("collect")
    c.add_argument("run_dir")
    c.add_argument("--plan", help = "PLAN.json: copy its cases.json into run_dir (report.py's expected case set)")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        print(json.dumps(prepare(json.loads(Path(a.plan).read_text()), a.out, plan_path = a.plan, pr = a.pr,
                                 merged = a.merged, base_sha = a.base_sha, bline_ref = a.bline_ref), indent = 1))
    else:
        print(json.dumps([{"tag": r.get("tag"), "verdict": r.get("verdict"), "state": r.get("amd_state")}
                          for r in collect(a.run_dir, a.plan)], indent = 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
