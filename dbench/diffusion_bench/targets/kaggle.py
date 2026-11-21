"""kaggle:T4x2: one kernel per plan VM running both T4s as two lanes, each lane pinned to its own half of the CPUs
(vm_driver.cpu_sets) and its own GPU; plan.py's same_stack / serial lane modes keep Studio and BlineUI from ever
being timed side by side. Launched with notebook_cloud_run.py --no-pair (the kernel's two T4s are both ours).

  python -m targets.kaggle prepare PLAN.json --out DIR --unsloth-sha SHA --bline-ref TAG [--hf-token]
  python -m targets.kaggle collect DIR/runs/<vm_id> [--plan PLAN.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import _remote as RM


def launch_line(vm: dict, nb: str, outdir: str, account, hf_token: bool) -> str:
    wall, cell = RM.timeouts(vm)
    wall, cell = min(wall, 42500), min(cell, 42500)
    return ('python3 -u "$WORKSPACE/notebook_cloud_run.py" --backend kaggle --gpu T4x2 --no-pair '
            '--no-pair-reason "two internal lanes, one per T4, CPU halves" --no-smoke-patch '
            f"--per-cell-timeout {cell} --wall-timeout {wall} --idle-timeout 5400 {RM.env_flags(hf_token)} "
            f"--outdir {RM.q(outdir)} {RM.q(nb)}")


def prepare(plan: dict, out_dir, **kw) -> dict:
    for vm in [v for v in plan["vms"] if v["target"].startswith("kaggle:")]:
        for slot in vm["slots"]:
            if len({it["stack"] for it in slot if it.get("timed")}) > 1:
                raise SystemExit(f"{vm['vm_id']}: a slot times two stacks side by side: {slot}")
    kw.setdefault("hf_token", False)  # public files by default; a Kaggle secret travels with the kernel
    return RM.prepare(plan, Path(out_dir), "kaggle", launch_line, **kw)


def collect(run_dir, plan=None) -> list:
    return RM.collect(Path(run_dir), plan = plan)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest = "cmd", required = True)
    p = sub.add_parser("prepare")
    p.add_argument("plan")
    p.add_argument("--out", required = True)
    p.add_argument("--unsloth-sha")
    p.add_argument("--bline-ref")
    p.add_argument("--zoo-sha", help = "unsloth-zoo commit for the Studio venv (a branch is resolved now)")
    p.add_argument("--bline-gguf-sha", help = "GGUF loader node commit for baseline GGUF cells (default: its main now)")
    p.add_argument("--workspace")
    p.add_argument("--vm", action = "append", default = [])
    p.add_argument("--hf-token", action = "store_true", help = "forward HF_TOKEN (gated / private files)")
    p.add_argument("--allow-dirty", action = "store_true")
    c = sub.add_parser("collect")
    c.add_argument("run_dir")
    c.add_argument("--plan", help = "PLAN.json: copy its cases.json into run_dir (report.py's expected case set)")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        meta = prepare(json.loads(Path(a.plan).read_text()), a.out, unsloth_sha = a.unsloth_sha, zoo_sha = a.zoo_sha, bline_gguf_sha = a.bline_gguf_sha,
                       bline_ref = a.bline_ref, workspace = a.workspace, allow_dirty = a.allow_dirty,
                       hf_token = a.hf_token, vm_ids = a.vm)
        print(json.dumps({k: v for k, v in meta.items() if k != "bundle"}, indent = 1))
    else:
        recs = collect(a.run_dir, a.plan)
        print(json.dumps([{"tag": r.get("tag"), "verdict": r.get("verdict"), "path": r["_path"]} for r in recs], indent = 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
