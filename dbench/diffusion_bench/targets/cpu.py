"""cpu: a Colab CPU High-RAM VM through cloud_pool.py run --cpu (the pool picks the account and keeps the books),
running the same notebook as the GPU targets (vm_setup.sh installs CPU torch when there is no nvidia-smi; the VM
driver runs every cell with no CUDA device). Generalised from scripts/cpumac/job.py + launch.sh, which drove the
stacks with their own harness; here the cells are the canonical run_cell.py.

  python -m targets.cpu prepare PLAN.json --out DIR --unsloth-sha SHA --bline-ref TAG
  python -m targets.cpu collect DIR/runs/<vm_id> [--plan PLAN.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import _remote as RM


def launch_line(vm: dict, nb: str, outdir: str, account, hf_token: bool) -> str:
    est = max(10, int(vm["est_min"]))
    tmo = int(min(720, est * 2 + 30))
    acc = f"--account {RM.q(account)} " if account else ""
    return (f'python3 -u "$WORKSPACE/cloud_pool.py" run --cpu {acc}--torch none --est-min {est} --timeout-min {tmo} '
            f"{'--pass-hf-token ' if hf_token else ''}--notebook {RM.q(nb)} --outdir {RM.q(outdir)}")


def prepare(plan: dict, out_dir, **kw) -> dict:
    return RM.prepare(plan, Path(out_dir), "cpu", launch_line, **kw)


def collect(run_dir, plan=None) -> list:
    return RM.collect(Path(run_dir), "cpu", plan)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest = "cmd", required = True)
    p = sub.add_parser("prepare")
    p.add_argument("plan")
    p.add_argument("--out", required = True)
    p.add_argument("--unsloth-sha")
    p.add_argument("--bline-ref")
    p.add_argument("--zoo-sha", help = "unsloth-zoo commit for the Studio venv (a branch is resolved now)")
    p.add_argument("--account")
    p.add_argument("--workspace")
    p.add_argument("--no-hf-token", action = "store_true")
    p.add_argument("--allow-dirty", action = "store_true")
    c = sub.add_parser("collect")
    c.add_argument("run_dir")
    c.add_argument("--plan", help = "PLAN.json: copy its cases.json into run_dir (report.py's expected case set)")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        meta = prepare(json.loads(Path(a.plan).read_text()), a.out, unsloth_sha = a.unsloth_sha, zoo_sha = a.zoo_sha,
                       bline_ref = a.bline_ref, account = a.account, workspace = a.workspace,
                       allow_dirty = a.allow_dirty, hf_token = not a.no_hf_token)
        print(json.dumps({k: v for k, v in meta.items() if k != "bundle"}, indent = 1))
    else:
        print(json.dumps([{"tag": r.get("tag"), "verdict": r.get("verdict")} for r in collect(a.run_dir, a.plan)], indent = 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
