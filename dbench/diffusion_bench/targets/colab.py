"""colab:{T4,L4,A100-40,A100-80,G4}: one notebook per plan VM, launched with notebook_cloud_run.py on the named tier
(the args the moss_speed / imgcmp_nov5 rounds used), the toolkit bundle at a recorded scripts SHA.

  python -m targets.colab prepare PLAN.json --out DIR --unsloth-sha SHA --bline-ref TAG [--account michael]
  python -m targets.colab collect DIR/runs/<vm_id> [--plan PLAN.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import _remote as RM

GPU_FLAG = {"T4": "T4", "L4": "L4", "A100": "A100", "A100-HM": "A100-HM", "G4": "G4"}
DEAR = {"A100", "A100-HM", "G4", "H100"}


def launch_line(vm: dict, nb: str, outdir: str, account: str | None, hf_token: bool) -> str:
    from . import target_info

    gpu = GPU_FLAG[target_info(vm["target"])["gpu"]]
    wall, cell = RM.timeouts(vm)
    acc = f"NBRUN_COLAB_ACCOUNT={RM.q(account)} " if account else 'NBRUN_COLAB_ACCOUNT="${ACCOUNT:?set ACCOUNT}" '
    return (f'{acc}COLAB_TIER_GUARD=1 COLAB_SHARE=0 python3 -u "$WORKSPACE/notebook_cloud_run.py" --backend colab '
            f"--gpu {gpu} --no-smoke-patch {'--dear-ok ' if gpu in DEAR else ''}--alloc-wait ${{ALLOC_WAIT:-1800}} "
            f"--per-cell-timeout {cell} --wall-timeout {wall} --idle-timeout 3600 {RM.env_flags(hf_token)} "
            f"--outdir {RM.q(outdir)} {RM.q(nb)}")


def prepare(plan: dict, out_dir, **kw) -> dict:
    return RM.prepare(plan, Path(out_dir), "colab", launch_line, **kw)


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
    p.add_argument("--account", help = "Colab account (else $ACCOUNT at launch time)")
    p.add_argument("--workspace", help = "default WORKSPACE baked into the launch script (else $WORKSPACE)")
    p.add_argument("--vm", action = "append", default = [])
    p.add_argument("--no-hf-token", action = "store_true")
    p.add_argument("--allow-dirty", action = "store_true")
    c = sub.add_parser("collect")
    c.add_argument("run_dir")
    c.add_argument("--plan", help = "PLAN.json: copy its cases.json into run_dir (report.py's expected case set)")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        meta = prepare(json.loads(Path(a.plan).read_text()), a.out, unsloth_sha = a.unsloth_sha, zoo_sha = a.zoo_sha, bline_gguf_sha = a.bline_gguf_sha,
                       bline_ref = a.bline_ref, account = a.account, workspace = a.workspace,
                       allow_dirty = a.allow_dirty, hf_token = not a.no_hf_token, vm_ids = a.vm)
        print(json.dumps({k: v for k, v in meta.items() if k != "bundle"}, indent = 1))
    else:
        recs = collect(a.run_dir, a.plan)
        print(json.dumps([{"tag": r.get("tag"), "verdict": r.get("verdict"), "path": r["_path"]} for r in recs], indent = 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
