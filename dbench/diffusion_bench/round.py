#!/usr/bin/env python3
"""One command for a Studio vs baseline round on one GPU type: request -> plan -> prepare -> launch -> collect ->
score -> report, every step the toolkit's own (plan.py, targets.<kind>, score.py, report.py).

  python diffusion_bench/round.py --target colab:L4 --models flux.3-klein:int8,bf16 qwen-image-2.1:int8 \\
      --account michael --out $WORKSPACE/outputs/dbench/l4_r1            # plan + prepare, prints the launch line
  ... same args ... --go        # also launch (foreground, hours: run it with run_in_background) and then collect
  python diffusion_bench/round.py --out DIR --collect                     # collect + score + report a finished run

--models FAMILY:FMT[,FMT] (card families in models/; each format gets a Studio arm and a baseline arm, so every
format is a timed pair and bf16 is each stack's own LPIPS reference). Revisions: --unsloth-sha / --zoo-sha take a
commit or a branch (resolved to its commit now and recorded); --bline-ref defaults to models/BLINE_PIN.json. The
round always carries the parity negative control (one Studio probe with a changed step count that the gate must
reject). Kaggle targets get no HF token unless --hf-token (gated weights only). Read the plan listing's NOT_RUN
lines before launching: each says why a model x format x GPU cannot run (card hints, capability, files).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def parse_models(specs: list) -> list:
    out = []
    for s in specs:
        fam, _, fmts = s.partition(":")
        entry = {"family": fam}
        if fmts:
            entry["formats"] = [f for f in fmts.split(",") if f]
        out.append(entry)
    return out


def build_request(a) -> dict:
    from targets import _remote as RM  # noqa: PLC0415

    pin = json.loads((HERE / "models" / "BLINE_PIN.json").read_text())
    models = parse_models(a.models)
    fmts = sorted({f for m in models for f in m.get("formats") or []}) or ["int8"]
    arms = []
    for f in fmts:
        arms.append({"stack": "studio", "format": f, "tier": a.tier})
        arms.append({"stack": "blineui", "format": f})
    req = {"name": a.name or Path(a.out).name,
           "revisions": {"unsloth_sha": RM.resolve_ref(RM.UNSLOTH_URL, a.unsloth_sha),
                         "zoo_sha": RM.resolve_ref(RM.ZOO_URL, a.zoo_sha),
                         "bline_ref": a.bline_ref or pin["tag"]},
           "models": models, "arms": arms, "targets": [a.target], "repeats": a.repeats, "n": a.n, "warmup": 1,
           "negative_control": True}
    if a.account:
        req["account"] = a.account
    if a.downloads:
        req["downloads"] = json.loads(Path(a.downloads).read_text())
    return req


def run(cmd: list, **kw) -> int:
    print("+", " ".join(str(c) for c in cmd), flush = True)
    return subprocess.call([str(c) for c in cmd], **kw)


def collect(out: Path, kind: str) -> int:
    plan = out / "plan" / "plan.json"
    rc = 0
    for rd in sorted((out / kind / "runs").glob("*")) if (out / kind / "runs").is_dir() else []:
        rc |= run([sys.executable, "-m", f"targets.{kind}", "collect", rd, "--plan", plan], cwd = str(HERE))
        rc |= run([sys.executable, HERE / "score.py", rd])
        rc |= run([sys.executable, HERE / "report.py", rd])
        print(f"report: {rd / 'report.md'}")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required = True, help = "round dir (request.json, plan/, <kind>/ with notebooks and runs)")
    ap.add_argument("--target", help = "colab:T4|L4|A100-40|A100-80|G4 or kaggle:T4x2")
    ap.add_argument("--models", nargs = "+", default = [], help = "FAMILY:FMT[,FMT] ...")
    ap.add_argument("--name")
    ap.add_argument("--account", help = "Colab account (michael, daniel, ...)")
    ap.add_argument("--unsloth-sha", default = "main")
    ap.add_argument("--zoo-sha", default = "main")
    ap.add_argument("--bline-ref", help = "baseline tag / sha (default models/BLINE_PIN.json)")
    ap.add_argument("--tier", default = "default", help = "Studio speed tier for every Studio arm")
    ap.add_argument("--repeats", type = int, default = 3, help = "timed fresh-process runs per arm (>= 3)")
    ap.add_argument("--n", type = int, default = 2, help = "new-prompt renders per timed run")
    ap.add_argument("--downloads", help = "JSON {file name: {repo, file, sha256}} staging baseline files a card "
                                         "lists without an official source (labelled in the request)")
    ap.add_argument("--hf-token", action = "store_true", help = "kaggle: forward HF_TOKEN (gated weights)")
    ap.add_argument("--allow-dirty", action = "store_true", help = "bundle an uncommitted toolkit (offline tests only)")
    ap.add_argument("--go", action = "store_true", help = "launch now (foreground), then collect + score + report")
    ap.add_argument("--collect", action = "store_true", help = "only collect + score + report finished runs")
    a = ap.parse_args(argv)
    # absolute before anything runs: prepare / collect run with cwd = this dir, so a relative --out would point at
    # two different places (the planner wrote it under the caller's cwd)
    out = Path(a.out).expanduser().resolve()
    a.out = str(out)
    out.mkdir(parents = True, exist_ok = True)
    if a.collect:
        kind = json.loads((out / "request.json").read_text())["targets"][0].split(":")[0]
        return collect(out, kind)
    if not a.target or not a.models:
        ap.error("--target and --models are required (or --collect)")
    kind = a.target.split(":")[0]
    if kind not in ("colab", "kaggle"):
        ap.error("round.py drives the rented-VM targets (colab, kaggle); others: README targets table")
    req = build_request(a)
    (out / "request.json").write_text(json.dumps(req, indent = 1))
    plan = out / "plan" / "plan.json"
    if run([sys.executable, HERE / "plan.py", out / "request.json", "--out", plan, "--dry-run"]):
        return 2
    if not json.loads(plan.read_text()).get("vms"):
        print("nothing to run: every comparison is NOT_RUN (reasons above)")
        return 3
    prep = [sys.executable, "-m", f"targets.{kind}", "prepare", plan, "--out", out / kind,
            "--workspace", os.environ.get("WORKSPACE", str(out))]
    if kind == "colab" and a.account:
        prep += ["--account", a.account]
    if kind == "kaggle" and a.hf_token:
        prep.append("--hf-token")
    if a.allow_dirty:
        prep.append("--allow-dirty")
    if run(prep, cwd = str(HERE)):
        return 2
    launch = out / kind / f"launch_{kind}.sh"
    print(f"\nlaunch (hours; background it): bash {launch} all\nthen: python {Path(__file__)} --out {out} --collect")
    if not a.go:
        return 0
    rc = run(["bash", launch, "all"], cwd = str(launch.parent))
    return collect(out, kind) or rc


if __name__ == "__main__":
    raise SystemExit(main())
