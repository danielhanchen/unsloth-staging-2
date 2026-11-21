#!/usr/bin/env python3
"""Build a ready-to-push AMD CI branch that runs diffusion_bench on gfx1151, through amd_ci/scaffold.py.

  python diffusion_bench/amd/scaffold_diffusion.py --pr 11766 --out $WORKSPACE/temp/dbench_amd/ci_dbench_pr11766
  python diffusion_bench/amd/scaffold_diffusion.py --pr 11766 --windows --out .../ci_dbench_pr11766w
  python diffusion_bench/amd/scaffold_diffusion.py --pr 11713 --mode differential \\
      --defect slow:heavy_q21_fbcache:heavy_q21_bf16:0.9 --only 'heavy_q21_*' --out .../ci_dbench_pr11713

What it adds on top of amd_ci/scaffold.py (which it calls, so the toolkit, the templates and the lint stay the
single source of truth):
  - probe diffusion_bench/amd/diffusion_probe.py and criteria amd/criteria_no_regression.py (default) or
    amd/criteria_differential.py, with the probe args after "--" so differential.py passes them through;
  - Linux: --no-suites (the GPU job is the whole point), and the GPU job's timeout raised to 330 min (installs
    plus base and head cells; the probe's --budget-min keeps each state inside it);
  - Windows: the template's commented-out concurrency group enabled as amd-ci-gfx1151-gpu-windows (every past
    Windows GPU measurement used it) and the job timeout raised the same way; the probe installs ROCm torch
    from repo.amd.com/rocm/whl/gfx1151 into the job's venv itself (--install auto);
  - a copy of diffusion_bench/ next to amd_ci/, because the runner checks out this branch and nothing else;
  - the lint again on the edited workflow, and the push command printed, never run.

Plan route (plan.py output; one amd VM = one branch):
  python diffusion_bench/amd/scaffold_diffusion.py --plan PLAN.json --vm amd-gfx1151-linux-1 --out DIR \
      [--base-sha SHA] [--bline-ref TAG]          # defaults: the plan's revisions
  Linux: amd/gen_workflow.py jobs: t0_trial (the VM's trial cell; a failed trial stops the rest) then the measured
  cells in plan order, base = unsloth at the plan's SHA, head = base + a no-op marker re-running one Studio cell
  (an extra A/A point). Windows needs an anchor PR for amd_ci's states (--pr N [--merged]) and runs the plan's
  cells through the --pr route.
Both routes get gen_workflow's harness invariants (AOTriton opt-in asserted, CI unset, Windows profiler off, a
step timeout and a kill step for arms that never exit).

It never pushes, never writes a secret into the workflow, and refuses an --out that already holds workflows.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLKIT = HERE.parent
sys.path.insert(0, str(TOOLKIT))
sys.path.insert(0, str(HERE))

import gen_workflow as GW  # noqa: E402
import public_bundle as PB  # noqa: E402 - the pushed toolkit is the rewritten public copy
AMD_CI = TOOLKIT.parent / "amd_ci"
HOST_REPO = "https://github.com/oobabooga/unsloth.git"

WIN_CONCURRENCY_OFF = ("    # concurrency:\n"
                       "    #   group: amd-ci-strix-halo-windows-gpu\n"
                       "    #   cancel-in-progress: false\n")
WIN_CONCURRENCY_ON = ("    concurrency:\n"
                      "      group: amd-ci-gfx1151-gpu-windows\n"
                      "      cancel-in-progress: false\n")


def quote_for(windows: bool, arg: str) -> str:
    # Single quotes are literal in both bash and PowerShell 5.1, which keeps fnmatch patterns from globbing.
    if any(ch in arg for ch in "*?[] '\"$;&|<>(){}"):
        return "'" + arg.replace("'", "''" if windows else "'\"'\"'") + "'"
    return arg


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pr", type = int, default = None, help = "required except for --plan on Linux")
    ap.add_argument("--plan", type = Path, default = None, help = "plan.py output (with --vm)")
    ap.add_argument("--vm", default = None, help = "the plan's amd VM id")
    ap.add_argument("--base-sha", default = None, help = "--plan Linux: unsloth commit (default: plan revisions)")
    ap.add_argument("--bline-ref", default = None, help = "--plan Linux: BlineUI tag / sha (default: plan revisions)")
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--branch", default = None, help = "default amd-ci-dbench-pr<N>[-win]")
    ap.add_argument("--merged", action = "store_true", help = "PR already merged: merge commit vs its parent")
    ap.add_argument("--windows", action = "store_true")
    ap.add_argument("--mode", choices = ["regression", "differential"], default = "regression")
    ap.add_argument("--defect", default = None, help = "differential only, e.g. broken:zimg_fp8")
    ap.add_argument("--spec", default = "diffusion_bench/specs/amd_strix_small.json",
                    help = "path inside the branch")
    ap.add_argument("--only", action = "append", default = [], help = "cell patterns (default: the probe's)")
    ap.add_argument("--budget-min", type = float, default = 100.0, help = "per state")
    ap.add_argument("--edge", default = None, help = "edge-suite command for the probe to record (TODO(verify))")
    ap.add_argument("--probe-extra", default = "", help = "more probe args, appended verbatim")
    ap.add_argument("--job-timeout-min", type = int, default = 330)
    ap.add_argument("--cell-timeout", type = float, default = 3600.0, help = "--plan: per cell, seconds (kills the arm)")
    ap.add_argument("--patch", default = None, help = "amd_ci/scaffold.py --patch (head = --patch-base + this diff)")
    ap.add_argument("--patch-base", default = None)
    args = ap.parse_args()

    plan_spec_rel = None
    if args.plan:
        if not args.vm:
            ap.error("--plan needs --vm")
        plan = json.loads(args.plan.read_text(encoding = "utf-8"))
        vm = next((v for v in plan["vms"] if v["vm_id"] == args.vm), None)
        if vm is None or not vm["target"].startswith("amd:"):
            ap.error(f"--vm {args.vm!r} is not an amd VM of the plan")
        if vm["target"].endswith("windows") != args.windows:
            ap.error(f"{args.vm} is {vm['target']}: pass --windows exactly when the target is Windows")
        if not args.windows:
            return plan_linux(args, plan, vm)
        if args.pr is None:
            ap.error("--plan on Windows needs an anchor --pr N (amd_ci states); add --merged for a merged PR")
        plan_spec_rel = f"diffusion_bench/specs/plan_{args.vm}.json"
        args.spec = plan_spec_rel
    elif args.pr is None:
        ap.error("--pr is required (or --plan --vm)")

    if args.mode == "differential" and not args.defect:
        ap.error("--mode differential needs --defect (the VOID rule needs to know what the base must show)")
    if args.mode == "regression" and args.defect:
        ap.error("--defect only means something with --mode differential")
    branch = args.branch or f"amd-ci-dbench-pr{args.pr}" + ("-win" if args.windows else "")
    criteria = f"diffusion_bench/amd/criteria_{'differential' if args.mode == 'differential' else 'no_regression'}.py"
    probe_args = ["--spec", args.spec, "--budget-min", str(args.budget_min)]
    for pat in args.only:
        probe_args += ["--only", pat]
    if args.defect:
        probe_args += ["--defect", args.defect]
    if args.edge:
        probe_args += ["--edge", args.edge]
    probe_str = "-- " + " ".join(quote_for(args.windows, a) for a in probe_args)
    if args.probe_extra:
        probe_str += " " + args.probe_extra

    cmd = [sys.executable, str(AMD_CI / "scaffold.py"), "--pr", str(args.pr), "--out", str(args.out),
           "--branch", branch, "--probe", "diffusion_bench/amd/diffusion_probe.py", "--criteria", criteria,
           "--probe-args", probe_str]
    if args.merged:
        cmd.append("--merged")
    if args.patch:
        cmd += ["--patch", args.patch, "--patch-base", args.patch_base or ""]
    cmd.append("--windows" if args.windows else "--no-suites")
    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush = True)
    rc = subprocess.run(cmd, capture_output = True, text = True)
    if rc.returncode != 0:
        print(rc.stdout + rc.stderr)
        print("amd_ci/scaffold.py failed; nothing else done")
        return rc.returncode

    wf = args.out / ".github" / "workflows" / f"{branch}.yml"
    text = wf.read_text(encoding = "utf-8")
    if args.windows:
        if WIN_CONCURRENCY_OFF not in text:
            raise SystemExit("the Windows template's commented concurrency block moved; update scaffold_diffusion.py")
        text = text.replace(WIN_CONCURRENCY_OFF, WIN_CONCURRENCY_ON, 1)
    text = (GW.harden_windows(text, args.job_timeout_min - 30) if args.windows
            else GW.harden_linux_template(text, args.job_timeout_min - 30))
    if "    timeout-minutes: 120\n" not in text:
        raise SystemExit("the GPU job's 'timeout-minutes: 120' moved; update scaffold_diffusion.py")
    text = text.replace("    timeout-minutes: 120\n", f"    timeout-minutes: {args.job_timeout_min}\n")
    text = text.replace(f'name: "AMD CI{" (Windows)" if args.windows else ""}: validate PR {args.pr}"',
                        f'name: "AMD CI{" (Windows)" if args.windows else ""}: diffusion_bench PR {args.pr} '
                        f'({args.mode})"', 1)
    text = GW.with_baseline_repo(PB.rewrite_text(text))  # paths / spec names as in the public copy
    bad = GW.public_problems(text)  # the branch is pushed to a public fork: no baseline-UI name in the workflow
    if bad:
        raise SystemExit(f"the generated workflow names the baseline UI (public fork): {bad[:5]}")
    wf.write_text(text, encoding = "utf-8")

    PB.write(args.out / "diffusion_bench", TOOLKIT)  # the rewritten public copy (public_bundle.py)
    from targets import common as TC  # noqa: PLC0415

    if plan_spec_rel:
        import matrix  # noqa: PLC0415

        pspec = matrix.plan_spec(args.plan, args.vm)
        spec_path = args.out / PB.rewrite_text(plan_spec_rel)
        spec_path.write_text(PB.rewrite_text(json.dumps(pspec, indent = 1)), encoding = "utf-8")
        TC.write_public_kit(args.out / "diffusion_bench", pspec["cells"])
    else:
        TC.write_public_kit(args.out / "diffusion_bench", spec_cells(args.spec))
    leak = PB.problems(args.out)
    if leak:
        raise SystemExit(f"{args.out} would name the baseline UI on a public push: {leak[:10]}")
    lint = subprocess.run([sys.executable, str(AMD_CI / "lib" / "lint_workflow.py"), str(wf)],
                          capture_output = True, text = True)
    print(lint.stdout + lint.stderr)
    if lint.returncode != 0:
        print("lint found error-level problems after the diffusion edits; fix them before pushing")
        return lint.returncode
    print(f"""wrote {wf}
branch contents: .github/workflows/{branch}.yml, amd_ci/, diffusion_bench/ (no other workflow can fire)
mode {args.mode}{' defect ' + args.defect if args.defect else ''}; probe args: {probe_str}

next (NOT run by this script; the push TRIGGERS the run):
  cd {args.out} && git init -q && git checkout -q -b {branch}
  git add -A && git commit -q -m "diffusion_bench PR {args.pr} on gfx1151 ({'Windows' if args.windows else 'Linux'})"
  git remote add ooba {HOST_REPO}
  git push -q ooba HEAD:refs/heads/{branch}

monitor and read the verdict (not the tick):
  RUN=$(gh run list --repo oobabooga/unsloth --branch {branch} --limit 1 --json databaseId --jq '.[0].databaseId')
  gh run download $RUN --repo oobabooga/unsloth -D $WORKSPACE/temp/dbench_amd/run_$RUN
  find $WORKSPACE/temp/dbench_amd/run_$RUN -name VERDICT.md -exec cat {{}} +
""")
    return 0


def spec_cells(spec: str | None) -> list:
    """The cells of a --spec (a path, or a name under diffusion_bench/specs/); [] when it cannot be read."""
    for cand in (Path(spec), TOOLKIT / spec, TOOLKIT / "specs" / f"{spec}.json") if spec else ():
        if cand.is_file():
            try:
                return json.loads(cand.read_text(encoding = "utf-8")).get("cells") or []
            except ValueError:
                return []
    return []


def plan_linux(args, plan: dict, vm: dict) -> int:
    """--plan on Linux: gen_workflow jobs over the VM's cells (trial first), the toolkit + amd_ci copied in."""
    import matrix  # noqa: PLC0415

    rev = plan.get("revisions") or {}
    base_sha = args.base_sha or rev.get("unsloth_sha")
    bline_ref = args.bline_ref or rev.get("bline_ref")
    if not base_sha or not bline_ref:
        raise SystemExit("--plan: no unsloth_sha / bline_ref in the plan's revisions; pass --base-sha / --bline-ref")
    if (args.out / ".github" / "workflows").exists() and any((args.out / ".github" / "workflows").glob("*.yml")):
        raise SystemExit(f"{args.out} already holds workflows; pick a fresh --out")
    branch = args.branch or f"amd-ci-dbench-{args.vm}"
    spec_rel = f"diffusion_bench/specs/plan_{args.vm}.json"
    spec = matrix.plan_spec(args.plan, args.vm)
    tags = [c["tag"] for c in spec["cells"]]
    trial = [t for t in tags if t.startswith("trial__")]
    rest = [t for t in tags if not t.startswith("trial__")]
    ref = next((c["tag"] for c in spec["cells"] if c["backend"] == "studio" and not c["tag"].startswith("trial__")),
               rest[0] if rest else None)
    budget = max(60.0, float(vm.get("est_min") or 0) * 1.3)
    jobs = []
    if trial:
        jobs.append({"id": "t0_trial", "title": f"Trial ({trial[0]})", "spec": spec_rel, "only": trial,
                     "head_only": trial, "criteria": "no_regression", "preflight_gib": 20, "budget_min": 60,
                     "cell_timeout": 2400, "trial": True})
    jobs.append({"id": "m1_cells", "title": f"{vm['target']} {len(rest)} cells", "spec": spec_rel, "only": rest,
                 "head_only": [ref] if ref else [], "criteria": "no_regression", "preflight_gib": 50,
                 "budget_min": budget, "cell_timeout": args.cell_timeout})
    text = GW.render_linux({"branch": branch, "title": plan.get("name", "plan"), "jobs": jobs}, base_sha, bline_ref)
    wf = GW.write(args.out, branch, PB.rewrite_text(text))
    PB.write(args.out / "diffusion_bench", TOOLKIT)  # the rewritten public copy (public_bundle.py)
    from targets import common as TC  # noqa: PLC0415

    TC.write_public_kit(args.out / "diffusion_bench", spec["cells"])  # Studio-over-HTTP launcher, when used
    (args.out / spec_rel).write_text(PB.rewrite_text(json.dumps(spec, indent = 1)), encoding = "utf-8")
    for sub in ("lib", "criteria", "probes"):
        shutil.copytree(AMD_CI / sub, args.out / "amd_ci" / sub, dirs_exist_ok = True,
                        ignore = shutil.ignore_patterns("__pycache__", "*.pyc"))
    lint = subprocess.run([sys.executable, str(AMD_CI / "lib" / "lint_workflow.py"), str(wf)],
                          capture_output = True, text = True)
    print(lint.stdout + lint.stderr)
    if lint.returncode != 0:
        print("lint found error-level problems in the generated workflow; fix gen_workflow.py before pushing")
        return lint.returncode
    (args.out / "plan_vm.json").write_text(PB.rewrite_text(json.dumps({"vm": vm, "base_sha": base_sha,
                                                                        "bline_ref": bline_ref, "branch": branch},
                                                                       indent = 1)), encoding = "utf-8")
    leak = PB.problems(args.out)
    if leak:
        raise SystemExit(f"{args.out} would name the baseline UI on a public push: {leak[:10]}")
    print(f"""wrote {wf}
jobs: {[j['id'] for j in jobs]}; base unsloth {base_sha[:12]}, baseline UI {bline_ref}; {len(rest)} cells + trial

next (NOT run by this script; the push TRIGGERS the run):
  cd {args.out} && git init -q && git checkout -q -b {branch}
  git add -A && git commit -q -m "diffusion_bench {args.vm} on gfx1151 (Linux)"
  git remote add ooba {HOST_REPO}
  git push -q ooba HEAD:refs/heads/{branch}
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
