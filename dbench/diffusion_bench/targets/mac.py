"""mac:macos-15: GitHub's Apple Silicon runner on an owned staging repo, generalised from
scripts/cpumac/mac/nov5-mac.yml + bench_mac.py: Studio (MPS; its MLX route when the arm enables it) vs BlineUI
(MPS, + BlineUI-GGUF for gguf cells) vs stable-diffusion.cpp (Metal prebuilt), one stack process at a time (7 GB
runner), cells through the canonical matrix.py, then targets/mac_check.py turns any CPU fallback into VOID.

  python -m targets.mac prepare PLAN.json --out DIR [--unsloth-sha SHA] [--bline-ref TAG] [--bline-gguf-sha SHA]
  python -m targets.mac collect DIR_OF_gh_run_download [--plan PLAN.json]

prepare writes a branch dir per VM (.github/workflows/dbench-mac-<vm>.yml + dbench/) and prints the push commands
for a staging repo; it never pushes. PYTORCH_ENABLE_MPS_FALLBACK=0: an op missing on MPS fails the cell instead of
running on the CPU unnoticed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import common as TC
from . import mac_check

WORKFLOW = r"""name: dbench-mac-__VM__
on:
  push:
    branches: ['__BRANCH__']
permissions:
  contents: read
env:
  STUDIO_SHA: __STUDIO_SHA__
  BASELINE_REF: __BASELINE_REF__
  BASELINE_REPO: ${{ vars.BASELINE_REPO }}
  BASELINE_GGUF_SHA: __GGUF_SHA__
  # one HF cache for the fetch step (Studio prefetch) and the bench step (Studio loads with local_files_only)
  HF_HOME: ${{ github.workspace }}/hf_home
jobs:
  bench:
    runs-on: macos-15
    timeout-minutes: __TIMEOUT__
    steps:
      - uses: actions/checkout@v4
        with: {path: head}
      - name: Host
        run: |
          mkdir -p results
          { sysctl hw.model hw.memsize hw.ncpu hw.perflevel0.physicalcpu machdep.cpu.brand_string; sw_vers; system_profiler SPDisplaysDataType; vm_stat; df -h .; } > results/host.txt 2>&1 || true
          cat results/host.txt
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - name: Studio venv (unsloth ${{ env.STUDIO_SHA }}, Mac pins)
        run: |
          set -eux
          python -m pip install -q uv
          git init -q unsloth && git -C unsloth remote add origin https://github.com/unslothai/unsloth
          git -C unsloth fetch -q --depth 1 origin $STUDIO_SHA && git -C unsloth checkout -q FETCH_HEAD
          uv venv -q --python "$(which python)" venv_studio
          R=unsloth/studio/backend/requirements
          O="--override $R/single-env/overrides-darwin-arm64.txt"
          PY=venv_studio/bin/python
          uv pip install -q --python $PY "torch>=2.4,<2.12.0" "torchvision>=0.19,<0.27.0"
          uv pip install -q --python $PY $O -r $R/no-torch-runtime.txt
          uv pip install -q --python $PY $O --no-deps -r $R/extras-no-deps.txt
          uv pip install -q --python $PY $O -r $R/studio.txt
          uv pip install -q --python $PY $O -r $R/diffusers-pin.txt
          grep -qv '^#' $R/diffusers-main.txt && uv pip install -q --python $PY $O -r $R/diffusers-main.txt || true
          uv pip install -q --python $PY $O "mlx==0.32.4" "mlx-metal==0.32.4" || true
          uv pip install -q --python $PY --no-deps "mflux==0.22.0" "pillow>=12.3.0,<13" "piexif>=1.1.3,<2" "platformdirs>=4,<5" "toml>=0.10.2,<1" || true
          uv pip install -q --python $PY gguf aiohttp psutil huggingface_hub scikit-image
          $PY -c "import torch, diffusers, transformers; print('STUDIO_VERSIONS', torch.__version__, diffusers.__version__, transformers.__version__, 'mps', torch.backends.mps.is_available())" | tee results/studio_versions.txt
      - name: Baseline UI venv (${{ env.BASELINE_REF }} + GGUF loader node)
        run: |
          set -eux
          python head/dbench/diffusion_bench/baseline_src.py --dir refrt --ref "$BASELINE_REF" \
            --gguf-sha "$BASELINE_GGUF_SHA" --commit-out results/baseline_commit.txt --reqs-out ref_req.txt
          uv venv -q --python "$(which python)" venv_ref
          PY=venv_ref/bin/python
          uv pip install -q --python $PY "torch>=2.4,<2.12.0" "torchvision>=0.19,<0.27.0" torchaudio
          uv pip install -q --python $PY -r ref_req.txt aiohttp psutil
          $PY -c "import torch; print('BASELINE_VERSIONS', torch.__version__, 'mps', torch.backends.mps.is_available())" | tee results/ref_versions.txt
      - name: Model files + cell localisation
        run: |
          set -eux
          venv_studio/bin/python head/dbench/diffusion_bench/targets/fetch_files.py --spec head/dbench/plan_vm.json \
            --root "$PWD/models" --write plan_local.json --venv "$PWD/venv_studio/bin/python" --studio-src "$PWD/unsloth" \
            --baseline-python "$PWD/venv_ref/bin/python" --baseline-dir "$PWD/refrt"
          ls -la models || true; df -h .
      - name: Bench (one stack process at a time)
        env:
          HF_HUB_DISABLE_PROGRESS_BARS: "1"
          PYTORCH_ENABLE_MPS_FALLBACK: "0"
          DIFFUSION_BENCH_SDCPP_METHOD: prebuilt
          DIFFUSION_BENCH_SDCPP_BACKEND: metal
        run: |
          set -o pipefail
          export WORKSPACE="$PWD" DIFFUSION_BENCH_BASELINE_PYTHON="$PWD/venv_ref/bin/python" DIFFUSION_BENCH_BASELINE_DIR="$PWD/refrt"
          export UNSLOTH_STUDIO_HOME="$PWD/studio_home" DBENCH_SCRIPTS_SHA=__SCRIPTS_SHA__
          venv_studio/bin/python -u head/dbench/diffusion_bench/matrix.py --plan plan_local.json --vm __VM__ \
            --out results/cells --no-gate --timeout __CELL_TIMEOUT__ 2>&1 | tee results/matrix.log
          vm_stat | head -12
      - name: CPU-fallback check (VOID on fallback)
        if: always()
        run: |
          venv_studio/bin/python head/dbench/diffusion_bench/targets/mac_check.py results/cells --write | tee results/mac_check.txt
      - uses: actions/upload-artifact@v4
        if: always()
        with: {name: "dbench-mac-__VM__", path: results/, retention-days: 7}
"""


def prepare(plan: dict, out_dir, unsloth_sha: str | None = None, bline_ref: str | None = None,
            bline_gguf_sha: str | None = None, allow_dirty: bool = False, staging: str = "OWNER/STAGING_REPO", **_) -> dict:
    import builders  # noqa: PLC0415
    import public_bundle as PB  # noqa: PLC0415 - private side only (targets.common put the toolkit root on sys.path)

    out_dir = Path(out_dir)
    vms = TC.vms_of(plan, "mac")
    if not vms:
        return {"kind": "mac", "vms": []}
    rev = dict(plan.get("revisions") or {})
    rev.update({k: v for k, v in (("unsloth_sha", unsloth_sha), ("bline_ref", bline_ref)) if v})
    from . import notebook

    notebook.check_revisions(rev)
    bundle = TC.build_bundle(out_dir / "bundle", allow_dirty = allow_dirty)
    made = {}
    for vm in vms:
        if any(len(s) > 1 for s in vm["slots"]):
            raise SystemExit(f"{vm['vm_id']}: the Mac runner runs one stack at a time")
        branch = f"dbench-mac-{vm['vm_id']}"
        d = out_dir / vm["vm_id"]
        if d.exists():
            shutil.rmtree(d)
        (d / ".github" / "workflows").mkdir(parents = True)
        # the public copy (staging repos can be public): the rewritten toolkit, not the bundle's own tree
        PB.write(d / "dbench" / "diffusion_bench", TC.TOOLKIT)
        TC.write_public_kit(d / "dbench" / "diffusion_bench", TC.vm_slice(plan, vm["vm_id"])["cells"].values())
        sl = TC.vm_slice(plan, vm["vm_id"])
        tool = builders.needs_tool(sl["cells"].values(), sl.get("downloads"))
        if tool:  # the public workflow has no route to the (not vendored) quantizer
            raise SystemExit(f"{vm['vm_id']}: {', '.join(tool)} need the int8 ConvRot quantizer, which the Mac runner "
                             "cannot fetch: plan those arms on a Colab / Kaggle / local target")
        # downloads, Studio prefetch (fetch_files.py pulls it into $HF_HOME), comparisons, revisions
        mini = {"name": plan["name"], **TC.C.plan_carry(plan, vm), "vms": [vm], "cells": list(sl["cells"].values())}
        mini["revisions"] = {**rev, "scripts_sha": bundle["scripts_sha"]}
        (d / "dbench" / "plan_vm.json").write_text(PB.rewrite_text(json.dumps(mini, indent = 1)))
        timeout = int(min(330, max(60, vm["est_min"] * 1.5 + 60)))
        wf = (WORKFLOW.replace("__VM__", vm["vm_id"]).replace("__BRANCH__", branch)
              .replace("__STUDIO_SHA__", rev["unsloth_sha"]).replace("__BASELINE_REF__", rev["bline_ref"])
              .replace("__GGUF_SHA__", bline_gguf_sha or "").replace("__TIMEOUT__", str(timeout))
              .replace("__CELL_TIMEOUT__", "3600").replace("__SCRIPTS_SHA__", bundle["scripts_sha"]))
        wf = PB.rewrite_text(wf)
        bad = TC.public_problems(wf)
        if bad:
            raise SystemExit(f"{branch}: the workflow names the baseline UI (staging repos are public): {bad[:5]}")
        (d / ".github" / "workflows" / f"{branch}.yml").write_text(wf)
        leak = PB.problems(d)
        if leak:
            raise SystemExit(f"{d} would name the baseline UI on a public push: {leak[:10]}")
        made[vm["vm_id"]] = {"dir": str(d), "branch": branch, "next": [
            f"cd {d} && git init -q -b {branch} && git add -A",
            f"git -c user.name='Daniel Han' -c user.email=danielhanchen@gmail.com commit -q -m 'dbench mac {vm['vm_id']}'",
            f"env -u GH_TOKEN git push -q https://github.com/{staging} HEAD:refs/heads/{branch}   # triggers the run"]}
    meta = {"kind": "mac", "vms": list(made), "branches": made, "bundle": bundle["path"],
            "scripts_sha": bundle["scripts_sha"], "note": "push is manual; read results/mac_check.txt and records"}
    (out_dir / "prepared_mac.json").write_text(json.dumps(meta, indent = 1))
    return meta


def collect(run_dir, plan=None) -> list:
    """Records from a downloaded artifact, with the CPU-fallback check (re)applied."""
    TC.unpack_artifacts(Path(run_dir))
    mac_check.run(Path(run_dir), write = True)
    return TC.collect_records(Path(run_dir), "mac:macos-15", plan)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest = "cmd", required = True)
    p = sub.add_parser("prepare")
    p.add_argument("plan")
    p.add_argument("--out", required = True)
    p.add_argument("--unsloth-sha")
    p.add_argument("--bline-ref")
    p.add_argument("--bline-gguf-sha")
    p.add_argument("--staging", default = "OWNER/STAGING_REPO", help = "repo the printed push targets")
    p.add_argument("--allow-dirty", action = "store_true")
    c = sub.add_parser("collect")
    c.add_argument("run_dir")
    c.add_argument("--plan", help = "PLAN.json: copy its cases.json into run_dir (report.py's expected case set)")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        print(json.dumps(prepare(json.loads(Path(a.plan).read_text()), a.out, unsloth_sha = a.unsloth_sha,
                                 bline_ref = a.bline_ref, bline_gguf_sha = a.bline_gguf_sha,
                                 allow_dirty = a.allow_dirty, staging = a.staging), indent = 1))
    else:
        print(json.dumps([{"tag": r.get("tag"), "verdict": r.get("verdict")} for r in collect(a.run_dir, a.plan)], indent = 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
