#!/usr/bin/env python3
"""Self test for the AMD layer, runnable on any host (no AMD GPU, no model, no push):

  1. the real amd_ci/lib/differential.py drives amd/diffusion_probe.py --dry over two states and both criteria
     modules: the observations must be valid JSON carrying the detected host, and the verdict must be
     INCONCLUSIVE, because a dry run (and, off the CI, a non-gfx1151 host) must never read as a result;
  2. the criteria logic on synthetic gfx1151 observations: a newly broken cell is a REGRESSION, an unchanged run is
     NO_REGRESSION, and in differential mode a base that does not show the defect is VOID while one that does and
     a head that clears it is CONFIRMED;
  3. scaffold_diffusion.py for Linux and Windows: the workflows lint clean, carry the probe, the criteria and the
     GPU concurrency group, and no secret.

  python diffusion_bench/amd/selftest_amd.py [--keep]     # outputs under $WORKSPACE/temp/dbench_amd/selftest
Exit 0 = pass.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLKIT = HERE.parent
AMD_CI = TOOLKIT.parent / "amd_ci"
sys.path.insert(0, str(TOOLKIT))

import common as C  # noqa: E402


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_obs(state: str, cells: dict, defect=None) -> dict:
    return {"state": state, "_state": state, "_probe_rc": 0, "dry": False, "defect": defect, "errors": [],
            "host": {"vendor": "amd", "hip": "7.13.99004", "torch": "2.11.0+rocm7.13.0", "arch": "gfx1151",
                     "is_gfx1151": True, "device_name": "AMD Radeon 8060S", "platform": "Linux"},
            "models": {"zimage_turbo": {"source": "downloaded"}},
            "spec": {"selected": sorted(cells)}, "cells": cells}


def cell(verdict="ok", new=10.0, steady=9.0, lpips=0.05, flags=None, **kw) -> dict:
    return {"verdict": verdict, "new_s": new, "steady_s": steady, "lpips": lpips, "ref": "zimg_bf16",
            "flags": flags or {}, "status": {"transformer_quant": kw.get("tq")}, "renders": [],
            "error": kw.get("error")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action = "store_true", help = "keep the previous output directory")
    args = ap.parse_args()
    root = C.WS / "temp" / "dbench_amd" / "selftest"
    if root.exists() and not args.keep:
        shutil.rmtree(root)
    root.mkdir(parents = True, exist_ok = True)
    failures: list = []

    # ------------------------------------------------------------------ 1. real differential.py, dry probe
    states = {"pr": 0, "merged": False, "commits": {"base": "0" * 40, "head": "1" * 40},
              "paths": {"base": str(TOOLKIT.parent), "head": str(TOOLKIT.parent)}}
    (root / "states.json").write_text(json.dumps(states), encoding = "utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    for crit in ("criteria_no_regression.py", "criteria_differential.py"):
        out = root / crit.replace(".py", "")
        extra = ["--dry", "--work", str(root / "work"), "--only", "tiny_*", "--only", "zimg_*", "--only", "wan_*"]
        if "differential" in crit:
            extra += ["--defect", "broken:zimg_fp8"]
        cmd = [sys.executable, str(AMD_CI / "lib" / "differential.py"), "--states", str(root / "states.json"),
               "--probe", str(HERE / "diffusion_probe.py"), "--criteria", str(HERE / crit), "--out-dir", str(out),
               "--", *extra]
        proc = subprocess.run(cmd, capture_output = True, text = True, env = env, timeout = 1800)
        (out / "differential.log").write_text(proc.stdout + proc.stderr, encoding = "utf-8")
        verdict = (json.loads((out / "verdict.json").read_text(encoding = "utf-8"))
                   if (out / "verdict.json").is_file() else {})
        if verdict.get("verdict") != "INCONCLUSIVE":
            failures.append(f"{crit}: dry run must be INCONCLUSIVE, got {verdict} (log {out / 'differential.log'})")
        for st in ("base", "head"):
            try:
                o = json.loads((out / f"obs_{st}.json").read_text(encoding = "utf-8"))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{crit}: obs_{st}.json unreadable: {exc}")
                continue
            n_ok = sum(1 for c in o.get("cells", {}).values() if c.get("verdict") == "ok")
            if not o.get("host", {}).get("vendor") or n_ok < 5 or o.get("errors"):
                failures.append(f"{crit}: obs_{st} incomplete: vendor={o.get('host', {}).get('vendor')} ok={n_ok} "
                                f"errors={o.get('errors')}")
            if not any(r.get("thumb") for c in o.get("cells", {}).values() for r in c.get("renders", [])):
                failures.append(f"{crit}: obs_{st} has no thumbnails")
        if "ran on ROCm gfx1151" not in (out / "VERDICT.md").read_text(encoding = "utf-8"):
            failures.append(f"{crit}: VERDICT.md does not show the gfx1151 gate")
        print(f"[1] {crit}: {verdict.get('verdict')} ({verdict.get('why')})")

    # ------------------------------------------------------------------ 2. criteria logic, synthetic observations
    diff = load(AMD_CI / "lib" / "differential.py", "amd_ci_differential")
    reg = load(HERE / "criteria_no_regression.py", "crit_reg")
    base_cells = {"zimg_bf16": cell(lpips = None), "zimg_bf16_repeat": cell(lpips = 0.0), "zimg_int8": cell(tq = "int8")}
    head_same = copy.deepcopy(base_cells)
    obs = {"base": fake_obs("base", base_cells), "head": fake_obs("head", head_same)}
    v, why = diff._decide(reg, obs, reg.gates(obs))
    if v != "NO_REGRESSION":
        failures.append(f"unchanged run should be NO_REGRESSION, got {v}: {why}")
    head_broken = copy.deepcopy(base_cells)
    head_broken["zimg_int8"] = cell("BROKEN", error = "RuntimeError: int8 load failed")
    obs = {"base": fake_obs("base", base_cells), "head": fake_obs("head", head_broken)}
    v, why = diff._decide(reg, obs, reg.gates(obs))
    if v != "REGRESSION":
        failures.append(f"newly broken cell should be REGRESSION, got {v}: {why}")
    head_slow = copy.deepcopy(base_cells)
    head_slow["zimg_int8"] = cell(new = 14.0, steady = 13.0, tq = "int8")
    obs = {"base": fake_obs("base", base_cells), "head": fake_obs("head", head_slow)}
    v, _ = diff._decide(reg, obs, reg.gates(obs))
    if v != "REGRESSION":
        failures.append(f"1.4x slower on both measures should be REGRESSION, got {v}")
    eb, eh = fake_obs("base", base_cells), fake_obs("head", copy.deepcopy(base_cells))
    eb["edge"] = {"checks": {"inproc/image.load": "PASS", "inproc/video.odd": "FAIL"}}
    eh["edge"] = {"checks": {"inproc/image.load": "FAIL", "inproc/video.odd": "FAIL"}}
    v, why = diff._decide(reg, {"base": eb, "head": eh}, reg.gates({"base": eb, "head": eh}))
    if v != "REGRESSION" or "image.load" not in why or "video.odd" in why.split("Also")[0]:
        failures.append(f"edge check PASS -> FAIL should be the (only) REGRESSION, got {v}: {why}")
    noisy = copy.deepcopy(base_cells)
    noisy["zimg_bf16_repeat"] = cell(lpips = 0.2)
    obs = {"base": fake_obs("base", noisy), "head": fake_obs("head", noisy)}
    v, _ = diff._decide(reg, obs, reg.gates(obs))
    if v != "INCONCLUSIVE":
        failures.append(f"a noise floor above the gate should be INCONCLUSIVE, got {v}")
    print("[2] regression criteria checks done")

    dcrit = load(HERE / "criteria_differential.py", "crit_diff")
    fixed_head = {**base_cells, "zimg_fp8": cell(tq = "fp8")}
    broken_base = {**base_cells, "zimg_fp8": cell("BROKEN", error = "fp8 refused")}
    ok_base = {**base_cells, "zimg_fp8": cell(tq = "fp8")}
    d = "broken:zimg_fp8"
    obs = {"base": fake_obs("base", ok_base, d), "head": fake_obs("head", fixed_head, d)}
    v, _ = diff._decide(dcrit, obs, dcrit.gates(obs))
    if v != "VOID":
        failures.append(f"base without the defect must be VOID, got {v}")
    obs = {"base": fake_obs("base", broken_base, d), "head": fake_obs("head", fixed_head, d),
           "merge": {"state": "merge", "skipped_state": True, "reason": "--skip-states merge", "_probe_rc": 0}}
    v, _ = diff._decide(dcrit, obs, dcrit.gates(obs))
    if v != "CONFIRMED":
        failures.append(f"base broken + head fixed should be CONFIRMED, got {v}")
    obs = {"base": fake_obs("base", broken_base, d), "head": fake_obs("head", broken_base, d)}
    v, _ = diff._decide(dcrit, obs, dcrit.gates(obs))
    if v != "FIX_INCOMPLETE":
        failures.append(f"broken at both should be FIX_INCOMPLETE, got {v}")
    slow_spec = "slow:zimg_fp8:zimg_bf16:0.9"
    obs = {"base": fake_obs("base", ok_base, slow_spec),
           "head": fake_obs("head", {**base_cells, "zimg_fp8": cell(new = 7.0, tq = "fp8")}, slow_spec)}
    v, _ = diff._decide(dcrit, obs, dcrit.gates(obs))
    if v != "CONFIRMED":
        failures.append(f"slow: base ratio 1.0 > 0.9 and head 0.7 should be CONFIRMED, got {v}")
    fast_base = {**base_cells, "zimg_fp8": cell(new = 7.0, tq = "fp8")}
    obs = {"base": fake_obs("base", fast_base, slow_spec), "head": fake_obs("head", fast_base, slow_spec)}
    v, _ = diff._decide(dcrit, obs, dcrit.gates(obs))
    if v != "VOID":
        failures.append(f"slow: a base already at 0.7 does not show the defect, must be VOID, got {v}")
    log_spec = "log:zimg_fp8:fell back:fused RoPE on"
    def lc(lines, trunc = False):
        return {**base_cells, "zimg_fp8": {**cell(tq = "fp8"), "log_lines": lines, "log_truncated": trunc}}
    fell, fixed = lc(["x fell back to eager"]), lc(["x fused RoPE on"])
    for name, b, h, want in (("base falls back, head engages", fell, fixed, "CONFIRMED"),
                             ("both fall back", fell, fell, "FIX_INCOMPLETE"),
                             ("head neither falls back nor engages", fell, lc(["x attention sdpa"]),
                              "FIX_INCOMPLETE"),
                             ("head's kept lines were truncated", fell, lc(["x fused RoPE on"], True),
                              "FIX_INCOMPLETE"),
                             ("base never fell back", fixed, fixed, "VOID"),
                             ("no kept log lines at head", fell, lc([]), "INCONCLUSIVE")):
        obs = {"base": fake_obs("base", b, log_spec), "head": fake_obs("head", h, log_spec)}
        v, _ = diff._decide(dcrit, obs, dcrit.gates(obs))
        if v != want:
            failures.append(f"log: {name} should be {want}, got {v}")
    print("[2] differential criteria checks done")

    # ------------------------------------------------------------------ 3. scaffold, Linux and Windows
    for windows in (False, True):
        out = root / ("ci_win" if windows else "ci_linux")
        cmd = [sys.executable, str(HERE / "scaffold_diffusion.py"), "--pr", "11766", "--out", str(out),
               "--only", "zimg_*"]
        if windows:
            cmd.append("--windows")
        proc = subprocess.run(cmd, capture_output = True, text = True, timeout = 600)
        (root / f"scaffold_{'win' if windows else 'linux'}.log").write_text(proc.stdout + proc.stderr,
                                                                           encoding = "utf-8")
        wfs = list((out / ".github" / "workflows").glob("*.yml"))
        text = wfs[0].read_text(encoding = "utf-8") if wfs else ""
        want = ["diffusion_bench/amd/diffusion_probe.py", "criteria_no_regression.py",
                "amd-ci-gfx1151-gpu-windows" if windows else "amd-ci-gfx1151-gpu", "timeout-minutes: 330",
                "'zimg_*'"]
        missing = [w for w in want if w not in text]
        if proc.returncode != 0 or missing or "secrets." in text or not (out / "diffusion_bench" / "run_cell.py").is_file():
            failures.append(f"scaffold {'windows' if windows else 'linux'}: rc {proc.returncode}, missing {missing}, "
                            f"secrets {'secrets.' in text}; see {root}")
        print(f"[3] scaffold {'windows' if windows else 'linux'}: rc {proc.returncode}, missing {missing or 'none'}")

    for f in failures:
        print("FAIL:", f)
    print("amd selftest", "FAILED" if failures else "passed", f"({root})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
