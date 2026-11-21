#!/usr/bin/env python3
"""End-to-end CPU test of the one process, with the fake backend on two targets:

  plan.py (a real model card, fake backend, colab:L4 + colab:A100-40)
    -> matrix.py --plan PLAN --vm <id> per VM (trial, parity probes + parity gate, ABBA timed cells)
    -> targets collect --plan (cases.json staged into the results dir)
    -> report.py RESULTS (and --expect)

Asserts: every planned case has a record; the parity gate ran on both targets and reported the card's accepted
precision difference without voiding the pair; report.py prints a FASTER / SLOWER / TIE verdict per target (FASTER
here: the fake Studio arm steps 3x quicker); report.py --expect with an extra tag refuses. Unit checks on the same
card: a precision difference the card does NOT accept voids the pair, a VOID group's timed cells are gated out, and
a cfg pair the card maps (guidance_semantics) is equal.

  python diffusion_bench/selftest_e2e.py [--keep DIR]        exit 0 = pass
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import parity as PG  # noqa: E402

FAMILY, FMT = "flux.1-schnell", "bf16"
FAKE_GPU = {"name": "FakeGPU", "uuid": "GPU-fake", "used_mib": 1000, "own_mib": 800, "sm_mhz": 1500, "temp_c": 50,
            "power_w": 100, "cc": "8.9", "total_mib": 24576}
FAILS: list = []


def check(cond, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg, flush = True)
    if not cond:
        FAILS.append(msg)


def request() -> dict:
    py = sys.executable
    return {"name": "e2e", "revisions": {"unsloth_sha": "0" * 40, "zoo_sha": "1" * 40, "bline_ref": "v0.45.6"},
            "models": [{"family": FAMILY, "formats": [FMT]}],
            "arms": [{"stack": "studio", "format": FMT, "backend": "fake", "venv": py,
                      "options": {"step_delay_s": 0.1}},
                     {"stack": "blineui", "format": FMT, "backend": "fake", "venv": py,
                      "options": {"step_delay_s": 0.3, "emulate": "blineui"}}],
            "targets": ["colab:L4", "colab:A100-40"], "repeats": 3, "n": 2, "warmup": 1, "negative_control": True}


def run(*args, env=None) -> subprocess.CompletedProcess:
    env = dict(env if env is not None else os.environ)
    # "-m targets.<kind>" needs the toolkit on sys.path: cwd alone is not there under PYTHONSAFEPATH=1
    env["PYTHONPATH"] = os.pathsep.join([str(HERE)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    return subprocess.run([sys.executable, *map(str, args)], capture_output = True, text = True, env = env,
                          cwd = str(HERE))


def e2e(work: Path) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "DBENCH_FAKE_GPU": json.dumps(FAKE_GPU),
           "WORKSPACE": str(work)}
    req = work / "request.json"
    req.write_text(json.dumps(request()))
    plan_path = work / "plan" / "plan.json"
    r = run(HERE / "plan.py", req, "--out", plan_path, env = env)
    check(r.returncode == 0 and (plan_path.parent / "cases.json").exists(), f"plan.py writes PLAN + cases.json "
                                                                             f"({r.stderr[-300:]})")
    plan = json.loads(plan_path.read_text())
    vms = [v["vm_id"] for v in plan["vms"]]
    check(len(vms) == 2 and {v["target"] for v in plan["vms"]} == {"colab:L4", "colab:A100-40"}, f"two VMs {vms}")
    check(all(c.get("parity_pairs") for c in plan["comparisons"]), "every comparison has a parity pair")
    results = work / "results"
    for vm in vms:
        r = run(HERE / "matrix.py", "--plan", plan_path, "--vm", vm, "--out", results, "--no-gate", env = env)
        check(r.returncode == 0, f"matrix.py --plan --vm {vm} exit {r.returncode}"
                                 + (f": {r.stdout[-400:]}{r.stderr[-800:]}" if r.returncode else ""))
    r = run("-m", "targets.colab", "collect", results, "--plan", plan_path, env = env)
    check(r.returncode == 0 and (results / "cases.json").exists(), f"collect --plan stages cases.json ({r.stderr[-300:]})")
    # the launcher runs the same adapter from any cwd with safe-path mode on
    r = subprocess.run([sys.executable, str(HERE / "target.py"), "colab", "--help"], capture_output = True, text = True,
                       env = {**env, "PYTHONSAFEPATH": "1"}, cwd = str(work))
    check(r.returncode == 0 and "collect" in r.stdout, f"target.py colab --help under PYTHONSAFEPATH=1 ({r.stderr[-300:]})")
    pj = sorted(results.rglob("parity.json"))
    check(len(pj) == 2, f"parity gate wrote {len(pj)} parity.json (one per target)")
    for p in pj:
        res = json.loads(p.read_text())
        acc = [x["id"] for x in res.get("accepted_differences") or []]
        check(res["verdict"] == "ok" and acc and res["checks"]["precision"]["status"] == "ok",
              f"{res['group']['target']}: parity ok with accepted precision difference(s) {acc} "
              f"({res.get('reasons')})")
        check("traps_unchecked" not in res["checks"] or "precision" not in res["checks"]["traps_unchecked"]["detail"],
              "precision traps map to a gate check (models.GATE_CHECK)")
    # negative control: the same Studio probe with a changed step count must come back VOID against the baseline
    # probe on every VM, without blocking the matched pair's timed cells
    nj = sorted(results.rglob("parity_negctl.json"))
    negs = [json.loads(p.read_text()) for p in nj]
    check(len(nj) == 2 and all(n["verdict"] == "VOID" and n["control_verdict"] == "rejected"
                               and any("steps" in x for x in n["reasons"]) for n in negs),
          f"parity negative control (steps changed) rejected on both VMs ({[n.get('reasons') for n in negs]})")
    # a control the gate PASSES (an identical record) means the gate cannot tell the pair apart: fail closed
    neg_pair = next(pr for pr in PG.plan_pairs(plan) if pr.get("expect") == "VOID")
    same = work / "same"
    for tag, src in ((neg_pair["a"], next(pr for pr in PG.plan_pairs(plan) if pr["comparison"] == neg_pair["comparison"]
                                         and not pr.get("expect"))["a"]), (neg_pair["b"], neg_pair["b"])):
        (same / tag).mkdir(parents = True, exist_ok = True)
        shutil.copy(results / src / "record.json", same / tag / "record.json")
    g = PG.gate_plan(plan, same, write = False)
    check((neg_pair["comparison"], neg_pair["format"]) in g["void"],
          "a negative control the gate passes voids its group (fail closed)")
    r = run(HERE / "report.py", results, "--no-lpips", env = env)
    check(r.returncode == 0, f"report.py exit {r.returncode} {r.stderr[-500:]}")
    rep = json.loads((results / "report.json").read_text()) if (results / "report.json").exists() else {}
    verdicts = {c["target"]: [x["verdict"] for x in c["comparisons"]] for c in rep.get("cards") or []}
    check(rep.get("complete") and len(verdicts) == 2, f"report complete over both targets {verdicts}")
    check(all(v and v[0] in ("FASTER", "SLOWER", "TIE") for v in verdicts.values()),
          f"FASTER / SLOWER / TIE per target: {verdicts}")
    check(all(v == ["FASTER"] for v in verdicts.values()), "the 3x quicker fake Studio arm is FASTER on both")
    md = (results / "report.md").read_text() if (results / "report.md").exists() else ""
    check("Parity negative control (steps" in md and ": rejected" in md, "report.md states the negative control")
    check("Accepted precision differences" in md and "accepted_precision_0" in md,
          "report.md lists the accepted precision difference")
    extra = work / "expect.json"
    extra.write_text(json.dumps(json.loads((results / "cases.json").read_text())["cases"] + [{"tag": "not_run_x"}]))
    r = run(HERE / "report.py", results, "--no-lpips", "--expect", extra, "--out", work / "x.md", "--json",
            work / "x.json", env = env)
    check(r.returncode == 2 and "not_run_x" in r.stderr, "report.py --expect with a missing case refuses (exit 2)")


def units() -> None:
    import models as M

    card = M.load_card(FAMILY)

    def rec(stack: str, prec: dict, cfg: float) -> dict:
        return {"tag": stack, "verdict": "ok", "arm": {"stack": stack},
                "cell": {"card": FAMILY, "variant": FMT, "options": {}}, "target": {"name": "colab:L4"},
                "parity": {"precision": prec, "resolved": {"steps": 4, "cfg": cfg, "width": 1024, "height": 1024}}}

    a, b = rec("studio", {}, card["studio"]["gen"]["guidance"]), rec("blineui", {}, card["defaults"]["cfg"])
    res = PG.check_pair(a, b, card, waive = ("sigmas", "passes", "size", "cache"))
    check(res["checks"]["settings"]["status"] == "ok" and "guidance_semantics" in res["checks"]["settings"]["detail"],
          f"cfg mapped by the card ({res['checks']['settings']['detail']})")
    res = PG.check_pair(a, rec("blineui", {}, 3.5), card, waive = ("sigmas", "passes", "size", "cache"))
    check(res["checks"]["settings"]["status"] == "mismatch", "a cfg pair the card does not map still voids")
    bad = copy.deepcopy(card)
    bad["formats"][FMT]["bline"]["precision"]["transformer"]["weights"] = "fp8_e4m3fn"
    res = PG.check_pair(a, b, bad, waive = ("sigmas", "passes", "size", "cache"))
    check(res["verdict"] == "VOID" and res["checks"]["precision"]["status"] == "mismatch",
          "an unaccepted transformer weights difference voids the pair")
    nul = copy.deepcopy(card)
    nul["formats"][FMT]["bline"]["precision"]["transformer"]["compute"] = None
    nul["formats"][FMT]["studio"]["precision"]["transformer"]["compute"] = "fp8"
    res = PG.check_pair(a, b, nul, waive = ("sigmas", "passes", "size", "cache"))
    check(res["checks"]["precision"]["status"] == "ok", "compute compared only when both sides state it")
    strong = rec("blineui", {"transformer": {"weights": "fp8_e4m3fn", "compute": "fp8"}}, card["defaults"]["cfg"])
    res = PG.check_pair(a, strong, card, waive = ("sigmas", "passes", "size", "cache"))
    check(res["checks"].get("precision_observed", {}).get("status") == "mismatch",
          "a backend's stated precision that contradicts its card side voids the pair")
    # fp16-only card (T4): the dense dtype is planned, not applied; each stack's own report must agree
    t4a = {**a, "cell": {**a["cell"], "options": {"dense_dtype": "fp16"}}, "status": {"dtype": "torch.float32"}}
    t4b = {**b, "cell": {**b["cell"], "options": {"dense_dtype": "fp16"}},
           "status": {"facts": {"model_weight_dtype": "torch.float16", "manual_cast": "None"}}}
    res = PG.check_pair(t4a, t4b, card, waive = ("sigmas", "passes", "size", "cache"))
    check(res["verdict"] == "VOID" and res["checks"]["dense_compute"]["status"] == "mismatch",
          f"T4: Studio fp32 vs baseline fp16 transformer compute voids the pair ({res['checks'].get('dense_compute')})")
    t4a["status"] = {"dtype": "torch.float16"}
    res = PG.check_pair(t4a, t4b, card, waive = ("sigmas", "passes", "size", "cache"))
    check(res["checks"]["dense_compute"]["status"] == "ok", "T4: both stacks report fp16 compute: dense_compute ok")
    t4b["status"] = {}
    res = PG.check_pair(t4a, t4b, card, waive = ("sigmas", "passes", "size", "cache"))
    check(res["verdict"] == "VOID" and res["checks"]["dense_compute"]["status"] == "missing",
          "T4: an unreported compute dtype is missing evidence (VOID), never a pass")
    check("dense_compute" not in PG.check_pair(a, b, card, waive = ("sigmas", "passes", "size", "cache"))["checks"],
          "dense_compute only applies to fp16-only runs")
    ta = {**a, "gpu": {"cc": "7.5"}, "status": {"dtype": "torch.float32"}}
    tb = {**b, "gpu": {"cc": "7.5"}, "status": {"facts": {"model_weight_dtype": "torch.float16", "manual_cast": "None"}}}
    res = PG.check_pair(ta, tb, card, waive = ("sigmas", "passes", "size", "cache"))
    check(res["checks"].get("dense_compute", {}).get("status") == "mismatch",
          "an unplanned bf16 pair on a cc 7.5 device (tester) still gets the dense compute check")
    # sigma tolerance: relative above 1 (SDXL's 14.6146469 vs 14.6146402 is float32 rounding), absolute below
    check(PG.sigma_delta(14.614646911621094, 14.614640235900879) <= PG.SIGMA_TOL
          and PG.sigma_delta(14.6146, 14.5) > PG.SIGMA_TOL and PG.sigma_delta(0.5, 0.500002) > PG.SIGMA_TOL,
          "sigmas: SDXL float32 rounding is equal, a real schedule difference is not")
    # Studio states its encoder decision (resolved.text_encoder_quant): an automatic int8 encoder that fell back to
    # the dense one (hosted file unavailable offline) is strong evidence and voids an int8 pair
    from backends.studio_inproc import studio_precision  # noqa: PLC0415

    fl = M.load_card("flux.3-klein")
    st_off = {"dtype": "bfloat16", "transformer_quant": "int8", "resolved": {"text_encoder_quant": {
        "value": "off", "status": "applied", "reason": "the hosted int8 ConvRot text encoder is not available"}}}
    sa = {"tag": "s", "verdict": "ok", "arm": {"stack": "studio"}, "cell": {"card": "flux.3-klein", "variant": "int8",
          "options": {}}, "target": {"name": "colab:L4"}, "parity": {"precision": studio_precision(st_off),
          "resolved": {"steps": 16, "cfg": 1.0, "width": 1024, "height": 1024}}}
    sb = {"tag": "c", "verdict": "ok", "arm": {"stack": "blineui"}, "cell": {"card": "flux.3-klein", "variant": "int8",
          "options": {}}, "target": {"name": "colab:L4"}, "parity": {"precision": {}, "resolved": {
          "steps": 16, "cfg": 1.0, "width": 1024, "height": 1024}}}
    res = PG.check_pair(sa, sb, fl, waive = ("sigmas", "passes", "size", "cache", "settings"))
    po = res["checks"].get("precision_observed") or {}
    check(res["verdict"] == "VOID" and po.get("status") == "mismatch" and "text_encoder" in po.get("detail", ""),
          f"Studio's dense-encoder fallback on an int8 pair voids it ({po})")
    st_on = {**st_off, "resolved": {"text_encoder_quant": {"value": "int8", "status": "applied", "reason": "auto"}}}
    sa["parity"]["precision"] = studio_precision(st_on)
    res = PG.check_pair(sa, sb, fl, waive = ("sigmas", "passes", "size", "cache", "settings"))
    check((res["checks"].get("precision_observed") or {}).get("status") != "mismatch",
          "Studio's applied int8 encoder matches the int8 card")
    doc = {"comparisons": [{"id": "x", "parity_pairs": [{"format": FMT, "a": "p1", "b": "p2"}], "family": FAMILY}]}
    cell = {"options": {"arm": {"comparison": "x", "format": FMT, "purpose": "timing"}}}
    check(PG.gated_out(doc, cell, [("x", FMT)]) and not PG.gated_out(doc, cell, []),
          "a VOID parity group gates out its timed cells")


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", default = None, help = "work dir to keep (default: a temp dir, removed)")
    a = ap.parse_args()
    root = Path(os.environ["WORKSPACE"]) / "temp" if os.environ.get("WORKSPACE") else None
    if root:
        root.mkdir(parents = True, exist_ok = True)
    work = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix = "dbench_e2e_", dir = root))
    work.mkdir(parents = True, exist_ok = True)
    try:
        units()
        e2e(work)
    finally:
        if not a.keep:
            shutil.rmtree(work, ignore_errors = True)
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
