#!/usr/bin/env python3
"""CPU-only self test, no model, no GPU (a fake device through DBENCH_FAKE_GPU). Exit 0 = pass.

  1. matrix.py + run_cell.py + score.py on specs/selftest_fake.json: records, resume, passes, video, scores
  2. setup_studio / bline_graphs plumbing checks
  3. the v2 path end to end: parity probes -> parity.py -> timed ABBA-style repeats -> report.py, which must call
     Studio FASTER than the slower fake BlineUI and score the int8 arm against each stack's own bf16
  4. negative controls the toolkit MUST reject: mismatched sigmas, a changed step count, a BlineUI cache hit, a
     lever not engaged (and one with no evidence), a missing reference, a missing VRAM metric, a cross-VM pair,
     non-finite JSON, an incomplete case set, an NVIDIA capability the device lacks, derived s/step under a step
     cache; plus the shim's simulated total / parity hooks and the cross-platform process kill
Needs numpy + pillow (+ psutil); LPIPS is not used (no weight download)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402

FAKE_GPU = {"name": "FakeGPU", "uuid": "GPU-fake", "used_mib": 1000, "own_mib": 800, "sm_mhz": 1500, "temp_c": 50,
            "power_w": 100, "power_limit_w": 300, "cc": "8.9", "total_mib": 24576}


def wipe(path: Path) -> None:
    for p in sorted(path.rglob("*"), reverse = True):
        p.unlink() if (p.is_file() or p.is_symlink()) else p.rmdir()


def legacy_checks(out: Path, failures: list) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "DBENCH_FAKE_GPU": json.dumps(FAKE_GPU)}
    run = lambda *a: subprocess.run([sys.executable, *map(str, a)], env = env, capture_output = True, text = True)  # noqa: E731
    first = run(HERE / "matrix.py", HERE / "specs/selftest_fake.json", "--out", out, "--no-gate")
    if first.returncode != 1:
        failures.append(f"matrix should exit 1 with one BROKEN cell, got {first.returncode}\n{first.stdout[-2000:]}")
    rec = C.read_record(out / "p1" / "ref")
    if not C.record_ok(rec) or len(rec["renders"]) != 4 or rec.get("step_s_derived") is None:
        failures.append(f"ref record incomplete: {rec and {k: rec.get(k) for k in ('verdict', 'reason', 'step_s_derived')}}")
    if (C.read_record(out / "p1" / "broken") or {}).get("verdict") != "BROKEN":
        failures.append("broken cell not recorded as BROKEN")
    if len((C.read_record(out / "p2" / "ref") or {}).get("renders", [])) != 2:
        failures.append("pass override n=2 not applied")
    if not (out / "p1" / "video" / "v0.npz").exists():
        failures.append("video frames not saved")
    second = run(HERE / "matrix.py", HERE / "specs/selftest_fake.json", "--out", out, "--no-gate", "--only", "ref")
    if "ok record exists" not in second.stdout:
        failures.append("resume did not skip an ok cell")
    sc = run(HERE / "score.py", out, "--no-lpips")
    if sc.returncode != 0:
        failures.append(f"score failed: {sc.stderr[-2000:]}")
    else:
        rows = {r["tag"]: r for r in json.loads((out / "p1" / "scores.json").read_text())["rows"]}
        if rows["same"]["identical"] != "4/4":
            failures.append(f"same-seed fake cell should be identical to ref: {rows['same']}")
        if rows["noisy"]["identical"] != "0/4" or (rows["noisy"]["psnr"] or 99) >= 99:
            failures.append(f"noisy cell should differ from ref: {rows['noisy']}")


def plumbing_checks(failures: list) -> None:
    import setup_studio as S
    for bad, exc in (("/nonexistent/unsloth", FileNotFoundError), ("./missing_checkout", FileNotFoundError),
                     ("not a ref", ValueError)):
        try:
            S.resolve_tree(bad)
            failures.append(f"setup_studio.resolve_tree({bad!r}) did not raise")
        except exc:
            pass
        except Exception as e:  # noqa: BLE001
            failures.append(f"setup_studio.resolve_tree({bad!r}) raised {type(e).__name__}: {e}")
    try:
        import bline_graphs as G
    except Exception as e:  # noqa: BLE001
        failures.append(f"bline_graphs import: {e}")
        return
    h3 = {"dit": "d", "te": "t", "vae": "v", "audio_vae": "a"}
    try:
        for fam in ("minimax-h3", "h3", "minimax_h3"):
            g = G.build(fam, G.GraphParams(prompt = "x", seed = 0, steps = 2, frames = 141, files = h3, nonce = 1))
            if g["cond"]["inputs"]["length"] != 141 or "adecode" not in g:
                failures.append(f"bline_graphs {fam}: unexpected graph {sorted(g)}")
        qf = {"dit": "d", "te": "t", "vae": "v"}
        plain = G.build("qwen21", G.GraphParams(prompt = "x", seed = 0, steps = 2, files = qf))
        if "prefix_cache" in plain:
            failures.append("bline_graphs qwen-image-2.1 added a prefix cache node without graph_extra.prefix_cache")
    except Exception as e:  # noqa: BLE001 - the graphs belong to the model-card toolkit; report, do not crash
        failures.append(f"bline_graphs checks raised {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------------------------- v2 helpers
def cell(tag: str, stack: str, variant: str = "bf16", vm: str = "vm1", delay: float = 0.004, block: int = 1,
         n: int = 3, **opts) -> dict:
    o = {"target": {"name": "colab:L4", "vm_id": vm}, "arm": {"stack": stack, "block": block, "order": "ABBA"},
         "step_delay_s": delay, "dtype": "int8" if variant == "int8" else "bf16", "foreign_wait": False}
    if stack == "blineui":
        o["emulate"] = "blineui"
    if variant == "int8":
        o["noise"] = 0.02
    o.update(opts)
    return C.merge_cell({}, {"tag": tag, "backend": "fake", "family": "fakefam", "variant": variant, "width": 64,
                             "height": 64, "steps": 6, "short_steps": 2, "short_n": 1, "n": n, "warmup": 1,
                             "prompts": "image_24", "options": o})


def run_cell(c: dict, out: Path, parity: bool = False, fake_gpu: dict | None = FAKE_GPU) -> dict:
    import run_cell as R

    saved = os.environ.get("DBENCH_FAKE_GPU")
    if fake_gpu is None:
        os.environ.pop("DBENCH_FAKE_GPU", None)
    else:
        os.environ["DBENCH_FAKE_GPU"] = json.dumps(fake_gpu)
    try:
        return R.run(c, out, quiet = True, parity = parity)
    finally:
        if saved is None:
            os.environ.pop("DBENCH_FAKE_GPU", None)
        else:
            os.environ["DBENCH_FAKE_GPU"] = saved


def write_parity(a: dict, b: dict, dest: Path, quality: bool = False) -> dict:
    import parity as P
    import schema

    res = P.check_pair(a, b, None, quality)
    dest.mkdir(parents = True, exist_ok = True)
    (dest / "parity.json").write_text(schema.dumps(res))
    return res


def v2_checks(root: Path, failures: list) -> None:
    import parity as P
    import report as Rp
    import schema

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    pos = root / "positive"
    cases = []
    # parity first, per variant, quality pairs included
    for var in ("bf16", "int8"):
        a = run_cell(cell(f"s_par_{var}", "studio", var), pos / "_parity", parity = True)
        b = run_cell(cell(f"c_par_{var}", "blineui", var), pos / "_parity", parity = True)
        res = write_parity(a, b, pos / "_parity" / var, quality = True)
        if res["verdict"] != "ok":
            failures.append(f"positive parity {var} should pass: {res['reasons']}")
    # a purpose "parity" plan's cases are parity-mode records: present ones count toward completeness (and stay
    # out of the timing tables)
    par_cases = [f"{s}_par_{v}" for v in ("bf16", "int8") for s in ("s", "c")]
    (pos / "_parity" / "cases.json").write_text(json.dumps(par_cases))
    prep = Rp.build(pos / "_parity", use_lpips = False)
    if not prep["complete"] or prep["missing"]:
        failures.append(f"report: parity-only records must count toward completeness, missing {prep['missing']}")
    (pos / "_parity" / "cases.json").write_text(json.dumps(par_cases + ["s_par_absent"]))
    if Rp.build(pos / "_parity", use_lpips = False)["missing"] != ["s_par_absent"]:
        failures.append("report: a parity-only case with no record must still be missing")
    (pos / "_parity" / "cases.json").unlink()
    # timed: 3 fresh-process repeats per arm (A/A band), interleaved S C C S
    for var in ("bf16", "int8"):
        for blk in (1, 2, 3):
            order = [("studio", 0.004), ("blineui", 0.012), ("blineui", 0.012), ("studio", 0.004)]
            for i, (st, dl) in enumerate(order[:2] if blk != 2 else order[2:]):
                tag = f"{st[0]}_{var}_b{blk}"
                rec = run_cell(cell(tag, st, var, delay = dl, block = blk), pos)
                cases.append(tag)
                if rec["verdict"] != "ok":
                    failures.append(f"positive {tag} not ok: {rec.get('reason')}")
    (pos / "cases.json").write_text(json.dumps(cases))
    rep = Rp.build(pos, use_lpips = False)
    cards = {(c["variant"]): c for c in rep["cards"]}
    bf = cards.get("bf16", {}).get("comparisons") or [{}]
    if not any("lpips" in x for x in rep["could_not_test"]):
        failures.append("report: LPIPS disabled but not listed under could not test")
    if not rep["complete"] or bf[0].get("verdict") != "FASTER" or bf[0].get("valid_pairs") != 9:
        failures.append(f"report positive path: complete={rep['complete']} bf16 comparison {bf[0]}")
    i8 = cards.get("int8", {}).get("stacks", {})
    if not i8 or any(i8[s]["psnr"] is None for s in ("studio", "blineui")):
        failures.append(f"report: int8 quality vs own bf16 missing: { {s: v.get('psnr') for s, v in i8.items()} }")
    st = cards.get("bf16", {}).get("stacks", {}).get("studio", {})
    for key in ("cold_s", "first_step_s", "step_polled", "vram_own", "rss_tree"):
        if st.get(key) is None:
            failures.append(f"report: studio bf16 {key} is UNAVAILABLE on the positive path")
    md = Rp.markdown(rep)
    if "FASTER" not in md or "## Coverage" not in md or "## Could not test" not in md:
        failures.append("report markdown is missing the verdict / coverage / could-not-test sections")
    cli = subprocess.run([sys.executable, str(HERE / "report.py"), str(pos), "--no-lpips"], capture_output = True,
                         text = True, env = {**os.environ})
    if cli.returncode != 0 or not (pos / "report.md").exists():
        failures.append(f"report.py CLI failed on a complete set: {cli.returncode} {cli.stderr[-500:]}")

    neg = root / "negative"
    # 1. mismatched sigmas / 2. changed step count
    s_par = run_cell(cell("s_par", "studio"), neg / "par", parity = True)
    for tag, opt, must in (("c_sig", {"sigma_shift": 0.01}, "sigmas"), ("c_steps", {"steps_offset": 1}, "settings")):
        c_par = run_cell(cell(tag, "blineui", **opt), neg / "par", parity = True)
        res = P.check_pair(s_par, c_par)
        if res["verdict"] != "VOID" or not any(r.startswith(must) for r in res["reasons"]):
            failures.append(f"negative control {tag}: parity must be VOID on {must}, got {res['verdict']} "
                            f"{res['reasons']}")
    # canvas mismatch: one stack renders at half size
    c_half = run_cell(cell("c_half", "blineui", downscale = 2), neg / "par", parity = True)
    res = P.check_pair(s_par, c_half)
    if res["verdict"] != "VOID" or not any(r.startswith("size") for r in res["reasons"]):
        failures.append(f"negative control canvas mismatch: {res['verdict']} {res['reasons']}")
    # GPU-drawn noise on a quality pair
    c_gpu = run_cell(cell("c_gpunoise", "blineui", noise_device = "cuda"), neg / "par", parity = True)
    if P.check_pair(s_par, c_gpu, quality = True)["verdict"] != "VOID":
        failures.append("negative control: GPU-drawn noise on a quality pair was not rejected")
    # card expected_passes: a card that expects CFG passes the fake does not make
    res = P.check_pair(s_par, s_par, {"expected_passes": "steps * 2"})
    if res["verdict"] != "VOID":
        failures.append("negative control: model calls != card expected_passes not rejected")
    if P.expected_passes("__import__('os').system('true')", {}) is not None:
        failures.append("expected_passes evaluated an unsafe expression")
    # 3. BlineUI cache hit
    hit = run_cell(cell("c_hit", "blineui", cache_hit = True), neg / "timed")
    if hit["verdict"] != "VOID" or "cache" not in (hit.get("reason") or ""):
        failures.append(f"negative control cache hit: {hit['verdict']} {hit.get('reason')}")
    # 4. lever not engaged / no evidence
    for tag, eng in (("s_lever_off", {"sage_attention": False}), ("s_lever_none", {})):
        r = run_cell(cell(tag, "studio", levers = {"sage_attention": True}, engaged = eng), neg / "timed")
        if r["verdict"] != "VOID" or "sage_attention" not in (r.get("reason") or ""):
            failures.append(f"negative control {tag}: {r['verdict']} {r.get('reason')}")
    ok_lever = run_cell(cell("s_lever_on", "studio", levers = {"sage_attention": True},
                             engaged = {"sage_attention": True}), neg / "timed")
    if ok_lever["verdict"] != "ok":
        failures.append(f"engaged lever wrongly rejected: {ok_lever.get('reason')}")
    # 5. missing reference
    mref = neg / "noref"
    run_cell(cell("s_int8_only", "studio", "int8"), mref)
    rep = Rp.build(mref, use_lpips = False)
    if not any("missing" in x and "reference" in x for x in rep["could_not_test"]):
        failures.append(f"negative control missing reference not reported: {rep['could_not_test']}")
    if rep["cards"][0]["stacks"]["studio"]["psnr"] is not None:
        failures.append("missing reference produced a quality number")
    # 6. missing VRAM metric
    novram = {k: v for k, v in FAKE_GPU.items() if k not in ("used_mib", "own_mib")}
    r = run_cell(cell("s_novram", "studio"), neg / "timed", fake_gpu = novram)
    if r["verdict"] != "VOID" or "VRAM" not in (r.get("reason") or ""):
        failures.append(f"negative control missing VRAM metric: {r['verdict']} {r.get('reason')}")
    # 7. cross-VM pair (parity ok, timing on two VMs)
    xvm = neg / "xvm"
    a = run_cell(cell("s_px", "studio"), xvm / "_parity", parity = True)
    b = run_cell(cell("c_px", "blineui"), xvm / "_parity", parity = True)
    write_parity(a, b, xvm / "_parity" / "bf16")
    for blk in (1, 2):
        run_cell(cell(f"s_x{blk}", "studio", vm = "vmA", block = blk), xvm)
        run_cell(cell(f"c_x{blk}", "blineui", vm = "vmB", delay = 0.012, block = blk), xvm)
    rep = Rp.build(xvm, use_lpips = False)
    comp = rep["cards"][0]["comparisons"][0]
    if comp["verdict"] != "VOID" or not any("cross-VM" in w for w in comp["invalid"]):
        failures.append(f"negative control cross-VM pair: {comp['verdict']} {comp['invalid'][:2]}")
    # clocks more than 3% apart on one VM, the slow arm also drawing less power (a throttled / contended device;
    # at equal power the difference is the workload's own on a power-capped card and is not voided)
    clk = neg / "clk"
    a = run_cell(cell("s_pk", "studio"), clk / "_parity", parity = True)
    b = run_cell(cell("c_pk", "blineui"), clk / "_parity", parity = True)
    write_parity(a, b, clk / "_parity" / "bf16")
    run_cell(cell("s_k", "studio"), clk)
    run_cell(cell("c_k", "blineui", delay = 0.012), clk, fake_gpu = {**FAKE_GPU, "sm_mhz": 1400, "power_w": 80})
    comp = Rp.build(clk, use_lpips = False)["cards"][0]["comparisons"][0]
    if comp["verdict"] != "VOID" or not any("3%" in w for w in comp["invalid"]):
        failures.append(f"negative control clocks > 3% apart: {comp['verdict']} {comp['invalid'][:2]}")
    # end to end: the same clocks at the same cap and the same draw are kept and the report prints both caps and
    # draws; the same pair at a different cap (equal draw) is still voided
    for name, gpu_o, keep in (("eqpow", {**FAKE_GPU, "sm_mhz": 1400}, True),
                              ("capdiff", {**FAKE_GPU, "sm_mhz": 1400, "power_limit_w": 250}, False),
                              ("belowcap", {**FAKE_GPU, "sm_mhz": 1400, "power_w": 70, "throttle_mask": 0}, "below"),
                              ("thermal", {**FAKE_GPU, "sm_mhz": 1400, "power_w": 70, "throttle_mask": 0x40}, False)):
        d = neg / f"clk_{name}"
        a = run_cell(cell("s_pk", "studio"), d / "_parity", parity = True)
        b = run_cell(cell("c_pk", "blineui"), d / "_parity", parity = True)
        write_parity(a, b, d / "_parity" / "bf16")
        run_cell(cell("s_k", "studio"), d, fake_gpu = {**FAKE_GPU, "throttle_mask": 0} if keep == "below" or
                 name == "thermal" else FAKE_GPU)
        run_cell(cell("c_k", "blineui", delay = 0.012), d, fake_gpu = gpu_o)
        rep = Rp.build(d, use_lpips = False)
        comp = rep["cards"][0]["comparisons"][0]
        md = Rp.markdown(rep)
        if keep == "below":
            if comp["valid_pairs"] != 1 or "Clock rule, below cap" not in md or \
                    [x.get("kind") for x in comp.get("clock_excused") or []] != ["below cap"]:
                failures.append(f"below-cap pair not kept / not listed end to end: {comp['valid_pairs']} "
                                f"{comp.get('clock_excused')} {comp['invalid'][:1]}")
            continue
        if name == "thermal":
            if comp["verdict"] != "VOID" or not any("throttled" in w for w in comp["invalid"]):
                failures.append(f"negative control (hw_thermal throttle below cap) passed: {comp['verdict']} "
                                f"{comp['invalid'][:1]}")
            continue
        if keep and (comp["valid_pairs"] != 1 or not comp.get("clock_excused") or "cap 300 vs 300 W" not in md):
            failures.append(f"equal-power pair not kept / not reported end to end: {comp['valid_pairs']} "
                            f"{comp.get('clock_excused')} {comp['invalid'][:1]}")
        if not keep and (comp["verdict"] != "VOID" or not any("power caps differ" in w for w in comp["invalid"])):
            failures.append(f"negative control (different power cap at equal draw) passed: {comp['verdict']} "
                            f"{comp['invalid'][:1]}")
    # Studio's default tier is measured as RESOLVED: the record says what ran, the report labels it, splits runs that
    # resolved differently, and voids a timed run whose compute-relevant tier differs from its parity probe's
    fast = {"speed_mode": "default", "speed_optims": ["compiled", "cuda_graph", "int8_gemm"],
            "attention_backend": "unsloth_int8", "transformer_quant": "int8", "offload_policy": "none"}
    eager = {**fast, "speed_optims": ["tf32"]}
    noint8 = {**fast, "attention_backend": "sdpa"}
    for name, probe_st, timed_st, want in (("same", fast, [fast, fast], "ok"), ("split", fast, [fast, eager], "split"),
                                          ("attn", fast, [noint8, noint8], "void")):
        d = neg / f"tier_{name}"
        a = run_cell(cell("s_pt", "studio", speed_mode = "default", studio_status = probe_st), d / "_parity",
                     parity = True)
        b = run_cell(cell("c_pt", "blineui"), d / "_parity", parity = True)
        write_parity(a, b, d / "_parity" / "bf16")
        recs = [run_cell(cell(f"s_t{i}", "studio", speed_mode = "default", studio_status = st, block = i), d)
                for i, st in enumerate(timed_st, 1)]
        for i in (1, 2):
            run_cell(cell(f"c_t{i}", "blineui", block = i), d)
        rep = Rp.build(d, use_lpips = False)
        card = rep["cards"][0]
        md = Rp.markdown(rep)
        studios = sorted(k for k in card["stacks"] if k.startswith("studio"))
        rt = recs[0].get("resolved_tier") or {}
        if rt.get("requested") != "default" or rt.get("compiled") is None:
            failures.append(f"tier {name}: record has no resolved tier with the request: {rt}")
        if want == "ok":
            comp = card["comparisons"][0]
            if studios != ["studio"] or comp["valid_pairs"] != 4 or \
                    "requested default): default: compiled + CUDA graph, attention unsloth_int8 (INT8), DiT int8" not in md:
                failures.append(f"tier same: not one resolved arm / not labelled ({studios}, {comp['valid_pairs']}, "
                                f"{[x for x in md.splitlines() if 'resolved tier' in x]})")
        elif want == "split":
            if len(studios) != 2 or not any("eager" in k for k in studios):
                failures.append(f"negative control: runs that resolved different tiers shared one arm ({studios})")
        else:
            comp = card["comparisons"][0]
            if comp["verdict"] != "VOID" or not any("different tier than its parity probe" in w
                                                    for w in comp["invalid"]):
                failures.append(f"negative control: a timed tier unlike its parity probe's (INT8 attention off) "
                                f"was not voided: {comp['verdict']} {comp['invalid'][:1]}")
    # a Studio arm whose tier label splits it ("studio{...}") is still a Studio arm and is compared (the Wan L4 round
    # lost its comparison to that); Studio's cuda_graph audit wins over a speed_optims list that disagrees
    import common as Cm
    st_list = {**fast, "speed_optims": ["compiled"],
               "resolved": {"speed_mode": {"steps": [{"step": "cuda_graph", "status": "engaged"}]}}}
    t2 = Cm.resolved_tier(st_list)
    if not t2.get("cuda_graph") or not t2.get("cuda_graph_conflict"):
        failures.append(f"Studio's cuda_graph audit did not win over speed_optims / the conflict was not kept: {t2}")
    d = neg / "tier_splitcmp"
    a = run_cell(cell("s_pt", "studio", speed_mode = "default", studio_status = fast), d / "_parity", parity = True)
    b = run_cell(cell("c_pt", "blineui"), d / "_parity", parity = True)
    write_parity(a, b, d / "_parity" / "bf16")
    for i, st in enumerate((fast, fast, {**fast, "speed_optims": ["compiled"]}), 1):
        run_cell(cell(f"s_t{i}", "studio", speed_mode = "default", studio_status = st, block = i), d)
        run_cell(cell(f"c_t{i}", "blineui", block = i), d)
    cmps = Rp.build(d, use_lpips = False)["cards"][0]["comparisons"]
    if len(cmps) != 2 or any(c["other"] != "blineui" for c in cmps):
        failures.append(f"split Studio tiers were not each compared with the baseline: {[(c['studio'], c['other']) for c in cmps]}")
    # two Studio arms of one format (SDPA and INT8 attention), each with its own passing parity probe: each arm is
    # held to its own probe, so the newer INT8 probe never voids the SDPA arm
    d = neg / "tier_twoarms"
    probes = {}
    for lab, st in (("sdpa", noint8), ("int8", fast)):
        arm_o = {"stack": "studio", "block": 1, "order": "ABBA", "label": lab}
        a = run_cell(cell(f"s_p{lab}", "studio", speed_mode = "default", studio_status = st, arm = arm_o),
                     d / "_parity", parity = True)
        b = run_cell(cell(f"c_p{lab}", "blineui"), d / "_parity", parity = True)
        probes[lab] = write_parity(a, b, d / "_parity" / lab)
        time.sleep(0.02)  # the INT8 probe is the newer one
    for i in (1, 2):
        for lab, st in (("sdpa", noint8), ("int8", fast)):
            run_cell(cell(f"s_{lab}{i}", "studio", speed_mode = "default", studio_status = st, block = i,
                          arm = {"stack": "studio", "block": i, "order": "ABBA", "label": lab}), d)
        run_cell(cell(f"c_t{i}", "blineui", block = i), d)
    cmps = Rp.build(d, use_lpips = False)["cards"][0]["comparisons"]
    bad = [(c["studio"], c["verdict"], c["invalid"][:1]) for c in cmps
           if c["verdict"] == "VOID" or any("different tier than its parity probe" in w for w in c["invalid"])]
    if any(r["verdict"] != "ok" for r in probes.values()) or len(cmps) != 2 or bad:
        failures.append(f"Studio arms with their own parity probes were not each held to their own: "
                        f"{[r['verdict'] for r in probes.values()]} {len(cmps)} {bad}")
    # one Studio probe paired with two baseline arms of the same format: each pair keeps its own parity result, and
    # each baseline comparison is held to its own (a VOID pair first, then an ok one, must not leave only the ok)
    import parity as PG

    d = neg / "parity_fanout"

    def _arm(stack_, lab, blk = 1):
        return {"stack": stack_, "block": blk, "order": "ABBA", "label": lab, "comparison": "fan"}

    run_cell(cell("s_pf", "studio", arm = _arm("studio", "st")), d / "_parity", parity = True)
    bad = cell("c_pbad", "blineui", arm = _arm("blineui", "cbad"))
    bad["steps"] = 5  # a mismatched step count: this pair must be VOID
    run_cell(bad, d / "_parity", parity = True)
    run_cell(cell("c_pok", "blineui", arm = _arm("blineui", "cok")), d / "_parity", parity = True)
    plan_doc = {"comparisons": [{"id": "fan", "family": "fakefam", "parity_pairs": [
        {"format": "bf16", "a": "s_pf", "b": "c_pbad"}, {"format": "bf16", "a": "s_pf", "b": "c_pok"}]}]}
    g = PG.gate_plan(plan_doc, d / "_parity")
    verdicts = {Path(x["path"]).parent.name: x["verdict"] for x in g["pairs"]}
    for i in (1, 2):
        run_cell(cell(f"s_f{i}", "studio", arm = _arm("studio", "st", i), block = i), d)
        run_cell(cell(f"c_fbad{i}", "blineui", arm = _arm("blineui", "cbad", i), block = i), d)
        run_cell(cell(f"c_fok{i}", "blineui", arm = _arm("blineui", "cok", i), block = i), d)
    cmps = {c["other"]: c for c in Rp.build(d, use_lpips = False)["cards"][0]["comparisons"]}
    ok_c = next((c for k, c in cmps.items() if "cok" in k), None)
    bad_c = next((c for k, c in cmps.items() if "cbad" in k), None)
    if verdicts != {"vs_c_pbad": "VOID", "vs_c_pok": "ok"} or ok_c is None or bad_c is None \
            or ok_c["verdict"] == "VOID" or bad_c["verdict"] != "VOID" \
            or not any("parity gate VOID" in w for w in bad_c["invalid"]):
        failures.append(f"a Studio probe paired with two baseline arms did not keep a result per pair: {verdicts} "
                        f"{[(k, c['verdict'], c['invalid'][:1]) for k, c in cmps.items()]}")
    # 8. non-finite JSON
    try:
        schema.loads('{"a": NaN}')
        failures.append("schema.loads accepted NaN")
    except ValueError:
        pass
    try:
        schema.dumps({"a": float("inf")})
        failures.append("schema.dumps wrote Infinity")
    except ValueError:
        pass
    nf = neg / "nonfinite"
    rec = run_cell(cell("s_ok", "studio"), nf)
    rec["timing"]["new_s_median"] = float("nan")
    C.write_record(nf / "s_ok", rec)
    back = C.read_record(nf / "s_ok")
    if not back or back["verdict"] != "VOID" or back["timing"]["new_s_median"] is not None:
        failures.append(f"NaN in an ok record was not VOIDed on write: {back and back.get('verdict')}")
    (nf / "raw").mkdir(parents = True, exist_ok = True)
    (nf / "raw" / C.RECORD).write_text('{"verdict": "ok", "timing": {"new_s_median": NaN}}')
    if C.read_record(nf / "raw") is not None:
        failures.append("read_record accepted a record with NaN")
    if not any("UNREADABLE" in x for x in Rp.build(nf, use_lpips = False)["could_not_test"]):
        failures.append("report did not list a non-finite record as unreadable")
    # incomplete case set: refused
    (nf / "cases.json").write_text(json.dumps(["s_ok", "c_never_ran"]))
    cli = subprocess.run([sys.executable, str(HERE / "report.py"), str(nf), "--no-lpips"], capture_output = True,
                         text = True)
    if cli.returncode != 2 or "REFUSED" not in cli.stderr:
        failures.append(f"report on an incomplete case set must refuse (exit 2), got {cli.returncode}")
    # capability: the fake card is 8.9, the cell needs 9.0
    r = run_cell({**cell("s_cc", "studio"), "min_compute_cap": "9.0"}, neg / "timed")
    if r["verdict"] != "NOT_RUN" or not str(r.get("reason")).startswith("unsupported"):
        failures.append(f"capability gate: {r['verdict']} {r.get('reason')}")
    if C.auto_dtype("7.5") != "fp16" or C.auto_dtype("8.0") != "bf16":
        failures.append("auto_dtype: T4 must resolve to fp16, sm80+ to bf16")
    # derived s/step flagged under a step cache
    r = run_cell(cell("s_cache", "studio", transformer_cache = "fbcache"), neg / "timed")
    if "step_cache_on" not in str(r["timing"].get("step_s_derived_flag")):
        failures.append(f"derived s/step not flagged under a step cache: {r['timing'].get('step_s_derived_flag')}")


def fairness_checks(root: Path, failures: list) -> None:
    """Negative controls for the PR #876 fairness holes; each FAILS on the tree before the fix (0c98da44)."""
    import types

    import parity as P
    import report as Rp
    import score as Sc

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    fr = root / "fair"
    # 1. precision: the backend's REPORT of the transformer weights vs the card, not card vs card
    q = {"transformer": {"weights": "int8_convrot", "compute": "int8"}, "vae": {"weights": "bf16", "compute": "bf16"}}
    card = {"formats": {"int8": {"studio": {"precision": q}, "bline": {"precision": q}},
                        "bf16": {"studio": {"precision": {"transformer": {"weights": "bf16", "compute": "bf16"}}},
                                 "bline": {"precision": {"transformer": {"weights": "bf16", "compute": "bf16"}}}}}}
    c_i8 = run_cell(cell("c_pi8", "blineui", "int8"), fr / "par", parity = True)
    for tag, rp, want in (("s_pi8_ok", {}, "ok"), ("s_pi8_fellback", {"transformer": "bf16"}, "VOID"),
                          ("s_pi8_noreport", {"transformer": None}, "VOID")):
        s_i8 = run_cell(cell(tag, "studio", "int8", report_precision = rp), fr / "par", parity = True)
        res = P.check_pair(s_i8, c_i8, card)
        hit = any(r.startswith("precision_observed") for r in res["reasons"])
        if res["verdict"] != want or (want == "VOID" and not hit):
            failures.append(f"precision observed-vs-card ({tag}): want {want} via precision_observed, got "
                            f"{res['verdict']} {res['reasons']}")
    # dense bf16 on an fp16-only card: fp16 reports pass only when the plan said so (options.dense_dtype)
    for tag, opts, want in (("t4_dense", {"dense_dtype": "fp16"}, "ok"), ("t4_nodecl", {}, "VOID")):
        a = run_cell(cell(f"s_{tag}", "studio", dtype = "fp16", **opts), fr / "par", parity = True)
        b = run_cell(cell(f"c_{tag}", "blineui", dtype = "fp16", **opts), fr / "par", parity = True)
        res = P.check_pair(a, b, card)
        if res["verdict"] != want:
            failures.append(f"T4 dense fp16 ({tag}): want {want}, got {res['verdict']} {res['reasons']}")
    # 2. a timed baseline-UI request with no execution facts is flagged
    nf = run_cell(cell("c_nofacts", "blineui", no_exec_facts = True), fr / "timed")
    probs = P.cache_problems(nf)
    if not any("execution facts missing" in x for x in probs):
        failures.append(f"timed baseline-UI record without execution facts not flagged: {probs}")
    # 3. the report groups by the plan's comparison id: a simulated-budget run never pairs with a full-card run
    grp = fr / "groups"
    for cmp_id in ("fam_l4", "fam_l4_b12"):
        for st, dl in (("studio", 0.004), ("blineui", 0.012)):
            run_cell(cell(f"{st[0]}_{cmp_id}", st, delay = dl, arm = {"stack": st, "comparison": cmp_id}), grp)
    cards = Rp.build(grp, use_lpips = False)["cards"]
    if sorted(c.get("comparison") for c in cards) != ["fam_l4", "fam_l4_b12"]:
        failures.append(f"report mixes comparisons in one group: {[(c['family'], c.get('comparison')) for c in cards]}")
    # an fp16-native card (SDXL-Turbo): the fp16 run is the stack's own reference, and an int8 run of the same
    # family scores against it instead of reporting a missing bf16 reference
    f16 = fr / "fp16ref"
    run_cell(cell("s_f16", "studio", variant = "fp16"), f16)
    run_cell(cell("s_i8f", "studio", variant = "int8"), f16)
    qrep = Rp.build(f16, use_lpips = False)
    qs = {(c["variant"]): c["stacks"].get("studio", {}).get("quality_note") for c in qrep["cards"]}
    if qs.get("fp16") != "dense reference" or any("missing reference" in x for x in qrep["could_not_test"]):
        failures.append(f"fp16 dense reference not used ({qs}, {[x for x in qrep['could_not_test'] if 'reference' in x][:2]})")
    # 4. the matrix gate's timeout (gate.matrix) voids the pair
    s_ok, o_ok = {"target": {"vm_id": "v"}, "gpu": {}, "telemetry": {}}, {"target": {"vm_id": "v"}, "gpu": {},
                                                                         "telemetry": {}, "tag": "o",
                                                                         "gate": {"gate": "ok",
                                                                                  "matrix": {"gate": "timeout"}}}
    if not any("matrix GPU gate" in x for x in Rp.pair_problems(s_ok, o_ok, {"verdict": "ok"})):
        failures.append("report ignores a matrix-level GPU gate timeout (gate.matrix)")
    # power-capped card (L4 FLUX.3 round): different SM clocks at equal power are the workloads' own; at unequal
    # power (a throttled / contended device) the 3% rule still voids the pair
    # the rule needs the SAME enforced power cap on both runs as well as the same draw; every other case still voids
    def tel(mhz, w, cap = 72.0, lo = None, hi = None, thr = None):
        t = {"sm_mhz_mean": mhz, "power_w_mean": w, "throttle_reasons": thr}
        if cap is not None:
            t.update(power_limit_w = cap, power_limit_w_min = cap if lo is None else lo,
                     power_limit_w_max = cap if hi is None else hi)
        return {"target": {"vm_id": "v"}, "gpu": {"name": "NVIDIA L4"}, "tag": f"t{mhz}", "telemetry": t}
    exc: list = []
    capped = Rp.pair_problems(tel(1222, 70.7), tel(1297, 67.9), {"verdict": "ok"}, exc)
    negs = {"unequal draw": Rp.pair_problems(tel(1222, 70.7), tel(1297, 50.0), {"verdict": "ok"}),
            "different cap": Rp.pair_problems(tel(1222, 70.7, cap = 72.0), tel(1297, 67.9, cap = 60.0), {"verdict": "ok"}),
            "cap not recorded": Rp.pair_problems(tel(1222, 70.7, cap = None), tel(1297, 67.9), {"verdict": "ok"}),
            "cap changed mid-run": Rp.pair_problems(tel(1222, 70.7, lo = 60.0, hi = 72.0), tel(1297, 67.9),
                                                    {"verdict": "ok"})}
    if any("SM clocks" in x for x in capped) or len(exc) != 1 or exc[0]["power_cap_w"] != [72.0, 72.0] or \
            exc[0]["power_w"] != [70.7, 67.9] or exc[0]["sm_mhz"] != [1222, 1297]:
        failures.append(f"clock rule: equal-cap equal-draw pair voided or its excuse not recorded ({capped}, {exc})")
    for name, got in negs.items():
        if not any("SM clocks" in x for x in got):
            failures.append(f"clock rule negative control passed ({name}): {got}")
    # below cap (the LTX-2.5 A100 round: 160-220 W of a 400 W cap, neither device throttled): a clock gap is the
    # stack's own duty cycle and the pair counts, labelled; any limiting throttle reason, an unread reason list or a
    # draw at 90% of the cap or more still voids it
    exc2: list = []
    kept = Rp.pair_problems(tel(1262, 182.9, cap = 400.0, thr = []), tel(1331, 217.6, cap = 400.0, thr = ["sync_boost"]),
                            {"verdict": "ok"}, exc2)
    if kept or len(exc2) != 1 or exc2[0].get("kind") != "below cap":
        failures.append(f"below-cap rule: an unthrottled pair under 90% of the cap was voided ({kept}, {exc2})")
    for name, a, b in (("hw_thermal fired", tel(1262, 182.9, cap = 400.0, thr = ["hw_thermal"]),
                        tel(1331, 217.6, cap = 400.0, thr = [])),
                       ("sw_power_cap fired", tel(1262, 182.9, cap = 400.0, thr = []),
                        tel(1331, 217.6, cap = 400.0, thr = ["sw_power_cap"])),
                       ("hw_slowdown fired", tel(1262, 182.9, cap = 400.0, thr = ["hw_slowdown"]),
                        tel(1331, 217.6, cap = 400.0, thr = [])),
                       ("reasons not recorded", tel(1262, 182.9, cap = 400.0, thr = None),
                        tel(1331, 217.6, cap = 400.0, thr = [])),
                       ("draw at 90% of the cap", tel(1262, 182.9, cap = 400.0, thr = []),
                        tel(1331, 365.0, cap = 400.0, thr = [])),
                       ("cap not recorded", tel(1262, 182.9, cap = None, thr = []),
                        tel(1331, 217.6, cap = 400.0, thr = []))):
        if not any("SM clocks" in x for x in Rp.pair_problems(a, b, {"verdict": "ok"})):
            failures.append(f"below-cap negative control passed ({name})")
    if Rp.pair_problems(tel(1222, 70.7, cap = None), tel(1240, 50.0, cap = None), {"verdict": "ok"}):
        failures.append("clock rule: clocks within 3% voided without a power cap (the cap is only needed past 3%)")
    # 5. score: unpaired / dropped renders reported, seeds checked; the report does not average over the rest
    qd = fr / "quality"
    ref = run_cell(cell("s_qref", "studio", n = 3), qd)
    i8 = run_cell(cell("s_qi8", "studio", "int8", n = 3), qd)
    i8["renders"][0]["seed"] = int(i8["renders"][0]["seed"]) + 1
    C.write_record(qd / "s_qi8", i8)
    qv = Sc.quality_vs_reference(qd / "s_qi8", i8, qd / "s_qref", ref, Sc.Metrics(use_lpips = False))
    if qv.get("n_paired") != 2 or qv.get("n_expected") != 3 or not any("seed" in d["reason"] for d in qv.get("dropped") or []):
        failures.append(f"score: a seed mismatch must be dropped and reported: {qv}")
    rep = Rp.build(qd, use_lpips = False)
    st = rep["cards"][0]["stacks"]["studio"] if rep["cards"] else {}
    if st.get("psnr") is not None or not any("2/3 renders paired" in x for x in rep["could_not_test"]):
        failures.append(f"report averaged quality over partly paired renders: psnr={st.get('psnr')}")
    # 6. the newest parity result wins; borrowing only within a target class
    old = {"group": {"family": "f", "variant": "bf16", "target": "colab:L4"}, "verdict": "ok", "created": 1.0}
    new = {**old, "verdict": "VOID", "created": 2.0}
    if (Rp.parity_for([new, old], "f", "bf16", "colab:L4") or {}).get("verdict") != "VOID" or \
            (Rp.parity_for([old, new], "f", "bf16", "colab:L4") or {}).get("verdict") != "VOID":
        failures.append("report did not pick the newest parity result")
    b200 = {"group": {"family": "f", "variant": "bf16", "target": "local:B200"}, "verdict": "ok", "created": 3.0}
    if Rp.parity_for([b200], "f", "bf16", "amd:gfx1151-linux") is not None or \
            Rp.parity_for([b200], "f", "bf16", "colab:L4") is not None or \
            (Rp.parity_for([b200], "f", "bf16", "colab:G4") or {}).get("from_target") != "local:B200":
        failures.append("parity borrowing crosses target classes (B200 -> AMD / L4) or misses a same-class one (G4)")
    # 7. an accepted sampler difference (card sampler_differs, timing only): timing pair ok, quality pair strict
    trap = {"id": "sampler", "check": "sampler_differs", "accepted_difference": True, "settings": ["sampler"],
            "purposes": ["timing"], "note": "equal evaluations"}
    a = run_cell(cell("s_samp", "studio", sampler = "euler"), fr / "par", parity = True)
    b = run_cell(cell("c_samp", "blineui", sampler = "res_multistep"), fr / "par", parity = True)
    rt, rq = P.check_pair(a, b, {"traps": [trap]}), P.check_pair(a, b, {"traps": [trap]}, quality = True)
    if rt["verdict"] != "ok" or not rt["accepted_differences"] or rq["verdict"] != "VOID":
        failures.append(f"accepted sampler difference: timing {rt['verdict']} {rt['reasons']}, quality {rq['verdict']}")
    if P.check_pair(a, b)["verdict"] != "VOID":
        failures.append("a sampler difference without the card trap must stay VOID")
    # 8. Studio's sampler from the generation record; "default" -> the resident scheduler class
    from backends import studio_inproc as SI

    FlowMatchEulerDiscreteScheduler = type("FlowMatchEulerDiscreteScheduler", (), {"config": {}})
    try:
        got = [SI.studio_sampler({"sampler": None, "schedule_type": None}, FlowMatchEulerDiscreteScheduler()),
               SI.studio_sampler({"sampler": None}, FlowMatchEulerDiscreteScheduler(), "res_multistep"),
               SI.studio_sampler({"sampler": "dpmpp_2m", "schedule_type": "karras"}, None)]
        if [(g["sampler"], g["scheduler"]) for g in got] != [("euler", "simple"), ("res_multistep", "simple"),
                                                              ("dpmpp_2m", "karras")]:
            failures.append(f"studio_sampler mapping wrong: {got}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"studio_sampler: {type(exc).__name__}: {exc}")
    # 8b. a scheduler that resolves its own sigma grid (MOSS-Image: sgm_uniform) reports that grid, not the record's
    MOSSImageScheduler = type("MOSSImageScheduler", (), {"config": {}, "resolved_sigma_schedule": lambda self: "sgm_uniform"})
    try:
        g = SI.studio_sampler({"sampler": None, "schedule_type": "simple"}, MOSSImageScheduler())
        if (g["sampler"], g["scheduler"]) != ("euler", "sgm_uniform"):
            failures.append(f"MOSS-Image scheduler reported as {g} (it runs sgm_uniform: the card's accepted "
                            "scheduler difference then covers the sigma grid; reported as simple the pair is VOID)")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"studio_sampler MOSS: {type(exc).__name__}: {exc}")
    # 8c. LTX-2.5 distilled: Studio states its ancestral sampler on the pipe; the hook's base class must not win
    try:
        LTX = type("LTX25EulerAncestralScheduler", (), {"config": {}})
        pipe = type("Pipe", (), {})()
        pipe.scheduler, pipe._unsloth_ltx25_sampler = LTX(), "euler_ancestral"
        eng = type("Eng", (), {})()
        eng._state = type("St", (), {})()
        eng._state.pipe = pipe
        g = SI.studio_sampler({"sampler": None}, SI.resident_scheduler(eng), SI.family_sampler(eng))
        g2 = SI.studio_sampler({"sampler": None}, LTX())
        if g["sampler"] != "euler_ancestral" or g2["sampler"] != "euler_ancestral":
            failures.append(f"LTX-2.5 sampler reported as {g['sampler']} / {g2['sampler']} (Studio runs euler_ancestral)")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"LTX-2.5 sampler: {type(exc).__name__}: {exc}")
    # 8d. a card's sigma_rounding trap accepts a published list's rounding (LTX-2.5 stage 2 0.4219 vs 0.421875)
    try:
        import models as MD  # noqa: PLC0415

        ltx = MD.load_card("ltx-2.5")

        def prec(stack, s2, sampler):
            return {"verdict": "ok", "arm": {"stack": stack},
                    "parity": {"resolved": {"steps": 8, "cfg": 1.0, "sampler": sampler, "scheduler": None,
                                            "width": 1280, "height": 704, "frames": 121},
                               "sigma_schedules": [[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0],
                                                   [0.85, 0.725, s2, 0.0]]}}
        r = P.check_pair(prec("studio", 0.421875, "euler_ancestral"), prec("blineui", 0.4219, "euler_ancestral"), ltx)
        far = P.check_pair(prec("studio", 0.421875, "euler_ancestral"), prec("blineui", 0.43, "euler_ancestral"), ltx)
        sig = (r.get("checks") or {}).get("sigmas") or {}
        if sig.get("status") != "ok" or (far.get("checks") or {}).get("sigmas", {}).get("status") != "mismatch" \
                or not any(x.get("component") == "sigmas" for x in r.get("accepted_differences") or []):
            failures.append(f"LTX-2.5 stage-2 rounding: {sig} / far {far.get('checks', {}).get('sigmas')}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"LTX-2.5 sigma rounding: {type(exc).__name__}: {exc}")
    # 9. a server backend's compile time is never this driver's dynamo 0.0
    import run_cell as R

    stub = {}
    if "torch" not in sys.modules:
        tu = types.ModuleType("torch._dynamo.utils")
        tu.compilation_time_metrics = {}
        stub = {"torch": types.ModuleType("torch"), "torch._dynamo": types.ModuleType("torch._dynamo"),
                "torch._dynamo.utils": tu}
        sys.modules.update(stub)
    try:
        srv = types.SimpleNamespace(in_process = False)
        try:
            got = R._compile_s(srv)
        except TypeError:  # the pre-fix signature took no backend: it read the driver's dynamo regardless
            got = R._compile_s()
        if got != (None, None):
            failures.append(f"server backend compile_s from the driver's dynamo: {got}")
    finally:
        for k in stub:
            sys.modules.pop(k, None)


def panel_checks(root: Path, failures: list) -> None:
    """Negative controls for the 19-voter panel's must-fix clusters (review876_tally.txt #N); each fails on 0c98da44."""
    import io
    import tarfile
    import types

    import parity as P
    import report as Rp
    import schema

    pc = root / "panel"
    pc.mkdir(parents = True, exist_ok = True)

    def guard(name, fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a crash on the old tree is that control failing
            failures.append(f"{name}: {type(exc).__name__}: {str(exc)[:300]}")

    # #2: a builder never deletes a shared Hugging Face cache blob (drop_bf16 behind a link into HF_HOME)
    def c2():
        hf = pc / "hf_home"
        blob = hf / "hub" / "models--x--y" / "blobs" / "abc"
        blob.parent.mkdir(parents = True, exist_ok = True)
        blob.write_bytes(b"shared weights")
        out = pc / "build_out"
        out.mkdir(parents = True, exist_ok = True)
        link = out / "q21_dit_bf16.safetensors"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(blob)
        r = subprocess.run([sys.executable, str(HERE / "builders" / "img_build_q5.py"), "--group", "q21", "--roles",
                            "drop_bf16", "--out", str(out)], capture_output = True, text = True,
                           env = {**os.environ, "HF_HOME": str(hf)})
        if not blob.exists():
            failures.append(f"#2 builder deleted a shared HF cache blob (rc {r.returncode})")
    guard("#2", c2)

    # #3: an artifact member never lands outside the extraction dir (existing symlink, drive letter, "..")
    def c3():
        from targets import common as TC

        run = pc / "art_run"
        outside = pc / "outside"
        outside.mkdir(parents = True, exist_ok = True)
        dest = run / "unpacked" / "artifact"
        dest.mkdir(parents = True, exist_ok = True)
        if not (dest / "cell").exists():
            (dest / "cell").symlink_to(outside, target_is_directory = True)
        buf = io.BytesIO()
        with tarfile.open(fileobj = buf, mode = "w:gz") as tf:
            for name in ("cell/record.json", "C:/Users/x/record.json", "../escape.json"):
                ti = tarfile.TarInfo(name)
                ti.size = 2
                tf.addfile(ti, io.BytesIO(b"{}"))
        (run / "artifact.tgz").write_bytes(buf.getvalue())
        TC.unpack_artifacts(run)
        if (outside / "record.json").exists() or (run / "unpacked" / "escape.json").exists() or \
                (dest / "C:").exists():
            failures.append("#3 artifact extraction wrote outside its directory")
    guard("#3", c3)

    # the artifact as Colab returns it: the base64 block split over many stream outputs of the executed notebook
    # (the JSON keys between them are base64-alphabet words), a progress line in the middle, and a broken copy
    # beside it that must not stop the good one from unpacking
    def c_art():
        import base64 as b64
        from targets import common as TC

        run = pc / "art_nb"
        run.mkdir(parents = True, exist_ok = True)
        buf = io.BytesIO()
        with tarfile.open(fileobj = buf, mode = "w:gz") as tf:
            ti = tarfile.TarInfo("cellx/record.json")
            body = json.dumps({"tag": "cellx"}).encode()
            ti.size = len(body)
            tf.addfile(ti, io.BytesIO(body))
        enc = b64.b64encode(buf.getvalue()).decode()
        chunks = [enc[k:k + 40] + "\n" for k in range(0, len(enc), 40)]
        outs = [{"output_type": "stream", "name": "stdout", "text": ["ARTIFACT_B64_BEGIN\n"]}]
        for k, ch in enumerate(chunks):
            outs.append({"output_type": "stream", "name": "stdout", "text": [ch]})
            if k == 1:
                outs.append({"output_type": "stream", "name": "stderr", "text": ["12:00:00 progress 3/5 steps\n"]})
        outs.append({"output_type": "stream", "name": "stdout", "text": ["ARTIFACT_B64_END\n"]})
        (run / "a.executed.ipynb").write_text(json.dumps({"cells": [{"cell_type": "code", "outputs": outs}]}))
        (run / "b.log").write_text("ARTIFACT_B64_BEGIN\nH4sIAAAAAAAAA\nARTIFACT_B64_END\n")
        TC.unpack_artifacts(run)
        if not (run / "unpacked" / "a.executed" / "cellx" / "record.json").exists():
            failures.append("artifact split over notebook stream outputs (with a progress line) was not unpacked")
    guard("artifact-nb", c_art)

    # #26 / #62: a tester-supplied name never writes outside --out
    def c26():
        import import_tester as IT

        res = pc / "tester_results"
        res.mkdir(parents = True, exist_ok = True)
        (res / "evil.json").write_text(json.dumps({"name": "../escaped_by_name", "stack": "studio", "outcome": "x",
                                                   "wl": {}}))
        out = pc / "tester_out"
        with open(os.devnull, "w") as dn:
            import contextlib

            with contextlib.redirect_stdout(dn):
                IT.main([str(res), "--out", str(out)])
        if (pc / "escaped_by_name").exists():
            failures.append("#26/#62 a tester-supplied name wrote outside --out")
    guard("#26", c26)

    # #14 / #19 / #27: the generated cleanup step kills nothing when the work root is unset
    def c14():
        import importlib.util as iu

        spec = iu.spec_from_file_location("gw_panel", HERE / "amd" / "gen_workflow.py")
        GW = iu.module_from_spec(spec)
        spec.loader.exec_module(GW)
        text = GW.JOB
        i = text.index("Stop anything this job started")
        block = text[text.index("run: |", i) + 6:text.index("- uses: actions/upload-artifact", i)]
        script = textwrap.dedent(block.replace("{{", "{").replace("}}", "}")).replace('kill -$sig "$pid"',
                                                                                     'echo WOULD_KILL "$pid"')
        script = script.replace("sleep 5", "true")
        r = subprocess.run(["bash", "-c", script], capture_output = True, text = True,
                           env = {k: v for k, v in os.environ.items() if k != "AMD_CI_WORK"})
        if "WOULD_KILL" in r.stdout:
            failures.append("#14 Linux cleanup with AMD_CI_WORK unset selects processes to kill")
        if shutil.which("pwsh"):
            ps = GW.STOP_PS.split("run: |", 1)[1]
            ps = textwrap.dedent(ps).replace("Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue",
                                             "Write-Output WOULD_STOP")
            ps = ps.replace("Get-CimInstance Win32_Process",
                            "@([pscustomobject]@{CommandLine='python unrelated.py'; ProcessId=987654; Name='p'})")
            r = subprocess.run(["pwsh", "-NoProfile", "-Command", ps], capture_output = True, text = True,
                               env = {k: v for k, v in os.environ.items() if k != "AMD_CI_WORK"})
            if "WOULD_STOP" in r.stdout:
                failures.append("#14 Windows cleanup with AMD_CI_WORK unset selects an unrelated process")
    guard("#14", c14)

    # #6: an ok CPU record is not VOID for lacking a GPU-only metric
    def c6():
        rec = run_cell(cell("s_cpu", "studio", target = {"name": "cpu", "vm_id": "cpu-1"}), pc / "cpu", fake_gpu = None)
        if rec["verdict"] != "ok":
            failures.append(f"#6 a CPU render is {rec['verdict']}: {rec.get('reason')}")
    guard("#6", c6)

    # #16 / #43: a flattened plan spec (positional, as the AMD probe runs it) with a parity pair whose probe never
    # wrote a record: its timed cell must not run
    def c16():
        tcell = cell("cmp1__s_bf16__k1p0", "studio", arm = {"stack": "studio", "comparison": "cmp1",
                                                            "format": "bf16", "purpose": "timing"})
        spec = {"name": "flat", "defaults": {}, "cells": [tcell],
                "comparisons": [{"id": "cmp1", "family": "fakefam",
                                 "parity_pairs": [{"format": "bf16", "a": "cmp1__s__parity", "b": "cmp1__c__parity",
                                                   "quality": False, "waive": []}]}]}
        sp = pc / "flat_spec.json"
        sp.write_text(json.dumps(spec))
        out = pc / "flat_out"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "DBENCH_FAKE_GPU": json.dumps(FAKE_GPU)}
        subprocess.run([sys.executable, str(HERE / "matrix.py"), str(sp), "--out", str(out), "--no-gate"],
                       capture_output = True, text = True, env = env)
        rec = C.read_record(out / tcell["tag"]) or {}
        if rec.get("verdict") != "NOT_RUN":
            failures.append(f"#16/#43 a timed cell ran with its parity probes missing ({rec.get('verdict')})")
    guard("#16", c16)

    # #17 / #45: an untimed trial and a different Studio arm never enter a comparison's medians / A/A band
    def c17():
        d = pc / "pool"
        for blk in (1, 2):
            run_cell(cell(f"s_def_{blk}", "studio", delay = 0.004, block = blk,
                          arm = {"stack": "studio", "label": "s_bf16", "comparison": "cp"}), d)
            run_cell(cell(f"s_max_{blk}", "studio", delay = 0.0005, block = blk,
                          arm = {"stack": "studio", "label": "s_bf16_max", "comparison": "cp"}), d)
            run_cell(cell(f"c_{blk}", "blineui", delay = 0.012, block = blk,
                          arm = {"stack": "blineui", "label": "c_bf16", "comparison": "cp"}), d)
        run_cell(cell("trial__vm1", "studio", delay = 0.05, arm = {"stack": "studio", "label": "s_bf16",
                                                                   "comparison": "cp", "purpose": "trial"}), d)
        pa = run_cell(cell("s_pp", "studio", arm = {"stack": "studio", "comparison": "cp"}), d / "_parity", parity = True)
        pb = run_cell(cell("c_pp", "blineui", arm = {"stack": "blineui", "comparison": "cp"}), d / "_parity",
                      parity = True)
        write_parity(pa, pb, d / "_parity" / "bf16")
        card = Rp.build(d, use_lpips = False)["cards"][0]
        tags = {t for c in card["comparisons"] for t in c["studio_tags"]}
        mixed = [c for c in card["comparisons"] if {"s_def_1", "s_max_1"} <= set(c["studio_tags"])]
        n = sum(st["n_total"] for st in card["stacks"].values())
        if not tags or "trial__vm1" in tags or n != 6 or mixed:
            failures.append(f"#17/#45 report pooled a trial or two Studio arms: records {n} (want 6), "
                            f"{[c['studio_tags'] for c in card['comparisons']]}")
    guard("#17", c17)

    # #20 / #44: the card's per-row count is compared with denoiser_items (batched CFG: 25 calls, 50 rows)
    def c20():
        a = run_cell(cell("s_cfg", "studio", cfg = 7.0), pc / "par", parity = True)
        b = run_cell(cell("c_cfg", "blineui", cfg = 7.0), pc / "par", parity = True)
        res = P.check_pair(a, b, {"expected_passes": "steps * (2 if cfg > 1 else 1)"})
        if res["checks"]["passes"]["status"] != "ok":
            failures.append(f"#20/#44 batched CFG rows rejected: {res['checks']['passes']['detail']}")
        bad = P.check_pair(a, b, {"expected_passes": "steps * 3"})
        if bad["checks"]["passes"]["status"] != "mismatch":
            failures.append("#20/#44 control: a wrong card count must still void")
    guard("#20", c20)

    # #39: one cached sampler stage of a two-stage graph fails the cache gate
    def c39():
        from backends import base as BB
        from backends.blineui import BlineUIBackend

        be = BlineUIBackend.__new__(BlineUIBackend)
        BB.Backend.__init__(be, cell("c_cache2", "blineui"), pc)
        be.graph = {"s1": {"class_type": "SamplerCustomAdvanced"}, "s2": {"class_type": "SamplerCustomAdvanced"}}
        be.ws_error = None
        facts = be._apply_events([(1.0, "execution_cached", {"nodes": ["s1"]}), (2.0, "executing", {"node": "s2"})])
        if not facts.get("sampler_cached"):
            failures.append(f"#39 a partly cached sampler passed: {facts}")
    guard("#39", c39)

    # #46 / #66: every sigma stage is compared; a null sigma is missing evidence, not a crash
    def c46():
        base = {"verdict": "ok", "arm": {"stack": "studio"}, "parity": {"resolved": {"steps": 2}}}
        mk = lambda st, sch: {**base, "arm": {"stack": st}, "parity": {**base["parity"], "sigma_schedules": sch,  # noqa: E731
                                                                       "sigmas": sch[-1]}}
        res = P.check_pair(mk("studio", [[1.0, 0.5, 0.0], [0.8, 0.0]]), mk("blineui", [[1.0, 0.4, 0.0], [0.8, 0.0]]))
        if res["checks"]["sigmas"]["status"] != "mismatch":
            failures.append(f"#46 a different first stage passed: {res['checks']['sigmas']}")
        res = P.check_pair(mk("studio", [[1.0, None, 0.0]]), mk("blineui", [[1.0, 0.5, 0.0]]))
        if res["checks"]["sigmas"]["status"] != "missing":
            failures.append(f"#66 a null sigma: {res['checks']['sigmas']}")
    guard("#46/#66", c46)

    # a shift one stack resolves dynamically (no fixed value to report: Studio's FLUX.3 klein) is held to the dumped
    # sigma schedules, which the shift only parameterizes; any other one-sided or differing shift still voids
    def cshift():
        sig = [1.0, 0.9, 0.5, 0.0]
        mk = lambda st, shift, s: {"verdict": "ok", "arm": {"stack": st}, "parity": {  # noqa: E731
            "resolved": {"steps": 3, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "shift": shift},
            "sigmas": s}}
        res = P.check_pair(mk("studio", None, sig), mk("blineui", 1.93586, list(sig)))
        if res["checks"]["settings"]["status"] != "ok":
            failures.append(f"shift: an unreported dynamic shift with equal sigmas voided: {res['checks']['settings']}")
        res = P.check_pair(mk("studio", None, sig), mk("blineui", 1.93586, [1.0, 0.8, 0.5, 0.0]))
        if res["checks"]["settings"]["status"] != "mismatch":
            failures.append(f"shift control: unreported shift with different sigmas passed: {res['checks']['settings']}")
        res = P.check_pair(mk("studio", 3.0, sig), mk("blineui", 1.93586, list(sig)))
        if res["checks"]["settings"]["status"] != "mismatch":
            failures.append(f"shift control: two reported shifts that differ passed: {res['checks']['settings']}")
        a, b = mk("studio", None, sig), mk("blineui", None, list(sig))
        b["parity"]["resolved"]["scheduler"] = None  # ManualSigmas: explicit sigmas, no scheduler name
        if P.check_pair(a, b)["checks"]["settings"]["status"] != "ok":
            failures.append("scheduler: explicit sigmas on one side with equal schedules voided")
        b["parity"]["sigmas"] = [1.0, 0.8, 0.5, 0.0]
        if P.check_pair(a, b)["checks"]["settings"]["status"] != "mismatch":
            failures.append("scheduler control: a one-sided scheduler with different sigmas passed")
    guard("shift-from-sigmas", cshift)

    # Studio over HTTP: the client sees no scheduler class, so the sampler / schedule come from the class the parity
    # hook saw set the sigmas in the server; a sampler the stack stated, or evidence that is not the hook's, wins
    def chooks():
        import run_cell as R

        ev = {"source": "hooks", "sigma_source": "FlowMatchEulerDiscreteScheduler"}
        res = R._sampler_from_hooks({"steps": 4, "sampler": None, "scheduler": None}, ev)
        if (res.get("sampler"), res.get("scheduler")) != ("euler", "simple"):
            failures.append(f"HTTP sampler not taken from the server's scheduler class: {res}")
        res = R._sampler_from_hooks({"sampler": "res_multistep", "scheduler": None}, ev)
        if res.get("sampler") != "res_multistep":
            failures.append(f"a stated sampler was overwritten by the hook's class: {res}")
        res = R._sampler_from_hooks({"sampler": None, "scheduler": None}, {**ev, "source": "backend"})
        if res.get("sampler") is not None:
            failures.append(f"non-hook evidence filled the sampler: {res}")
    guard("http-sampler-from-hooks", chooks)

    # Studio over HTTP: the image route rewrites the prompt with an LLM by default (FLUX.3 klein, MiniMax-H3-Image);
    # the baseline arm runs with the enhancer off, so every image request says enhance_prompt=false unless asked
    def cenhance():
        from backends.studio_http import generate_fields

        cell = {"width": 1024, "height": 1024, "guidance": 1.0}
        img = generate_fields(cell, {}, 6, 7, "image")
        asked = generate_fields(cell, {"generate": {"enhance_prompt": True}}, 6, 7, "image")
        vid = generate_fields(cell, {}, 6, 7, "video")
        if img.get("enhance_prompt") is not False or asked.get("enhance_prompt") is not True or "enhance_prompt" in vid:
            failures.append(f"HTTP enhance_prompt default wrong: {img} / {asked} / {vid}")
    guard("http-enhance-off", cenhance)

    # no NVML in the driver venv (a tester's own Studio venv has no nvidia-ml-py): the SMI fallback still reads the
    # SM clock, or the reporter's 3% clock rule voids every pair on an NVIDIA card
    def csmiclk():
        import subprocess as SP

        import telemetry as T

        orig_smi, orig_vendor = C.smi, C.gpu_vendor
        try:
            C.gpu_vendor = lambda: "nvidia"
            T._SMI_FIELDS.clear()
            C.smi = lambda args, timeout = 20: SP.CompletedProcess(args, 0, "1065, 89, 293.40, 300.00, 0x0000000000000024\n", "")
            thr = T.smi_clocks("0")
            T._SMI_FIELDS.clear()
            # a driver that rejects the reasons field: an error line first, then the query without it
            calls = []

            def smi_old(args, timeout = 20):
                calls.append(args)
                bad = any("clocks_event_reasons" in a for a in args)
                return SP.CompletedProcess(args, 0, "Field \"clocks_event_reasons.active\" is not a valid field\n"
                                           if bad else "1065, 89, 293.40, 300.00\n", "")
            C.smi = smi_old
            got = T.smi_clocks("0")
            C.smi = lambda args, timeout = 20: SP.CompletedProcess(args, 0, "1065, 89, [N/A], [N/A]\n", "")
            na = T.smi_clocks("0")
        finally:
            C.smi, C.gpu_vendor = orig_smi, orig_vendor
        if thr != (1065.0, 89.0, 293.4, 300.0, 0x24) or T.throttle_names(0x24) != ["sw_power_cap", "sw_thermal"] \
                or len(calls) != 2:
            failures.append(f"SMI throttle reasons wrong: {thr} {T.throttle_names(0x24)} ({len(calls)} queries)")
        T._SMI_FIELDS.clear()
        # the enforced power cap rides with the clock: the equal-power clock rule needs it on both arms
        if got != (1065.0, 89.0, 293.4, 300.0, None) or na != (1065.0, 89.0, None, None, None):
            failures.append(f"SMI clock fallback wrong: {got} / {na}")
        # the cooldown gate reads the temperature the same way, or it is "unavailable" and the pair is refused
        orig_handle = T.handle
        try:
            C.gpu_vendor = lambda: "nvidia"
            C.smi = lambda args, timeout = 20: SP.CompletedProcess(args, 0, "1065, 54, 80.10\n", "")
            T.handle = lambda device: None
            gate = T.cooldown("0", 57, timeout_s = 1, poll_s = 0.1, log = lambda *a: None)
        finally:
            C.smi, C.gpu_vendor, T.handle = orig_smi, orig_vendor, orig_handle
        if gate.get("gate") != "ok" or gate.get("temp_c") != 54:
            failures.append(f"cooldown without NVML: {gate}")
    guard("smi-clocks", csmiclk)

    # Wan2.2-TI2V-5B: Studio applies the baseline's shift 8 to the resident UniPC at load and its video result says
    # flow_shift None; the reader takes the resident scheduler's shift, and diffusers' UniPC flow grid is labelled as
    # its own schedule (it is not the baseline's "simple": the dumps differ by 0.159 relative)
    def cwanshift():
        from backends import studio_inproc as SI

        class UniPCMultistepScheduler:  # noqa: N801 - the diffusers class name is the lookup key
            config = {"use_flow_sigmas": True, "flow_shift": 8.0, "prediction_type": "flow_prediction"}

        class FlowMatchEulerDiscreteScheduler:  # noqa: N801 - FLUX.3 klein on the Nov 19 L4: static 6.93 in config
            config = {"shift": 6.93, "use_dynamic_shifting": False}

        cell = {"steps": 20, "guidance": 5.0, "width": 1280, "height": 704, "frames": 121, "options": {}}
        got = SI.studio_resolved({"resolved": {"steps": 20}}, cell, 20,
                                 {"flow_shift": None, "resident_shift": SI.resident_shift(UniPCMultistepScheduler())})
        names = SI.scheduler_names(UniPCMultistepScheduler())
        if got.get("shift") != 8.0 or names[:2] != ("uni_pc", "flow_linspace"):
            failures.append(f"Wan shift / UniPC grid not read from the resident scheduler: {got.get('shift')} {names}")
        if SI.resident_shift(FlowMatchEulerDiscreteScheduler()) is not None:
            # its config shift is another parameterisation than the baseline's flux shift (1.936): reading it voided
            # an equal FLUX.3 pair (sigmas agree) on the Nov 19 L4 round
            failures.append("a FlowMatch scheduler's config shift was reported as the run's shift")
        req = SI.studio_resolved({}, {**cell, "options": {"generate": {"flow_shift": 5.0}}}, 20,
                                 {"resident_shift": 8.0})
        if req.get("shift") != 5.0:
            failures.append(f"a requested shift lost to the resident one: {req.get('shift')}")
    guard("wan-shift", cwanshift)

    # MiniMax-H3 (A100 round, Nov 19): Studio's H3 scheduler builds its grid in h3_simple_sigmas and never reaches
    # the stock set_timesteps the hook wraps, so the probe dumped no sigmas; and the card leaves Studio's audio VAE
    # dtype unknown, which the gate read as missing. The hook now emits that grid; the loaded VAEs' dtypes fill a
    # card component left null
    def ch3():
        import types

        import torch

        import parity as P
        from backends import studio_inproc as SI
        from shim import dbench_hooks as H

        got = []
        orig = H._emit
        H._emit = got.append
        try:
            mod = types.ModuleType("core.inference.video_minimax_h3_sampling")
            mod.h3_simple_sigmas = lambda shift, steps: torch.tensor([1.0, 0.5, 0.0])
            H._patch_h3_sampling(mod)
            mod.h3_simple_sigmas(12.0, 2)
        finally:
            H._emit = orig
        saved = os.environ.get("DBENCH_PARITY_DUMP")
        os.environ["DBENCH_PARITY_DUMP"] = "1"
        try:
            targets = H._targets()
        finally:
            if saved is None:
                os.environ.pop("DBENCH_PARITY_DUMP", None)
            else:
                os.environ["DBENCH_PARITY_DUMP"] = saved
        if not got or got[0].get("sigmas") != [1.0, 0.5, 0.0] or got[0].get("stack") != "studio" or \
                "core.inference.video_minimax_h3_sampling" not in targets:
            failures.append(f"H3 sigma grid not emitted by the hook: {got}")
        pipe = types.SimpleNamespace(vae = torch.nn.Linear(2, 2).half(), audio_vae = torch.nn.Linear(2, 2))
        md = SI.module_dtypes(pipe)
        if md.get("audio_vae", {}).get("weights") != "fp32" or md.get("vae", {}).get("weights") != "fp16":
            failures.append(f"loaded VAE dtypes not read: {md}")
        card = {"formats": {"x": {"studio": {"precision": {"audio_vae": {"weights": None, "compute": None},
                                                           "vae": {"weights": "fp16", "compute": "fp16"}}}}}}
        rec = {"arm": {"stack": "studio"}, "cell": {"variant": "x"},
               "parity": {"precision": {"audio_vae": md["audio_vae"], "vae": {"weights": "bf16", "compute": "bf16"}}}}
        side, src = P.side_precision(rec, card)
        if side["audio_vae"]["weights"] != "fp32" or side["vae"]["weights"] != "fp16" or "backend report" not in src:
            failures.append(f"a card-unknown component did not take the backend's strong report: {side} {src}")
        weak = {**rec, "parity": {"precision": {"audio_vae": "fp32"}}}
        if P.side_precision(weak, card)[0]["audio_vae"]["weights"] is not None:
            failures.append("a weak (bare token) report filled a card-unknown component")
    guard("h3-evidence", ch3)

    # MiniMax-H3-Image (3090 recheck, Nov 20): Studio's MiniMaxH3ImageScheduler lives in
    # core.inference.diffusion_minimax_h3_image, which the diffusers.schedulers hook never reached, so the probe had
    # no sigma dump and no sampler / scheduler in process or over HTTP. The hook now wraps that module's scheduler and
    # the class name resolves to euler / simple (in process from the pipeline, over HTTP from the dumped class)
    def ch3i():
        import types

        import torch

        import run_cell as R
        from backends import studio_inproc as SI
        from shim import dbench_hooks as H

        name = "core.inference.diffusion_minimax_h3_image"
        saved = os.environ.get("DBENCH_PARITY_DUMP")
        os.environ["DBENCH_PARITY_DUMP"] = "1"
        try:
            fn = H._targets().get(name)
        finally:
            if saved is None:
                os.environ.pop("DBENCH_PARITY_DUMP", None)
            else:
                os.environ["DBENCH_PARITY_DUMP"] = saved
        if fn is None:
            failures.append(f"no parity hook targets {name}")
            return
        mod = types.ModuleType(name)

        class MiniMaxH3ImageScheduler:
            def set_timesteps(self, num_inference_steps = None, device = None, sigmas = None):
                self.sigmas = torch.tensor([1.0, 0.75, 0.25, 0.0])

        MiniMaxH3ImageScheduler.__module__ = name
        mod.MiniMaxH3ImageScheduler = MiniMaxH3ImageScheduler
        got = []
        orig = H._emit
        H._emit = got.append
        try:
            fn(mod)
            mod.MiniMaxH3ImageScheduler().set_timesteps(3)
        finally:
            H._emit = orig
        ev = {**H.summarize(got), "source": "hooks"}
        if ev.get("sigmas") != [1.0, 0.75, 0.25, 0.0] or ev.get("sigma_source") != "MiniMaxH3ImageScheduler":
            failures.append(f"H3-Image sigma grid not emitted by the hook: {got}")
        res = R._sampler_from_hooks({"steps": 3, "sampler": None, "scheduler": None}, ev)
        if (res.get("sampler"), res.get("scheduler")) != ("euler", "simple"):
            failures.append(f"H3-Image sampler / scheduler unresolved over HTTP: {res}")
        inproc = SI.studio_sampler({"sampler": None, "schedule_type": None}, MiniMaxH3ImageScheduler())
        if (inproc.get("sampler"), inproc.get("scheduler")) != ("euler", "simple"):
            failures.append(f"H3-Image sampler / scheduler unresolved in process: {inproc}")
    guard("h3-image-sigmas", ch3i)

    # a mirror-rule VOID carries the card's own account of what would unblock the pair (MiniMax-H3's NVFP4 AWQ
    # encoder: published only by the baseline UI's org, a quantised derivative no official file verifies)
    def cunblock():
        import mirror as M
        import models as MC

        card = MC.load_card("minimax-h3")
        te = card["formats"]["int8"]["bline"]["files"]["te"]
        rec = {"arm": {"stack": "blineui"}, "settings": {"files": {"applied": {"te": "x/" + te}}}}
        probs = M.record_problems(rec, card, "int8")
        if not probs or "unblocks when:" not in probs[0] or "int8_teint8" not in probs[0]:
            failures.append(f"mirror VOID does not say what unblocks it: {probs}")
    guard("mirror-unblock", cunblock)

    # mirror rule at run time: a mirror copy is used only after every tensor equals the official checkpoint's (key
    # map for renames / fused tensors, optional cast); a changed, extra or unmapped tensor fails it; a baseline record
    # that ran a file without an official source and without a recorded verification is VOID at the parity gate
    def cmirror():
        import tempfile

        import torch
        from safetensors.torch import save_file

        import mirror as M
        import parity as P

        d = Path(tempfile.mkdtemp(dir = os.environ.get("TMPDIR")))
        q, k, v, w = (torch.randn(4, 3, dtype = torch.bfloat16) for _ in range(4))
        f32 = torch.randn(3, dtype = torch.float32)
        save_file({"l.0.q": q, "l.0.k": k, "l.0.v": v, "l.0.w": w, "n": f32.to(torch.bfloat16)}, str(d / "off1.st"))
        save_file({"extra.unused": torch.zeros(2)}, str(d / "off2.st"))
        km = {"l.{i}.qkv": {"cat": ["l.{i}.q", "l.{i}.k", "l.{i}.v"], "dim": 0}, "x.{i}.w": "l.{i}.w"}
        offs = [d / "off1.st", d / "off2.st"]
        save_file({"l.0.qkv": torch.cat([q, k, v]), "x.0.w": w, "n": f32.to(torch.bfloat16)}, str(d / "good.st"))
        bad_w = w.clone()
        bad_w[0, 0] += 1
        save_file({"l.0.qkv": torch.cat([q, k, v]), "x.0.w": bad_w}, str(d / "changed.st"))
        save_file({"l.0.qkv": torch.cat([q, k, v]), "x.0.w": w, "mirror.only": torch.ones(1)}, str(d / "extra.st"))
        save_file({"l.0.qkv": torch.cat([q, v, k]), "x.0.w": w}, str(d / "order.st"))
        good = M.verify_tensors(d / "good.st", offs, km)
        if not good["verified"] or good["compared"] != 3 or good["n_official_unused"] != 1:
            failures.append(f"mirror: an exact repackage (rename + fused qkv) was not verified: {good}")
        for name in ("changed", "extra", "order"):
            if M.verify_tensors(d / f"{name}.st", offs, km)["verified"]:
                failures.append(f"mirror negative control passed: {name} tensor")
        save_file({"l.0.w": w, "n": f32}, str(d / "f32.st"))
        save_file({"l.0.w": w, "n": f32.to(torch.bfloat16)}, str(d / "off_bf16.st"))
        if M.verify_tensors(d / "f32.st", [d / "off_bf16.st"])["verified"]:
            failures.append("mirror: a dtype change passed without cast")
        # cast compares the official tensor cast to the mirror's dtype: bf16 -> f32 of a rounded value != f32 value
        if M.verify_tensors(d / "f32.st", [d / "off_bf16.st"], cast = True)["verified"]:
            failures.append("mirror: cast hid a precision loss (official bf16 vs mirror f32 values)")
        entry = {"repo": "mir/x", "file": "m.st", "sha256": "a" * 64, "mirror": True,
                 "official": {"repo": "off/x", "files": ["o.st"]}, "verify": {"method": "tensors", "key_map": km}}
        try:
            M.verify_entry(entry, d / "changed.st", offs)
            failures.append("mirror: verify_entry let a changed tensor through")
        except RuntimeError:
            pass
        ok = M.verify_entry(entry, d / "good.st", offs)
        card = {"formats": {"bf16": {"bline": {"sources": {"dit.st": {"source": None, "reason": "mirror only"}}}}}}
        rec = {"arm": {"stack": "blineui"}, "cell": {"variant": "bf16"},
               "settings": {"files": {"applied": {"dit": "dit.st", "vae": "vae.st"}}}}
        if not M.record_problems(rec, card, "bf16"):
            failures.append("mirror: a baseline record that ran an unsourced file unverified was not flagged")
        if M.record_problems({**rec, "staged_files": {"dit.st": ok}}, card, "bf16") or \
                "verified by tensors: 3/3 tensors equal" not in " ".join(M.staged_lines({"staged_files": {"dit.st": ok}})):
            failures.append(f"mirror: a verified staged file was flagged or not described: {ok}")
        a = {"tag": "s", "verdict": "ok", "arm": {"stack": "studio"}, "cell": {"variant": "bf16"}, "parity": {}}
        res = P.check_pair(a, {**rec, "tag": "c", "verdict": "ok", "parity": {}}, card)
        if "mirror_blineui" not in res["checks"] or res["checks"]["mirror_blineui"]["status"] != "mismatch":
            failures.append(f"parity gate passed an unverified mirror file: {res['checks'].get('mirror_blineui')}")
    guard("mirror", cmirror)

    # #15: a keyword-only denoiser input keeps its batch rows; a nested denoiser is not a second evaluation
    def c15():
        dump = pc / "hooks.jsonl"
        dump.unlink(missing_ok = True)
        code = textwrap.dedent(f"""
            import sys, torch
            sys.path.insert(0, {str(HERE / 'shim')!r})
            import dbench_hooks as H
            H._patch_torch_module_hook(torch.nn.modules.module)
            class ModelMixin(torch.nn.Module):
                pass
            class InnerTransformer2DModel(ModelMixin):
                def forward(self, hidden_states):
                    return hidden_states
            class OuterUNet2DConditionModel(ModelMixin):
                def __init__(self):
                    super().__init__()
                    self.inner = InnerTransformer2DModel()
                def forward(self, sample=None):
                    return self.inner(sample)
            OuterUNet2DConditionModel()(sample=torch.zeros(2, 4))
        """)
        subprocess.run([sys.executable, "-c", code], capture_output = True, text = True,
                       env = {**os.environ, "DBENCH_PARITY_DUMP": str(dump)})
        sys.path.insert(0, str(HERE / "shim"))
        import dbench_hooks as H

        ev = H.summarize(H.read_events(dump))
        if ev.get("denoiser_calls") != 1 or ev.get("denoiser_items") != 2:
            failures.append(f"#15 keyword input / nested denoiser miscounted: calls {ev.get('denoiser_calls')} "
                            f"items {ev.get('denoiser_items')}")
    guard("#15", c15)

    # the extnames module the hooks load by path is importable by name: torch.compile imports the module of every
    # function it traces, and an unregistered one made the Studio parity probes run eager
    def cext():
        code = ("import sys, importlib; sys.path.insert(0, sys.argv[1]); import dbench_hooks as H; H._ext('x'); "
                "importlib.import_module('dbench_extnames'); print('OK')")
        r = subprocess.run([sys.executable, "-c", code, str(HERE / "shim")], capture_output = True, text = True)
        if "OK" not in r.stdout:
            failures.append(f"dbench_extnames not importable after the hooks load it: {r.stderr[-300:]}")
    guard("#ext", cext)

    # the sigmas hook keeps set_timesteps' signature: diffusers' retrieve_timesteps reads it before passing custom
    # sigmas, and refuses a scheduler whose signature lacks them (FLUX.3 klein parity probe, Studio 96bf6de124d)
    def csig():
        import inspect
        import types

        sys.path.insert(0, str(HERE / "shim"))
        import dbench_hooks as H

        mod = types.ModuleType("diffusers.schedulers.scheduling_dbench_selftest")

        class FakeScheduler:
            def set_timesteps(self, num_inference_steps = None, device = None, sigmas = None, mu = None):
                self.sigmas = sigmas

        FakeScheduler.__module__ = mod.__name__
        mod.FakeScheduler = FakeScheduler
        H._patch_scheduler_module(mod)
        params = inspect.signature(FakeScheduler.set_timesteps).parameters
        if not getattr(FakeScheduler.set_timesteps, "_dbench", False) or "sigmas" not in params:
            failures.append(f"sigmas hook hid set_timesteps' signature: {list(params)}")
    guard("sigmas-signature", csig)

    # #65: settings the backend did not read back (source "cell") are missing, not equal
    def c65():
        a = {"verdict": "ok", "arm": {"stack": "studio"}, "parity": {"resolved": {"steps": 4, "source": "studio"}}}
        b = {"verdict": "ok", "arm": {"stack": "sdcpp"},
             "parity": {"resolved": {"steps": 4, "source": "cell (not read back from the stack)"}}}
        if P.check_pair(a, b)["checks"]["settings"]["status"] != "missing":
            failures.append("#65 cell-sourced settings were compared as read back")
    guard("#65", c65)

    # #67: missing precision evidence on one side is never an accepted difference
    def c67():
        q = {"transformer": {"weights": "int8_convrot", "compute": "int8"},
             "text_encoder": {"weights": "bf16", "compute": "bf16"}}
        card = {"formats": {"int8": {"studio": {"precision": q}, "bline": {}}},
                "traps": [{"id": "te", "check": "precision_equal", "accepted_difference": True,
                           "components": ["text_encoder"], "formats": ["int8"], "note": "x"}]}
        a = {"verdict": "ok", "arm": {"stack": "studio"}, "cell": {"variant": "int8"},
             "parity": {"precision": {"transformer": "int8", "text_encoder": "bf16"}}}
        b = {"verdict": "ok", "arm": {"stack": "blineui"}, "cell": {"variant": "int8"},
             "parity": {"precision": {"transformer": "int8"}}}
        row, acc, _ = P.check_precision(a, b, card)
        if any(x["component"] == "text_encoder" for x in acc) or row["status"] == "ok":
            failures.append(f"#67 one-sided missing evidence accepted: {row} {acc}")
    guard("#67", c67)

    # #75: a gate that could not run is "unavailable", never "ok"
    def c75():
        import run_cell as R
        import telemetry as T

        saved = T.cooldown
        T.cooldown = lambda *a, **k: {"gate": "unavailable"}
        try:
            g = R._gates({"options": {"cool_to_c": 60, "foreign_wait": False}}, None, lambda m: None)
        finally:
            T.cooldown = saved
        if g.get("gate") != "unavailable":
            failures.append(f"#75 an unavailable gate summarised as {g.get('gate')}")
    guard("#75", c75)

    # #52: own-tree VRAM NVML cannot attribute is UNAVAILABLE (None), not 0.0
    def c52():
        import telemetry as T

        s = T.Sampler.__new__(T.Sampler)
        s.fake, s.h, s.device, s.own_vram_fallback = None, object(), "0", None
        saved = T.compute_procs
        T.compute_procs = lambda device: {999999: 512}
        try:
            got = s._own({os.getpid()})
        finally:
            T.compute_procs = saved
        if got is not None:
            failures.append(f"#52 unattributed own VRAM reported as {got}")
    guard("#52", c52)

    # #13: a direct fetch with the wrong bytes fails the card's sha256 (the VM driver and fetch_files)
    def c13():
        import huggingface_hub

        from targets import fetch_files as FF
        from targets import vm_driver as VD

        bad = pc / "dl" / "w.safetensors"
        bad.parent.mkdir(parents = True, exist_ok = True)
        bad.write_bytes(b"not the pinned file")
        entry = {"repo": "org/repo", "file": "w.safetensors", "sha256": "0" * 64}
        saved = huggingface_hub.hf_hub_download
        huggingface_hub.hf_hub_download = lambda *a, **k: str(bad)
        try:
            doc = {"cells": [{"tag": "t", "options": {"files": {"dit": "w.safetensors"}}}],
                   "downloads": {"w.safetensors": entry}}
            for name, fn in (("fetch_files", lambda: FF.resolve(doc, pc / "ffroot")),
                             ("vm_driver", lambda: VD.Files(pc, ["t"], {"t": doc["cells"][0]}, None,
                                                            doc["downloads"]).resolve(doc["cells"][0]))):
                try:
                    fn()
                    failures.append(f"#13 {name} accepted a download whose sha256 differs from the card")
                except RuntimeError:
                    pass
        finally:
            huggingface_hub.hf_hub_download = saved
    guard("#13", c13)

    # #47: a token inside a compressed artifact is caught by the leak scan
    def c47():
        from targets import _remote as RM

        run = pc / "leak_run"
        run.mkdir(parents = True, exist_ok = True)
        buf = io.BytesIO()
        tok = ("hf_" + "".join(chr(c) for c in range(65, 91)) + "abcdefgh").encode()
        with tarfile.open(fileobj = buf, mode = "w:gz") as tf:
            ti = tarfile.TarInfo("cell/server.log")
            ti.size = len(tok)
            tf.addfile(ti, io.BytesIO(tok))
        (run / "artifact.tgz").write_bytes(buf.getvalue())
        try:
            RM.collect(run)
            failures.append("#47 a token inside artifact.tgz passed the leak scan")
        except SystemExit:
            pass
    guard("#47", c47)

    # #10: the sampler thread stops and joins (Thread._stop not shadowed: Python 3.12 calls it in join())
    def c10():
        import telemetry as T

        s = T.Sampler(None)
        s.start()
        s.stop()
        s.join(timeout = 5)
    guard("#10", c10)
    del schema, types


def shim_checks(root: Path, failures: list) -> None:
    """The PYTHONPATH shim on fake torch / bline modules: simulated total, parity events, sitecustomize chaining."""
    import budget as B
    import extnames as X

    fake = root / "fakemods"
    # the fake server package is named as the REAL one (extnames.ext: identity here, the real spelling in a public copy)
    for rel, text in {
        "torch/__init__.py": "",
        "torch/cuda/__init__.py": "def mem_get_info(device=None):\n    return (5 * 2**30, 80 * 2**30)\n",
        "bline/__init__.py": "",
        "bline/samplers.py": textwrap.dedent("""
            class CFGGuider:
                cfg = 1.0
                def sample(self, noise, latent_image, sampler, sigmas, *a, **k):
                    return sigmas
        """),
        "bline/model_base.py": textwrap.dedent("""
            class BaseModel:
                def apply_model(self, x, t, **k):
                    return x
        """),
        "chain/sitecustomize.py": "import os\nos.environ['CHAINED_SITE'] = '1'\n",
    }.items():
        p = fake / X.ext(rel)
        p.parent.mkdir(parents = True, exist_ok = True)
        p.write_text(text)
    dump = root / "dump.jsonl"
    env = B.child_env({k: v for k, v in os.environ.items() if k != "PYTHONPATH"}, {"sim_total_gib": 1}, str(dump))
    env["PYTHONPATH"] += os.pathsep + str(fake) + os.pathsep + str(fake / "chain")
    code = textwrap.dedent("""
        import importlib, os, torch.cuda
        S = importlib.import_module(os.environ["DBENCH_TEST_PKG"] + ".samplers")
        M = importlib.import_module(os.environ["DBENCH_TEST_PKG"] + ".model_base")
        class X:
            shape = (2, 4)
        S.CFGGuider().sample([0.0], None, None, [1.0, 0.5, 0.0])
        for _ in range(3):
            M.BaseModel().apply_model(X(), 0)
        print(torch.cuda.mem_get_info()[1], os.environ.get('CHAINED_SITE'))
    """)
    env["DBENCH_TEST_PKG"] = X.ext("bline")
    r = subprocess.run([sys.executable, "-c", code], env = env, capture_output = True, text = True, cwd = str(root))
    if r.returncode != 0 or r.stdout.split() != [str(2**30), "1"]:
        failures.append(f"shim: simulated total / chained sitecustomize wrong: {r.stdout!r} {r.stderr[-800:]}")
    sys.path.insert(0, str(B.SHIM))
    import dbench_hooks as H

    ev = H.summarize(H.read_events(dump))
    if ev.get("sigmas") != [1.0, 0.5, 0.0] or ev.get("denoiser_calls") != 3 or ev.get("denoiser_items") != 6:
        failures.append(f"shim parity events wrong: {ev}")


def kill_check(failures: list) -> None:
    code = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"
    proc = subprocess.Popen([sys.executable, "-c", code], start_new_session = os.name != "nt")
    time.sleep(0.5)
    try:
        import psutil

        kids = psutil.Process(proc.pid).children(recursive = True)
    except Exception:  # noqa: BLE001
        kids = []
    C.kill_tree(proc, timeout = 5)
    if proc.poll() is None or any(k.is_running() and k.status() != "zombie" for k in kids):
        failures.append("kill_tree left a process running")


def cluster_checks(root: Path, failures: list) -> None:
    """Negative controls for panel clusters #28, #29, #30, #35, #68 and #78 (review876_tally.txt); each fails on
    6d9eb747."""
    import bline_graphs as G
    import plan as P
    import score as S
    from backends.blineui import BlineUIBackend, BlineUIError

    cc = root / "clusters"
    cc.mkdir(parents = True, exist_ok = True)

    def guard(name, fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a crash on the old tree is that control failing
            failures.append(f"{name}: {type(exc).__name__}: {str(exc)[:300]}")

    def backend(cell, names, info):
        b = BlineUIBackend(cell, cc)
        b.names, b.info = names, {"commit": "0" * 10}
        b.api = lambda path, payload = None, timeout = 30: info
        return b

    def spec(*names_):  # an object_info entry per class: a COMBO list for the file inputs
        return {"input": {"required": {k: [list(v)] for k, v in names_}}}

    # #28: the load-time node check validates the graph the cell renders (GGUF loader rewrite, the cell's frames)
    def c28():
        z = {"family": "z-image", "width": 1024, "height": 1024, "steps": 8, "options": {"family": "z-image"}}
        names = {"dit": "z-Q4_K_M.gguf", "te": "te.safetensors", "vae": "vae.safetensors"}
        g = G.build("z-image", G.GraphParams(prompt = "p", seed = 0, steps = 8, width = 1024, height = 1024,
                                             files = names))
        info = {n["class_type"]: {"input": {"required": {}}} for n in g.values()}
        info.update({"UNETLoader": spec(("unet_name", ["other.safetensors"])),
                     "UnetLoaderGGUF": spec(("unet_name", [names["dit"]]))})
        backend(z, names, info)._check_nodes(G.module("z-image"))  # base: "UNETLoader.unet_name ... not in BlineUI's list"
    guard("#28", c28)

    def c28b():
        ltx = {"family": "ltx-2.3", "kind": "video", "width": 1280, "height": 704, "frames": 121, "fps": 25,
               "steps": 8, "options": {"family": "ltx-2.3"}}
        b = backend(ltx, {"checkpoint": "c.safetensors", "te": "t.safetensors"}, {})
        try:
            b._check_nodes(G.module("ltx-2.3"))
            failures.append(f"#28 the ltx-2.3 node check was skipped ({getattr(b, 'check_skipped', None)})")
        except BlineUIError as exc:
            if "has no node" not in str(exc):
                failures.append(f"#28 unexpected node-check error: {exc}")
    guard("#28b", c28b)

    # #29: resolved settings are the render params first (stage-1 latent / CFGOverride are not the cell's values)
    def c29():
        ltx = {"family": "ltx-2.3", "kind": "video", "width": 1280, "height": 704, "frames": 121, "fps": 25,
               "steps": 8, "options": {"family": "ltx-2.3", "cfg": 4.0}}
        b = backend(ltx, {"checkpoint": "c.safetensors", "te": "t.safetensors"}, {})
        b._params = G.GraphParams(prompt = "p", seed = 0, steps = 8, width = 1280, height = 704, frames = 121,
                                  cfg = 4.0, files = b.names)
        b.graph = G.build("ltx-2.3", b._params)
        r = b.resolved_settings()
        want = {"width": 1280, "height": 704, "frames": 121, "steps": 8, "cfg": 4.0}
        got = {k: r.get(k) for k in want}
        if got != want:
            failures.append(f"#29 resolved settings {got} != the cell's {want}")
    guard("#29", c29)

    # #30: CFGNorm (a MODEL -> MODEL patch) gets no per-render nonce, like ModelSamplingAuraFlow
    def c30():
        g = {"n": {"class_type": "CFGNorm", "inputs": {"strength": 1.0}}, G.SAVE: {"class_type": "SaveImage", "inputs": {}}}
        G.apply_nonce(g, G.GraphParams(prompt = "p", seed = 0, steps = 1, width = 64, height = 64, files = {}, nonce = 3,
                                       nonce_nodes = "all"))
        if "_bench_nonce" in g["n"]["inputs"]:
            failures.append("#30 CFGNorm carries a per-render nonce (re-patched every render)")
    guard("#30", c30)

    cards = P.load_cards(["qwen-image-2.1"], None)
    base = {"name": "x", "revisions": {"unsloth_sha": "0" * 40, "zoo_sha": "1" * 40, "bline_ref": "v0.45.6"}, "models": ["qwen-image-2.1"],
            "repeats": 3, "n": 2, "targets": ["colab:A100-80"],
            "arms": [{"stack": "studio", "format": "int8"}, {"stack": "blineui", "format": "int8"}]}

    # #35: an arm on a stack the card cannot map (diffusers) is refused alone, not the whole comparison
    def c35():
        pl = P.build_plan({**base, "arms": base["arms"] + [{"stack": "diffusers", "format": "int8"}]}, cards)
        why = [r["reason"] for r in pl["not_run"] if r["arm"] == "d_int8"]
        if len(pl["comparisons"]) != 1 or not any("diffusers model" in w for w in why):
            failures.append(f"#35 a diffusers arm without a model dropped the comparison "
                            f"({len(pl['comparisons'])} comparisons, {why})")
    guard("#35", c35)

    # #68: a target listed twice is refused by name; arm generate merges with the card's; fp8 cells carry the
    # compute capability their kernels need (tester:<label> names no GPU)
    def c68():
        try:
            P.build_plan({**base, "targets": ["colab:A100-80", "colab:A100-80"]}, cards)
            failures.append("#68b a duplicate target was planned")
        except SystemExit as exc:
            if "listed twice" not in str(exc):
                failures.append(f"#68b duplicate target: {exc}")
        except AssertionError as exc:
            failures.append(f"#68b duplicate target fails as an internal assert: {str(exc)[:120]}")
        card = {"family": "x", "kind": "image", "defaults": {"steps": 20, "width": 512, "height": 512},
                "formats": {"bf16": {"studio": {"model": "m"}}, "fp8": {"studio": {"model": "m8"}}},
                "studio": {"gen": {"true_cfg_scale": 4.0}}}
        cell = P.card_to_cell(card, {"stack": "studio", "format": "bf16",
                                     "options": {"generate": {"steps": 9, "extra": 1}}}, 2, 1)
        if cell["options"].get("generate") != {"true_cfg_scale": 4.0, "extra": 1} or cell.get("steps") != 9:
            failures.append(f"#68d arm generate replaced the card's or kept a request field: "
                            f"{cell['options'].get('generate')} steps={cell.get('steps')}")
        cell = P.card_to_cell(card, {"stack": "studio", "format": "fp8"}, 2, 1)
        if cell.get("min_compute_cap") != "8.9":
            failures.append(f"#68e an fp8 cell carries no min_compute_cap ({cell.get('min_compute_cap')})")
    guard("#68", c68)

    # #78: Studio's max tier is a step cache for the derived s/step, and score.py never falls back to a flagged one
    def c78():
        if not C.step_cache_on({"options": {"speed_mode": "max"}}):
            failures.append("#78 speed_mode=max is not flagged as a step cache")
        d = cc / "score78"
        (d / "c1").mkdir(parents = True, exist_ok = True)
        (d / "c1" / C.RECORD).write_text(json.dumps({"tag": "c1", "verdict": "NOT_RUN", "reason": "x",
                                                     "step_s_derived": 0.5, "timing": {
                                                         "step_s_derived": 0.5,
                                                         "step_s_derived_flag": "step_cache_on:speed_mode=max"}}))
        rows = S.score_dir(d, {}, None, None, [])["rows"]
        if rows[0].get("step_s") is not None:
            failures.append(f"#78 score.py used a flagged derived s/step ({rows[0].get('step_s')})")
    guard("#78", c78)

    # #79: fetch_files pulls the plan's Studio prefetch repos into $HF_HOME/hub (vm_driver.prefetch), not only files
    def c79():
        import huggingface_hub

        from targets import fetch_files as FF

        got = []
        saved = (huggingface_hub.snapshot_download, huggingface_hub.hf_hub_download, os.environ.get("HF_HOME"))
        hub_saved = os.environ.pop("HF_HUB_CACHE", None)  # HF_HOME/hub is the cache when no HF_HUB_CACHE is set
        huggingface_hub.snapshot_download = lambda repo, **k: got.append(("snap", repo, k.get("cache_dir")))
        huggingface_hub.hf_hub_download = lambda repo, f, **k: got.append(("file", repo, k.get("cache_dir")))
        os.environ["HF_HOME"] = str(cc / "hf79")
        try:
            doc = {"cells": [], "prefetch": {"a/pipe": {"repo": "a/pipe"},
                                             "b/one:x.gguf": {"repo": "b/one", "file": "x.gguf"}}}
            FF.prefetch(doc, download = False)
            if got:
                failures.append(f"#79 a dry run prefetched {got}")
            FF.prefetch(doc)
        finally:
            huggingface_hub.snapshot_download, huggingface_hub.hf_hub_download = saved[:2]
            if saved[2] is None:
                os.environ.pop("HF_HOME", None)
            else:
                os.environ["HF_HOME"] = saved[2]
            if hub_saved is not None:
                os.environ["HF_HUB_CACHE"] = hub_saved
        hub = os.path.join(str(cc / "hf79"), "hub")
        if sorted(got) != [("file", "b/one", hub), ("snap", "a/pipe", hub)]:
            failures.append(f"#79 fetch_files prefetch fetched {got}, want both repos into {hub}")
    guard("#79", c79)

    # #80: a relative --out still gives stack servers (own cwd) an absolute parity dump and work dirs
    def c80():
        import budget as Bu
        from backends.base import Backend

        rel = os.path.relpath(str(cc / "rel80" / "_parity.jsonl"))
        env = Bu.child_env({}, None, rel)
        if not os.path.isabs(env.get("DBENCH_PARITY_DUMP", "")):
            failures.append(f"#80 child_env exported a relative parity dump ({env.get('DBENCH_PARITY_DUMP')})")
        import types

        saved = {k: os.environ.get(k) for k in ("DBENCH_PARITY_DUMP", "DBENCH_SIM_TOTAL_MIB")}
        real = sys.modules.get("dbench_hooks")
        sys.modules["dbench_hooks"] = types.SimpleNamespace(install = dict)  # the env only, no import hooks here
        try:
            os.environ.pop("DBENCH_SIM_TOTAL_MIB", None)
            Bu.install_in_process(None, rel)
            if not os.path.isabs(os.environ.get("DBENCH_PARITY_DUMP", "")):
                failures.append("#80 install_in_process set a relative parity dump")
        finally:
            if real is None:
                sys.modules.pop("dbench_hooks", None)
            else:
                sys.modules["dbench_hooks"] = real
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        b = Backend({"tag": "t", "options": {}}, Path(os.path.relpath(str(cc / "rel80"))))
        if not b.out.is_absolute():
            failures.append(f"#80 backend work root stayed relative ({b.out})")
    guard("#80", c80)

    def q21(fmt):  # the real card's baseline file sources for one format
        card = json.loads((HERE / "models" / "qwen-image-2.1.json").read_text())
        return card["formats"][fmt]["bline"]["sources"]

    # #81: the VM driver and fetch_files hand builders the baseline python / checkout / ConvRot tool explicitly
    # (builders run in the driver process, whose environment has neither); BASELINE_* names are read too
    def c81():
        import builders
        from targets import fetch_files as FF
        from targets import vm_driver as VD

        root = cc / "vm81"
        (root / "src" / "BlineUI").mkdir(parents = True, exist_ok = True)
        (root / "bline_py.txt").write_text("/vm/venvs/base/bin/python\n")
        (root / "convrot_tool.txt").write_text("/vm/src/convrot_tool/q.py\n")
        want = {"python": "/vm/venvs/base/bin/python", "bline_dir": str(root / "src" / "BlineUI"),
                "tool": "/vm/src/convrot_tool/q.py"}
        name = "qwen_image_2.1_vae_bf16.safetensors"
        downloads = {name: q21("bf16")[name]}
        cell = {"tag": "t", "options": {"files": {"vae": name}}}
        got = []
        saved = builders.build_file
        builders.build_file = lambda entry, n, out, **k: got.append(k) or Path(out) / n
        try:
            VD.Files(root, ["t"], {"t": cell}, None, downloads, build_kw = VD.builder_kw(root)).resolve(cell)
            FF.resolve({"cells": [cell], "downloads": downloads}, root / "ff", build_kw = want)
        finally:
            builders.build_file = saved
        for who, k in zip(("vm_driver", "fetch_files"), got):
            if any(k.get(x) != v for x, v in want.items()):
                failures.append(f"#81 {who} called a builder with {k}, want {want}")
        if len(got) != 2:
            failures.append(f"#81 expected 2 builder calls, got {len(got)}")
        envs = {k: os.environ.pop(k, None) for k in ("DIFFUSION_BENCH_BLINE_PYTHON", "DIFFUSION_BENCH_BASELINE_PYTHON")}
        os.environ["DIFFUSION_BENCH_BASELINE_PYTHON"] = "/b/python"
        try:
            py = builders.plan_build(downloads[name], name, cc / "pb81")["cmds"][0][0]
        finally:
            for k, v in envs.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v
        if py != "/b/python":
            failures.append(f"#81 builders ignored DIFFUSION_BENCH_BASELINE_PYTHON (ran {py})")
    guard("#81", c81)

    # #82: a VM whose plan needs the int8 ConvRot quantizer gets its pinned copy from setup (or is refused at prepare)
    def c82():
        import extnames
        from targets import notebook as NB

        src = q21("int8")
        cells = {"t": {"tag": "t", "options": {"files": {k: k for k in src}}}}
        sl = {"vm": {"vm_id": "v82"}, "cells": cells, "downloads": src}
        envs = {k: os.environ.pop(k, None) for k in ("DBENCH_CONVROT_REPO", "DBENCH_CONVROT_SHA", "DBENCH_CONVROT_PATH")}
        # the private pin when present; a public copy has none, so the environment route stands in for it
        pin = extnames.convrot_tool(required = False) or {"repo": "https://example.invalid/tools", "sha": "a" * 40,
                                                           "path": "q.py"}
        os.environ.update({f"DBENCH_CONVROT_{k.upper()}": v for k, v in pin.items()})
        try:
            line = NB.convrot_env(sl)
        finally:
            for k in envs:
                os.environ.pop(k, None)
        if f"CONVROT_SHA={pin['sha']}" not in line or f"CONVROT_REPO={pin['repo']}" not in line:
            failures.append(f"#82 the setup cell does not ask for the pinned quantizer ({line!r})")
        if NB.convrot_env({**sl, "downloads": {k: v for k, v in src.items() if "convrot" not in k}}):
            failures.append("#82 a slice without int8 ConvRot builds still fetches the quantizer")
        sh = (HERE / "targets" / "vm_setup.sh").read_text()
        if "convrot_tool.txt" not in sh or "CONVROT_SHA" not in sh:
            failures.append("#82 vm_setup.sh does not provision the quantizer for the driver")
        saved = extnames.private_defaults
        extnames.private_defaults = dict
        try:
            NB.convrot_env(sl)
            failures.append("#82 no quantizer pin, yet the int8 arms were prepared")
        except SystemExit as exc:
            if "DBENCH_CONVROT_REPO" not in str(exc):
                failures.append(f"#82 the refusal is not actionable: {exc}")
        finally:
            extnames.private_defaults = saved
            for k, v in envs.items():
                if v is not None:
                    os.environ[k] = v
    guard("#82", c82)

    # #83: the Metal target reads Studio's own memory from the MPS driver (vram_source "mps_driver"); a stack with no
    # Metal reading is an explicit UNAVAILABLE there, not VOID; a missing VRAM metric elsewhere still VOIDs
    def c83():
        import types

        import run_cell as RC
        import schema
        import telemetry as T
        from backends.base import Backend

        def ok_rec(target, backend):
            r = schema.new_record({"tag": "m83", "backend": backend, "options": {"target": {"name": target}}})
            r.update({"verdict": "ok", "requests": [{"kind": "new", "total_s": 1.0}]})
            r["timing"]["new_s_median"] = 1.0
            return r

        def none_sampler(hits = 0):
            s = T.Sampler.__new__(T.Sampler)
            s.source, s.fallback_hits = "none", hits
            return s

        vram = lambda r: [x for x in schema.validate(r) if "VRAM metric" in x]  # noqa: E731
        mac = ok_rec("mac:macos-15", "blineui")
        RC.vram_source(mac, none_sampler(), None)
        if vram(mac) or mac["memory"]["vram_source"] != "unavailable":
            failures.append(f"#83 a Metal baseline record with no device memory is not UNAVAILABLE: {vram(mac)}")
        nv = ok_rec("colab:L4", "blineui")
        RC.vram_source(nv, none_sampler(), None)
        if not vram(nv):
            failures.append("#83 an NVIDIA ok record without a VRAM metric was accepted")
        nv["memory"].update({"vram_source": "unavailable", "vram_unavailable_reason": "claimed"})
        if not vram(nv):
            failures.append("#83 an UNAVAILABLE claim waived the VRAM metric on an NVIDIA target")
        fake = types.SimpleNamespace(
            cuda = types.SimpleNamespace(is_available = lambda: False),
            backends = types.SimpleNamespace(mps = types.SimpleNamespace(is_available = lambda: True)),
            mps = types.SimpleNamespace(driver_allocated_memory = lambda: 3 * 2**30, synchronize = lambda: None))
        real = sys.modules.get("torch")
        sys.modules["torch"] = fake
        try:
            b = Backend({"tag": "s", "options": {}}, cc / "m83")
            b.in_process = True
            s = none_sampler()
            s.fake, s.h, s.device, s.own_vram_fallback = None, None, None, b.own_vram_mib
            own = s._own(set())
        finally:
            if real is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = real
        st = ok_rec("mac:macos-15", "studio")
        st["memory"]["peak_vram_own_gib"] = None if own is None else round(own / 1024, 3)
        RC.vram_source(st, s, b)
        if own != 3072 or st["memory"]["vram_source"] != "mps_driver" or vram(st):
            failures.append(f"#83 Studio on MPS: own {own} MiB, source {st['memory']['vram_source']}, {vram(st)}")
    guard("#83", c83)

    # #84: one run unpacked from two copies (artifact.tgz + the base64 block in a log / notebook) is one repeat:
    # the collector and report.py dedup on run_id, so duplicates never widen aa_n or narrow the A/A band
    def c84():
        import base64
        import io
        import shutil
        import tarfile

        import report as Rp
        from targets import common as TC

        d = cc / "dup84"
        shutil.rmtree(d, ignore_errors = True)
        a = run_cell(cell("s_pd", "studio"), d / "_parity", parity = True)
        b = run_cell(cell("c_pd", "blineui"), d / "_parity", parity = True)
        write_parity(a, b, d / "_parity" / "bf16")
        run_cell(cell("s_d", "studio"), d)
        run_cell(cell("c_d", "blineui", delay = 0.012), d)
        before = Rp.build(d, use_lpips = False)["cards"][0]["comparisons"][0]["aa_n"]
        for tag in ("s_d", "c_d"):  # the same records again, as a second unpack of the same payload
            shutil.copytree(d / tag, d / "unpacked" / "log_copy" / tag)
        rep = Rp.build(d, use_lpips = False)
        after = rep["cards"][0]["comparisons"][0]["aa_n"]
        if after != before:
            failures.append(f"#84 duplicated payloads changed aa_n {before} -> {after}")
        if not any("duplicate record copies" in x for x in rep["could_not_test"]):
            failures.append("#84 report.py does not warn about dropped duplicate records")
        # the collector (private side only: it restores a public copy's names): one tarball, found as artifact.tgz
        # and inside a log -> one record per run
        import extnames

        if extnames.is_public():
            return
        buf = io.BytesIO()
        with tarfile.open(fileobj = buf, mode = "w:gz") as tf:
            tf.add(str(d / "s_d" / C.RECORD), arcname = "s_d/record.json")
            tf.add(str(d / "c_d" / C.RECORD), arcname = "c_d/record.json")
        run = cc / "dup84_run"
        shutil.rmtree(run, ignore_errors = True)
        run.mkdir(parents = True)
        (run / "artifact.tgz").write_bytes(buf.getvalue())
        (run / "driver.log").write_text(f"x\n{TC.MARK_BEGIN}\n{base64.b64encode(buf.getvalue()).decode()}\n"
                                        f"{TC.MARK_END}\n")
        recs = TC.collect_records(run)
        if sorted(r.get("tag") for r in recs) != ["c_d", "s_d"]:
            failures.append(f"#84 the collector returned {len(recs)} records for 2 runs: "
                            f"{sorted(r.get('tag') for r in recs)}")
    guard("#84", c84)


def nov21_checks(root: Path, failures: list) -> None:
    """Controls for the 2026-11-21 decisions on scripts#876's open questions (each fails on the tree before them)."""
    import parity as P
    import plan as PL

    def guard(name, fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a crash on the old tree is that control failing
            failures.append(f"{name}: {type(exc).__name__}: {str(exc)[:300]}")

    sig = [1.0, 0.75, 0.4, 0.0]

    def mk(stack, sampler, scheduler, sigmas, items):
        return {"verdict": "ok", "tag": stack[0], "arm": {"stack": stack}, "parity": {
            "resolved": {"steps": 3, "cfg": 1.0, "sampler": sampler, "scheduler": scheduler, "width": 64,
                         "height": 64, "source": "x"},
            "sigmas": sigmas, "denoiser_items": items, "denoiser_calls": items}}

    # (1) an unstated Studio sampler / scheduler is covered by equal sigmas AND equal evaluation rows; a stated
    # mismatch, a sigma mismatch, a pass mismatch, or a silent baseline still void
    def c1():
        res = P.check_pair(mk("studio", None, None, sig, 3), mk("blineui", "euler", "simple", list(sig), 3))
        st = res["checks"]["settings"]
        if st["status"] != "ok" or "covered" not in st["detail"]:
            failures.append(f"open q1: unstated Studio sampler with equal sigmas + passes voided: {st}")
        bad = P.check_pair(mk("studio", None, None, sig, 2), mk("blineui", "euler", "simple", list(sig), 3))
        if bad["checks"]["settings"]["status"] != "mismatch":
            failures.append(f"open q1 control: unstated sampler with unequal passes passed: {bad['checks']['settings']}")
        bad = P.check_pair(mk("studio", None, None, sig, 3), mk("blineui", "euler", "simple", [1.0, 0.7, 0.4, 0.0], 3))
        if bad["checks"]["settings"]["status"] != "mismatch":
            failures.append("open q1 control: unstated sampler with different sigmas passed")
        bad = P.check_pair(mk("studio", "dpmpp_2m", "simple", sig, 3), mk("blineui", "euler", "simple", list(sig), 3))
        if bad["checks"]["settings"]["status"] != "mismatch":
            failures.append("open q1 control: a stated, different Studio sampler passed")
        bad = P.check_pair(mk("studio", "euler", "simple", sig, 3), mk("blineui", None, "simple", list(sig), 3))
        if bad["checks"]["settings"]["status"] != "mismatch":
            failures.append("open q1 control: an unstated baseline sampler was covered (only Studio's silence is)")
        from backends.studio_inproc import scheduler_names
        if scheduler_names("FlowMatchEulerDiscreteScheduler", {"FlowMatchEulerDiscreteScheduler": ["x", "y"]})[:2] \
                != ("x", "y"):
            failures.append("open q1: the card's studio.scheduler_map does not win over the generic table")
    guard("open q1 sampler names", c1)

    # (8) the planner's per-run overhead comes from the card's measured est_load_s, per GPU and stack
    def c8():
        card = {"family": "f", "kind": "image", "formats": {"int8": {"est_s_per_render": {"L4": 10.0},
                "est_load_s": {"L4": {"studio": 30.0, "blineui": 90.0}}}}}
        cell = {"variant": "int8", "warmup": 1, "n": 2, "short_n": 0, "options": {"arm": {"stack": "studio"}}}
        if PL.est_cell_min(card, "colab:L4", cell) != round((30.0 + 3 * 10.0) / 60, 2):
            failures.append(f"open q8: card est_load_s not used: {PL.est_cell_min(card, 'colab:L4', cell)}")
        cell["options"]["arm"]["stack"] = "blineui"
        if PL.load_s_estimate(card, "int8", "colab:A100-40", "blineui")[0] != 90.0:
            failures.append("open q8: another GPU's measured load not used when the target has none")
        if PL.load_s_estimate({"kind": "video", "formats": {}}, "x", "colab:L4", "studio")[0] != 240.0:
            failures.append("open q8: the default for an unmeasured card changed")
    guard("open q8 load time", c8)

    # a request "frames" override runs a short clip on every arm of its own comparison id
    def cframes():
        cards = PL.load_cards(["wan2.2-5b"], None)
        req = {"name": "x", "revisions": {"unsloth_sha": "a" * 40, "bline_ref": "v0"},
               "models": [{"family": "wan2.2-5b", "frames": 41}],
               "arms": [{"stack": "studio", "format": "fp16"}, {"stack": "blineui", "format": "fp16"}],
               "targets": ["colab:A100-40"], "repeats": 3, "n": 1}
        plan = PL.build_plan(req, cards)
        cells = [c for c in plan["cells"] if c.get("family") == "wan2.2-5b"]
        if not cells or any(c.get("frames") != 41 for c in cells) or \
                not all("_f41" in (c["options"]["arm"]["comparison"]) for c in cells):
            failures.append(f"frames override not applied to every arm: {[(c.get('tag'), c.get('frames')) for c in cells]}")
    guard("frames override", cframes)

    # a T4 Studio load reads the dense transformer: its prefetch keeps the weights other GPUs skip
    def cdense():
        import models as M
        card = M.load_card("flux.3-klein")
        l4 = PL.studio_prefetch(card, "int8", "colab:L4")
        t4 = PL.studio_prefetch(card, "int8", "kaggle:T4x2")
        snap = lambda pf: [e for e in pf.values() if e.get("file") is None]  # noqa: E731
        if not snap(l4) or not all(e.get("ignore_patterns") for e in snap(l4)):
            failures.append(f"dense_on: the L4 snapshot lost its ignore patterns: {l4}")
        if not snap(t4) or any(e.get("ignore_patterns") for e in snap(t4)) or set(l4) == set(t4):
            failures.append(f"dense_on: the T4 snapshot still skips the dense transformer: {t4}")
    guard("dense_on T4 prefetch", cdense)

    # card_timings: timed records only (no trial / probe / budget run), overhead = load + first-render excess
    def ctimings():
        import card_timings as CT
        d = root / "nov21_timings"

        def rec(tag, purpose, load, first, steady, budget=None):
            r = {"verdict": "ok", "run_id": tag, "tag": tag, "timing": {"load_s": load, "first_render_s": first,
                                                                         "steady_s_median": steady},
                 "cell": {"card": "qwen-image-2.1", "options": {"target": {"name": "colab:L4",
                                                                           "simulated_budget_gib": budget}}},
                 "arm": {"stack": "studio", "format": "int8", "purpose": purpose}}
            (d / tag).mkdir(parents = True, exist_ok = True)
            (d / tag / "record.json").write_text(json.dumps(r))
        rec("a", "timing", 60.0, 30.0, 10.0)
        rec("b", "timing", 80.0, 10.0, 10.0)
        rec("trial__x", "trial", 999.0, 1.0, 1.0)
        rec("p", "parity", 999.0, 1.0, 1.0)
        rec("bud", "timing", 999.0, 1.0, 1.0, budget = 8)
        g = CT.gather([d])
        vals = sorted(v for v, _ in g.get(("qwen-image-2.1", "int8", "L4", "studio"), {}).values())
        if vals != [80.0, 80.0]:
            failures.append(f"card_timings: expected two timed runs at 80 s overhead, got {vals} ({list(g)})")
    guard("card_timings", ctimings)


def main() -> int:
    t0 = time.time()
    import tempfile

    # a private dir per run: two selftests on one host never wipe each other's state (kept when a check fails)
    (C.WS / "temp" / "diffusion_bench").mkdir(parents = True, exist_ok = True)
    base = Path(tempfile.mkdtemp(prefix = "selftest_", dir = str(C.WS / "temp" / "diffusion_bench")))
    out, v2 = base / "selftest", base / "selftest_v2"
    for d in (out, v2):
        d.mkdir(parents = True, exist_ok = True)
    failures: list = []
    legacy_checks(out, failures)
    plumbing_checks(failures)
    v2_checks(v2, failures)
    fairness_checks(v2, failures)
    panel_checks(v2, failures)
    cluster_checks(v2, failures)
    shim_checks(v2, failures)
    nov21_checks(v2, failures)
    kill_check(failures)
    for f in failures:
        print("FAIL:", f)
    print("selftest", "FAILED" if failures else "passed", f"in {time.time() - t0:.0f}s ({out}, {v2})")
    if not failures:
        wipe(base)
        base.rmdir()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
