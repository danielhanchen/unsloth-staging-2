#!/usr/bin/env python3
"""CPU self-test of the scheduler and every target adapter (offline: nothing is launched, pushed or downloaded).

  python diffusion_bench/targets/selftest_targets.py [--keep DIR]

  1. plan.py on targets/fixtures/request_all_targets.json (every target): the scheduling rules hold, the listing
     prints, a timing arm on local:B200 is refused, NOT_RUN reasons are given;
  2. matrix.py --plan --vm --dry-run lists the local VM; a two-lane VM is refused there;
  3. prepare() of colab, kaggle, cpu, local, amd (Linux scaffold + lint), mac produces its artifacts offline;
  4. vm_driver.py runs a two-lane slice of fake-backend cells end to end on the CPU (trial gate, lanes, artifact),
     and colab.collect() reads the records back out of the printed base64 artifact;
  5. a failing trial stops the VM: no timed cell runs.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TK = HERE.parent
sys.path.insert(0, str(TK))

import plan as P  # noqa: E402
from targets import amd, colab, common as TC, cpu, kaggle, local, mac  # noqa: E402
from targets import notebook as NB  # noqa: E402

FIX = HERE / "fixtures"
FAILS: list = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush = True)
    if not cond:
        FAILS.append(msg)


def build_fixture_plan() -> dict:
    req = json.loads((FIX / "request_all_targets.json").read_text())
    cards = P.load_cards(sorted({P.norm_model(m)["family"] for m in req["models"]}), str(FIX / "cards"))
    return P.build_plan(req, cards)


def test_plan(work: Path) -> dict:
    plan = build_fixture_plan()
    kinds = {v["target"] for v in plan["vms"]}
    check(kinds >= {"colab:L4", "colab:T4", "colab:A100-40", "kaggle:T4x2", "local:B200", "amd:gfx1151-linux",
                    "amd:gfx1151-windows", "mac:macos-15", "cpu", "tester:3090"}, f"plan covers every target ({len(kinds)})")
    cell = {c["tag"]: c for c in plan["cells"]}
    for c in plan["cells"]:
        o = c["options"]
        assert {"name", "vm_id", "lane", "account", "simulated_budget_gib"} <= set(o["target"]), c["tag"]
        assert {"stack", "block", "order", "aa_of"} <= set(o["arm"]), c["tag"]
    check(True, "every cell carries options.target / options.arm with the CONTRACT fields")
    ok = all(v["slots"][0][0].get("trial") for v in plan["vms"])
    check(ok, "a trial cell is first on every VM")
    rented = [v for v in plan["vms"] if v["kind"] in ("colab", "kaggle")]
    others = [v for v in plan["vms"] if v["kind"] in ("amd", "tester", "local", "mac")]
    merged = [v["vm_id"] for v in rented if v["slots"][0][0]["tag"].endswith("__parity")]
    check(merged and all(not any(it["tag"].startswith("trial__") for s_ in v["slots"] for it in s_)
                         for v in rented if v["vm_id"] in merged)
          and all(v["slots"][0][0]["tag"].startswith("trial__") for v in others),
          f"rented VMs use their first parity probe as the trial (no extra cold load: {merged}); AMD / tester / "
          "local / Mac keep a trial__ cell")
    import matrix as MX  # noqa: PLC0415

    pc = {c["tag"]: c for c in plan["cells"]}
    check(all(MX.is_trial_cell(pc[v["slots"][0][0]["tag"]]) for v in plan["vms"]),
          "matrix.py recognises every VM's first item as its trial (merged probe or trial__ cell)")
    mixed = [s for v in plan["vms"] for s in v["slots"] if len({i["stack"] for i in s if i.get("timed")}) > 1]
    check(not mixed, "no slot times two stacks side by side")
    kag = next(v for v in plan["vms"] if v["target"] == "kaggle:T4x2")
    # lanes of different workloads are NOT lockstepped (a run's neighbour would depend on the other lane's queue):
    # the VM falls back to serial, so every arm of a comparison sees the same (no) concurrency
    check(kag["lane_mode_effective"].startswith("serial") and all(len(s) == 1 for s in kag["slots"]),
          f"kaggle: unmirrored lanes run serial ({kag['lane_mode_effective']})")
    tr = [s[0] for s in kag["slots"] if s[0].get("trial")]
    check(sorted(t["lane"] for t in tr) == [0, 1] and kag["slots"][0][0].get("trial") and kag["slots"][1][0].get("trial"),
          f"kaggle: one trial cell per lane, both first ({[t['tag'] for t in tr]})")
    lanes_check(work)
    remote_check()
    split = [c["id"] for c in plan["comparisons"]
             if len({cell[t]["options"]["target"]["vm_id"] for tl in c["runs"].values() for t in tl}) != 1]
    check(not split, "every comparison's arms are on one VM")
    timed = [c for c in plan["comparisons"] if c["purpose"] == "timing"]
    check(all(len(tl) >= 3 for c in timed for tl in c["runs"].values()), "timing arms have >= 3 runs each")
    aa = all(c["aa_pairs"] for c in timed)
    check(aa, "every timing comparison has A/A pairs")
    abba = next(c for c in timed if len(c["arms"]) == 2)
    seq = [cell[t]["backend"] for t in abba["order"][:4]]
    check(seq[0] == seq[3] and seq[1] == seq[2] and seq[0] != seq[1], f"ABBA order in a block ({seq})")
    check(any("no fp8" in r["reason"] for r in plan["not_run"]), "fp8 on T4 / A100 is NOT_RUN with a reason")
    check(plan["estimate"]["total_cu"] > 0, f"estimate present ({plan['estimate']['total_cu']} CU)")
    json.dumps(plan, allow_nan = False)
    # refusal: timing on local
    req = json.loads((FIX / "request_all_targets.json").read_text())
    req["targets"] = [{"name": "local:B200", "purpose": "timing"}]
    try:
        P.build_plan(req, P.load_cards(["qwen-image-2.1", "flux3-klein", "minimax-h3"], str(FIX / "cards")))
        check(False, "local:B200 refuses a timing purpose")
    except SystemExit as exc:
        check("refuses timing" in str(exc), "local:B200 refuses a timing purpose")
    # serial lane mode never runs two at once
    req = json.loads((FIX / "request_all_targets.json").read_text())
    req["targets"] = [{"name": "kaggle:T4x2", "lane_mode": "serial"}]
    sp = P.build_plan(req, P.load_cards(["qwen-image-2.1", "flux3-klein", "minimax-h3"], str(FIX / "cards")))
    check(all(len(s) == 1 for v in sp["vms"] for s in v["slots"]), "kaggle serial mode: one item per slot")
    pp = work / "plan.json"
    pp.write_text(json.dumps(plan, indent = 1))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        P.main([str(FIX / "request_all_targets.json"), "--cards-dir", str(FIX / "cards"), "--dry-run"])
    check("TRIAL" in buf.getvalue() and "estimate:" in buf.getvalue(), "plan.py --dry-run prints the listing")
    return plan


def public_bundle_check(work: Path) -> None:
    """The rewritten public toolkit copy: no path / content names the baseline UI (with a planted control), the
    rewrite inverts, the real names come back at runtime from BASELINE_REPO only, and its fake-backend selftest
    passes from inside the copy."""
    import public_bundle as PB  # noqa: PLC0415

    check(PB.check_private() == [], "the sentinel never occurs in the private toolkit (the rewrite inverts)")
    dst = PB.write(work / "public_copy")
    bad = PB.problems(dst)
    check(not bad and (dst / "PUBLIC_BUNDLE").exists() and not (dst / "private_defaults.json").exists()
          and not (dst / "public_bundle.py").exists(), f"public copy: every path and file clean ({bad[:3]})")
    (dst / "planted").mkdir(exist_ok = True)
    (dst / "planted" / "note.txt").write_text("see " + PB.WORD.upper() + " here\n")
    check(PB.problems(dst) != [], "public copy grep: a planted name is caught (control)")
    shutil.rmtree(dst / "planted")
    texts = [(TK / r).read_text() for r in PB.source_files() if r.endswith((".py", ".json", ".md", ".sh"))
             and r != "extnames.py"]
    check(all(PB.restore_text(PB.rewrite_text(t)) == t for t in texts), "restore_text inverts rewrite_text")
    import extnames  # noqa: PLC0415

    real = extnames.baseline_repo(True)
    S = extnames.SENTINEL
    probe = ("import sys, json; sys.path.insert(0, sys.argv[1]); import extnames as X; "
             f"print(json.dumps([X.is_public(), X.ext('{S}.samplers'), X.norm(X.ext('x-{S}-int8.safetensors')), "
             f"X.ext_env({{'{S.upper()}_KITCHEN_X': '1'}})]))")
    r = subprocess.run([sys.executable, "-c", probe, str(dst)], capture_output = True, text = True,
                       env = {**os.environ, "BASELINE_REPO": real})
    got = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else None
    w = PB.WORD
    check(got and got[0] is True and got[1] == f"{w}.samplers" and got[2] == f"x-{S}-int8.safetensors"
          and f"{w.upper()}_KITCHEN_X" in got[3], f"public copy restores the real names at runtime ({r.stderr[-200:]})")
    r = subprocess.run([sys.executable, "-c", probe, str(dst)], capture_output = True, text = True,
                       env = {k: v for k, v in os.environ.items() if k != "BASELINE_REPO"})
    check(r.returncode != 0 and "BASELINE_REPO is not set" in r.stderr,
          "public copy without BASELINE_REPO fails clearly (no literal URL to fall back to)")
    r = subprocess.run([sys.executable, str(dst / "selftest.py")], capture_output = True, text = True,
                       env = {**os.environ, "BASELINE_REPO": real}, timeout = 1200)
    check(r.returncode == 0 and "selftest passed" in r.stdout,
          f"public copy: its fake-backend selftest passes from inside the copy ({(r.stdout + r.stderr)[-300:]})")


def remote_check() -> None:
    """The notebook targets' leak scan and timeouts."""
    from targets import _remote as RM  # noqa: PLC0415

    body = "".join(chr(c) for c in range(65, 91)) + "abcdefghijklmnop0123456789"  # 42 distinct chars
    shapes = {"hf": "hf_" + body[:34], "github classic": "ghp_" + body[:36], "github oauth": "gho_" + body[:36],
              "github fine-grained": "github_pat_" + (body * 2)[:70], "kaggle": "KGAT_" + body[:30],
              "kaggle.json": '{"username": "u", "key": "' + "0123456789abcdef" * 2 + '"}'}
    missed = [k for k, v in shapes.items() if not RM.leaks_in(v.encode(), [])]
    check(not missed and not RM.leaks_in(b"plain log line ghp_short", []),
          f"leak scan catches HF / GitHub / Kaggle token shapes (missed {missed})")
    check(RM.leaks_in(b"x live-value-0123456789 y", [b"live-value-0123456789"]), "leak scan catches a live token value")
    wall, cell = RM.timeouts({"est_min": 100})
    check(cell < wall, f"per-cell timeout ({cell}s) below the wall timeout ({wall}s)")


def lanes_check(work: Path) -> None:
    """merge_lanes negative control: two lanes of different workloads (a heavy video run on one GPU, a light image
    run on the other) must never share a slot; mirrored lanes (the same arm sequence) run in lockstep."""
    def item(tag, stack, fam, lane):
        return {"tag": f"{fam}_k__{tag}__k1p0", "lane": lane, "stack": stack, "timed": True, "family": fam,
                "est_min": 1.0}
    q0 = [item("s_bf16", "studio", "wan", 0), item("c_bf16", "blineui", "wan", 0)]
    q1 = [item("s_bf16", "studio", "sdxl", 1), item("s_bf16", "studio", "sdxl", 1), item("c_bf16", "blineui", "sdxl", 1)]
    slots = P.merge_lanes([q0, q1], "same_stack")
    mixed = [s for s in slots if len(s) > 1 and len({i["family"] for i in s}) > 1]
    check(not mixed, f"same_stack never runs two different workloads side by side ({len(mixed)} mixed slots)")
    m0 = [item("s_bf16", "studio", "sdxl", 0), item("c_bf16", "blineui", "sdxl", 0)]
    m1 = [item("s_bf16", "studio", "sdxl", 1), item("c_bf16", "blineui", "sdxl", 1)]
    slots, mode = P.merge_lanes([m0, m1], "same_stack", with_mode = True)
    check(len(slots) == 2 and all(len(s) == 2 for s in slots) and "lockstep" in mode,
          f"same_stack: mirrored lanes run in lockstep ({mode})")


def test_real_cards(work: Path) -> None:
    """plan.py on the model cards (diffusion_bench/models): the card's planning hints refuse infeasible cells with a
    reason, parity probes precede the timed cells, file sources become downloads / builds, Studio repos prefetch."""
    from targets import fetch_files as FF  # noqa: PLC0415

    req = {"name": "real", "revisions": {"unsloth_sha": "0" * 40, "zoo_sha": "1" * 40, "bline_ref": "v0.45.6"},
           "models": ["qwen-image-2.1", "minimax-h3"],
           "arms": [{"stack": "studio", "format": "int8"}, {"stack": "blineui", "format": "int8"},
                    {"stack": "studio", "format": "int8_teint8"}, {"stack": "blineui", "format": "int8_teint8"}],
           "targets": ["colab:L4", "kaggle:T4x2", "colab:T4"], "repeats": 3, "n": 2}
    plan = P.build_plan(req, P.load_cards(["qwen-image-2.1", "minimax-h3"], None))
    why = {(r["family"], r["target"], r["arm"]): r["reason"] for r in plan["not_run"]}
    check(("qwen-image-2.1", "kaggle:T4x2", "c_int8") not in why,
          "a measured T4 time (est_s_per_render.T4) means qwen int8 is NOT refused for VRAM on a 15 GB T4")
    check("min_host_ram_gib" in why.get(("qwen-image-2.1", "colab:T4", "s_int8"), ""),
          "host RAM is per target: qwen int8 (21.2 GiB) refused on Colab T4 (12.7 GB) though measured on a T4 GPU")
    check("min_host_ram_gib" not in why.get(("qwen-image-2.1", "kaggle:T4x2", "s_int8"), ""),
          "host RAM is per target: qwen int8 fits Kaggle T4x2's 29 GB host")
    check("min_host_ram_gib" in why.get(("minimax-h3", "kaggle:T4x2", "s_int8"), ""),
          "card min_host_ram_gib refuses H3 on a 29 GB Kaggle host (never measured there)")
    mb = P.build_plan({**req, "models": ["moss-image-base"], "arms": [{"stack": "studio", "format": "int8"}],
                       "targets": ["kaggle:T4x2"]}, P.load_cards(["moss-image-base"], None))
    check(any("min_vram_gib" in r["reason"] for r in mb["not_run"]),
          "min_vram_gib still refuses a format never measured on the GPU (moss-image-base int8 on a 15 GB T4)")
    # dense bf16 on an fp16-only T4: both stacks run fp16 (explicit, recorded), never refused
    dn = P.build_plan({**req, "models": ["z-image-turbo"], "arms": [{"stack": "studio", "format": "bf16"},
                                                                    {"stack": "blineui", "format": "bf16"}],
                       "targets": ["kaggle:T4x2"]}, P.load_cards(["z-image-turbo"], None))
    t4 = [c for c in dn["cells"] if not c["tag"].startswith("trial__")]
    check(t4 and not any("bf16 support" in r["reason"] for r in dn["not_run"])
          and all(c["options"].get("dense_dtype") == "fp16" for c in t4),
          f"dense bf16 on a T4 runs at fp16 on both stacks ({len(t4)} cells, NOT_RUN {[r['reason'][:60] for r in dn['not_run']]})")
    # a comparison left with one stack is not timed (FLUX.3 klein NVFP4 on G4: Studio refuses the family's NVFP4;
    # Studio runs INT8 on sm75 since #17949 / #18441, so qwen int8 on Kaggle is a real pair now)
    nv = P.build_plan({**req, "models": [{"family": "flux.3-klein", "formats": ["nvfp4"]}],
                       "arms": [{"stack": "studio", "format": "nvfp4"}, {"stack": "blineui", "format": "nvfp4"}],
                       "targets": ["colab:G4"]}, P.load_cards(["flux.3-klein"], None))
    check(not nv["comparisons"] and any(r["arm"] == "comparison" and "not timed" in r["reason"]
                                        for r in nv["not_run"]),
          "a one-stack comparison (the other stack blocked) is NOT_RUN, not timed alone")
    q = [c for c in plan["comparisons"] if c["family"] == "qwen-image-2.1" and c["target"] == "kaggle:T4x2"]
    check(bool(q), "qwen int8 on Kaggle T4x2 is a timed pair (Studio INT8 on sm75)")
    st = P.build_plan({**req, "models": ["minimax-h3"], "arms": [{"stack": "studio", "format": "int8_teint8"},
                                                                 {"stack": "blineui", "format": "int8"}],
                       "targets": ["tester:big"]}, P.load_cards(["minimax-h3"], None))  # no disk / VRAM hint there
    check(not st["comparisons"] and any("not timed" in r["reason"] for r in st["not_run"]),
          "a Studio-only comparison (every baseline arm blocked: no official source) is not timed")
    check("no official source" in why.get(("minimax-h3", "colab:L4", "c_int8"), ""),
          "a template file with no official source is NOT_RUN for that arm, with the card's reason")
    # mirror rule: a baseline file the card has no official source for runs from a mirror only when the request's
    # entry verifies it against the official checkpoint (mirror.py); every unverified form stays NOT_RUN
    sha = "f96738d8116d78b7316b57c663d8c98591ddafab7488f48e3099b7655e34ab6f"
    good = {"repo": "Bline-Org/MOSS-Image", "file": "diffusion_models/moss_image_sft_bf16.safetensors", "sha256": sha,
            "mirror": True, "official": {"repo": "OpenMOSS-Team/MOSS-Image-SFT",
                                         "files": ["transformer/diffusion_pytorch_model-00001-of-00003.safetensors"]},
            "verify": {"method": "tensors", "key_map": None}}
    others = {n: {**good, "file": f"x/{n}"} for n in ("qwen3.5_9b_bf16.safetensors", "flux2-vae_small.safetensors")}
    variants = {
        "verified": {"moss_image_sft_bf16.safetensors": good},
        "legacy staged_from_mirror": {"moss_image_sft_bf16.safetensors": {
            k: v for k, v in good.items() if k not in ("mirror", "verify", "official")} | {"staged_from_mirror": True}},
        "no verify block": {"moss_image_sft_bf16.safetensors": {k: v for k, v in good.items() if k != "verify"}},
        "unflagged, unverified": {"moss_image_sft_bf16.safetensors": {"repo": good["repo"], "file": good["file"],
                                                                      "sha256": sha}},
        "published hash mismatch": {"moss_image_sft_bf16.safetensors": {
            **good, "verify": {"method": "sha256"}, "official": {**good["official"], "sha256": "0" * 64}}},
        "tensors without pinned bytes": {"moss_image_sft_bf16.safetensors": {
            k: v for k, v in good.items() if k != "sha256"}},
    }
    for name, dl in variants.items():
        mp = P.build_plan({**req, "models": [{"family": "moss-image-sft", "formats": ["bf16"]}],
                           "arms": [{"stack": "studio", "format": "bf16"}, {"stack": "blineui", "format": "bf16"}],
                           "targets": ["colab:A100-40"], "downloads": {**others, **dl}},
                          P.load_cards(["moss-image-sft"], None))
        reasons = [r["reason"] for r in mp["not_run"] if r["family"] == "moss-image-sft"]
        mirror_refused = any("not verified against the official checkpoint" in r for r in reasons)
        if name == "verified":
            check(not mirror_refused and not any("no official source" in r for r in reasons) and mp["comparisons"],
                  f"a mirror file verified against the official checkpoint is staged ({reasons[:2]})")
        else:
            check(mirror_refused and not mp["comparisons"],
                  f"negative control: an unverified mirror file ({name}) is refused ({reasons[:1]})")
    vm = next(v for v in plan["vms"] if v["target"] == "colab:L4")
    order = [it for s in vm["slots"] for it in s]
    first_timed = {}
    for i, it in enumerate(order):
        if it.get("timed"):
            first_timed.setdefault(it["comparison"], i)
    par = {it["comparison"]: i for i, it in enumerate(order) if it.get("parity")}
    check(par and all(par[c] < first_timed[c] for c in par), "parity probes run before each comparison's timed cells")
    built = {k: v for k, v in plan["downloads"].items() if v.get("build")}
    fetched = {k: v for k, v in plan["downloads"].items() if not v.get("build")}
    check(built and all(not str(v.get("repo", "")).lower().startswith("bline-org/") for v in plan["downloads"].values()),
          f"card sources -> {len(built)} built + {len(fetched)} fetched files, none from Bline-Org")
    check(any(k.startswith("Qwen/Qwen-Image-2.1") for k in plan["prefetch"]), "Studio sources become prefetch repos")
    est = [c["est_min"] for c in plan["cells"] if c["tag"].startswith("qwen-image-2-1_colab-l4__s_int8__k")]
    check(est and est[0] > 2.0, f"estimates use the card's est_s_per_render.L4 ({est[:1]} min/cell)")
    doc = FF.resolve({"cells": [c for c in plan["cells"] if c["backend"] == "blineui"], "downloads": plan["downloads"]},
                     work / "cm", download = False)
    files = doc["cells"][0]["options"]["files"]
    check(all(isinstance(v, str) and str(work / "cm") in v for v in files.values()),
          "fetch_files --dry-run resolves fetched and built card files to paths")


def test_matrix(work: Path) -> None:
    env = {**os.environ, "WORKSPACE": str(work)}
    r = subprocess.run([sys.executable, str(TK / "matrix.py"), "--plan", str(work / "plan.json"), "--vm", "local-b200-1",
                        "--out", str(work / "mx"), "--dry-run"], capture_output = True, text = True, env = env)
    check(r.returncode == 0 and "[dry]" in r.stderr + r.stdout, "matrix.py --plan --vm --dry-run (local)")
    r = subprocess.run([sys.executable, str(TK / "matrix.py"), "--plan", str(work / "plan.json"), "--vm", "kaggle-t4x2-1",
                        "--out", str(work / "mx"), "--dry-run"], capture_output = True, text = True, env = env)
    check(r.returncode != 0 and "vm_driver" in r.stderr + r.stdout, "matrix.py refuses a two-lane VM")
    # --plan: a failed trial stops the VM (NOT_RUN records with the reason); --gpus a,b refused for a timed VM
    def fc(tag, **opt):
        return {"tag": tag, "backend": "fake", "width": 64, "height": 64, "steps": 3, "short_steps": 2, "short_n": 0,
                "n": 1, "warmup": 1, "prompts": "image_24",
                "options": {**opt, "foreign_wait": False, "target": {"name": "colab:L4", "vm_id": "v1"},
                            "arm": {"stack": "studio", "purpose": "trial" if tag.startswith("trial") else "timing"}}}
    cells = [fc("trial__v1", fail_on_load = True), fc("a1"), fc("a2")]
    tp = {"name": "t", "cells": cells, "comparisons": [],
          "vms": [{"vm_id": "v1", "target": "colab:L4", "purpose": "timing", "lanes": 1,
                   "slots": [[{"tag": c["tag"], "lane": 0, "stack": "studio", "timed": c["tag"] != "trial__v1",
                               **({"trial": True} if c["tag"] == "trial__v1" else {})}] for c in cells]}]}
    (work / "tplan.json").write_text(json.dumps(tp))
    env2 = {**env, "CUDA_VISIBLE_DEVICES": "", "DBENCH_FAKE_GPU": json.dumps({"name": "FakeGPU", "used_mib": 1000,
                                                                          "own_mib": 800, "sm_mhz": 1500, "cc": "8.9",
                                                                          "total_mib": 24576})}
    r = subprocess.run([sys.executable, str(TK / "matrix.py"), "--plan", str(work / "tplan.json"), "--vm", "v1",
                        "--out", str(work / "tmx"), "--no-gate"], capture_output = True, text = True, env = env2)
    recs = {t: json.loads((work / "tmx" / t / "record.json").read_text()) for t in ("a1", "a2")
            if (work / "tmx" / t / "record.json").exists()}
    check(len(recs) == 2 and all(x["verdict"] == "NOT_RUN" and "trial" in x["reason"] for x in recs.values()),
          f"matrix --plan: a failed trial stops the VM ({ {t: x['verdict'] for t, x in recs.items()} })")
    # one cell per process (tester runner, AMD probe): a trial that dies before its record still stops the next call
    cells3 = [fc("trial__v1", exit_on_load = True), fc("a1"), fc("a2")]
    (work / "tplan3.json").write_text(json.dumps({**tp, "cells": cells3}))
    for tag in ("trial__v1", "a1"):
        subprocess.run([sys.executable, str(TK / "matrix.py"), "--plan", str(work / "tplan3.json"), "--vm", "v1",
                        "--only", tag, "--out", str(work / "tmx3"), "--no-gate"], capture_output = True, text = True,
                       env = env2)
    a1 = json.loads((work / "tmx3" / "a1" / "record.json").read_text()) if (work / "tmx3" / "a1" / "record.json").exists() else {}
    check(a1.get("verdict") == "NOT_RUN" and "trial" in (a1.get("reason") or ""),
          f"#50 a trial that dies with no record stops a later one-cell call ({a1.get('verdict')})")
    import export_tester_bundle as ETB  # noqa: PLC0415

    compile(ETB.RUNNER, "run_tester.py", "exec")
    check("TRIAL_FAILED" in ETB.RUNNER and "trial__" in ETB.RUNNER, "#50 run_tester.py checks the trial record")
    # pr876 recheck (Nov 20): the Ada idled at 63-65 C beside a busy RTX 3090, so a fixed 60 C gate never opened
    # (the tester used 68 C by hand). The gate follows the card's own idle baseline + 3 C (capped), a per-card value
    # wins, and the effective limit is the one each cell's cooldown gets
    ns: dict = {"__name__": "run_tester_selftest", "__file__": str(TK / "run_tester.py")}
    exec(compile(ETB.RUNNER, "run_tester.py", "exec"), ns)  # noqa: S102 - the bundle's own runner source
    gl = ns.get("gate_limit")
    check(gl is not None and gl(60, {}, 0, 64.0, 3.0) == 67.0 and gl(60, {}, 0, 40.0, 3.0) == 60.0
          and gl(60, {}, 0, 90.0, 3.0) == 70.0 and gl(60, {"0": 68}, 0, 64.0, 3.0) == 68.0
          and gl(60, {}, 0, None, 3.0) == 60.0 and gl(None, {}, 0, 64.0, 3.0) is None
          and gl(60, {}, 0, 64.0, None) == 60.0,
          "run_tester's start-temperature gate follows the card's idle baseline, capped; a per-card value wins")
    check('env["DBENCH_START_TEMP_C"] = str(temp_c' in ETB.RUNNER and "wait_gates(gpu, lim, temp_c)" in ETB.RUNNER
          and "--start-temp-c" in ETB.RUNNER,
          "run_tester passes the effective start temperature to its gate and every cell's cooldown")
    r = subprocess.run([sys.executable, str(TK / "matrix.py"), "--plan", str(work / "tplan.json"), "--vm", "v1",
                        "--out", str(work / "tmx2"), "--gpus", "0,1", "--dry-run"], capture_output = True, text = True,
                       env = env2)
    check(r.returncode != 0 and "refused" in r.stderr + r.stdout, "matrix --plan refuses --gpus a,b for a timed VM")


def test_cases(work: Path) -> None:
    out = work / "plan_out" / "plan.json"
    with contextlib.redirect_stdout(io.StringIO()):
        P.main([str(FIX / "request_all_targets.json"), "--cards-dir", str(FIX / "cards"), "--out", str(out)])
    doc = json.loads((out.parent / "cases.json").read_text())
    plan = json.loads(out.read_text())
    n_runs = sum(len(c["order"]) for c in plan["comparisons"])
    c0 = doc["cases"][0]
    check(len(doc["cases"]) == n_runs and not any(c["tag"].startswith("trial__") for c in doc["cases"])
          and c0["seeds"] and len(c0["seeds"]) == c0["n"] and {"arm", "repeat", "block", "prompt_ids"} <= set(c0),
          f"plan.py writes cases.json: {len(doc['cases'])} cases with prompts / seeds / arm / repeat")
    bdir = os.environ.get("DBENCH_SCHEMA_DIR") or str(TK)
    if bdir and (Path(bdir) / "report.py").exists():
        code = ("import sys, json; sys.path.insert(0, sys.argv[1]); import report; from pathlib import Path; "
                "print(json.dumps(report.expected_cases(Path(sys.argv[2]), None)))")
        r = subprocess.run([sys.executable, "-c", code, bdir, str(out.parent)], capture_output = True, text = True)
        got = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else None
        check(got == [c["tag"] for c in doc["cases"]], "report.expected_cases reads plan.py's cases.json")
    else:
        print("NOTE report.py not importable (set DBENCH_SCHEMA_DIR): cases.json parse by report.py skipped")


def test_matrix_gate(work: Path) -> None:
    import common as C
    import matrix as M

    saved = {k: getattr(C, k, None) for k in ("gpu_vendor", "gpu_gate", "capability_gate")}
    try:
        C.gpu_vendor = lambda: "nvidia"
        C.gpu_gate = lambda **kw: {"gate": "ok", "util": 0, "used_mib": 1}
        cell = C.merge_cell({}, {"tag": "g1", "backend": "fake", "width": 64, "height": 64, "steps": 3,
                                 "short_steps": 2, "short_n": 1, "n": 1, "warmup": 1})
        out = work / "gate"
        M.run_job("", cell, out, "", {}, 300, True, False)
        rec = C.read_record(out / "g1") or {}
        check(isinstance(rec.get("gate"), dict) and rec["gate"].get("matrix", {}).get("gate") == "ok",
              f"matrix merges its gate reading under gate.matrix ({rec.get('gate')})")
        C.capability_gate = lambda need, gpu = None, device = None: (False, "unsupported: fake AMD")
        r = M.run_job("", {**cell, "tag": "g2", "min_compute_cap": "8.9"}, out, "", {}, 300, True, False)
        check(r == "skipped", "matrix uses common.capability_gate for min_compute_cap")
    finally:
        for k, v in saved.items():
            if v is None:
                if hasattr(C, k):
                    delattr(C, k)
            else:
                setattr(C, k, v)


def PB_rw(obj):
    """obj as the public copy writes it (plan_vm.json goes through public_bundle.rewrite_text)."""
    import public_bundle as PB  # noqa: PLC0415

    return json.loads(PB.rewrite_text(json.dumps(obj)))


def test_prefetch_skip() -> None:
    from targets import vm_driver as VD  # noqa: PLC0415

    cells = {"trial": {"backend": "studio", "model": "org/a", "options": {"local_files_only": False}},
             "s1": {"backend": "studio", "model": "org/a", "options": {}},
             "s2": {"backend": "studio", "model": "org/b", "options": {}},
             "s3": {"backend": "studio", "model": "org/b", "options": {"local_files_only": False}}}
    entries = {"org/a:*": {"repo": "org/a", "file": None}, "org/b:*": {"repo": "org/b", "file": None},
               "org/a:x.gguf": {"repo": "org/a", "file": "x.gguf"}}
    got = VD.fetched_by_studio(entries, ["trial", "s1", "s2", "s3"], cells)
    check(set(got) == {"org/a:*"}, f"prefetch skips only a whole repo an online Studio cell fetches first ({sorted(got)})")
    import types  # noqa: PLC0415
    from unittest import mock  # noqa: PLC0415

    fake = types.SimpleNamespace(siblings = [types.SimpleNamespace(size = 400 * 2**30)])
    with mock.patch("huggingface_hub.HfApi.model_info", return_value = fake), \
            mock.patch.object(VD.shutil, "disk_usage", return_value = types.SimpleNamespace(free = 220 * 2**30)):
        big = VD.snapshot_too_big("org/huge", str(TK))
    with mock.patch("huggingface_hub.HfApi.model_info", return_value = fake), \
            mock.patch.object(VD.shutil, "disk_usage", return_value = types.SimpleNamespace(free = 2000 * 2**30)):
        ok = VD.snapshot_too_big("org/huge", str(TK))
    check(big and "400 GiB" in big and ok is None, f"a repo snapshot bigger than the free disk is skipped ({big})")
    test_prefetch_narrow()


def test_prefetch_narrow() -> None:
    """pr876 recheck (Nov 20): an online run_tester.py run pulled the whole Qwen/Qwen-Image-2.1 snapshot (28 files)
    for an int8 arm that loads Studio's hosted transformer, and a gated flux-3-image-klein snapshot no arm needed
    failed with 403. The int8 cards prefetch the base repo without the dense transformer plus the hosted file, the
    tester path skips a repo its online Studio trial fetches itself, and a 403 is a skip, not a failure."""
    import types  # noqa: PLC0415
    from unittest import mock  # noqa: PLC0415

    import models as MC  # noqa: PLC0415
    import plan as PL  # noqa: PLC0415
    from targets import fetch_files as FF  # noqa: PLC0415
    from targets import vm_driver as VD  # noqa: PLC0415

    pf = PL.studio_prefetch(MC.load_card("qwen-image-2.1"), "int8")
    base = pf.get("Qwen/Qwen-Image-2.1:*") or {}
    check("transformer/*.safetensors" in (base.get("ignore_patterns") or [])
          and pf.get("unsloth/Qwen-Image-2.1-FP8:Qwen-Image-2.1-INT8-ConvRot.safetensors"),
          f"int8 Qwen-Image-2.1 prefetch skips the dense transformer and names the hosted file ({pf})")
    bf = PL.studio_prefetch(MC.load_card("qwen-image-2.1"), "bf16")
    check(not (bf.get("Qwen/Qwen-Image-2.1:*") or {}).get("ignore_patterns"),
          "bf16 Qwen-Image-2.1 prefetch keeps the dense transformer it loads")
    calls = []

    def snap(repo, **kw):
        calls.append(("snap", repo, kw.get("ignore_patterns")))

    def one(repo, filename, **kw):
        calls.append(("file", repo, filename))

    class Gated(Exception):
        pass

    Gated.__name__ = "GatedRepoError"

    def gated(repo, **kw):
        raise Gated("403 Client Error")

    logs = []
    fake = types.SimpleNamespace(siblings = [types.SimpleNamespace(rfilename = "transformer/a.safetensors",
                                                                   size = 900 * 2**30),
                                             types.SimpleNamespace(rfilename = "vae/b.safetensors", size = 2**30)])
    with mock.patch("huggingface_hub.snapshot_download", snap), mock.patch("huggingface_hub.hf_hub_download", one), \
            mock.patch("huggingface_hub.HfApi.model_info", return_value = fake), \
            mock.patch.object(VD.shutil, "disk_usage", return_value = types.SimpleNamespace(free = 200 * 2**30)), \
            mock.patch.object(VD, "log", logs.append):
        VD.prefetch(pf, {"HF_HOME": str(TK / "temp" / "hf_unused")})
    check(("snap", "Qwen/Qwen-Image-2.1", ["transformer/*.safetensors", "transformer/*.bin"]) in calls
          and any(c[0] == "file" and c[2] == "Qwen-Image-2.1-INT8-ConvRot.safetensors" for c in calls),
          f"prefetch passes the card's ignore_patterns and fetches the hosted file ({calls}, {logs})")
    logs.clear()
    with mock.patch("huggingface_hub.snapshot_download", gated), \
            mock.patch.object(VD, "snapshot_too_big", return_value = None), mock.patch.object(VD, "log", logs.append):
        VD.prefetch({"bfl/klein:*": {"repo": "bfl/klein", "file": None}}, {"HF_HOME": str(TK / "temp" / "x")})
    check(logs and "skipped: no access" in logs[-1] and "failed" not in logs[-1],
          f"a 403 on a gated prefetch repo is a skip ({logs})")
    nf = types.SimpleNamespace(response = types.SimpleNamespace(status_code = 404))
    check(not VD.no_access(type("E", (Exception,), {})()) and not VD.no_access(Exception()) and
          not VD.no_access(type("H", (Exception,), {"response": nf.response})()),
          "a 404 / plain error stays a prefetch failure")
    doc = {"prefetch": {"bfl/klein:*": {"repo": "bfl/klein", "file": None}},
           "cells": [{"tag": "trial", "backend": "studio", "model": "bfl/klein", "options": {"local_files_only": False}},
                     {"tag": "s1", "backend": "studio", "model": "bfl/klein", "options": {}}],
           "vms": [{"slots": [[{"tag": "trial"}], [{"tag": "s1"}]]}]}
    with mock.patch.object(VD, "prefetch") as pre:
        kept = FF.prefetch(doc)
    check(kept == [] and not pre.called,
          f"the tester path skips a repo its online Studio trial fetches itself ({kept}, {pre.call_args})")


def test_cache_janitor(work: Path) -> None:
    from targets import vm_driver as VD  # noqa: PLC0415

    cache = work / "hub"
    cache.mkdir(parents = True, exist_ok = True)
    arm = lambda c: {"options": {"arm": {"comparison": c}}}  # noqa: E731
    cells = {"trial": {**arm(None), "backend": "studio", "model": "org/a"},
             "a_p": {**arm("A"), "backend": "studio", "model": "org/a", "options": {"arm": {"comparison": "A"}}},
             "a_t": {"backend": "studio", "model": "org/a", "options": {"arm": {"comparison": "A"}}},
             "b_p": {"backend": "studio", "model": "org/b", "options": {"arm": {"comparison": "B"}}},
             "c_p": {"backend": "studio", "model": "org/shared", "options": {"arm": {"comparison": "C"}}}}
    slots = [[{"tag": "trial", "trial": True}], [{"tag": "a_p"}], [{"tag": "a_t"}], [{"tag": "b_p"}], [{"tag": "c_p"}]]
    j = VD.CacheJanitor(str(cache), slots, cells)
    (cache / "models--org--old").mkdir()
    gone = []
    for si, slot in enumerate(slots):
        j.start(si)
        tag = slot[0]["tag"]
        if tag == "trial":
            (cache / "models--org--a").mkdir()
            (cache / "models--unsloth--a-FP8").mkdir()
        if tag == "b_p":
            (cache / "models--org--b").mkdir()
            (cache / "models--org--shared").mkdir()
        gone += j.finish(si)
    left = sorted(d.name for d in cache.iterdir())
    # org/shared: created by B, kept for C, freed after C (C's snapshot already held it, so C must not disown it)
    check(j.cmp_of[0] == "A" and set(gone) == {"models--org--a", "models--unsloth--a-FP8", "models--org--b",
                                               "models--org--shared"}
          and gone.index("models--org--shared") == len(gone) - 1 and left == ["models--org--old"],
          f"hub cache janitor frees a comparison's downloads after their last consumer, keeps older ones ({gone}, {left})")
    # a repo a later comparison names is kept
    cache2 = work / "hub2"
    cache2.mkdir(parents = True, exist_ok = True)
    j2 = VD.CacheJanitor(str(cache2), slots, cells)
    for si, slot in enumerate(slots[:4]):
        j2.start(si)
        if slot[0]["tag"] == "b_p":
            (cache2 / "models--org--shared").mkdir()
        j2.finish(si)
    check((cache2 / "models--org--shared").exists(), "hub cache janitor keeps a repo a later cell loads")
    # interleaved comparisons (a serial fallback alternates them): A's last slot must not free what B's online
    # probe fetched (a hosted file no later cell names) while B still has timed slots to run
    cache3 = work / "hub3"
    cache3.mkdir(parents = True, exist_ok = True)
    cells3 = {"a1": {"backend": "studio", "model": "org/a", "options": {"arm": {"comparison": "A"}}},
              "b1": {"backend": "studio", "model": "org/b", "options": {"arm": {"comparison": "B"}}},
              "a2": {"backend": "studio", "model": "org/a", "options": {"arm": {"comparison": "A"}}},
              "b2": {"backend": "studio", "model": "org/b", "options": {"arm": {"comparison": "B"}}}}
    slots3 = [[{"tag": t}] for t in ("a1", "b1", "a2", "b2")]
    j3 = VD.CacheJanitor(str(cache3), slots3, cells3)
    freed3 = {}
    for si, slot in enumerate(slots3):
        j3.start(si)
        if slot[0]["tag"] == "a1":
            (cache3 / "models--unsloth--a-INT8").mkdir()
        if slot[0]["tag"] == "b1":
            (cache3 / "models--unsloth--b-INT8").mkdir()
        freed3[slot[0]["tag"]] = j3.finish(si)
    check(freed3["a2"] == ["models--unsloth--a-INT8"] and freed3["b2"] == ["models--unsloth--b-INT8"],
          f"hub cache janitor frees an interleaved comparison's downloads only after its own last slot ({freed3})")
    # a repo freed after one comparison and fetched again by a later one is that comparison's to free
    cache4 = work / "hub4"
    cache4.mkdir(parents = True, exist_ok = True)
    cells4 = {t: {"backend": "studio", "model": f"org/{t}", "options": {"arm": {"comparison": t.upper()}}}
              for t in ("a", "b", "c")}
    slots4 = [[{"tag": t}] for t in ("a", "b", "c")]
    j4 = VD.CacheJanitor(str(cache4), slots4, cells4)
    freed4 = {}
    for si, slot in enumerate(slots4):
        j4.start(si)
        if slot[0]["tag"] in ("a", "b"):
            (cache4 / "models--unsloth--shared-INT8").mkdir(exist_ok = True)
        freed4[slot[0]["tag"]] = j4.finish(si)
    check(freed4["a"] == ["models--unsloth--shared-INT8"] and freed4["b"] == ["models--unsloth--shared-INT8"]
          and not (cache4 / "models--unsloth--shared-INT8").exists(),
          f"hub cache janitor frees a repo a later comparison fetched again after it freed it ({freed4})")


def test_pack_keeps_per_pair_parity(work: Path) -> None:
    """pack() ships the per-pair parity results parity.gate_plan writes one level down, else report.py finds no
    parity evidence for those pairs after collection."""
    import io
    import tarfile

    from targets import vm_driver as VD  # noqa: PLC0415

    out = work / "pack_pp"
    for rel in ("s_p/record.json", "s_p/parity.json", "s_p/vs_c_a/parity.json", "s_p/vs_c_b/parity.json"):
        (out / rel).parent.mkdir(parents = True, exist_ok = True)
        (out / rel).write_text("{}")
    with tarfile.open(fileobj = io.BytesIO(VD.pack(out)), mode = "r:gz") as tf:
        names = set(tf.getnames())
    want = {"s_p/parity.json", "s_p/vs_c_a/parity.json", "s_p/vs_c_b/parity.json"}
    check(want <= names, f"pack ships per-pair parity results ({sorted(n for n in names if 'parity' in n)})")


def test_release_built_payload(work: Path) -> None:
    """Files.release frees a built file's payload, not just the link build_file returns (root/<name> ->
    root/_built/<payload>): shared payloads wait for their last consumer, a link out of the root loses only the link."""
    from targets import vm_driver as VD  # noqa: PLC0415

    root = work / "release_vm"
    f = VD.Files(root, [], {}, None)
    built, ext = f.root / "_built", work / "release_outside"
    built.mkdir(parents = True, exist_ok = True)
    ext.mkdir(parents = True, exist_ok = True)
    for name in ("a.st", "shared.st"):
        (built / name).write_bytes(b"x" * 1024)
    (ext / "hub.st").write_bytes(b"y" * 1024)
    links = {"a": built / "a.st", "b": built / "shared.st", "c": built / "shared.st", "x": ext / "hub.st"}
    for name, target in links.items():
        (f.root / name).symlink_to(target)
        f.built[("build", name)] = str(f.root / name)
    f.last_build = {("build", "a"): 0, ("build", "b"): 0, ("build", "x"): 0, ("build", "c"): 1}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        f.release(0)
    step0 = (not (built / "a.st").exists() and not (f.root / "a").is_symlink() and (built / "shared.st").exists()
             and not (f.root / "x").is_symlink() and (ext / "hub.st").exists())
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        f.release(1)
    check(step0 and not (built / "shared.st").exists() and not (f.root / "c").is_symlink(),
          "releasing a built file deletes its payload under the VM's file root; a shared payload waits for its last "
          "consumer; a link out of the root loses only the link")


def test_online_probe_token() -> None:
    from targets import vm_driver as VD  # noqa: PLC0415

    probe = {"backend": "studio", "options": {"local_files_only": False, "parity_probe": True}}
    timed = {"backend": "studio", "options": {}}
    base = {"backend": "blineui", "options": {"local_files_only": False}}
    check(VD.online_probe_keeps_token(probe, True) and not VD.online_probe_keeps_token(probe, False)
          and not VD.online_probe_keeps_token(timed, True) and not VD.online_probe_keeps_token(base, True),
          "the online Studio probe (it downloads the repos whose prefetch was dropped) keeps the HF token in its "
          "environment; timed and baseline cells stay scrubbed")
    src = (TK / "targets" / "vm_driver.py").read_text()
    check('cell["keep_tokens"] = True' in src and "online_probe_keeps_token(cell" in src,
          "the driver applies it to every resolved cell (a boolean on the cell, never the token)")


def test_stage_cases_per_vm(work: Path) -> None:
    """A multi-VM plan: each VM's results dir expects only its own cases (report.py refused a complete VM because
    the other VMs' cases were missing)."""
    plan_dir = work / "multi_vm_plan"
    plan_dir.mkdir(parents = True, exist_ok = True)
    case = lambda tag, vm, tgt: {"tag": tag, "vm_id": vm, "target": tgt}  # noqa: E731
    (plan_dir / "cases.json").write_text(json.dumps({"plan": "p", "cases": [
        case("a1", "colab-l4-1", "colab:L4"), case("a2", "colab-l4-1", "colab:L4"),
        case("b1", "colab-t4-1", "colab:T4")], "not_run": [
        {"family": "f", "target": "colab:L4", "arm": "s_x", "reason": "r"},
        {"family": "f", "target": "colab:T4", "arm": "s_y", "reason": "r"}]}))
    (plan_dir / "plan.json").write_text("{}")
    run = work / "multi_vm_runs" / "colab-l4-1"
    TC.stage_cases(run, plan_dir / "plan.json")
    got = json.loads((run / "cases.json").read_text())
    check([c["tag"] for c in got["cases"]] == ["a1", "a2"] and [r["target"] for r in got["not_run"]] == ["colab:L4"],
          f"staged cases are the collected VM's own ({[c['tag'] for c in got['cases']]})")
    one = work / "single_runs" / "whatever"
    TC.stage_cases(one, plan_dir / "plan.json")
    check(len(json.loads((one / "cases.json").read_text())["cases"]) == 3,
          "a results dir not named after a plan VM gets the whole case set (local / single-VM layouts)")


def test_parity_only_gate() -> None:
    """A purpose "parity" target marks its ordinary cells parity_probe (no parity_order items): the VM driver must
    still run the gate after those slots, or the probes render and no parity.json verdict is written."""
    from targets import vm_driver as VD  # noqa: PLC0415

    req = json.loads((FIX / "request_all_targets.json").read_text())
    req["targets"] = [{"name": "colab:L4", "purpose": "parity"}]
    cards = P.load_cards(sorted({P.norm_model(m)["family"] for m in req["models"]}), str(FIX / "cards"))
    plan = P.build_plan(req, cards)
    cells = {c["tag"]: c for c in plan["cells"]}
    pairs = [p for c in plan["comparisons"] for p in c.get("parity_pairs") or []]
    probe_tags = {t for p in pairs for t in (p["a"], p["b"])}
    hit = {it["tag"] for v in plan["vms"] for s in v["slots"] if VD.slot_runs_parity(s, cells) for it in s}
    check(pairs and probe_tags <= hit and not any(it.get("parity") for v in plan["vms"] for s in v["slots"] for it in s),
          f"parity-only plan: every slot holding a parity probe runs the gate ({len(probe_tags)} probes, "
          f"{len(probe_tags - hit)} missed)")
    check(not VD.slot_runs_parity([{"tag": "x"}], {"x": {"options": {}}}), "a plain timed slot does not run the gate")


def test_prefetch_per_vm() -> None:
    """A multi-VM plan: each VM's slice prefetches only the Studio repos its own comparisons load."""
    req = {"name": "pf", "revisions": {"unsloth_sha": "0" * 40, "zoo_sha": "1" * 40, "bline_ref": "v0.45.6"},
           "models": [{"family": "flux.3-klein", "formats": ["int8"]}, {"family": "sdxl-turbo", "formats": ["fp16"]}],
           "arms": [{"stack": "studio", "format": "int8"}, {"stack": "blineui", "format": "int8"},
                    {"stack": "studio", "format": "fp16"}, {"stack": "blineui", "format": "fp16"}],
           "targets": ["colab:L4", "colab:T4"], "repeats": 3, "n": 1}
    with contextlib.redirect_stdout(io.StringIO()):
        pl = P.build_plan(req, P.load_cards(["flux.3-klein", "sdxl-turbo"], None))
    got = {}
    for vm in (pl or {}).get("vms") or []:
        sl = TC.vm_slice(pl, vm["vm_id"])
        repos = {r for c in sl["cells"].values() if c.get("backend") == "studio"
                 for r in (c.get("model"), (c.get("options") or {}).get("base_repo")) if r}
        got[vm["vm_id"]] = (sorted(sl["prefetch"]), repos)
    # whole-repo entries are the cells' own model repos; a single hosted file (an int8 card's unsloth/*-FP8
    # transformer) rides with the comparison whose card names it
    ok = len(got) >= 2 and all(pf and {k.split(":")[0] for k in pf if k.endswith(":*")} <= repos
                               for pf, repos in got.values())
    check(ok, f"each VM prefetches only its own comparisons' Studio repos ({ {k: v[0] for k, v in got.items()} })")


def test_disk_hint() -> None:
    req = {"name": "d", "revisions": {"unsloth_sha": "0" * 40, "zoo_sha": "1" * 40, "bline_ref": "v0.45.6"},
           "models": [{"family": "minimax-h3", "formats": ["int8_teint8"]}],
           "arms": [{"stack": "studio", "format": "int8_teint8"}, {"stack": "blineui", "format": "int8_teint8"}],
           "targets": ["colab:A100-40"], "repeats": 3, "n": 1}
    # the real card now fits a Colab disk (180 GiB once the builders free their own shards): a copy of it with a
    # peak above the VM's disk keeps the refusal under test
    cards = P.load_cards(["minimax-h3"], None)
    big = json.loads(json.dumps(cards))
    for c in big.values():
        c["formats"]["int8_teint8"]["min_disk_gib"] = 260
    with contextlib.redirect_stdout(io.StringIO()):
        pl = P.build_plan(req, big)
        fits = P.build_plan(req, cards)
    reasons = [n["reason"] for n in (pl or {}).get("not_run", [])]
    check(pl is not None and not pl["vms"] and any("GiB of disk" in r for r in reasons),
          f"a format whose card disk peak exceeds the VM's disk is NOT_RUN with the reason ({reasons[:1]})")
    check(fits is not None and fits["vms"] and not any("GiB of disk" in n["reason"] for n in fits["not_run"]),
          "MiniMax-H3 int8_teint8 fits a Colab A100 disk at the builders' freed-shard peak (min_disk_gib 180)")


def test_speed_default() -> None:
    import run_cell as RC  # noqa: PLC0415

    def rec(applied):
        st = {"speed_mode": applied}
        return {"arm": {"stack": "studio"}, "status_after": st, "settings": {}}

    for req, applied, void in (("default", "off", False), ("default", "default", False), ("fast", "off", True),
                               ("off", "default", True)):
        cell = {"options": {"speed_mode": req}}
        r = rec(applied)
        bad = RC._settings(cell, None, r)
        check(bool(bad) == void, f"speed_mode requested {req!r}, Studio ran {applied!r}: "
                                 f"{'VOID' if void else 'measured'} ({bad})")
        if req == "default" and applied != "default":
            check("default resolved" in str(r["settings"]["speed_mode"].get("flag")),
                  "Studio's default resolving to another tier is labelled on the record")


def test_time_accounting(work: Path) -> None:
    import report as Rp  # noqa: PLC0415

    d = work / "acct" / "vm1"
    d.mkdir(parents = True, exist_ok = True)
    (d / "setup_log.txt").write_text("INSTALL_SH_S 200\nSETUP_S 300\n")
    (d / "driver_summary.json").write_text(json.dumps({
        "vm": "colab-l4-1", "target": "colab:L4", "elapsed_s": 1000.0,
        "time": {"prefetch_s": 100.0, "downloads_s": 50.0, "builds_s": 0.0},
        "results": {"trial__colab-l4-1": {"s": 100}, "f_colab-l4__s_int8__parity": {"s": 80},
                    "f_colab-l4__c_int8__parity": {"s": 70}, "f_colab-l4__s_int8__k1p0": {"s": 600}}}))
    rows = Rp.time_accounting(work / "acct")
    r = rows[0] if rows else {}
    check(r.get("total_s") == 1400.0 and r.get("timed_s") == 600.0 and r.get("parity_s") == 150.0
          and r.get("other_s") == 100.0 and r.get("cu") == round(1400 / 3600 * 1.54, 2),
          f"report time accounting: setup + prefetch + driver, split by kind, CU at the tier rate ({r})")


def test_round(work: Path) -> None:
    """round.py: request -> plan -> prepare in one command (no launch), offline with pinned revisions."""
    out = work / "round"
    r = subprocess.run([sys.executable, str(TK / "round.py"), "--out", str(out), "--target", "colab:L4", "--models",
                        "flux.3-klein:int8,bf16", "--account", "michael", "--unsloth-sha", "0" * 40, "--zoo-sha", "1" * 40,
                        "--allow-dirty"], capture_output = True, text = True, env = {**os.environ, "WORKSPACE": str(work)})
    req = json.loads((out / "request.json").read_text()) if (out / "request.json").exists() else {}
    nb = sorted((out / "colab" / "nb").glob("*.ipynb")) if (out / "colab" / "nb").is_dir() else []
    rp = json.loads((out / "plan" / "plan.json").read_text()) if (out / "plan" / "plan.json").exists() else {}
    check(rp and all(c.get("short_n") == 1 for c in rp["cells"] if (c["options"].get("arm") or {}).get("purpose") == "timing"),
          "plan default short_n is 1 (one same-prompt pair per run, not run_cell's 3)")
    check(r.returncode == 0 and req.get("negative_control") and len(req.get("arms") or []) == 4 and nb
          and (out / "colab" / "launch_colab.sh").exists() and "launch (hours" in r.stdout,
          f"round.py: request + plan + notebooks + launch script in one command ({r.stderr[-300:]})")


def test_round_relative_out(work: Path) -> None:
    """round.py with a relative --out, launched from outside the toolkit dir: plan, prepare and collect all use one
    directory (prepare runs with cwd = the toolkit dir)."""
    cwd = work / "round_rel_cwd"
    cwd.mkdir(parents = True, exist_ok = True)
    env = {**os.environ, "WORKSPACE": str(work)}
    r = subprocess.run([sys.executable, str(TK / "round.py"), "--out", "rel_round", "--target", "colab:L4", "--models",
                        "flux.3-klein:bf16", "--account", "michael", "--unsloth-sha", "0" * 40, "--zoo-sha", "1" * 40,
                        "--allow-dirty"], capture_output = True, text = True, env = env, cwd = str(cwd))
    out = cwd / "rel_round"
    launch = out / "colab" / "launch_colab.sh"
    c = subprocess.run([sys.executable, str(TK / "round.py"), "--out", "rel_round", "--collect"], capture_output = True,
                       text = True, env = env, cwd = str(cwd))
    check(r.returncode == 0 and launch.exists() and f"bash {launch}" in r.stdout and c.returncode == 0,
          f"round.py: a relative --out resolves once, before prepare / collect change cwd ({(r.stderr or c.stderr)[-300:]})")


def exec_bundle_cells(m: dict, vm_root: Path, kind: str) -> None:
    """Execute every generated notebook's pure-Python cells (bundle unpack, slice, bundle record) with the VM root
    moved under the work dir: a JSON literal written as Python source (`false`) fails here, not on a rented VM."""
    for vm_id, nbp in m["notebooks"].items():
        nb = json.loads(Path(nbp).read_text())
        py = [c["source"] for c in nb["cells"] if not c["source"].startswith(("%%", "!"))]
        try:
            for src in py[:1]:
                exec(compile(src.replace(NB.R, str(vm_root)), f"{kind}_{vm_id}", "exec"), {})  # noqa: S102
            rec = json.loads((vm_root / f"out_{vm_id}" / "setup_bundle.txt").read_text())
            ok = rec["scripts_sha"] == m["bundle"]["scripts_sha"] and isinstance(rec.get("dirty"), bool)
        except Exception as exc:  # noqa: BLE001
            ok, rec = False, exc
        check(ok, f"{kind}: notebook {vm_id}'s bundle cell executes and records the bundle ({str(rec)[:160]})")


def test_localize_studio() -> None:
    import run_cell as RC  # noqa: PLC0415
    from targets import vm_driver as VD  # noqa: PLC0415

    root, pys, rev = Path("/vm"), {"studio": "/vm/studio/bin/python", "bline": "/vm/c/bin/python"}, {"unsloth_sha": "a" * 40}
    http = VD.localize_studio({"backend": "studio_http", "options": {}}, root, pys, rev)["options"]
    inproc = VD.localize_studio({"backend": "studio", "options": {}}, root, pys, rev)["options"]
    bline = VD.localize_studio({"backend": "blineui", "options": {}}, root, pys, rev)["options"]
    own = VD.localize_studio({"backend": "studio_http", "options": {"studio_src": "/x"}}, root, pys, rev)["options"]
    check(http.get("studio_src") == "/vm/src/unsloth" and http.get("studio_python") == pys["studio"]
          and http.get("expect_studio_sha") == rev["unsloth_sha"],
          "HTTP Studio arms run the VM's pinned tree with its install's python and state the pinned commit")
    check(inproc.get("studio_src") == "/vm/src/unsloth" and inproc.get("expect_studio_sha") == rev["unsloth_sha"],
          "in-process Studio arms: pinned tree and commit")
    check(not bline and "expect_studio_sha" not in own, "baseline arms untouched; an explicit studio_src is not pinned")
    cell = {"backend": "studio_http", "options": {"expect_studio_sha": "a" * 40}}
    check(RC._studio_pin(cell, {"revisions": {"studio": "a" * 10}}) is None, "the pinned Studio commit passes")
    check(bool(RC._studio_pin(cell, {"revisions": {"studio": "b" * 10}}))
          and bool(RC._studio_pin(cell, {"revisions": {}})),
          "a Studio arm that ran another commit (or reported none) is VOID")


def test_prepare(plan: dict, work: Path, allow_dirty: bool) -> None:
    kw = {"allow_dirty": allow_dirty}
    tc = [c for c in plan["cells"] if (c["options"].get("arm") or {}).get("purpose") == "timing"]
    check(tc and all(c.get("short_n") == 0 for c in tc),
          "timed cells state the request's short_n (fixture 0): the estimate counts the renders that run")
    sp = [c for c in plan["cells"] if c.get("backend") == "studio" and c["options"].get("parity_probe")]
    st = [c for c in plan["cells"] if c.get("backend") == "studio" and (c["options"].get("arm") or {}).get("purpose") == "timing"]
    check(sp and st and all(c["options"].get("local_files_only") is False for c in sp)
          and all(c["options"].get("local_files_only", True) is not False for c in st),
          "Studio parity probes may download (Studio's own hosted prequant files), timed Studio cells load offline")
    m = colab.prepare(plan, work / "colab", account = "michael", **kw)
    nb = json.loads(Path(m["notebooks"]["colab-l4-1"]).read_text())
    src = "".join(c["source"] for c in nb["cells"])
    check(nb["metadata"]["dbench"]["scripts_sha"] == m["bundle"]["scripts_sha"] and m["bundle"]["sha256"] in src,
          f"colab: {len(m['notebooks'])} notebooks, bundle {m['bundle']['scripts_sha'][:12]} sha256-checked")
    check(Path(m["bundle"]["path"]).exists(), "colab: bundle tarball kept in the outputs dir")
    # every Python cell of the notebook runs: executed here with the VM root moved into the work dir (the bundle
    # unpack + slice + bundle-record cell is pure Python; the shell / nvidia-smi / driver cells are not run)
    vm_root = work / "vm_root"
    py_cells = [c["source"] for c in nb["cells"] if not c["source"].startswith("%%")]
    try:
        ns: dict = {}
        exec(compile(py_cells[0].replace(NB.R, str(vm_root)), "nb_cell", "exec"), ns)  # noqa: S102
        rec = json.loads((vm_root / "out_colab-l4-1" / "setup_bundle.txt").read_text())
        ok = rec["scripts_sha"] == m["bundle"]["scripts_sha"] and (vm_root / "bench" / "diffusion_bench" / "run_cell.py").exists()
    except Exception as exc:  # noqa: BLE001
        ok, rec = False, exc
    check(ok, f"colab: the notebook's bundle cell executes (unpack, slice, bundle record) ({str(rec)[:200]})")
    for i, cell_src in enumerate(py_cells):
        try:
            compile(cell_src.replace(NB.R, str(vm_root)), f"nb_cell_{i}", "exec")
            good = True
        except SyntaxError as exc:
            good = exc
        check(good is True, f"colab: notebook Python cell {i} compiles ({good})")
    # the Studio arm's unsloth-zoo is pinned (install.sh --local overlays zoo from git main unless UNSLOTH_ZOO_REF)
    zoo = plan["revisions"]["zoo_sha"]
    check(f"ZOO_SHA={zoo}" in src and "UNSLOTH_ZOO_REF=$ZOO_SHA" in (TK / "targets" / "vm_setup.sh").read_text()
          and "ZOO_COMMIT" in (TK / "targets" / "vm_setup.sh").read_text(),
          "colab: notebook exports ZOO_SHA, vm_setup installs zoo at it and verifies the installed commit")
    nozoo = json.loads(json.dumps(plan))
    nozoo["revisions"].pop("zoo_sha", None)
    try:
        colab.prepare(nozoo, work / "colab_nozoo", account = "michael", **kw)
        check(False, "colab: prepare refuses a plan without an unsloth-zoo pin")
    except SystemExit as exc:
        check("zoo_sha" in str(exc), "colab: prepare refuses a plan without an unsloth-zoo pin")
    # a baseline cell loading a .gguf file: the GGUF loader node is installed at a recorded commit
    gp = json.loads(json.dumps(plan))
    cc = next(c for c in gp["cells"] if c.get("backend") == "blineui" and c["options"]["target"].get("vm_id") == "colab-l4-1")
    cc["options"].setdefault("files", {})["dit"] = "model-Q4_K_M.gguf"
    check(not NB.needs_gguf_node(c for c in plan["cells"]) and NB.needs_gguf_node([cc]),
          "needs_gguf_node: only a baseline cell with a .gguf file")
    mg = colab.prepare(gp, work / "colab_gguf", account = "michael", bline_gguf_sha = "2" * 40, vm_ids = ["colab-l4-1"], **kw)
    gsrc = "".join(c["source"] for c in json.loads(Path(mg["notebooks"]["colab-l4-1"]).read_text())["cells"])
    vs = (TK / "targets" / "vm_setup.sh").read_text()
    check(f"BLINE_GGUF_SHA={'2' * 40}" in gsrc and "BLINE_GGUF_SHA" not in src and mg["revisions"]["bline_gguf_sha"] == "2" * 40
          and vs.count("gguf_node") >= 3, "colab: a baseline GGUF cell installs the GGUF node at its pin (fresh and reused VM)")
    from backends import blineui as BC  # noqa: PLC0415

    cn = work / "bline_cn" / "custom_nodes" / "N1"
    cn.mkdir(parents = True, exist_ok = True)
    subprocess.run(["git", "init", "-q", str(cn)], check = True)
    subprocess.run(["git", "-C", str(cn), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty",
                    "-m", "x"], check = True)
    got = BC.custom_nodes(work / "bline_cn")
    check(got and len(got.get("N1") or "") == 40 and BC.custom_nodes(work / "nowhere") is None,
          f"baseline records its custom node commits ({got})")
    from targets import _remote as RMx  # noqa: PLC0415

    check(RMx.resolve_ref("https://invalid.example/x", "a" * 40) == "a" * 40, "resolve_ref keeps a full sha offline")
    import common as Cx  # noqa: PLC0415

    check(Cx.vcs_commit("definitely-not-installed-dist") is None, "vcs_commit: None for a missing distribution")
    check("HF_TOKEN=" not in Path(m["launch"]).read_text() and "--env HF_TOKEN" in Path(m["launch"]).read_text(),
          "colab: token only via --env NAME")
    check(subprocess.run(["bash", "-n", m["launch"]]).returncode == 0, "colab: launch script parses")
    check("--gpu A100 " in Path(m["launch"]).read_text() and "--dear-ok" in Path(m["launch"]).read_text(),
          "colab: A100-40 maps to --gpu A100 with --dear-ok")
    m = kaggle.prepare(plan, work / "kaggle", **kw)
    exec_bundle_cells(m, work / "vm_root_kaggle", "kaggle")
    t = Path(m["launch"]).read_text()
    check("--backend kaggle --gpu T4x2 --no-pair" in t and "--env HF_TOKEN" not in t, "kaggle: T4x2 kernel, no token by default")
    m = cpu.prepare(plan, work / "cpu", **kw)
    exec_bundle_cells(m, work / "vm_root_cpu", "cpu")
    check("cloud_pool.py\" run --cpu" in Path(m["launch"]).read_text(), f"cpu: cloud_pool --cpu for {m['vms']}")
    m = local.prepare(plan, work / "local")
    check("--exclusive" not in Path(m["launch"]).read_text() and "matrix.py" in Path(m["launch"]).read_text(),
          "local: shared gpu_queue lease, matrix --plan")
    bad = json.loads(json.dumps(plan))
    for v in bad["vms"]:
        if v["target"] == "local:B200":
            v["slots"][1][0]["timed"] = True
    try:
        local.prepare(bad, work / "local_bad")
        check(False, "local: prepare refuses a timed item")
    except SystemExit:
        check(True, "local: prepare refuses a timed item")
    m = amd.prepare(plan, work / "amd", plan_path = str(work / "plan.json"))
    wf = m["scaffolds"]["amd-gfx1151-linux-1"]["workflow"][0]
    t = Path(wf).read_text()
    check("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1" in t and "unset CI" in t and "kill -$sig" in t,
          "amd linux: AOTriton asserted, CI unset, kill step")
    check("amd-gfx1151-windows-1" in m["skipped"], "amd windows: needs an anchor PR (skipped with a reason)")
    try:
        import yaml

        yaml.safe_load(t)
        check(True, "amd linux workflow is valid YAML")
    except ImportError:
        print("NOTE pyyaml absent: YAML parse skipped")
    # the fixture plan has no Studio prefetch repos: give the Mac slice one (#79)
    mplan = {**plan, "prefetch": {"stabilityai/sdxl-turbo": {"repo": "stabilityai/sdxl-turbo"}},
             # each comparison names the prefetch entries it needs (plan.py): the slice keeps only those
             "comparisons": [{**c, "prefetch": ["stabilityai/sdxl-turbo"]} for c in plan.get("comparisons") or []]}
    m = mac.prepare(mplan, work / "mac", **kw)
    t = next((work / "mac").glob("*/.github/workflows/*.yml")).read_text()
    check("macos-15" in t and "PYTORCH_ENABLE_MPS_FALLBACK: \"0\"" in t and "mac_check.py" in t,
          "mac: macos-15 workflow with MPS fallback off and the CPU-fallback check")
    # #79: the Mac slice carries the Studio prefetch map and one HF_HOME spans the fetch and bench steps
    mv = next(iter(m["branches"].values()))
    mp = json.loads((Path(mv["dir"]) / "dbench" / "plan_vm.json").read_text())
    want = PB_rw(TC.vm_slice(mplan, m["vms"][0]).get("prefetch") or {})
    check(want and mp.get("prefetch") == want and "HF_HOME: ${{ github.workspace }}/hf_home" in t,
          "#79 mac: plan_vm.json carries the Studio prefetch map, HF_HOME is job-wide")
    # public pushes (AMD fork, staging repos): no generated workflow names the baseline UI anywhere, and neither does
    # any other path or file of the pushed branch dirs (the rewritten public toolkit copy, specs, plan slices)
    import public_bundle as PB  # noqa: PLC0415

    wfs = list((work / "amd").rglob(".github/workflows/*.yml")) + list((work / "mac").rglob(".github/workflows/*.yml"))
    branch_dirs = sorted({w.parents[2] for w in wfs})
    leaks = {str(d.relative_to(work)): PB.problems(d)[:3] for d in branch_dirs if PB.problems(d)}
    check(branch_dirs and not leaks, f"public branch dirs: {len(branch_dirs)}, paths and contents clean ({leaks})")
    check(all("vars.BASELINE_REPO" in w.read_text() for w in wfs),
          "public workflows read the baseline repo from the repository variable (no literal URL)")
    hits = {str(w.relative_to(work)): TC.public_problems(w.read_text()) for w in wfs}
    check(wfs and not any(hits.values()), f"public workflows: {len(wfs)} generated, none names the baseline UI "
                                          f"({ {k: v[:2] for k, v in hits.items() if v} })")
    check(TC.public_problems("      - name: BlineUI venv\n") != [], "public workflow grep: catches the word (control)")
    # the AMD writer replaces only workflows it generated; a checkout's own workflows survive
    import importlib.util as _iu

    spec = _iu.spec_from_file_location("gw_test", TK / "amd" / "gen_workflow.py")
    GWm = _iu.module_from_spec(spec)
    spec.loader.exec_module(GWm)
    co = work / "checkout"
    (co / ".github" / "workflows").mkdir(parents = True, exist_ok = True)
    (co / ".github" / "workflows" / "ci.yml").write_text("name: ci\n")
    (co / ".github" / "workflows" / "old.yml").write_text(GWm.GENERATED_MARK + " old\n")
    GWm.write(co, "amd-ci-x", GWm.GENERATED_MARK + "\nname: x\n")
    left = sorted(p.name for p in (co / ".github" / "workflows").glob("*.yml"))
    check(left == ["amd-ci-x.yml", "ci.yml"], f"amd write keeps unrelated workflows, replaces its own: {left}")
    try:
        GWm.write(co, "amd-ci-y", "name: BlineUI run\n")
        check(False, "amd write refuses a workflow naming the baseline UI")
    except SystemExit:
        check(True, "amd write refuses a workflow naming the baseline UI")


def fake_slice(work: Path, trial_fail: bool) -> Path:
    cells = {}
    slots = []

    def cell(tag, stack, **opt):
        cells[tag] = {"tag": tag, "backend": "fake", "width": 64, "height": 64, "steps": 3, "short_steps": 2,
                      "short_n": 1, "n": 2, "warmup": 1, "prompts": "image_24",
                      "options": {**opt, "foreign_wait": False, "target": {"name": "kaggle:T4x2"},
                                  "arm": {"stack": stack}}}
    cell("trial__x", "studio", fail_on_load = trial_fail)
    slots.append([{"tag": "trial__x", "lane": 0, "stack": "studio", "timed": False, "trial": True}])
    for k in range(2):
        for st in ("studio", "blineui"):
            a, b = f"a_{st}_{k}", f"b_{st}_{k}"
            cell(a, st)
            cell(b, st)
            slots.append([{"tag": a, "lane": 0, "stack": st, "timed": True},
                          {"tag": b, "lane": 1, "stack": st, "timed": True}])
    sl = {"plan_name": "fake", "plan_sha256": "0", "revisions": {},
          "vm": {"vm_id": "fake-1", "target": "kaggle:T4x2", "kind": "kaggle", "lanes": 2, "slots": slots},
          "cells": cells}
    p = work / f"slice_{int(trial_fail)}.json"
    p.write_text(json.dumps(sl))
    return p


def test_driver(work: Path) -> None:
    # the fake backend runs in this interpreter: bline_py / studio_py absent -> sys.executable for both
    bench = work / "bench"
    if not (bench / "diffusion_bench").exists():
        bench.mkdir(parents = True, exist_ok = True)
        os.symlink(TK, bench / "diffusion_bench")
    root = work / "vmroot"
    for fail in (False, True):
        out = work / f"vmout_{int(fail)}"
        shutil.rmtree(out, ignore_errors = True)
        # offline and off the host's GPUs: a fake card for telemetry, no launcher gates from the caller's shell, and a
        # hard cap so a regression fails the check instead of hanging the suite
        env = {k: v for k, v in os.environ.items() if k not in ("DBENCH_START_TEMP_C", "HF_TOKEN")}
        env.update({"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
                    "DBENCH_FAKE_GPU": json.dumps({"name": "FakeGPU", "uuid": "GPU-fake", "used_mib": 1000,
                                                   "own_mib": 800, "sm_mhz": 1500, "temp_c": 50, "power_w": 100,
                                                   "cc": "7.5", "total_mib": 15360})})
        try:
            r = subprocess.run([sys.executable, str(HERE / "vm_driver.py"), "--slice", str(fake_slice(work, fail)),
                                "--root", str(root), "--bench", str(bench), "--out", str(out), "--timeout", "120"],
                               capture_output = True, text = True, env = env, timeout = 300)
        except subprocess.TimeoutExpired:
            check(False, f"vm_driver (trial_fail={fail}) finished within 300 s")
            continue
        summ = json.loads((out / "driver_summary.json").read_text())
        verdicts = {k: v.get("verdict") for k, v in summ["results"].items()}
        if not fail:
            check(r.returncode == 0 and all(v == "ok" for v in verdicts.values()),
                  f"vm_driver: 2 lanes x {len(verdicts)} fake cells ok ({verdicts if r.returncode else ''})")
            run = work / "runs" / "fake-1"
            run.mkdir(parents = True, exist_ok = True)
            (run / "driver.log").write_text(r.stdout)
            recs = colab.collect(run)
            check(len({x.get("tag") or (x.get("cell") or {}).get("tag") for x in recs}) == len(verdicts),
                  f"collect: {len(recs)} records back from the base64 artifact")
        else:
            nr = [k for k, v in verdicts.items() if v == "NOT_RUN"]
            check(summ["stopped"] and len(nr) == 8 and len(verdicts) == 9, f"failed trial stops the VM ({len(nr)} NOT_RUN)")
            recs = {k: json.loads((out / k / "record.json").read_text()) for k in nr if (out / k / "record.json").exists()}
            check(len(recs) == len(nr) and all(x["verdict"] == "NOT_RUN" and "trial" in (x.get("reason") or "")
                                               for x in recs.values()),
                  f"every cell after a failed trial gets a NOT_RUN record with the reason ({len(recs)}/{len(nr)})")


def test_export_keys(plan: dict, work: Path, allow_dirty: bool) -> None:
    """#85: every per-target export of a plan (the notebook slice for colab / kaggle / cpu, matrix.plan_spec for
    local / AMD Linux / AMD Windows, the AMD workflow's spec file, the Mac plan_vm.json, the tester's plans/<vm>.json)
    carries downloads, the Studio prefetch map, the VM's comparisons and the plan revisions."""
    import zipfile

    import common as C  # noqa: PLC0415
    import export_tester_bundle as XB  # noqa: PLC0415
    import matrix  # noqa: PLC0415

    ep = json.loads(json.dumps(plan))
    ep["prefetch"] = {"stabilityai/sdxl-turbo:*": {"repo": "stabilityai/sdxl-turbo", "file": None}}
    ep["downloads"] = {**(ep.get("downloads") or {}), "x85.safetensors": {"repo": "org/x85", "file": "x85.safetensors"}}
    pp = work / "plan_keys.json"
    pp.write_text(json.dumps(ep))
    vm_of = {v["vm_id"]: v for v in ep["vms"]}

    def lost(doc: dict, vm_id: str, public: bool = False) -> list:
        want = C.plan_carry(ep, vm_of[vm_id])
        if public:
            want = PB_rw(want)
        bad = [k for k in ("downloads", "prefetch", "comparisons") if doc.get(k) != want[k]]
        rev = doc.get("revisions") or {}
        if any(rev.get(k) != v for k, v in want["revisions"].items()):
            bad.append("revisions")
        return bad

    got = {}
    for vm in ep["vms"]:  # what the VM driver reads (colab / kaggle / cpu notebooks embed it)
        got[f"vm_slice {vm['vm_id']}"] = lost(TC.vm_slice(ep, vm["vm_id"]), vm["vm_id"])
        if int(vm.get("lanes") or 1) == 1 and all(len(sl) == 1 for sl in vm["slots"]):  # local / AMD / Mac / cpu
            got[f"plan_spec {vm['vm_id']}"] = lost(matrix.plan_spec(pp, vm["vm_id"]), vm["vm_id"])
    m = colab.prepare(ep, work / "keys_colab", account = "michael", allow_dirty = allow_dirty)
    for vm_id, nbp in m["notebooks"].items():
        src = "".join(c["source"] for c in json.loads(Path(nbp).read_text())["cells"])
        lit = src.split(f"slice_{vm_id}.json', 'w').write(", 1)[1].split(")\n", 1)[0]
        got[f"colab notebook {vm_id}"] = lost(json.loads(json.loads(lit)), vm_id)
    m = amd.prepare(ep, work / "keys_amd", plan_path = str(pp))
    for vm_id, sc in m["scaffolds"].items():
        spec = next(Path(sc["dir"]).glob("diffusion_bench/specs/plan_*.json"))
        got[f"amd spec {vm_id}"] = lost(json.loads(spec.read_text()), vm_id, public = True)
    m = mac.prepare(ep, work / "keys_mac", allow_dirty = allow_dirty)
    for vm_id, br in m["branches"].items():
        got[f"mac plan_vm {vm_id}"] = lost(json.loads((Path(br["dir"]) / "dbench" / "plan_vm.json").read_text()),
                                           vm_id, public = True)
    z = work / "keys_tester" / "bundle.zip"
    res = XB.build(ep, z, {"0": "Ada 48GB", "1": "3090"}, 60.0, {"0": 2400, "1": 500}, [12.0], allow_dirty)
    with zipfile.ZipFile(z) as zf:
        for vm_id in res["vms"]:
            got[f"tester plan {vm_id}"] = lost(json.loads(zf.read(f"plans/{vm_id}.json")), vm_id)
    kinds = {k.split(" ")[0] + " " + k.split(" ")[1] if k.startswith(("colab", "amd", "mac", "tester")) else
             k.split(" ")[0] for k in got}
    bad = {k: v for k, v in got.items() if v}
    check(not bad and {"vm_slice", "plan_spec", "colab notebook", "amd spec", "mac plan_vm", "tester plan"} <= kinds,
          f"#85 every target export keeps downloads / prefetch / comparisons / revisions ({len(got)} exports): {bad}")


def test_kit(plan: dict, work: Path, allow_dirty: bool) -> None:
    """#86: every bundle / branch dir whose cells run Studio over HTTP (backend studio_http) ships
    studio_test_kit/lifecycle.py beside diffusion_bench (setup_studio.launch loads it), the public copies rewritten
    with no private name; a plan without such a cell ships no kit."""
    import tarfile
    import zipfile

    import export_tester_bundle as XB  # noqa: PLC0415
    import public_bundle as PB  # noqa: PLC0415

    http = json.loads(json.dumps(plan))
    for c in http["cells"]:
        if c.get("backend") == "studio":
            c["backend"] = "studio_http"
    pp = work / "plan_http.json"
    pp.write_text(json.dumps(http))
    LC = "studio_test_kit/lifecycle.py"

    def tgz_has(meta):
        with tarfile.open(meta["bundle"]["path"]) as tf:
            return LC in tf.getnames()

    got = {}
    got["colab"] = tgz_has(colab.prepare(http, work / "kit_colab", account = "michael", allow_dirty = allow_dirty))
    got["kaggle"] = tgz_has(kaggle.prepare(http, work / "kit_kaggle", allow_dirty = allow_dirty))
    got["cpu"] = tgz_has(cpu.prepare(http, work / "kit_cpu", allow_dirty = allow_dirty))
    m = mac.prepare(http, work / "kit_mac", allow_dirty = allow_dirty)
    dirs = [Path(b["dir"]) for b in m["branches"].values()]
    got["mac"] = bool(dirs) and all((d / "dbench" / LC).is_file() and not PB.problems(d) for d in dirs)
    m = amd.prepare(http, work / "kit_amd", plan_path = str(pp))
    dirs = [Path(sc["dir"]) for sc in m["scaffolds"].values()]
    got["amd"] = bool(dirs) and all((d / LC).is_file() and not PB.problems(d) for d in dirs)
    z = work / "kit_tester" / "bundle.zip"
    XB.build(http, z, {"0": "Ada 48GB", "1": "3090"}, 60.0, {"0": 2400, "1": 500}, [12.0], allow_dirty)
    got["tester"] = LC in zipfile.ZipFile(z).namelist()
    plain = not any(c.get("backend") == "studio_http" for c in plan["cells"]) and \
        not tgz_has(kaggle.prepare(plan, work / "kit_kaggle_plain", allow_dirty = allow_dirty))
    check(all(got.values()) and plain,
          f"#86 studio_http plans ship studio_test_kit/lifecycle.py in every bundle ({got}); none without ({plain})")


def test_tester(plan_path: Path, work: Path, allow_dirty: bool) -> None:
    import zipfile

    import export_tester_bundle as XB

    plan = json.loads(plan_path.read_text())
    z = work / "tester" / "bundle.zip"
    res = XB.build(plan, z, {"0": "Ada 48GB", "1": "3090"}, 60.0, {"0": 2400, "1": 500}, [12.0], allow_dirty)
    names = zipfile.ZipFile(z).namelist()
    check({"README.md", "run_tester.py", "tester.json", "MANIFEST.json"} <= set(names)
          and any(n.startswith("plans/") for n in names) and "diffusion_bench/run_cell.py" in names,
          f"tester bundle: {len(names)} files, plans {res['vms']}, missing-at-export {res['toolkit_missing']}")
    ex = work / "tester" / "x"
    shutil.rmtree(ex, ignore_errors = True)
    zipfile.ZipFile(z).extractall(ex)
    r = subprocess.run([sys.executable, str(ex / "run_tester.py"), str(ex / "plans" / f"{res['vms'][0]}.json"),
                        "--studio-py", sys.executable, "--studio-src", str(ex), "--bline-py", sys.executable,
                        "--bline-dir", str(ex), "--dry-run"], capture_output = True, text = True, cwd = str(ex))
    check(r.returncode == 0 and "TESTER_PLAN_DONE" in r.stdout and "[dry] gpu=1" in r.stdout + r.stderr,
          "tester bundle: run_tester.py --dry-run maps label 3090 -> GPU 1 and lists every cell")
    # Studio over HTTP on the tester's box: the server starts from the tester's Studio python / checkout (not a venv
    # built from Studio main) and is launched through studio_test_kit, which the bundle then ships
    check("DIFFUSION_BENCH_STUDIO_PYTHON" in XB.RUNNER and "DIFFUSION_BENCH_STUDIO_SRC" in XB.RUNNER,
          "tester bundle: run_tester.py hands the Studio python and checkout to Studio-over-HTTP cells")
    from targets import fetch_files as FF  # noqa: PLC0415

    loc = FF.localise({"cells": [{"backend": "studio_http"}, {"backend": "studio"}, {"backend": "blineui"}]},
                      None, "/src/unsloth")
    check([(c.get("options") or {}).get("studio_src") for c in loc["cells"]] == ["/src/unsloth", "/src/unsloth", None],
          "fetch_files.localise points Studio-over-HTTP cells at the tester's checkout too")
    tester_cells = [c for c in plan["cells"] if c.get("backend") == "studio"
                    and str((c.get("options") or {}).get("target", {}).get("name", "")).startswith("tester:")]
    if tester_cells:
        http = json.loads(json.dumps(plan))
        tag = tester_cells[0]["tag"]
        for c in http["cells"]:
            if c["tag"] == tag:
                c["backend"] = "studio_http"
        z2 = work / "tester" / "bundle_http.zip"
        XB.build(http, z2, {"0": "Ada 48GB", "1": "3090"}, 60.0, {"0": 2400, "1": 500}, [12.0], allow_dirty)
        check("studio_test_kit/lifecycle.py" in zipfile.ZipFile(z2).namelist()
              and not any(n.startswith("studio_test_kit/") for n in names),
              "tester bundle ships studio_test_kit only when a tester cell runs Studio over HTTP")


def tester_nov18_result(stack: str) -> dict:
    """One result JSON as the Nov 18 tester package's bench.py main() writes it (keys, nesting and value kinds copied
    from its writer: code_versions(), studio_packages(), Monitor.clocks() per request, summary); synthetic values."""
    wl = {"model": "klein", "w": 1024, "h": 1024, "steps": 16, "guidance": 1.0,
          "unet": "flux3-image-klein-int8-convrot.safetensors", "label": "FLUX.3 Klein 1024\u00b2, 16 steps"}
    cfg = {"flags": []} if stack == "bline" else {"load": {"model_path": "black-forest-labs/flux-3-image-klein"}}
    reqs = []
    for idx, kind in enumerate(["cold", "warmup", "warmup", "new_prompt", "new_prompt", "new_prompt"]):
        t0 = 1000.0 + 20 * idx
        total = 30.0 if kind == "cold" else 5.0 + 0.1 * idx
        r = {"t_submit": t0, "t_end": t0 + total, "total_s": total, "first_step_t": t0 + 0.8,
             "last_step_t": t0 + total - 0.4, "s_per_step": 0.25, "pre_sampler_s": 0.8, "post_sampler_s": 0.4,
             "kind": kind, "idx": idx, "seed": 1000 + idx, "peak_vram_mib": 14000.0, "peak_host_mib": 52000.0,
             "clocks": None if idx == 1 else {"sm_mhz_mean": 2500.0 + idx, "temp_max": 60 + idx, "power_mean": 250.0},
             "peak_vram_over_base_mib": 13500.0}
        if stack == "bline":
            r["executed_nodes"] = ["1", "2", "3", "4", "5", "6", "7", "8"]
        else:
            r["resp_meta"] = {"images": [{"url": "/x.png"}]}
        reqs.append(r)
    news = sorted(r["total_s"] for r in reqs if r["kind"] == "new_prompt")
    name = f"{stack}_{'default' if stack == 'bline' else 'klein'}_K1_gpu0_full"
    d = {"name": name, "stack": stack, "config": "default" if stack == "bline" else "klein", "cfg": cfg,
         "workload": "K1", "wl": wl, "gpu": 0, "gpu_name": "NVIDIA RTX 6000 Ada Generation", "driver": "580.95",
         "free_mib_target": 0, "sim_total_mib": 0, "started": "2026-11-18 05:00:00", "requests": reqs,
         "code": {"studio_rev": "96bf6de124d" + "0" * 29, "studio_dirty": False, "bline_rev": "2aa4d535" + "1" * 32,
                  "bline_describe": "v0.45.4", "bline_dirty": False},
         "mem_max": "70G", "env": {}, "start_temp_c": 45, "idle_used_mib": 500.0, "base_used_mib": 500.0,
         "free_at_start_mib": 48000.0, "server_ready_s": 12.0,
         "load": {"load_s": 40.0, "peak_vram_mib": 9000.0, "peak_host_mib": 50000.0} if stack == "studio" else
         {"peak_vram_mib": None, "peak_host_mib": None},
         "outcome": "ok", "cell_peak_vram_over_base_mib": 13600.0, "cell_peak_host_mib": 53000.0,
         "finished": "2026-11-18 05:10:00",
         "summary": {"outcome": "ok", "load_s": 40.0 if stack == "studio" else None, "cold_total_s": 30.0,
                     "new_prompt_totals": [round(x, 2) for x in news], "new_prompt_median_s": news[1],
                     "cell_peak_vram_over_base_mib": 13600.0}}
    if stack == "studio":
        d["studio_packages"] = {"torch": "2.10.0", "triton": "3.6.0", "diffusers": "0.37.0", "transformers": "5.2.0",
                                "torchao": None, "unsloth-studio-i8gemm": "0.3.1", "unsloth_zoo": "2026.11.3"}
    return d


def test_import_nov18(work: Path) -> None:
    import import_tester as IT  # noqa: PLC0415

    res = work / "nov18_results"
    res.mkdir(parents = True, exist_ok = True)
    for stack in ("studio", "bline"):
        d = tester_nov18_result(stack)
        (res / f"{d['name']}.json").write_text(json.dumps(d))
    out = work / "nov18_import"
    with contextlib.redirect_stdout(io.StringIO()):
        rc = IT.main([str(res), "--out", str(out), "--schema-dir", str(TK)])
    summ = json.loads((out / "import_summary.json").read_text())
    recs = {r["arm"]["stack"]: r for r in (json.loads(p.read_text()) for p in out.rglob("record.json"))}
    s, c = recs.get("studio") or {}, recs.get("blineui") or {}
    check(rc == 0 and summ["valid"] == 2, f"Nov 18 tester results import and validate ({summ['rows']})")
    tel = s.get("telemetry") or {}
    check(tel.get("sm_mhz_mean") == 2504.0 and tel.get("power_w_mean") == 250.0 and tel.get("temp_c_max") == 65,
          f"requests[].clocks -> telemetry over the new-prompt requests ({tel})")
    check((c.get("revisions") or {}).get("blineui_tag") == "v0.45.4"
          and (s.get("revisions") or {}).get("zoo") == "2026.11.3"
          and str((s.get("revisions") or {}).get("studio", "")).startswith("96bf6de124d"),
          "baseline tag from code.bline_describe, Studio zoo / torch from studio_packages, Studio commit kept")
    mem = s.get("memory") or {}
    check(mem.get("peak_rss_tree_gib") is None and (s.get("legacy") or {}).get("host_used_peak_gib")
          and abs((mem.get("peak_vram_device_gib") or 0) - 14100 / 1024) < 1e-3
          and (s.get("timing") or {}).get("new_s_median") == 5.4,
          "host used RAM stays legacy (not RSS); device VRAM = over base + base; new-prompt median kept")


def test_import(work: Path) -> None:
    ws = Path(os.environ.get("WORKSPACE") or TK.parents[2])
    res = ws / "outputs" / "bline_r5_nov17" / "archive" / "results"
    if not res.exists():
        print(f"NOTE {res} absent: tester import skipped")
        return
    schema_dir = os.environ.get("DBENCH_SCHEMA_DIR") or str(TK)
    import import_tester as IT

    out = work / "r5_import"
    args = [str(res), "--out", str(out), "--bline-tag", "2aa4d535=v0.45.4"]
    if schema_dir:
        args += ["--schema-dir", schema_dir]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = IT.main(args)
    summ = json.loads((out / "import_summary.json").read_text())
    if str(summ["validated_with"]).startswith("SKIPPED"):
        print("NOTE schema.py not importable (set DBENCH_SCHEMA_DIR or merge B): records written, validation skipped")
        check(summ["count"] == 131, f"tester import: {summ['count']} files converted (validation skipped)")
    else:
        check(rc == 0 and summ["valid"] == 131, f"tester import: {summ['valid']}/{summ['count']} valid against "
                                                f"{summ['validated_with']}")
    recs = [json.loads(p.read_text()) for p in out.rglob("record.json")]
    sim = [r for r in recs if (r.get("memory") or {}).get("budget", {}).get("simulated")]
    check(recs and all((r.get("cell") or {}).get("variant") and (r.get("arm") or {}).get("comparison") for r in recs)
          and sim and all(f"b{r['memory']['budget']['gib']:g}" in r["arm"]["comparison"] for r in sim)
          and not any((r.get("timing") or {}).get("step_s_derived_flag") for r in recs),
          f"tester import keeps config (variant) and budget (comparison id), no blanket step-cache flag "
          f"({len(recs)} records, {len(sim)} simulated)")
    import report as Rp  # noqa: PLC0415

    cards = Rp.build(out, use_lpips = False)["cards"]
    mixed = [c["comparison"] for c in cards if isinstance(c.get("budget_gib"), list)]
    check(cards and not mixed, f"report on the tester import: {len(cards)} groups, none mixes budgets {mixed[:3]}")


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", default = None, help = "work dir to keep (default: a temp dir, removed)")
    ap.add_argument("--allow-dirty", action = "store_true", help = "bundle an uncommitted tree (dev only)")
    a = ap.parse_args()
    tmp_root = Path(os.environ["WORKSPACE"]) / "temp" if os.environ.get("WORKSPACE") else None
    if tmp_root:
        tmp_root.mkdir(parents = True, exist_ok = True)
    work = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix = "dbench_targets_", dir = tmp_root))
    work.mkdir(parents = True, exist_ok = True)
    rev = TC.toolkit_revision()
    allow = a.allow_dirty or rev["dirty"]
    if rev["dirty"]:
        print("NOTE toolkit tree is dirty: bundles are built from the working tree (prepare refuses that by default)")
    try:
        plan = test_plan(work)
        test_matrix(work)
        test_real_cards(work)
        test_cases(work)
        test_matrix_gate(work)
        test_prepare(plan, work, allow)
        test_round(work)
        test_round_relative_out(work)
        test_prefetch_skip()
        test_time_accounting(work)
        test_cache_janitor(work)
        test_pack_keeps_per_pair_parity(work)
        test_disk_hint()
        test_speed_default()
        test_localize_studio()
        test_import_nov18(work)
        test_release_built_payload(work)
        test_online_probe_token()
        test_stage_cases_per_vm(work)
        test_prefetch_per_vm()
        test_parity_only_gate()
        test_driver(work)
        test_tester(work / "plan.json", work, allow)
        test_export_keys(plan, work, allow)
        test_kit(plan, work, allow)
        test_import(work)
        public_bundle_check(work)
    finally:
        if not a.keep:
            shutil.rmtree(work, ignore_errors = True)
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
