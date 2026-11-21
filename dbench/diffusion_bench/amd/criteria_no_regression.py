#!/usr/bin/env python3
"""Criteria: does the head make any diffusion cell worse than the base on gfx1151?

MODE regression, so the verdict is NO_REGRESSION or REGRESSION (or INCONCLUSIVE when a gate fails). This is a
measurement run and says so: it cannot CONFIRM a fix, only that nothing measured got worse. Use
criteria_differential.py when the PR claims to fix something.

A head cell regresses when, against the same cell at the base:
  - it was ok and now is BROKEN / MISSING (a load that fails closed counts: an explicit quant scheme never falls
    back silently, so BROKEN is the honest outcome);
  - it now shows a sanity flag (black, constant, blank frame, frozen clip, missing media);
  - its LPIPS against its own state's reference rose by more than DBENCH_AMD_LPIPS_TOL (default 0.03);
  - its median new-prompt s/image AND its steady s/image both rose by more than DBENCH_AMD_MAX_SLOWDOWN (1.25x).
    Both, because one noisy number on this APU has been off by 2x (a 524 s outlier on Windows).
Cells newly fixed at the head, and levers that resolved differently (quant, cache, offload), are reported.
Cells with "tier": "tiny" (hf-internal-testing random pipelines) are judged on plumbing only: broken or newly
flagged counts, LPIPS and speed do not (their output is noise).

Pairs with amd/diffusion_probe.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import criteria_common as CC  # noqa: E402

TITLE = "diffusion_bench on gfx1151: head vs base, no-regression"
MODE = "regression"
NEEDS = CC.NEEDS


def gates(obs: dict) -> list:
    return CC.host_gates(obs)


def table(obs: dict) -> str:
    return CC.side_by_side(obs)


def head_is_worse(base: dict, head: dict) -> tuple:
    worse, notes = [], []
    cb_all, ch_all = CC.cells(base), CC.cells(head)
    for tag, cb in cb_all.items():
        ch = ch_all.get(tag)
        if ch is None:
            notes.append(f"`{tag}` not run at the head (budget or selection)")
            continue
        if CC.ok(cb) and not CC.ok(ch):
            worse.append(f"`{tag}` ok at base, {ch.get('verdict')} at head: {str(ch.get('error'))[:160]}")
            continue
        if not CC.ok(cb) and CC.ok(ch):
            notes.append(f"`{tag}` newly works at the head (base: {str(cb.get('error'))[:100]})")
            continue
        if not CC.ok(ch):
            notes.append(f"`{tag}` broken at BOTH states, so it pre-dates this change: {str(ch.get('error'))[:100]}")
            continue
        new_flags = CC.bad_flags(ch) - CC.bad_flags(cb)
        if new_flags:
            worse.append(f"`{tag}` new sanity flags at the head: {', '.join(sorted(new_flags))}")
        stb, sth = cb.get("status") or {}, ch.get("status") or {}
        moved = [k for k in ("transformer_quant", "text_encoder_quant", "transformer_cache", "offload_policy",
                             "attention_backend", "speed_optims") if stb.get(k) != sth.get(k)]
        if moved:
            notes.append(f"`{tag}` resolved differently at the head: "
                         + ", ".join(f"{k} {stb.get(k)} -> {sth.get(k)}" for k in moved))
        if "tiny" in (ch.get("tier"), cb.get("tier")):
            continue  # random-weight pipelines: plumbing only, their pixels and timings carry no quality or speed claim
        lb, lh = cb.get("lpips"), ch.get("lpips")
        if lb is not None and lh is not None and lh - lb > CC.LPIPS_TOL:
            worse.append(f"`{tag}` LPIPS vs {ch.get('ref')} {lb:.4f} -> {lh:.4f} (+{lh - lb:.4f} > {CC.LPIPS_TOL})")
        nb, nh = cb.get("new_s"), ch.get("new_s")
        sb, sh = cb.get("steady_s"), ch.get("steady_s")
        if nb and nh and nh / nb > CC.MAX_SLOWDOWN:
            if sb and sh and sh / sb > CC.MAX_SLOWDOWN:
                worse.append(f"`{tag}` slower: new {nb:.2f} -> {nh:.2f} s, steady {sb:.2f} -> {sh:.2f} s "
                             f"(> {CC.MAX_SLOWDOWN}x on both)")
            else:
                notes.append(f"`{tag}` new-prompt s/image {nb:.2f} -> {nh:.2f} s but steady did not agree; noise")
    # Edge suite (probe --edge): compared by check id, never by count, like the pytest criteria.
    eb, eh = (base.get("edge") or {}).get("checks") or {}, (head.get("edge") or {}).get("checks") or {}
    for check, status in sorted(eh.items()):
        if status == "FAIL" and eb.get(check) == "PASS":
            worse.append(f"edge `{check}` PASS at base, FAIL at head")
        elif status == "FAIL" and eb.get(check) == "FAIL":
            notes.append(f"edge `{check}` fails at BOTH states")
        elif status == "PASS" and eb.get(check) == "FAIL":
            notes.append(f"edge `{check}` newly passes")
    if base.get("edge") and head.get("edge") and not eh:
        notes.append(f"edge suite produced no results at the head (rc {head['edge'].get('rc')})")
    if worse:
        return True, "; ".join(worse) + (". Also: " + "; ".join(notes) if notes else "")
    detail = "no cell that worked at the base is broken, flagged, less accurate or slower at the head"
    if notes:
        detail += ". Notes: " + "; ".join(notes)
    return False, detail
