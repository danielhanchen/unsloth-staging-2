#!/usr/bin/env python3
"""The one reporter: every record.json under ROOT (schema v2; v1 upgraded and marked legacy) -> a discussion-ready
Markdown report (and JSON). Replaces the per-round table scripts (r5b_summary.py, table5.py, matrix5.py, tables.py,
the per-round LPIPS scripts).

  python diffusion_bench/report.py ROOT [--expect cases.json] [--out ROOT/report.md] [--json ROOT/report.json]
                                        [--allow-incomplete] [--no-lpips] [--baseline blineui]

What it writes:
  per comparison (the plan's arm.comparison: family x target x budget, then the format / config variant; a
    simulated-budget run never shares a group with a full-card run): Studio vs each other stack (BlineUI by default first), ratio of median
    new-prompt seconds (other / Studio: > 1 = Studio faster) against the A/A band (the spread of each stack's
    repeated fresh-process medians; the wider stack's band is used), verdict FASTER / SLOWER / TIE, or UNDECIDED when
    a band cannot be formed (fewer than 2 repeats of either arm)
  only valid pairs count: both records ok and schema-valid, the same VM (target.vm_id, never null), SM clocks within
    3% (UNAVAILABLE on an NVIDIA card voids the pair), no gate timeout, no BlineUI cache hit on a timed request, and
    a parity-gate result for the group that is ok (parity.json files under ROOT, written by parity.py)
  per stack: cold (load + first image), first step, first image, steady s/image, s/step (polled; derived only when
    no step cache), compile, peak VRAM (own tree / over baseline), peak host RSS (process tree), quality vs the
    stack's OWN bf16 (LPIPS / PSNR, paired by prompt id and seed); a missing number prints UNAVAILABLE
  coverage matrix: family x target: measured / VOID (why) / NOT_RUN (why) / unsupported / BROKEN / MISSING
  could not test: every VOID, NOT_RUN, BROKEN, missing case, missing reference and UNAVAILABLE metric, by name

It refuses (exit 2, nothing written) when the expected case set is incomplete: --expect (a JSON list of tags or of
{"tag": ...}, or {"cases": [...]}) or, when absent, ROOT/cases.json, else ROOT/spec.json's cells x passes. A NOT_RUN
record counts as accounted for; a missing record does not. --allow-incomplete writes it with an INCOMPLETE banner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import mirror  # noqa: E402
import parity as P  # noqa: E402
import schema  # noqa: E402

CLOCK_TOL = 0.03
UNAV = "UNAVAILABLE"
DUP = "(duplicate record copies)"  # load_records' marker entry: copies of an already-counted run were dropped


# ---------------------------------------------------------------------------------------------- loading
def load_records(root: Path, untimed: Optional[list] = None) -> tuple:
    """([(dir, record, effective_verdict, reasons)], unreadable[(path, error)]). Parity-mode record dirs are appended
    to `untimed`: a purpose "parity" plan's cases are those records, so they count toward completeness only."""
    out, bad = [], []
    for path in sorted(root.rglob(C.RECORD)):
        try:
            rec = schema.load(path)
        except Exception as exc:  # noqa: BLE001 - incl. NaN / Infinity in the file
            bad.append((str(path.relative_to(root)), f"{type(exc).__name__}: {exc}"))
            continue
        arm = rec.get("arm") or ((rec.get("cell") or {}).get("options") or {}).get("arm") or {}
        if rec.get("mode") in ("parity", "trial") or arm.get("purpose") in ("trial", "parity") or \
                str(rec.get("tag") or "").startswith("trial__"):
            if untimed is not None and (rec.get("mode") == "parity" or arm.get("purpose") == "parity"):
                untimed.append(path.parent)
            continue  # untimed: never a repeat, a median, an A/A point or a memory peak
        verdict = rec.get("verdict")
        reasons = [rec.get("reason") or rec.get("void_reason")] if verdict != "ok" else []
        if verdict == "ok":
            probs = schema.validate(rec)
            probs += P.cache_problems(rec)
            if probs:
                verdict, reasons = "VOID", probs
        out.append((path.parent, rec, verdict, [r for r in reasons if r]))
    # one entry per run (schema.run_identity): an artifact unpacked twice is one repeat, not two
    out, dropped = schema.dedup(out, lambda x: x[1])
    if dropped:
        bad.append((DUP, f"dropped {dropped} duplicate record copies of runs already counted"))
    return out, bad


def load_parity(root: Path, name: str = "parity.json") -> list:
    out = []
    for path in sorted(root.rglob(name)):
        try:
            res = schema.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        if res.get("created") is None:  # older results: the file's write time orders them
            res["created"] = path.stat().st_mtime
        out.append(res)
    return out


def expected_cases(root: Path, expect: Optional[str]) -> Optional[list]:
    path = Path(expect) if expect else (root / "cases.json" if (root / "cases.json").exists() else None)
    if path is not None:
        data = json.loads(path.read_text())
        data = data.get("cases", data) if isinstance(data, dict) else data
        return [d["tag"] if isinstance(d, dict) else str(d) for d in data]
    spec_path = root / "spec.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text())
        # untimed plan cells (trial, parity probes) leave no timed record: not expected cases
        tags = [c["tag"] for c in spec.get("cells") or [] if not str(c["tag"]).startswith("trial__")
                and not (c.get("options") or {}).get("parity_probe")]
        passes = [p["name"] for p in spec.get("passes") or []]
        return [f"{p}/{t}" for p in passes for t in tags] if passes else tags
    return None


def planner_refusals(root: Path, expect: Optional[str]) -> list:
    """plan.py's NOT_RUN list carried in cases.json ({"not_run": [{family, target, arm, reason}]}), else []."""
    path = Path(expect) if expect else root / "cases.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    return [r for r in (data.get("not_run") or []) if isinstance(r, dict)] if isinstance(data, dict) else []


def case_key(root: Path, cell_dir: Path) -> str:
    return str(cell_dir.relative_to(root)).replace("\\", "/")


# ---------------------------------------------------------------------------------------------- facts per record
def family(rec: dict) -> str:
    return P.group_of(rec)["family"]


def variant(rec: dict) -> str:
    return P.group_of(rec)["variant"]


def target(rec: dict) -> str:
    t = rec.get("target") or {}
    name = t.get("name") or "unknown"
    b = (rec.get("memory") or {}).get("budget") or {}
    if b.get("simulated") and b.get("gib"):
        name += f" (simulated {b['gib']} GiB)"
    return name


def budget_of(rec: dict) -> Optional[float]:
    """The simulated budget (GiB) a record ran under, None for the full card."""
    b = ((rec.get("memory") or {}).get("budget") or {}).get("gib")
    cell = rec.get("cell") or {}
    b = b or ((cell.get("options") or {}).get("target") or {}).get("simulated_budget_gib") or cell.get("budget_gib")
    return float(b) if b else None


def comparison_of(rec: dict) -> str:
    """The plan's comparison id (family x target x budget; plan.py arm.comparison), else a key built from the same
    facts, so a simulated-budget run and a full-card run never share a group."""
    cmp = (rec.get("arm") or {}).get("comparison") or (((rec.get("cell") or {}).get("options") or {}).get("arm")
                                                       or {}).get("comparison")
    if cmp:
        return str(cmp)
    b = budget_of(rec)
    return f"{family(rec)}@{(rec.get('target') or {}).get('name') or 'unknown'}@" + (f"b{b:g}" if b else "full")


def group_key(rec: dict) -> tuple:
    return (family(rec), variant(rec), (rec.get("target") or {}).get("name") or "unknown", comparison_of(rec))


def stack(rec: dict) -> str:
    return (rec.get("arm") or {}).get("stack") or schema.stack_of(rec.get("backend")) or "?"


def tier_of(rec: dict) -> Optional[dict]:
    """The tier a Studio run RESOLVED (run_cell's resolved_tier; older records: derived from status_after)."""
    if stack(rec) != "studio":
        return None
    # derived from the record's own Studio status with the current reader (a record's stored resolved_tier is the
    # reader of the day it ran); the stored copy only when the status is gone
    return C.resolved_tier(rec.get("status_after") or rec.get("status")) or rec.get("resolved_tier")


def probe_labels(untimed: list) -> dict:
    """{parity probe tag: its arm label} for every parity record under ROOT (Studio and baseline)."""
    out = {}
    for d in untimed:
        try:
            rec = schema.load(Path(d) / C.RECORD)
        except Exception:  # noqa: BLE001
            continue
        if rec.get("tag"):
            out[rec["tag"]] = (rec.get("arm") or {}).get("label")
    return out


def probe_tiers(untimed: list) -> dict:
    """{parity probe tag: resolved tier} for the Studio parity records under ROOT."""
    out = {}
    for d in untimed:
        try:
            rec = schema.load(Path(d) / C.RECORD)
        except Exception:  # noqa: BLE001
            continue
        if stack(rec) == "studio" and rec.get("tag"):
            out[rec["tag"]] = tier_of(rec)
    return out


def t(rec: dict, key: str):
    return (rec.get("timing") or {}).get(key)


def fmt(v, unit: str = "", nd: int = 3) -> str:
    if v is None:
        return UNAV
    if isinstance(v, float):
        return f"{v:.{nd}f}{unit}"
    return f"{v}{unit}"


# ---------------------------------------------------------------------------------------------- pairing
POWER_TOL = 0.05      # the two runs' mean power draw: within 5% of the higher one
POWER_CAP_TOL_W = 1.0  # the two runs' enforced power cap: within 1 W (a cap is a setting, not a measurement)
DENSE_REF = ("bf16", "fp16")  # a stack's quality reference: its own dense run, bf16 first


def power_facts(r: dict) -> dict:
    """The clock rule's inputs of one run: SM clock, mean power draw, enforced power cap (and its range)."""
    t = r.get("telemetry") or {}
    return {"sm_mhz": t.get("sm_mhz_mean"), "power_w": t.get("power_w_mean"), "power_cap_w": t.get("power_limit_w"),
            "power_cap_range": [t.get("power_limit_w_min"), t.get("power_limit_w_max")]}


def equal_power(s: dict, o: dict) -> tuple:
    """(True, facts) when both runs ran at the SAME enforced power cap (within POWER_CAP_TOL_W, steady within each
    run) and drew the SAME power (within POWER_TOL); else (False, why). On a power-capped card (L4 72 W, T4 70 W)
    the governor sets the SM clock from the workload, so a clock difference between two stacks at one cap and one
    draw is the stacks' own effect (L4 FLUX.3: Studio 1222 MHz at 70.7 W, the baseline 1297 MHz at 67.9 W), not a
    throttled or different device; a device-side drop (thermal, contention) shows as a draw difference, and a
    different cap (a changed limit, another card) is never excused. A run without a recorded cap (records before the
    cap was sampled, a card without power management) cannot be excused."""
    fs, fo = power_facts(s), power_facts(o)
    facts = {"studio_or_a": fs, "other_or_b": fo}
    for name, f in (("a", fs), ("b", fo)):
        if not f["power_w"]:
            return False, f"power draw {UNAV} on run {name}"
        if f["power_cap_w"] is None:
            return False, f"power cap {UNAV} on run {name} (not recorded)"
        lo, hi = f["power_cap_range"]
        if lo is not None and hi is not None and hi - lo > POWER_CAP_TOL_W:
            return False, f"power cap changed during run {name} ({lo:.0f}-{hi:.0f} W)"
    if abs(fs["power_cap_w"] - fo["power_cap_w"]) > POWER_CAP_TOL_W:
        return False, f"power caps differ ({fs['power_cap_w']:.0f} vs {fo['power_cap_w']:.0f} W)"
    ps, po = fs["power_w"], fo["power_w"]
    if abs(ps - po) / max(ps, po) > POWER_TOL:
        return False, f"power draw differs by more than {POWER_TOL:.0%} ({ps:.1f} vs {po:.1f} W)"
    return True, facts


BELOW_CAP_FRAC = 0.90  # below-cap rule: each run's mean draw under 90% of its enforced cap
LIMITING_THROTTLE = ("sw_power_cap", "hw_slowdown", "sw_thermal", "hw_thermal", "hw_power_brake")


def below_cap(s: dict, o: dict) -> tuple:
    """(True, facts) when BOTH runs drew under BELOW_CAP_FRAC of their enforced power cap and the clock throttle
    reasons sampled over each timed window (telemetry.throttle_reasons) show no power-cap, thermal or hardware
    slowdown: neither device held its clock down, so a clock gap is the stack's own duty cycle (an A100 at 160-220 W
    of 400 W: Studio's offloaded, host-bound LTX-2.5 runs average a lower SM clock), not a throttled or limited card.
    Unread throttle reasons (records before they were sampled, a driver without them) cannot prove it: (False, why)."""
    fs, fo = power_facts(s), power_facts(o)
    reasons = []
    for name, f, r in (("a", fs, s), ("b", fo, o)):
        if not f["power_w"] or f["power_cap_w"] is None:
            return False, f"power draw or cap {UNAV} on run {name}"
        if f["power_w"] >= BELOW_CAP_FRAC * f["power_cap_w"]:
            return False, (f"run {name} drew {f['power_w']:.0f} W, not below {BELOW_CAP_FRAC:.0%} of its "
                           f"{f['power_cap_w']:.0f} W cap")
        thr = (r.get("telemetry") or {}).get("throttle_reasons")
        if thr is None:
            return False, f"clock throttle reasons {UNAV} on run {name} (not recorded)"
        hit = [x for x in thr if x in LIMITING_THROTTLE]
        if hit:
            return False, f"run {name} throttled during the timed window ({', '.join(hit)})"
        reasons.append(list(thr))
    return True, {"studio_or_a": fs, "other_or_b": fo, "throttle": reasons}


def power_capped(s: dict, o: dict) -> bool:
    """Compatibility wrapper: equal_power()'s verdict only."""
    return equal_power(s, o)[0]


def clock_rule(s: dict, o: dict) -> tuple:
    """(problem or None, excuse or None) for the SM-clock rule: clocks within CLOCK_TOL of the slower one pass;
    a larger difference passes only at equal power (equal_power), and the excuse (both clocks, caps and draws)
    is returned so the report can print it next to the pair."""
    cs, co = (s.get("telemetry") or {}).get("sm_mhz_mean"), (o.get("telemetry") or {}).get("sm_mhz_mean")
    nvidia = any("nvidia" in str((r.get("gpu") or {}).get("name") or "").lower() or
                 (r.get("gpu") or {}).get("source") in ("nvml", "fake") for r in (s, o))
    if cs is None or co is None:
        return (f"SM clock {UNAV} ({cs}, {co}): the 3% rule cannot be checked" if nvidia else None), None
    if abs(cs - co) / min(cs, co) <= CLOCK_TOL:  # within 3% of the slower clock
        return None, None
    ok, why = equal_power(s, o)
    kind = "equal power"
    if not ok:
        ok2, why2 = below_cap(s, o)
        if not ok2:
            return (f"SM clocks differ by more than 3% ({cs:.0f} vs {co:.0f} MHz), not at equal power ({why}) "
                    f"and not below cap unthrottled ({why2})"), None
        why, kind = why2, "below cap"
    fs, fo = why["studio_or_a"], why["other_or_b"]
    return None, {"kind": kind, "pair": [s.get("tag"), o.get("tag")], "sm_mhz": [cs, co],
                  "power_cap_w": [fs["power_cap_w"], fo["power_cap_w"]], "power_w": [fs["power_w"], fo["power_w"]],
                  "throttle": why.get("throttle")}


def pair_problems(s: dict, o: dict, parity_ok: Optional[dict], excused: Optional[list] = None) -> list:
    out = []
    vs, vo = (s.get("target") or {}).get("vm_id"), (o.get("target") or {}).get("vm_id")
    if not vs or not vo:
        out.append("vm_id missing: a pair is only valid on one VM")
    elif vs != vo:
        out.append(f"cross-VM pair ({vs} vs {vo})")
    clk, excuse = clock_rule(s, o)
    if clk:
        out.append(clk)
    elif excuse is not None and excused is not None:
        excused.append(excuse)
    for r in (s, o):
        g = r.get("gate") or {}
        if g.get("gate") == "timeout":
            out.append(f"{r.get('tag')}: GPU gate timed out (contended device)")
        if g.get("gate") == "unavailable" or (g.get("matrix") or {}).get("gate") == "unavailable":
            out.append(f"{r.get('tag')}: GPU gate could not run (unavailable): the device state is unproven")
        if (g.get("matrix") or {}).get("gate") == "timeout":
            out.append(f"{r.get('tag')}: matrix GPU gate timed out before the cell (contended device)")
    if parity_ok is None:
        out.append("no parity-gate result for this model / variant / target")
    elif parity_ok.get("verdict") != "ok":
        out.append("parity gate VOID: " + "; ".join(parity_ok.get("reasons") or [])[:300])
    return out


def band(values: list) -> tuple:
    """(relative spread of repeated medians, n): (max - min) / median."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None, len(vals)
    med = C.stat_median(vals)
    return ((max(vals) - min(vals)) / med if med else None), len(vals)


def target_class(name: Optional[str]) -> str:
    """Targets whose parity results are interchangeable: the same kind and the same resolved dtype set (fp16-only
    T4, bf16 A100, fp8 L4, nvfp4 G4 / B200); AMD, Mac, CPU and tester targets are each their own class."""
    name = name or "unknown"
    try:
        from targets import target_info  # noqa: PLC0415

        info = target_info(name)
    except Exception:  # noqa: BLE001
        return name
    if info["kind"] in ("amd", "mac", "cpu", "tester"):
        return name
    return "nvidia:" + "/".join(sorted(info.get("dtypes") or []))


def parity_candidates(results: list, fam: str, var: str, tgt: str, cmp: Optional[str] = None) -> list:
    """The parity results for this group, NEWEST first (``created``): the same comparison id when results carry
    one, else the same (family, variant, target name); else the same family and variant from another target of the
    same target class (target_class: same kind and resolved dtypes; never AMD / Mac from NVIDIA, never a T4 from a
    bf16 card), each marked ``from_target``. Only the first of those tiers that has any result is returned."""
    def newest_first(rs):
        return sorted(rs, key = lambda r: -float(r.get("created") or 0))

    same = [r for r in results if (r.get("group") or {}).get("family") == fam
            and str((r.get("group") or {}).get("variant") or "default") == var]
    if cmp:
        hits = newest_first([r for r in same if r.get("comparison") == cmp])
        if hits:
            return hits
    hits = newest_first([r for r in same if (r.get("group") or {}).get("target") == tgt
                         and (not cmp or not r.get("comparison") or r.get("comparison") == cmp)])
    if hits:
        return hits
    cls = target_class(tgt)
    other = newest_first([r for r in same if (r.get("group") or {}).get("target") != tgt
                          and target_class((r.get("group") or {}).get("target")) == cls])
    return [{**r, "from_target": (r.get("group") or {}).get("target")} for r in other]


def parity_for(results: list, fam: str, var: str, tgt: str, cmp: Optional[str] = None) -> Optional[dict]:
    """The NEWEST parity result for this group (parity_candidates' first), or None."""
    hits = parity_candidates(results, fam, var, tgt, cmp)
    return hits[0] if hits else None


# ---------------------------------------------------------------------------------------------- report
def build(root: Path, expect: Optional[str] = None, use_lpips: bool = True, baseline: str = "blineui") -> dict:
    untimed: list = []
    recs, unreadable = load_records(root, untimed)
    parities = load_parity(root)
    ptiers = probe_tiers(untimed)
    plabels = probe_labels(untimed)
    negctls = load_parity(root, "parity_negctl.json")  # deliberately mismatched pairs the gate must reject
    want = expected_cases(root, expect)
    have = {case_key(root, d) for d, *_ in recs} | {case_key(root, d) for d in untimed}
    # a bare tag in the case set matches a record's last path part only when the set itself names no pass
    # (pass/tag): with passes, the same tag under another pass must not satisfy it
    if not any("/" in c for c in want or []):
        have |= {case_key(root, d).split("/")[-1] for d in [d for d, *_ in recs] + untimed}
    missing = [c for c in want or [] if c not in have]
    problems: list = []
    groups: dict = {}
    for d, rec, verdict, reasons in recs:
        groups.setdefault(group_key(rec), []).append((d, rec, verdict, reasons))
    # quality vs own bf16
    metrics = None
    try:
        import score

        metrics = score.Metrics(use_lpips = use_lpips)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"quality scoring unavailable: {exc}")
    quality: dict = {}
    for d, rec, verdict, _ in recs:
        if verdict != "ok":
            continue
        if variant(rec) in DENSE_REF:
            quality[str(d)] = {"ref": f"self ({variant(rec)} reference)"}
            continue
        ref = None
        ref_tag = (rec.get("cell") or {}).get("ref")
        # the stack's own dense run of the same family on the same target: bf16, else fp16 (an fp16-native card such
        # as SDXL-Turbo has no bf16 format)
        for want in ([None] if ref_tag else list(DENSE_REF)):
            for d2, r2, v2, _ in recs:
                if v2 != "ok" or d2 == d:
                    continue
                if (ref_tag and r2.get("tag") == ref_tag) or (not ref_tag and family(r2) == family(rec) and
                                                             stack(r2) == stack(rec) and variant(r2) == want and
                                                             (r2.get("target") or {}).get("name") ==
                                                             (rec.get("target") or {}).get("name")):
                    ref = (d2, r2)
                    break
            if ref is not None:
                break
        if ref is None:
            quality[str(d)] = {"error": f"missing reference: no ok {stack(rec)} bf16 / fp16 cell for {family(rec)}"}
            problems.append(f"{rec.get('tag')}: quality {UNAV} (missing {stack(rec)} bf16 reference)")
            continue
        if metrics is None:
            quality[str(d)] = {"error": "no scorer"}
            continue
        import score

        q = score.quality_vs_reference(d, rec, ref[0], ref[1], metrics)
        q["ref"] = ref[1].get("tag")
        drop = "; ".join(f"{x['id']}: {x['reason']}" for x in (q.get("dropped") or [])[:6])
        if q.get("size_mismatch"):
            q["error"] = f"VOID: rendered on a different canvas than {q['ref']} ({q['size_mismatch'][:6]})"
            problems.append(f"{rec.get('tag')}: quality VOID (size mismatch vs {q['ref']}: {drop})")
            q = {**q, "lpips": None, "psnr": None, "ssim": None}
        elif not q["n"]:
            q["error"] = "no renders paired with the reference"
            problems.append(f"{rec.get('tag')}: quality {UNAV} (nothing paired with {q['ref']})")
        elif q.get("n_paired", q["n"]) < q.get("n_expected", q["n"]):
            q["error"] = f"{q['n_expected'] - q['n_paired']} of {q['n_expected']} renders unpaired"
            problems.append(f"{rec.get('tag')}: quality {UNAV} ({q['n_paired']}/{q['n_expected']} renders paired "
                            f"with {q['ref']}: {drop})")
            q = {**q, "lpips": None, "psnr": None, "ssim": None}
        quality[str(d)] = q
    cards = []
    for (fam, var, tgt, cmp_id), items in sorted(groups.items()):
        cands = parity_candidates(parities, fam, var, tgt, cmp_id)
        par = cands[0] if cands else None
        # one arm per key: a stack with several arms here (Studio default vs max tier, a config suffix) is split by
        # its arm label, so different arms never share a median or an A/A band
        labels: dict = {}
        tiers: dict = {}
        for _, rec, verdict, _ in items:
            labels.setdefault(stack(rec), set()).add((rec.get("arm") or {}).get("label"))
            if verdict == "ok" and stack(rec) == "studio":
                tiers.setdefault((rec.get("arm") or {}).get("label"), set()).add(C.tier_label(tier_of(rec)))
        by_stack: dict = {}
        for d, rec, verdict, reasons in items:
            st = stack(rec)
            lab = (rec.get("arm") or {}).get("label")
            key = st if len(labels[st]) <= 1 else f"{st}[{lab}]"
            if st == "studio" and len(tiers.get(lab) or ()) > 1:
                # Studio resolved different tiers for one requested arm (a default that compiled in one process and
                # fell back to eager in another): never one median or one A/A band, each tier is its own arm
                key = f"{key}{{{C.tier_label(tier_of(rec))}}}"
            by_stack.setdefault(key, []).append((d, rec, verdict, reasons))
        card_doc = P.load_card(fam)
        staged = sorted({x for _, r, v, _ in items if v == "ok" for x in mirror.staged_lines(r)})
        # the tier the parity gate proved: a timed Studio run whose compute-relevant tier differs from its parity
        # probe's (INT8 attention on in one, off in the other) computed something the gate never compared
        def probe_tier_of(pr: dict):
            tag = next((x for x in (pr.get("pair") or []) if x in ptiers), None)
            return (ptiers.get(tag) if tag else pr.get("studio_tier")), tag

        if par is not None:
            _, probe_tag = probe_tier_of(par)
            if probe_tag is None and not par.get("studio_tier") and any(stack(r) == "studio" and v == "ok" for _, r, v, _ in items):
                problems.append(f"{fam} / {var} / {tgt}: Studio's timed tier was not compared with its parity "
                                f"probe's (probe record {par.get('pair')} not under this root)")

        def parity_of_arm(arm: list, other: Optional[list] = None) -> Optional[dict]:
            """Each Studio x baseline arm pair is held to its OWN parity evidence: among the results whose baseline
            probe belongs to the paired baseline arm (when the probes say which), the newest whose Studio probe
            resolved the same compute tier as every ok timed run of the Studio arm (two arms of one format, SDPA and
            INT8 attention, each with its own probe). None matching: the newest of those, so a mismatch voids the
            arm as before."""
            runs = [tier_of(r) for _, r, v, _ in arm if v == "ok"]
            pool = cands
            labels = {(r.get("arm") or {}).get("label") for _, r, _, _ in other or []} - {None}
            if labels:
                mine = [pr for pr in cands if any(plabels.get(x) in labels for x in (pr.get("pair") or [])[1:])]
                pool = mine or cands
            for pr in pool:
                pt, _ = probe_tier_of(pr)
                if pt and runs and not any(C.tier_compute_diff(pt, x) for x in runs):
                    return pr
            return pool[0] if pool else par
        def base_of(k: str) -> str:  # "studio[arm]{tier}" -> "studio"
            return k.split("[")[0].split("{")[0]

        studios = sorted(k for k in by_stack if base_of(k) == "studio")
        others = sorted((k for k in by_stack if base_of(k) != "studio"),
                        key = lambda k: (base_of(k) != baseline, k))
        comparisons = []
        for skey, other in [(a, b) for a in studios for b in others]:
            studio = by_stack[skey]
            spar = parity_of_arm(studio, by_stack[other])
            probe_tier = probe_tier_of(spar)[0] if spar is not None else None
            valid, why, excused = [], [], []
            for ds, s, vs_, rs in studio:
                for do, o, vo_, ro in by_stack[other]:
                    if vs_ != "ok" or vo_ != "ok":
                        continue
                    exc: list = []
                    pp = pair_problems(s, o, spar, exc)
                    pp += [f"{o.get('tag')}: {x}" for x in mirror.record_problems(o, card_doc, var)]
                    tdiff = C.tier_compute_diff(probe_tier, tier_of(s))
                    if tdiff:
                        pp.append("Studio's timed run resolved a different tier than its parity probe (" +
                                  "; ".join(tdiff) + ")")
                    if not pp:
                        excused += exc
                    if pp:
                        why.append(f"{s.get('tag')} x {o.get('tag')}: " + "; ".join(pp))
                    else:
                        valid.append((s, o))
            s_ok = [s for s, _ in valid]
            o_ok = [o for _, o in valid]
            s_meds = [t(x, "new_s_median") for x in {id(x): x for x in s_ok}.values()]
            o_meds = [t(x, "new_s_median") for x in {id(x): x for x in o_ok}.values()]
            bs, ns = band(s_meds)
            bo, no = band(o_meds)
            sm, om = C.stat_median(s_meds), C.stat_median(o_meds)
            ratio = (om / sm) if sm and om else None
            if not valid:
                verdict = "VOID"
            elif bs is None or bo is None:
                verdict = "UNDECIDED (no A/A band: fewer than 2 fresh-process repeats of an arm)"
            else:
                b = max(bs, bo)
                verdict = "FASTER" if ratio > 1 + b else ("SLOWER" if ratio < 1 / (1 + b) else "TIE")
            comparisons.append({"studio": skey, "other": other, "valid_pairs": len(valid), "invalid": why, "ratio": ratio,
                                "parity_pair": (spar or {}).get("pair"),
                                "band": max(bs, bo) if bs is not None and bo is not None else None,
                                "aa_n": [ns, no], "verdict": verdict, "clock_excused": excused, "studio_tags": sorted({x.get("tag") for x in s_ok}),
                                "other_tags": sorted({x.get("tag") for x in o_ok})})
        stacks = {}
        for name, lst in by_stack.items():
            ok = [(d, r) for d, r, v, _ in lst if v == "ok"]
            def med(key, ok = ok):
                return C.stat_median([t(r, key) for _, r in ok])

            def memk(key, ok = ok):
                vals = [(r.get("memory") or {}).get(key) for _, r in ok]
                return max([v for v in vals if v is not None], default = None)

            flags = sorted({t(r, "step_s_derived_flag") for _, r in ok if t(r, "step_s_derived_flag")})
            q = [quality.get(str(d)) or {} for d, _ in ok]
            lp = [x.get("lpips") for x in q if x.get("lpips") is not None]
            ps = [x.get("psnr") for x in q if x.get("psnr") is not None]
            qerr = sorted({x["error"] for x in q if x.get("error")})
            qref = any(str(x.get("ref", "")).startswith("self") for x in q)
            # a setting the stack ran differently from the request but within the rules (run_cell._settings flags):
            # measured, and said next to the numbers
            sflags = sorted({f"{k}: {v['flag']}" for _, r in ok for k, v in (r.get("settings") or {}).items()
                             if isinstance(v, dict) and v.get("flag")})
            stacks[name] = {
                "n_ok": len(ok), "n_total": len(lst), "cold_s": med("cold_s"), "first_step_s": med("first_step_s"),
                "first_render_s": med("first_render_s"), "new_s": med("new_s_median"),
                "steady_s": med("steady_s_median"), "step_polled": med("step_s_polled_median"),
                "step_derived": None if flags else med("step_s_derived"), "derived_flags": flags,
                "compile_s": med("compile_s"), "compile_s_derived": med("compile_s_derived"),
                "vram_own": memk("peak_vram_own_gib"), "vram_over_base": memk("peak_vram_over_base_gib"),
                "rss_tree": memk("peak_rss_tree_gib"),
                "lpips": C.stat_median(lp), "psnr": C.stat_median(ps),
                "quality_note": "dense reference" if qref and not lp else ("; ".join(qerr) if qerr and not lp else None),
                "simulated": any(((r.get("memory") or {}).get("budget") or {}).get("simulated") for _, r in ok),
                "setting_flags": sflags,
                "resolved_tier": sorted({C.tier_label(tier_of(r)) for _, r in ok}) if name.startswith("studio") else [],
                "requested_tier": sorted({str(((r.get("cell") or {}).get("options") or {}).get("speed_mode"))
                                          for _, r in ok}) if name.startswith("studio") else [],
                # the clock rule's inputs, per stack (medians over its ok runs; UNAVAILABLE when never recorded)
                "sm_mhz": C.stat_median([(r.get("telemetry") or {}).get("sm_mhz_mean") for _, r in ok]),
                "power_w": C.stat_median([(r.get("telemetry") or {}).get("power_w_mean") for _, r in ok]),
                "power_cap_w": C.stat_median([(r.get("telemetry") or {}).get("power_limit_w") for _, r in ok])}
        budgets = sorted({budget_of(r) or 0 for _, r, _, _ in items})
        cards.append({"family": fam, "variant": var, "target": tgt, "comparison": cmp_id, "staged_files": staged,
                      "budget_gib": budgets[0] or None if len(budgets) == 1 else budgets,
                      "parity": par and par.get("verdict"),
                      "parity_from": (par or {}).get("from_target"),
                      "parity_waived": (par or {}).get("waived") or [],
                      "parity_accepted": (par or {}).get("accepted_differences") or [],
                      "parity_negctl": parity_for(negctls, fam, var, tgt, cmp_id),
                      "comparisons": comparisons, "stacks": stacks})
    # coverage
    coverage: dict = {}
    for d, rec, verdict, reasons in recs:
        fam, tg = family(rec), (rec.get("target") or {}).get("name") or "unknown"
        cur = coverage.setdefault(fam, {}).get(tg)
        if verdict == "ok":
            state = "measured"
        elif verdict == "NOT_RUN" and str(reasons[0] if reasons else "").startswith("unsupported"):
            state = "unsupported"
        else:
            state = f"{verdict}: {(reasons or ['?'])[0][:80]}"
        rank = {"measured": 0, "unsupported": 2}.get(state, 1)
        if cur is None or rank > cur[0]:
            coverage[fam][tg] = (rank, state)
    for card in cards:
        if card["comparisons"] and all(c["verdict"] == "VOID" for c in card["comparisons"]):
            cell = coverage.get(card["family"], {}).get(card["target"])
            if cell and cell[1] == "measured":
                coverage[card["family"]][card["target"]] = (1, "VOID: no valid pair (see could not test)")
    # the planner's refusals (cases.json "not_run"): a requested cell that never got a record is still named
    for r in planner_refusals(root, expect):
        fam, tg = r.get("family") or "?", r.get("target") or "?"
        if fam not in coverage or tg not in coverage[fam]:
            coverage.setdefault(fam, {})[tg] = (1, f"NOT_RUN: {str(r.get('reason'))[:80]}")
    could_not = [f"{r.get('family')} {r.get('target')} {r.get('arm')}: NOT_RUN (planner): {r.get('reason')}"
                 for r in planner_refusals(root, expect)]
    for d, rec, verdict, reasons in recs:
        if verdict != "ok":
            could_not.append(f"{case_key(root, d)} [{stack(rec)}, {target(rec)}]: {verdict}: "
                             f"{'; '.join(reasons)[:400] or '?'}")
    for c in missing:
        could_not.append(f"{c}: MISSING (expected case, no record)")
    for path, err in unreadable:
        could_not.append(f"WARNING: {err}" if path == DUP else f"{path}: UNREADABLE ({err[:200]})")
    for card in cards:
        where = f"{card['family']} {card['variant']} {card['target']} [{card['comparison']}]"
        for comp in card["comparisons"]:
            for w in comp["invalid"]:
                could_not.append(f"{where} {comp.get('studio', 'studio')} vs {comp['other']}: pair not counted: "
                                 f"{w[:400]}")
            if comp["verdict"].startswith("UNDECIDED"):
                could_not.append(f"{where} {comp.get('studio', 'studio')} vs {comp['other']}: {comp['verdict']}")
        for name, st in card["stacks"].items():
            gaps = [k for k in ("cold_s", "first_step_s", "new_s", "step_polled", "vram_own", "rss_tree")
                    if st.get(k) is None and st["n_ok"]]
            if st["n_ok"] and card["variant"] not in DENSE_REF and st.get("lpips") is None:
                gaps.append("lpips (quality vs own bf16)")
            if gaps:
                could_not.append(f"{where} {name}: {UNAV}: {gaps}")
    could_not += problems
    return {"root": str(root), "complete": not missing, "missing": missing, "expected": want is not None,
            "cards": cards, "coverage": {f: {k: v[1] for k, v in d.items()} for f, d in coverage.items()},
            "could_not_test": sorted(set(could_not)), "n_records": len(recs), "time": time_accounting(root)}


def time_accounting(root: Path) -> list:
    """Where each VM's time went, from the driver's driver_summary.json (item seconds, prefetch, baseline-file
    downloads and builds) and setup_log.txt's SETUP_S: one row per VM with the Colab CU it cost at the target's
    rate (targets.TARGETS cu_per_h). Allocation waits and the notebook transport are outside the VM's own clock and
    are not counted."""
    import re  # noqa: PLC0415

    try:
        from targets import TARGETS  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        TARGETS = {}
    rows, seen = [], set()
    for path in sorted(root.rglob("driver_summary.json")):
        try:
            d = json.loads(path.read_text())
        except ValueError:
            continue
        if d.get("vm") in seen:  # the same artifact unpacked from the notebook and the log copy
            continue
        seen.add(d.get("vm"))
        setup = None
        sl = path.parent / "setup_log.txt"
        if sl.exists():
            m = re.findall(r"^SETUP_S (\d+)", sl.read_text(errors = "replace"), re.M)
            setup = float(m[-1]) if m else (0.0 if "SETUP_REUSED" in sl.read_text(errors = "replace") else None)
        kinds = {"trial": 0.0, "parity": 0.0, "timed": 0.0}
        for tag, r in (d.get("results") or {}).items():
            sec = float((r or {}).get("s") or 0)
            kinds["trial" if tag.startswith("trial__") else "parity" if "__parity" in tag else "timed"] += sec
        t = d.get("time") or {}
        elapsed = float(d.get("elapsed_s") or 0)
        other = elapsed - sum(kinds.values()) - float(t.get("downloads_s") or 0) - float(t.get("builds_s") or 0)
        total = (setup or 0) + float(t.get("prefetch_s") or 0) + elapsed
        rate = (TARGETS.get(d.get("target")) or {}).get("cu_per_h")
        rows.append({"vm": d.get("vm"), "target": d.get("target"), "setup_s": setup, "prefetch_s": t.get("prefetch_s"),
                     "downloads_s": t.get("downloads_s"), "builds_s": t.get("builds_s"), **{f"{k}_s": round(v, 1)
                                                                                           for k, v in kinds.items()},
                     "other_s": round(other, 1), "total_s": round(total, 1),
                     "cu": round(total / 3600 * rate, 2) if rate is not None else None})
    return rows


def markdown(rep: dict) -> str:
    L = [f"# Diffusion benchmark report: {Path(rep['root']).name}", ""]
    if not rep["complete"]:
        L += [f"> INCOMPLETE: {len(rep['missing'])} expected case(s) have no record. Not for discussion.", ""]
    if not rep["expected"]:
        L += ["> No expected case set (cases.json / spec.json / --expect): completeness could not be checked.", ""]
    L += ["Ratio = other stack's median new-prompt s/image over Studio's (> 1: Studio faster). A/A band = the wider "
          "relative spread of either arm's repeated fresh-process medians; a ratio inside it is a TIE.", ""]
    for card in rep["cards"]:
        bud = card.get("budget_gib")
        L += [f"## {card['family']} / {card['variant']} / {card['target']}"
              + (f" (simulated {bud:g} GiB)" if isinstance(bud, (int, float)) and bud else "")
              + f" [{card.get('comparison')}]", ""]
        par = card["parity"] or "not run"
        L.append(f"Parity gate: {par}" + (f" (from {card['parity_from']})" if card.get("parity_from") else "")
                 + (f" (waived: {', '.join(card['parity_waived'])})" if card["parity_waived"] else ""))
        neg = card.get("parity_negctl")
        if neg:
            L.append(f"Parity negative control ({neg.get('negative_control')}): {neg.get('control_verdict')}"
                     + (f" -- {'; '.join(neg.get('reasons') or [])[:200]}" if neg.get("reasons") else ""))
        if card.get("parity_accepted"):
            L.append("")
            L.append("Accepted precision differences (model card traps; reported, not voided):")
            for x in card["parity_accepted"]:
                L.append(f"- {x.get('id')} [{x.get('component')}]: {x.get('detail')}"
                         + (f" -- {x['note']}" if x.get("note") else ""))
        if card.get("staged_files"):
            L.append("")
            L.append("Baseline files staged from a mirror (verified against the official checkpoint they repackage):")
            L += [f"- {x}" for x in card["staged_files"]]
        L.append("")
        if card["comparisons"]:
            L += ["| comparison | valid pairs | ratio | A/A band | A/A n (studio, other) | verdict |",
                  "|---|---:|---:|---:|---|---|"]
            for c in card["comparisons"]:
                L.append(f"| {c.get('studio', 'studio')} vs {c['other']} | {c['valid_pairs']} | {fmt(c['ratio'], 'x')} | "
                         f"{fmt(None if c['band'] is None else c['band'] * 100, '%', 1)} | {c['aa_n'][0]}, "
                         f"{c['aa_n'][1]} | {c['verdict']} |")
            L.append("")
            for c in card["comparisons"]:
                for kind, rule in (("equal power", f"at equal power (same enforced cap within {POWER_CAP_TOL_W:g} W, "
                                                    f"draw within {POWER_TOL:.0%})"),
                                   ("below cap", f"below cap (each run under {BELOW_CAP_FRAC:.0%} of its enforced "
                                                 "cap, no power-cap / thermal / HW slowdown throttle reason in its "
                                                 "timed window)")):
                    ex = [x for x in c.get("clock_excused") or [] if x.get("kind", "equal power") == kind]
                    if not ex:
                        continue
                    # every pair a clock rule kept, with both arms' clocks, caps and draws
                    L.append(f"Clock rule, {kind} ({c.get('studio', 'studio')} vs {c['other']}): {len(ex)} pair(s) "
                             f"kept with SM clocks more than {CLOCK_TOL:.0%} apart {rule}:")
                    for x in ex[:8]:
                        L.append(f"- {x['pair'][0]} x {x['pair'][1]}: {x['sm_mhz'][0]:.0f} vs {x['sm_mhz'][1]:.0f} MHz, "
                                 f"cap {x['power_cap_w'][0]:.0f} vs {x['power_cap_w'][1]:.0f} W, draw "
                                 f"{x['power_w'][0]:.1f} vs {x['power_w'][1]:.1f} W")
                    if len(ex) > 8:
                        L.append(f"- ... {len(ex) - 8} more (report.json clock_excused)")
                    L.append("")
        L += ["| stack | ok/total | cold s | first step s | first image s | steady s/img | s/step polled | "
              "s/step derived | compile s | peak VRAM own GiB | over base GiB | peak RSS tree GiB | LPIPS vs own bf16 "
              "| PSNR | SM MHz / draw W / cap W |", "|---|---|" + "---:|" * 12 + "---|"]
        for name, s in sorted(card["stacks"].items(), key = lambda kv: (kv[0] != "studio", kv[0])):
            der = fmt(s["step_derived"], "", 4) if not s["derived_flags"] else "flagged: step cache on"
            comp = s["compile_s"] if s["compile_s"] is not None else s["compile_s_derived"]
            comp_txt = fmt(comp) + ("" if s["compile_s"] is not None or comp is None else " (derived)")
            lp = fmt(s["lpips"], "", 4) if s["lpips"] is not None else (s["quality_note"] or UNAV)
            L.append(f"| {name}{' (simulated budget)' if s['simulated'] else ''} | {s['n_ok']}/{s['n_total']} | "
                     f"{fmt(s['cold_s'])} | {fmt(s['first_step_s'])} | {fmt(s['first_render_s'])} | {fmt(s['new_s'])} | "
                     f"{fmt(s['step_polled'], '', 4)} | {der} | {comp_txt} | {fmt(s['vram_own'])} | "
                     f"{fmt(s['vram_over_base'])} | {fmt(s['rss_tree'])} | {lp} | {fmt(s['psnr'], '', 2)} | "
                     f"{fmt(s.get('sm_mhz'), '', 0)} / {fmt(s.get('power_w'), '', 1)} / {fmt(s.get('power_cap_w'), '', 0)} |")
        for name, s in sorted(card["stacks"].items()):
            for lab in s.get("resolved_tier") or []:
                # what Studio ran, not what the cell asked for: the numbers above are this tier's
                L.append(f"\n{name} resolved tier (requested {', '.join(s.get('requested_tier') or ['?'])}): {lab}")
            for f in s.get("setting_flags") or []:
                L.append(f"\n{name} setting note: {f}")
        L.append("")
    fams = sorted(rep["coverage"])
    tgts = sorted({tg for f in fams for tg in rep["coverage"][f]})
    L += ["## Coverage (model x target)", "", "| model | " + " | ".join(tgts) + " |", "|---|" + "---|" * len(tgts)]
    for f in fams:
        L.append(f"| {f} | " + " | ".join(rep["coverage"][f].get(tg, "-") for tg in tgts) + " |")
    if rep.get("time"):
        L += ["", "## Where the VM time went", "",
              "Seconds on the VM's own clock (allocation waits and the notebook transport excluded); CU at the "
              "target's Colab rate.", "",
              "| VM | target | setup | prefetch | file downloads | file builds | trial | parity probes | timed runs | "
              "other (gate, gaps) | total | CU |", "|---|---|" + "---:|" * 10]
        for r in rep["time"]:
            L.append(f"| {r['vm']} | {r['target']} | " + " | ".join(fmt(r.get(k), "", 0) for k in (
                "setup_s", "prefetch_s", "downloads_s", "builds_s", "trial_s", "parity_s", "timed_s", "other_s",
                "total_s")) + f" | {fmt(r.get('cu'), '', 2)} |")
    L += ["", "## Could not test", ""]
    L += [f"- {x}" for x in rep["could_not_test"]] or ["- nothing: every expected case measured, every metric present"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--expect", default = None)
    ap.add_argument("--out", default = None)
    ap.add_argument("--json", default = None)
    ap.add_argument("--allow-incomplete", action = "store_true")
    ap.add_argument("--no-lpips", action = "store_true")
    ap.add_argument("--baseline", default = "blineui")
    args = ap.parse_args()
    root = Path(C.expand_env(args.root)).resolve()
    rep = build(root, args.expect, use_lpips = not args.no_lpips, baseline = args.baseline)
    if not rep["complete"] and not args.allow_incomplete:
        print(f"REFUSED: {len(rep['missing'])} expected case(s) without a record: {rep['missing'][:20]}",
              file = sys.stderr)
        return 2
    md = markdown(rep)
    (Path(args.out) if args.out else root / "report.md").write_text(md, encoding = "utf-8")
    (Path(args.json) if args.json else root / "report.json").write_text(schema.dumps(rep), encoding = "utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
