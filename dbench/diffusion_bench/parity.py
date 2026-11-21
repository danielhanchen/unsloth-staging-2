#!/usr/bin/env python3
"""Parity gate: run before any timing. Two stacks' parity records (run_cell.py --parity) must describe the same
computation, or the pair is VOID with reasons before a second of timing is spent.

Checks (each a named row in the result; a missing piece of evidence is VOID unless the pair waives that check by
name, and waivers are printed in every report):
  settings     resolved steps / cfg / sampler / scheduler / shift / width / height / frames equal (numbers 1e-6);
               a card sampler_differs trap (accepted_difference, purposes ["timing"]) lists a sampler / scheduler
               difference on a TIMING pair instead of voiding it (and a same-length sigma difference with an
               accepted scheduler); a quality pair stays strict. A sampler / scheduler Studio does not state
               (HTTP cells see no scheduler class) is covered when the dumped sigmas AND the evaluation rows are
               equal on both stacks (named in the row); a stated, different name still voids
  sigmas      the dumped sigma schedules have the same length and agree to 1e-6 (the H3 19-vs-20 pass bug)
  passes       denoiser evaluations equal (batch items: CFG batched or not costs the same work), and each stack's
               model calls equal the model card's ``expected_passes`` when the card states it
  size         output width / height / frames as rendered equal on both and equal to the resolved settings
  precision    each component's (transformer, text_encoder, vae, audio_vae) precision from the model card's
               formats.<fmt>.studio / .bline ``precision`` blocks (else the backend's report): ``weights`` always
               compared, ``compute`` only when both sides state it; a difference the card lists as an accepted
               difference (models.accepted_differences) is reported under ``accepted_differences``, never voids;
               then each backend's own report of the weights it RAN is checked against its card side
               (precision_observed): a transformer whose reported weights family differs (an int8 card side that
               ran bf16) is VOID even on a bare token, a quantized transformer with no report is VOID as missing,
               other components' bare tokens inform. --waive precision:<component> waives one component
  noise        quality pairs only: both stacks draw the initial noise on the CPU in fp32, and the hashes agree when
               both stacks report one in the same layout
  cache        BlineUI only: the parity request (and every timed request in timed records) executed nodes > 0 and
               the sampler was not served from cache; a request with no execution facts is flagged
  traps        the model card's traps with a machine-checkable ``check`` name map onto the checks above through
               models.GATE_CHECK; checks enforced outside the pair gate are named (ENFORCED_ELSEWHERE); any other
               name is listed as unchecked, never passed

  python diffusion_bench/parity.py STUDIO_CELL_DIR BLINE_CELL_DIR [--quality] [--family F] [--waive sigmas,...]
  exit 0 = ok, 1 = VOID (parity.json written next to the first record), 2 = usage / unreadable
  python diffusion_bench/parity.py --plan PLAN.json --results DIR     # every pair plan.py scheduled
  exit 0 = all ok, 1 = a group VOID, 3 = records still missing (matrix.py --plan / vm_driver.py call gate_plan
  after the parity slot and turn the VOID groups' timed cells into NOT_RUN)
"""

from __future__ import annotations

import argparse
import ast
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import schema  # noqa: E402

SETTING_KEYS = ("steps", "cfg", "sampler", "scheduler", "shift", "width", "height", "frames")
SIGMA_TOL = 1e-6  # relative to the sigma's own size once it exceeds 1 (sigma_delta)


def sigma_delta(x: float, y: float) -> float:
    """|x - y| scaled to sigmas above 1: SDXL starts at sigma 14.6, where the two stacks' float32 schedules differ by
    6.7e-6 (4.6e-7 relative), the same rounding a flow-matching sigma in [0, 1] shows below 1e-6 absolute."""
    return abs(x - y) / max(1.0, abs(x), abs(y))
# Sampler / scheduler names differ across stacks for the same algorithm; compare these canonical names.
ALIASES = {"euler": "euler", "flowmatcheulerdiscretescheduler": "euler", "flow_match_euler": "euler",
           "euler_discrete": "euler", "eulerdiscretescheduler": "euler", "simple": "simple", "normal": "normal",
           "beta": "beta", "sgm_uniform": "sgm_uniform", "karras": "karras", "euler_a": "euler_ancestral",
           "eulerancestraldiscretescheduler": "euler_ancestral", "dpmsolvermultistepscheduler": "dpmpp_2m",
           "unipcmultistepscheduler": "uni_pc", "unipc": "uni_pc"}
PRECISION_ALIASES = {"bfloat16": "bf16", "torch.bfloat16": "bf16", "float16": "fp16", "half": "fp16",
                     "float32": "fp32", "float": "fp32", "int8_convrot": "int8", "convrot": "int8",
                     "fp8_e4m3fn": "fp8", "fp8_e4m3fn_fast": "fp8", "float8_e4m3fn": "fp8", "nvfp4": "nvfp4",
                     "default": None, "none": None, "": None}
TRAP_CHECKS = {"pass_count": "passes", "passes": "passes", "sigma_match": "sigmas", "sigmas": "sigmas",
               "cpu_noise": "noise", "gpu_vs_cpu_noise": "noise", "noise": "noise", "size_match": "size",
               "canvas": "size", "precision": "precision", "settings": "settings", "cache_defeat": "cache"}


def sigma_stages(p: dict) -> Optional[list]:
    """[[float, ...] per stage] from parity evidence (``sigma_schedules``, else the single ``sigmas``); None when
    absent or when any entry is not a finite number (a sanitized NaN became null: missing, never a crash)."""
    stages = p.get("sigma_schedules") or ([p["sigmas"]] if p.get("sigmas") else None)
    if not stages:
        return None
    out = []
    for st in stages:
        if not isinstance(st, list) or not st:
            return None
        vals = []
        for x in st:
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
                return None
            vals.append(float(x))
        out.append(vals)
    return out


def norm_name(v: Any) -> Any:
    if isinstance(v, str):
        k = v.strip().lower()
        return ALIASES.get(k, k)
    return v


def norm_precision(v: Any) -> Optional[str]:
    if v is None:
        return None
    k = str(v).strip().lower().replace("torch.", "")
    return PRECISION_ALIASES.get(k, k)


def same_value(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol = 0, abs_tol = 1e-6)
    return norm_name(a) == norm_name(b)


# ---------------------------------------------------------------------------------------------- expected passes
_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Compare, ast.BoolOp, ast.Name, ast.Load,
            ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Gt, ast.GtE, ast.Lt,
            ast.LtE, ast.Eq, ast.NotEq, ast.And, ast.Or, ast.Not, ast.USub, ast.Call)


def expected_passes(expr: Any, env: dict) -> Optional[int]:
    """Evaluate a card's ``expected_passes`` (an int, or an arithmetic expression over steps / cfg / cfg_batched /
    frames / width / height) without eval's reach. None when it cannot be evaluated."""
    if isinstance(expr, (int, float)):
        return int(expr)
    if not isinstance(expr, str) or not expr.strip():
        return None
    try:
        tree = ast.parse(expr, mode = "eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            return None
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in ("int", "min", "max")):
            return None
    names = {"int": int, "min": min, "max": max, **env}
    try:
        return int(eval(compile(tree, "<expected_passes>", "eval"), {"__builtins__": {}}, names))  # noqa: S307
    except Exception:  # noqa: BLE001
        return None


def load_card(family: Optional[str]) -> Optional[dict]:
    """Agent A's model card through models.load_card, None when the package or the card is absent."""
    if not family:
        return None
    try:
        from models import load_card as _load  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        return _load(family)
    except Exception:  # noqa: BLE001
        return None


def family_of(rec: dict) -> Optional[str]:
    cell = rec.get("cell") or {}
    # the model card's family first: options.family is the baseline UI's graph key (flux3-klein for flux.3-klein)
    return cell.get("card") or cell.get("family") or (cell.get("options") or {}).get("family")


def group_of(rec: dict) -> dict:
    """{family, variant, target}: what a parity result and a timed pair are matched on (report.py)."""
    cell = rec.get("cell") or {}
    o = cell.get("options") or {}
    return {"family": family_of(rec) or cell.get("model") or rec.get("tag") or "?",
            "variant": str(cell.get("variant") or o.get("format") or o.get("variant") or "default"),
            "target": (rec.get("target") or {}).get("name") or "unknown"}


# ---------------------------------------------------------------------------------------------- checks
def _row(ok: Optional[bool], detail: str, waived: bool = False) -> dict:
    status = "waived" if waived and ok is not True else ("ok" if ok else ("missing" if ok is None else "mismatch"))
    return {"status": status, "detail": detail}


def check_pair(a: dict, b: dict, card: Optional[dict] = None, quality: bool = False, waive: tuple = ()) -> dict:
    """{verdict: ok | VOID, reasons: [...], checks: {name: {status, detail}}, waived: [...]} for two parity records
    (either order; labels come from each record's arm.stack)."""
    la, lb = (a.get("arm") or {}).get("stack") or a.get("tag"), (b.get("arm") or {}).get("stack") or b.get("tag")
    pa, pb = a.get("parity") or {}, b.get("parity") or {}
    checks: dict = {}
    for rec, lab in ((a, la), (b, lb)):
        if rec.get("verdict") != "ok":
            checks[f"record_{lab}"] = _row(False, f"{lab} parity record verdict {rec.get('verdict')}: "
                                                  f"{rec.get('reason')}")
    # settings
    ra, rb = pa.get("resolved") or {}, pb.get("resolved") or {}
    from_cell = [lab for r, lab in ((ra, la), (rb, lb)) if str(r.get("source") or "").startswith("cell")]
    if not ra or not rb:
        checks["settings"] = _row(None, f"resolved settings missing ({la}: {bool(ra)}, {lb}: {bool(rb)})",
                                  "settings" in waive)
    elif from_cell:
        # the backend did not read its settings back (sd.cpp / diffusers fall back to the cell): comparing them
        # would compare the cell with itself
        checks["settings"] = _row(None, f"resolved settings not read back from the stack for {from_cell} "
                                        "(source: cell)", "settings" in waive)
    acc_settings = accepted_settings(card, quality)
    accepted: list = []
    # the shift and the scheduler only build the sigma grid: a stack that resolves the shift dynamically (Studio's
    # FLUX.3 klein pipeline) or takes explicit sigmas (BlineUI's ManualSigmas, Qwen-Image-2.1-Turbo) has no value to
    # report, and then the dumped schedules, when both agree, are the evidence
    ss_a, ss_b = sigma_stages(pa), sigma_stages(pb)
    tol_trap = sigma_tol_trap(card)
    tol = max(SIGMA_TOL, float(tol_trap["sigma_tol"])) if tol_trap else SIGMA_TOL
    sigmas_equal = bool(ss_a and ss_b and len(ss_a) == len(ss_b) and all(len(x) == len(y) for x, y in zip(ss_a, ss_b))
                        and max(sigma_delta(x, y) for st_a, st_b in zip(ss_a, ss_b) for x, y in zip(st_a, st_b)) <= tol)
    # an unstated Studio sampler / scheduler (HTTP cells see no scheduler class; some pipelines keep a family's own
    # sampler) is covered by the work it did: equal dumped sigmas AND equal denoiser evaluation rows on both sides.
    # A stated, different name still voids (unless the card accepts it for timing)
    ia0, ib0 = pa.get("denoiser_items"), pb.get("denoiser_items")
    passes_equal = ia0 is not None and ib0 is not None and ia0 == ib0
    covered: list = []
    if ra and rb and not from_cell:
        mapped = cfg_mapped(a, b, ra, rb, card)
        diff, took = [], []
        for k in SETTING_KEYS:
            if (ra.get(k) is None and rb.get(k) is None) or same_value(ra.get(k), rb.get(k)) or \
                    (k == "cfg" and mapped):
                continue
            one_unstated = (ra.get(k) is None) != (rb.get(k) is None)
            if k in ("sampler", "scheduler") and one_unstated and sigmas_equal and passes_equal:
                silent = la if ra.get(k) is None else lb
                rec_silent = a if ra.get(k) is None else b
                if (rec_silent.get("arm") or {}).get("stack", silent) == "studio":
                    other = rb.get(k) if ra.get(k) is None else ra.get(k)
                    covered.append(f"{k} unstated by studio (other {other!r}): covered by equal sigmas and "
                                   f"{ia0} evaluation rows on both")
                    continue
            if k in ("shift", "scheduler") and one_unstated and sigmas_equal:
                continue
            line = f"{k}: {la}={ra.get(k)!r} {lb}={rb.get(k)!r}"
            if k in acc_settings and ra.get(k) is not None and rb.get(k) is not None:
                took.append(k)
                accepted += [{"id": t.get("id"), "component": k, "detail": line, "note": t.get("note")}
                             for t in acc_settings[k]]
            else:
                diff.append(line)
        checks["settings"] = _row(not diff, "; ".join(diff) or ("equal" + (f" ({mapped})" if mapped else "")
                                                                + (f" (accepted for timing: {took})" if took else "")
                                                                + (f" ({'; '.join(covered)})" if covered else "")),
                                  "settings" in waive)
    else:
        took = []
    # sigmas: every stage (two-stage graphs dump one schedule per stage), each compared; a non-number is missing
    sa, sb = sigma_stages(pa), sigma_stages(pb)
    if not sa or not sb:
        checks["sigmas"] = _row(None, f"sigma dump missing or non-numeric ({la}: {bool(sa)}, {lb}: {bool(sb)})",
                                "sigmas" in waive)
    elif len(sa) != len(sb) or any(len(x) != len(y) for x, y in zip(sa, sb)):
        checks["sigmas"] = _row(False, f"schedule stages / lengths {la}={[len(x) for x in sa]} "
                                       f"{lb}={[len(x) for x in sb]}", "sigmas" in waive)
    else:
        worst = max(sigma_delta(x, y) for st_a, st_b in zip(sa, sb) for x, y in zip(st_a, st_b))
        detail = (f"max |delta| (relative above 1) {worst:.2e} over {sum(len(x) for x in sa)}"
                  + (f" in {len(sa)} stages" if len(sa) > 1 else "") + f" (tol {SIGMA_TOL})")
        if worst > SIGMA_TOL and "scheduler" in took:
            # an accepted schedule difference changes the grid, never its length (the work): timing pairs only
            accepted += [{"id": t.get("id"), "component": "sigmas", "detail": detail, "note": t.get("note")}
                         for t in acc_settings["scheduler"]]
            checks["sigmas"] = _row(True, detail + " (accepted: the card's scheduler difference, same length)")
        elif SIGMA_TOL < worst <= tol:
            # the card's measured rounding of a published sigma list (LTX-2.5 template stage 2: 0.4219 for
            # 0.421875): same grid length and work, reported, never silently passed
            accepted.append({"id": tol_trap.get("id"), "component": "sigmas", "detail": detail,
                             "note": tol_trap.get("note")})
            checks["sigmas"] = _row(True, detail + f" (accepted: card sigma_tol {tol:g})")
        else:
            checks["sigmas"] = _row(worst <= SIGMA_TOL, detail, "sigmas" in waive)
    # passes
    ia, ib = pa.get("denoiser_items"), pb.get("denoiser_items")
    detail = []
    ok: Optional[bool] = None
    if ia is None or ib is None:
        detail.append(f"denoiser evaluation count missing ({la}: {ia}, {lb}: {ib})")
    else:
        ok = ia == ib
        detail.append(f"denoiser evaluations {la}={ia} {lb}={ib}")
    expr = (card or {}).get("expected_passes")
    if expr is not None:
        # the card counts evaluations per sample ROW (a CFG pair is 2 even batched as one call): compare it with
        # denoiser_items; the env is the card's own defaults + pass_vars (stage2_sigmas ...) under the resolved values
        base = {k: v for k, v in ((card or {}).get("defaults") or {}).items()
                if isinstance(v, (int, float, bool)) and not isinstance(v, str)}
        base.update({k: v for k, v in ((card or {}).get("pass_vars") or {}).items() if isinstance(v, (int, float))})
        for p, r, lab in ((pa, ra, la), (pb, rb, lb)):
            env = {**base, **{k: r[k] for k in ("steps", "width", "height") if r.get(k) is not None},
                   "cfg": float(r.get("cfg") if r.get("cfg") is not None else base.get("cfg", 1.0)),
                   "frames": r.get("frames") or base.get("frames") or 1,
                   "cfg_batched": bool(p.get("cfg_batched", True))}
            want = expected_passes(expr, env)
            got = p.get("denoiser_items")
            if want is None:
                detail.append(f"card expected_passes {expr!r} not evaluable for {lab}")
            elif got is None:
                if ok is not False:
                    ok = None  # unknown rows: missing evidence, not a pass
                detail.append(f"{lab} evaluation rows unknown, card expects {want}")
            elif got != want:
                ok = False
                detail.append(f"{lab} evaluated {got} rows ({p.get('denoiser_calls')} calls), card expects {want} "
                              f"({expr})")
            else:
                detail.append(f"{lab} rows {got} == card")
    checks["passes"] = _row(ok, "; ".join(detail), "passes" in waive)
    # size
    oa, ob = pa.get("output") or {}, pb.get("output") or {}
    if not oa or not ob:
        checks["size"] = _row(None, "rendered output size missing", "size" in waive)
    else:
        diff = [f"{k}: {la}={oa.get(k)} {lb}={ob.get(k)}" for k in ("width", "height", "frames") if oa.get(k) != ob.get(k)]
        for o, r, lab in ((oa, ra, la), (ob, rb, lb)):
            for k in ("width", "height", "frames"):
                if r.get(k) is not None and o.get(k) is not None and int(o[k]) != int(r[k]):
                    diff.append(f"{lab} rendered {k}={o[k]} but resolved {r[k]}")
        checks["size"] = _row(not diff, "; ".join(diff) or f"{oa.get('width')}x{oa.get('height')}"
                              + (f"x{oa.get('frames')}" if oa.get("frames") else ""), "size" in waive)
    # precision
    checks["precision"], acc_prec, observed = check_precision(a, b, card, waive)
    accepted += acc_prec
    if observed:
        checks["precision_observed"] = observed
    dense = dense_compute_row(a, b, la, lb)
    if dense is not None:
        checks["dense_compute"] = _row(dense[0], dense[1], "dense_compute" in waive)
    # noise
    if quality:
        na, nb = pa.get("noise") or {}, pb.get("noise") or {}
        if not na or not nb:
            checks["noise"] = _row(None, "initial-noise evidence missing", "noise" in waive)
        else:
            bad = []
            for n, lab in ((na, la), (nb, lb)):
                dev = str(n.get("generator_device") or n.get("device") or "")
                if not dev.startswith("cpu"):
                    bad.append(f"{lab} noise drawn on {dev or '?'}")
                if str(n.get("dtype")).replace("torch.", "") not in ("float32", "fp32"):
                    bad.append(f"{lab} noise dtype {n.get('dtype')}")
            if na.get("sha") and nb.get("sha") and na.get("shape") == nb.get("shape") and na["sha"] != nb["sha"]:
                bad.append(f"noise hashes differ ({na['sha']} vs {nb['sha']})")
            checks["noise"] = _row(not bad, "; ".join(bad) or "CPU fp32 on both", "noise" in waive)
    # cache defeat (BlineUI)
    for rec, p, lab in ((a, pa, la), (b, pb, lb)):
        if (rec.get("arm") or {}).get("stack") != "blineui" and not p.get("bline"):
            continue
        bad = cache_problems(rec)
        c = p.get("bline") or {}
        if not c:
            bad.append("no execution facts for the parity request")
        elif not c.get("executed_nodes"):
            bad.append("parity request executed 0 nodes")
        elif c.get("sampler_cached"):
            bad.append("parity request sampler served from cache")
        checks[f"cache_{lab}"] = _row(not bad, "; ".join(bad) or f"{c.get('executed_nodes')} nodes executed",
                                      "cache" in waive)
    # traps: each card trap's check name -> the gate check that enforces it (models.GATE_CHECK), a check that
    # lives outside the gate (ENFORCED_ELSEWHERE), or unchecked (listed, never passed)
    mapped, elsewhere, unchecked = trap_coverage(card)
    if mapped:
        checks["traps_gated"] = {"status": "info", "detail": "; ".join(f"{t} -> {g}" for t, g in mapped)}
    if elsewhere:
        checks["traps_elsewhere"] = {"status": "info", "detail": "; ".join(f"{t} -> {w}" for t, w in elsewhere)}
    if unchecked:
        checks["traps_unchecked"] = {"status": "info", "detail": f"card traps without a gate check: {unchecked}"}
    # a baseline file the card has no official source for must carry its mirror verification (mirror.py)
    import mirror  # noqa: PLC0415

    for rec, lab in ((a, la), (b, lb)):
        mp = mirror.record_problems(rec, card, group_of(rec)["variant"])
        if mp:
            checks[f"mirror_{lab}"] = _row(False, "; ".join(mp))
        elif mirror.staged_of(rec):
            checks[f"mirror_{lab}"] = {"status": "ok", "detail": "; ".join(mirror.staged_lines(rec))}
    ga, gb = group_of(a), group_of(b)
    if (ga["family"], ga["variant"]) != (gb["family"], gb["variant"]):
        checks["group"] = _row(False, f"different model / variant: {ga} vs {gb}")
    reasons = [f"{k}: {v['detail']}" for k, v in checks.items() if v["status"] in ("mismatch", "missing")]
    return {"verdict": "VOID" if reasons else "ok", "reasons": reasons, "checks": checks,
            "waived": [k for k, v in checks.items() if v["status"] == "waived"], "pair": [a.get("tag"), b.get("tag")],
            "group": group_of(a), "comparison": (a.get("arm") or {}).get("comparison"), "quality": quality,
            "accepted_differences": accepted, "created": time.time(),
            # the tier the Studio probe RESOLVED: the report holds every timed Studio run of the group to it
            "studio_tier": next((C.resolved_tier(r.get("status_after") or r.get("status")) for r, lab in
                                 ((a, la), (b, lb)) if (r.get("arm") or {}).get("stack") == "studio"), None)}


def sigma_tol_trap(card: Optional[dict]) -> Optional[dict]:
    """The card trap that accepts a sigma difference up to ``sigma_tol`` (check "sigma_rounding"), else None. Capped
    at 1e-4: a tolerance is for a published list's rounding, never for a different schedule."""
    for t in (card or {}).get("traps") or []:
        if t.get("check") == "sigma_rounding" and t.get("accepted_difference") and \
                isinstance(t.get("sigma_tol"), (int, float)) and 0 < float(t["sigma_tol"]) <= 1e-4:
            return t
    return None


def accepted_settings(card: Optional[dict], quality: bool) -> dict:
    """setting -> [trap] of the card's accepted sampler / scheduler differences (models.accepted_settings): timing
    pairs only, so a quality pair never accepts one."""
    if not card or quality:
        return {}
    try:
        from models import accepted_settings as _acc  # type: ignore

        traps = _acc(card, "timing")
    except Exception:  # noqa: BLE001
        traps = [t for t in card.get("traps") or [] if t.get("accepted_difference")
                 and t.get("check") == "sampler_differs" and "timing" in (t.get("purposes") or [])]
    out: dict = {}
    for t in traps:
        for k in t.get("settings") or []:
            out.setdefault(k, []).append(t)
    return out


# ---------------------------------------------------------------------------------------------- precision
CARD_SIDE = {"studio": "studio", "blineui": "bline"}
# Trap checks a card names that are enforced outside the pair gate (where, for the report).
ENFORCED_ELSEWHERE = {
    "gguf_bline_refused": "plan.py: a baseline-UI GGUF arm of a refused family is NOT_RUN (bline_graphs.GGUF_REFUSED)",
    "h3_qkv_regroup": "builders: H3 DiT files are built through models.convert.h3_qkv_regroup",
    "speed_tier_stated": "record: arm.tier and options.speed_mode are stored per cell",
    "cache_defeated": "gate check cache (nonce + executed nodes > 0 on every timed request)",
}


def gate_check_name(check: Optional[str]) -> Optional[str]:
    """A card trap's check name -> this gate's check name (models.GATE_CHECK, then this module's own aliases)."""
    if not check:
        return None
    try:
        from models import GATE_CHECK  # type: ignore
    except Exception:  # noqa: BLE001
        GATE_CHECK = {}
    return GATE_CHECK.get(check) or TRAP_CHECKS.get(check)


def trap_coverage(card: Optional[dict]) -> tuple:
    """([(trap id, gate check)], [(trap id, where)], [unchecked trap ids]) for the card's machine-checkable traps."""
    mapped, elsewhere, unchecked = [], [], []
    for trap in (card or {}).get("traps") or []:
        name = trap.get("check")
        if not name:
            continue
        tid = trap.get("id") or name
        gate = gate_check_name(name)
        if gate:
            mapped.append((tid, gate))
        elif name in ENFORCED_ELSEWHERE:
            elsewhere.append((tid, ENFORCED_ELSEWHERE[name]))
        else:
            unchecked.append(tid)
    return mapped, elsewhere, unchecked


def norm_weights(v: Any) -> Optional[str]:
    """Storage precision for comparison: lower case, fp8_e4m3fn == fp8, int8 ConvRot == int8_convrot."""
    if v is None:
        return None
    k = str(v).strip().lower().replace("torch.", "")
    k = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32", "fp8_e4m3fn": "fp8", "float8_e4m3fn": "fp8",
         "convrot": "int8_convrot"}.get(k, k)
    return k or None


def _same_weights(x: str, y: str) -> bool:
    if x == y:
        return True
    # a bare "gguf" names the container only; it matches any gguf:<quant>
    return "gguf" in (x, y) and all(v.startswith("gguf") for v in (x, y))


def _observed(rec: dict) -> dict:
    """The record's own precision report as {component: {weights, compute, strong}}. A dict entry (the backend
    states weights / compute) is strong evidence; a bare string (a file-name token or the pipeline dtype) is weak:
    it is checked against the card but only informs."""
    out = {}
    for comp, v in ((rec.get("parity") or {}).get("precision") or {}).items():
        if isinstance(v, dict):
            out[comp] = {"weights": v.get("weights"), "compute": v.get("compute"), "strong": True}
        elif v is not None:
            out[comp] = {"weights": v, "compute": v, "strong": False}
    return out


def side_precision(rec: dict, card: Optional[dict]) -> tuple:
    """({component: {weights, compute}}, source) for one record: the card's declared precision for the record's
    stack and format when the card has it, else what the backend reported."""
    stack = (rec.get("arm") or {}).get("stack") or schema.stack_of(rec.get("backend"))
    fmt = group_of(rec)["variant"]
    side = CARD_SIDE.get(stack or "")
    fentry = ((card or {}).get("formats") or {}).get(fmt) or {}
    declared = ((fentry.get(side) or {}) if side else {}).get("precision")
    if isinstance(declared, dict) and declared:
        out = {c: {"weights": v.get("weights"), "compute": v.get("compute")} for c, v in declared.items()
                if isinstance(v, dict)}
        # a component the card leaves unknown (weights null: never confirmed) takes the backend's STRONG report
        # of what it loaded (Studio's H3 audio VAE: the module's own parameter dtype)
        obs = _observed(rec)
        filled = [c for c, v in out.items() if v["weights"] is None and (obs.get(c) or {}).get("strong")]
        for c in filled:
            out[c] = {"weights": obs[c]["weights"], "compute": obs[c]["compute"]}
        return out, f"card {fmt}.{side}" + (f" + backend report ({', '.join(filled)})" if filled else "")
    obs = _observed(rec)
    return {c: {"weights": v["weights"], "compute": v["compute"] if v["strong"] else None} for c, v in obs.items()}, \
        "backend report"


def accepted_for(card: Optional[dict], fmt: str) -> dict:
    """component -> [trap] of the card's accepted precision differences for this format (models.accepted_differences:
    traps with ``accepted_difference: true`` whose ``formats`` hold ``fmt``)."""
    out: dict = {}
    if not card:
        return out
    try:
        from models import accepted_differences  # type: ignore

        traps = accepted_differences(card, fmt)
    except Exception:  # noqa: BLE001
        traps = [t for t in card.get("traps") or [] if t.get("accepted_difference") and fmt in (t.get("formats") or [])]
    for t in traps:
        if gate_check_name(t.get("check")) not in (None, "precision"):
            continue
        for comp in t.get("components") or []:
            out.setdefault(comp, []).append(t)
    return out


def check_precision(a: dict, b: dict, card: Optional[dict], waive: tuple = ()) -> tuple:
    """(row, accepted[], observed_row | None). Per component: ``weights`` always compared, ``compute`` only when
    both sides state it; a difference the card lists as an accepted difference for this format is reported in
    ``accepted`` and does not void the pair. ``waive`` may hold "precision" (all) or "precision:<component>"."""
    la, lb = (a.get("arm") or {}).get("stack") or a.get("tag"), (b.get("arm") or {}).get("stack") or b.get("tag")
    qa, src_a = side_precision(a, card)
    qb, src_b = side_precision(b, card)
    fmt = group_of(a)["variant"]
    acc_map = accepted_for(card, fmt)
    comps = sorted(set(qa) | set(qb))
    if not comps:
        return _row(None, "component precision missing on both", "precision" in waive), [], None
    diff, miss, accepted, waived = [], [], [], []
    for c in comps:
        x, y = qa.get(c) or {}, qb.get(c) or {}
        wx, wy = norm_weights(x.get("weights")), norm_weights(y.get("weights"))
        cx, cy = norm_weights(x.get("compute")), norm_weights(y.get("compute"))
        problems = []
        if wx is None or wy is None:
            gap = f"{c} weights unknown ({la}={x.get('weights')!r}, {lb}={y.get('weights')!r})"
        else:
            gap = None
            if not _same_weights(wx, wy):
                problems.append(f"{c} weights {la}={x.get('weights')} {lb}={y.get('weights')}")
        if cx is not None and cy is not None and cx != cy:
            problems.append(f"{c} compute {la}={x.get('compute')} {lb}={y.get('compute')}")
        if f"precision:{c}" in waive and (problems or gap):
            waived.append(c)
            continue
        if problems and not gap and c in acc_map:
            accepted += [{"id": t.get("id"), "component": c, "detail": "; ".join(problems), "note": t.get("note")}
                         for t in acc_map[c]]
            continue
        diff += problems  # (missing evidence is never an accepted difference: only a stated, differing value is)
        if gap:
            miss.append(gap)
    status: Optional[bool] = False if diff else (None if miss else True)
    parts = diff + miss + ([f"accepted: {', '.join(sorted({x['id'] for x in accepted}))}"] if accepted else []) + \
        ([f"waived components: {waived}"] if waived else [])
    row = _row(status, "; ".join(parts) or "equal", "precision" in waive)
    row["sources"] = {la: src_a, lb: src_b}
    # the stack's own report (what ran) checked against the card side it was declared by: the observed WEIGHTS
    # against the card's weights, so a quantized arm that silently loaded dense weights is VOID
    observed = observed_problems(a, b, qa, qb, src_a, src_b, la, lb, fmt)
    if observed is not None and observed["status"] == "mismatch" and "precision" in waive:
        observed = _row(False, observed["detail"], True)
    return row, accepted, observed


DENSE = ("bf16", "fp16", "fp32")


def weight_family(v: Any) -> Optional[str]:
    """A weights token's storage family: bf16 / fp16 / fp32 / fp8 / int8 / nvfp4 / mxfp4 / nf4 / gguf / other."""
    k = norm_weights(v)
    if not k or k in ("default", "none", "?"):
        return None
    if k.startswith("gguf") or re.match(r"^(q\d|iq\d)", k):
        return "gguf"
    for fam in ("fp8", "int8", "nvfp4", "mxfp4", "nf4", "int4", "w6a8", "w4a8"):
        if k.startswith(fam):
            return fam
    if k.startswith(("float8", "fp4_mixed")):
        return "fp8" if k.startswith("float8") else k
    return k


def dense_override(rec: dict) -> Optional[str]:
    """The dense dtype a record ran a "bf16" card format in when the target has no bf16 (plan.py sets
    options.dense_dtype = "fp16" on sm75): both stacks run the bf16 files at fp16 there."""
    return ((rec.get("cell") or {}).get("options") or {}).get("dense_dtype")


DTYPE_TOKENS = {"float16": "fp16", "half": "fp16", "fp16": "fp16", "bfloat16": "bf16", "bf16": "bf16",
                "float32": "fp32", "float": "fp32", "fp32": "fp32"}


def dense_token(v: Any) -> Optional[str]:
    """fp16 / bf16 / fp32 for a dtype string ("torch.float16", "bf16"), else None."""
    if v is None:
        return None
    return DTYPE_TOKENS.get(str(v).strip().lower().replace("torch.", ""))


def observed_dense_compute(rec: dict) -> Optional[str]:
    """The transformer compute dtype the stack itself reported: the baseline UI's diffusion-model load line
    ("model weight dtype X, manual cast: Y": Y when set, else X; status_after / status facts), Studio's pipeline
    dtype (status.dtype), else a dense token in the backend's precision report. None when the record states none."""
    for st in (rec.get("status_after") or {}, rec.get("status") or {}):  # the baseline UI loads lazily: after first
        facts = st.get("facts") or {}
        cast = dense_token(facts.get("manual_cast"))
        if cast or facts.get("model_weight_dtype"):
            return cast or dense_token(facts.get("model_weight_dtype"))
    for st in (rec.get("status_after") or {}, rec.get("status") or {}):
        if dense_token(st.get("dtype")):
            return dense_token(st.get("dtype"))
    obs = _observed(rec).get("transformer") or {}  # the backend's own precision report (dense token)
    return dense_token(obs.get("compute")) or dense_token(obs.get("weights"))


def dense_compute_row(a: dict, b: dict, la: str, lb: str) -> Optional[tuple]:
    """(ok, detail) when a dense format runs on an fp16-only card (options.dense_dtype set by plan.py): the dtype is
    planned, not applied by the backends, so each stack's own report must show the same transformer compute dtype
    (a stack that silently ran fp32 on a T4 is not comparable with one that ran fp16). None when not applicable."""
    planned = dense_override(a) or dense_override(b)
    if not planned and not (str(group_of(a).get("variant")) in ("bf16", "fp16") and fp16_only(a, b)):
        return None
    planned = planned or "unplanned: a device without bf16 (tester / unknown target)"
    xa, xb = observed_dense_compute(a), observed_dense_compute(b)
    if xa is None or xb is None:
        return None, f"transformer compute dtype not reported ({la}={xa}, {lb}={xb}; planned {planned})"
    return xa == xb, f"transformer compute {la}={xa} {lb}={xb} (planned {planned})"


def fp16_only(*recs) -> bool:
    """A record ran on an NVIDIA device below compute capability 8.0 (no bf16: T4, V100, a tester's 20xx), from its
    own gpu block: the plan may not have known the card (tester:<label>), so the dense check applies anyway."""
    for rec in recs:
        cc = (rec.get("gpu") or {}).get("cc")
        try:
            if cc and float(str(cc)) < 8.0:
                return True
        except ValueError:
            continue
    return False


def observed_problems(a: dict, b: dict, qa: dict, qb: dict, src_a: str, src_b: str, la: str, lb: str,
                      fmt: str) -> Optional[dict]:
    """The observed-vs-card row, or None. Per record whose precision came from its card side: every component the
    backend reports is compared by weights family against the card's weights. A transformer mismatch is VOID even
    on a bare token (Studio's transformer_quant / dtype, the baseline UI's weight_dtype / file name); a quantized
    transformer with no report at all is VOID as missing; other components' bare tokens only inform."""
    void, miss, notes = [], [], []
    for rec, q, src, lab in ((a, qa, src_a, la), (b, qb, src_b, lb)):
        if not src.startswith("card"):
            continue
        obs = _observed(rec)
        dense = dense_override(rec)
        for c, decl in q.items():
            want = weight_family((decl or {}).get("weights"))
            if want is None:
                continue
            allowed = {want} | ({"bf16", "fp16"} if want == "bf16" and dense == "fp16" else set())
            o = obs.get(c)
            parts = [] if o is None else [p for p in str(o["weights"]).split("+") if p not in ("?", "None", "")]
            got = [g for g in (weight_family(p) for p in parts) if g is not None]
            if not got:
                if c == "transformer" and want not in DENSE:
                    miss.append(f"{lab} transformer: no backend report of the weights it ran (card {fmt} says "
                                f"{decl.get('weights')}): a silent fallback cannot be ruled out")
                continue
            if all(g in allowed for g in got):
                continue
            msg = f"{lab} {c}: backend reported {o['weights']!r}, card {fmt} says {decl.get('weights')}"
            (void if (c == "transformer" or o["strong"]) else notes).append(msg)
    if void or miss:
        return _row(False if void else None, "; ".join(void + miss))
    if notes:
        return {"status": "info", "detail": "weak evidence (file-name token / pipeline dtype) differs from the "
                                            "card: " + "; ".join(notes)}
    return None


def cfg_mapped(a: dict, b: dict, ra: dict, rb: dict, card: Optional[dict]) -> Optional[str]:
    """When the card has a guidance_semantics trap, Studio's guidance and the baseline UI's cfg are different scales
    (FLUX.1 embedded guidance, Z-Image g = cfg - 1, true-CFG off = 1.0): the like-for-like pair is the card's own
    ``studio.gen.guidance`` against ``defaults.cfg``. Returns a note when the two resolved values are exactly that
    pair, else None (then cfg is compared as numbers)."""
    if not card or not any(gate_check_name(t.get("check")) == "settings" and t.get("check") == "guidance_semantics"
                           for t in card.get("traps") or []):
        return None
    want = {"studio": ((card.get("studio") or {}).get("gen") or {}).get("guidance"),
            "blineui": (card.get("defaults") or {}).get("cfg")}
    if None in want.values():
        return None
    got = {}
    for rec, r in ((a, ra), (b, rb)):
        st = (rec.get("arm") or {}).get("stack")
        if st in want:
            got[st] = r.get("cfg")
    if set(got) == set(want) and all(got[k] is not None and same_value(float(got[k]), float(want[k])) for k in want):
        return f"cfg mapped by the card's guidance_semantics: studio guidance {want['studio']} = baseline cfg {want['blineui']}"
    return None


def cache_problems(rec: dict) -> list:
    """Timed BlineUI requests that were (partly) served from cache, or whose execution count is unknown (no
    execution facts attached at all: progress feed down, request id mismatch, a node path that reports none)."""
    out = []
    bline = ((rec.get("arm") or {}).get("stack") or schema.stack_of(rec.get("backend"))) == "blineui" or \
        ((rec.get("cell") or {}).get("options") or {}).get("emulate") == "blineui"
    for i, r in enumerate(rec.get("requests") or []):
        if r.get("kind") not in ("new", "same"):
            continue
        c = r.get("bline")
        if c is None:
            if bline:
                out.append(f"request {i} ({r.get('kind')}): execution facts missing (cache defeat unproven)")
            continue
        if c.get("executed_nodes") is None:
            out.append(f"request {i} ({r.get('kind')}): execution count unknown (no progress feed)")
        elif c.get("executed_nodes") == 0:
            out.append(f"request {i} ({r.get('kind')}): 0 nodes executed (cache hit)")
        elif c.get("sampler_cached"):
            out.append(f"request {i} ({r.get('kind')}): sampler cached")
    return out


# ---------------------------------------------------------------------------------------------- plan gate
def plan_pairs(plan: dict) -> list:
    """Every parity pair plan.py scheduled: [{comparison, family, format, a, b, quality, waive}]."""
    out = []
    for cmp in plan.get("comparisons") or []:
        for pr in cmp.get("parity_pairs") or []:
            out.append({"comparison": cmp["id"], "family": cmp["family"], **pr})
    return out


def gate_plan(plan: dict, results: Path, write: bool = True) -> dict:
    """Run the gate on every planned pair whose two parity records exist under ``results`` (<tag>/record.json).
    {"pairs": [{..., "verdict", "reasons", "path"}], "void": [(comparison, format)], "pending": [...]}; each
    result is written to results/<a tag>/parity.json (report.py finds it there)."""
    results = Path(results)
    done, void, pending = [], [], []
    pairs = plan_pairs(plan)
    # a Studio probe paired with several baseline arms gets one result per pair (results/<a>/vs_<b>/parity.json):
    # one parity.json per Studio probe would keep only the last pair's verdict
    fan_out = {a for a in {pr["a"] for pr in pairs} if sum(1 for x in pairs if x["a"] == a
                                                              and x.get("expect") != "VOID") > 1}
    for pr in pairs:
        paths = [results / pr["a"] / "record.json", results / pr["b"] / "record.json"]
        neg = pr.get("expect") == "VOID"  # a negative control: a deliberately mismatched pair the gate must reject
        if not all(p.exists() for p in paths):
            if not neg:  # a negative control that never ran proves nothing, but blocks nothing either
                pending.append(pr)
            continue
        try:
            ra, rb = schema.load(paths[0]), schema.load(paths[1])
        except Exception as exc:  # noqa: BLE001 - an unreadable parity record voids the group
            res = {"verdict": "VOID", "reasons": [f"unreadable parity record: {exc}"], "checks": {}, "waived": [],
                   "pair": [pr["a"], pr["b"]], "group": {"family": pr["family"], "variant": pr["format"]}}
        else:
            try:
                res = check_pair(ra, rb, load_card(pr["family"]), bool(pr.get("quality")),
                                 tuple(pr.get("waive") or ()))
            except Exception as exc:  # noqa: BLE001 - a gate that cannot decide voids the group, never passes it
                res = {"verdict": "VOID", "reasons": [f"parity gate error: {type(exc).__name__}: {exc}"],
                       "checks": {}, "waived": [], "pair": [pr["a"], pr["b"]],
                       "group": {"family": pr["family"], "variant": pr["format"]}}
        res["comparison"] = pr["comparison"]
        name = "parity.json"
        if neg:
            # kept out of parity.json (report.parity_for takes the newest result of a group); a control the gate
            # passes means the gate cannot tell this pair apart: the group is VOID, fail closed
            res["negative_control"] = pr.get("negative_control") or True
            res["control_verdict"] = "rejected" if res["verdict"] != "ok" else "NOT_REJECTED"
            name = "parity_negctl.json"
        where = paths[0].parent / f"vs_{pr['b']}" if pr["a"] in fan_out and not neg else paths[0].parent
        if write:
            where.mkdir(parents = True, exist_ok = True)
            (where / name).write_text(schema.dumps(res))
        done.append({**pr, "verdict": res["verdict"], "reasons": res["reasons"],
                     "accepted": [x["id"] for x in res.get("accepted_differences") or []],
                     "path": str(where / name),
                     **({"control_verdict": res["control_verdict"]} if neg else {})})
        if (res["verdict"] == "ok") if neg else (res["verdict"] != "ok"):
            void.append((pr["comparison"], pr["format"]))
    return {"pairs": done, "void": void, "pending": pending}


def blocked_groups(gate_result: dict) -> list:
    """(comparison, format) groups whose timed cells must not run: VOID, and PENDING (a probe that never wrote its
    record: its download or process failed) -- no parity result is never a pass."""
    return list(gate_result["void"]) + [(p["comparison"], p["format"]) for p in gate_result["pending"]]


def gated_out(plan: dict, cell: dict, void: list) -> Optional[str]:
    """Why a timed / quality cell must not run: its (comparison, format) group failed the parity gate."""
    arm = (cell.get("options") or {}).get("arm") or {}
    key = (arm.get("comparison"), arm.get("format"))
    if arm.get("purpose") in ("trial", "parity") or key not in {tuple(v) for v in void}:
        return None
    return (f"parity gate VOID or pending (no ok result) for {key[0]} / {key[1]}: no timing is spent on a pair "
            "that is not proven to compute the same thing")


def main() -> int:
    if "--plan" in sys.argv[1:]:
        return main_plan()
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--quality", action = "store_true", help = "a quality pair: require shared CPU fp32 noise")
    ap.add_argument("--family", default = None, help = "model card for expected_passes / traps (default: the cell's)")
    ap.add_argument("--waive", default = "", help = "comma list of checks to waive (printed in every report)")
    args = ap.parse_args()
    recs = []
    for d in (args.a, args.b):
        path = Path(d) / "record.json" if Path(d).is_dir() else Path(d)
        try:
            recs.append(schema.load(path))
        except Exception as exc:  # noqa: BLE001
            print(f"cannot read {path}: {exc}", file = sys.stderr)
            return 2
    card = load_card(args.family or family_of(recs[0]))
    res = check_pair(recs[0], recs[1], card, args.quality, tuple(w for w in args.waive.split(",") if w))
    out = (Path(args.a) if Path(args.a).is_dir() else Path(args.a).parent) / "parity.json"
    out.write_text(schema.dumps(res))
    for k, v in res["checks"].items():
        print(f"{v['status']:9s} {k:14s} {v['detail']}")
    print(f"parity {res['verdict']} -> {out}")
    return 0 if res["verdict"] == "ok" else 1


def main_plan() -> int:
    ap = argparse.ArgumentParser(description = "parity.py --plan PLAN.json --results DIR: gate every planned pair")
    ap.add_argument("--plan", required = True)
    ap.add_argument("--results", required = True)
    a = ap.parse_args()
    import json

    res = gate_plan(json.loads(Path(a.plan).read_text()), Path(a.results))
    for pr in res["pairs"]:
        print(f"{pr['verdict']:5s} {pr['comparison']} / {pr['format']}: {pr['a']} x {pr['b']}"
              + (f" (accepted: {pr['accepted']})" if pr["accepted"] else "")
              + (f" -- {'; '.join(pr['reasons'])[:300]}" if pr["reasons"] else ""))
    for pr in res["pending"]:
        print(f"PENDING {pr['comparison']} / {pr['format']}: records missing for {pr['a']} / {pr['b']}")
    return 1 if res["void"] else (3 if res["pending"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
