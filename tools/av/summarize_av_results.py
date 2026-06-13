#!/usr/bin/env python3
"""Merge AV results (CI keyless engines, static + dynamic + a _defender_control
EICAR proof) and local VirusTotal into a variant x engine matrix, and gate.
Defender static is reported as a detection RATE (cloud verdicts are
non-deterministic) and only real threats count, never RPC/scan errors.

  --results DIR  tree of <os>/{<variant>.json,_defender_control.json,dynamic/*}
  --vt FILE      local VirusTotal results (vt_scan.py)
  --out-md / --out-json  outputs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def load_results(results_dir: str):
    static: dict = {}        # variant -> os -> {engine -> rec}
    dynamic: dict = {}       # variant -> os -> rec
    controls: dict = {}      # os -> control rec
    for path in glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True):
        base = os.path.basename(path)
        try:
            rec = json.load(open(path))
        except Exception as e:
            print(f"[warn] skip {path}: {e}", file=sys.stderr)
            continue
        norm = path.replace("\\", "/")
        if base == "_defender_control.json":
            osl = _os_from_path(norm)
            controls[osl] = rec
            continue
        if "/dynamic/" in norm:
            variant = rec.get("variant") or base[:-5]
            dynamic.setdefault(variant, {})[_os_from_path(norm)] = rec
            continue
        variant = rec.get("variant")
        if not variant:
            continue
        osl = rec.get("os", _os_from_path(norm))
        static.setdefault(variant, {}).setdefault(osl, {})
        static[variant][osl] = rec
    return static, dynamic, controls


def _os_from_path(p: str) -> str:
    for tok in p.split("/"):
        if tok.startswith("av-results-"):
            return tok[len("av-results-"):]
    return "unknown"


def defender_rate(rec: dict):
    """Return (detected_passes, total_passes, verdict, threats)."""
    eng = (rec.get("engines") or {}).get("defender", {})
    perfile = eng.get("perfile") or {}
    det = tot = 0
    for f, c in perfile.items():
        det += int(c.get("detected", 0))
        tot += len(c.get("passes", []) or [])
    return det, tot, eng.get("verdict", "?"), eng.get("threats", [])


def clamav_verdict(rec: dict) -> str:
    return ((rec.get("engines") or {}).get("clamav") or {}).get("verdict", "-")


def kvrt_verdict(rec: dict) -> str:
    return ((rec.get("engines") or {}).get("kvrt") or {}).get("verdict", "-")


def vt_by_variant(vt_file):
    out: dict = {}
    if not vt_file or not os.path.exists(vt_file):
        return out
    for rec in json.load(open(vt_file)):
        parts = rec.get("path", "").replace("\\", "/").split("/")
        variant = parts[-2] if len(parts) >= 2 else "?"
        out.setdefault(variant, []).append(rec)
    return out


def vt_kaspersky(records):
    verdict = "n/a"
    for rec in records or []:
        kav = (rec.get("engines_of_interest") or {}).get("Kaspersky")
        if kav and kav.lower() not in ("undetected", "clean", "none", "type-unsupported"):
            return "detected"
        if verdict == "n/a":
            verdict = "clean"
    return verdict


def dynamic_rate(variant: str, dynamic: dict):
    """(detected_iters, total_iters) aggregated across OSes for a variant."""
    det = tot = 0
    for osl, rec in dynamic.get(variant, {}).items():
        det += int(rec.get("detected_count", 0))
        tot += int(rec.get("iterations", 0))
    return det, tot


def is_clean_everywhere(variant, static, dynamic, vt) -> bool:
    for osl, rec in static.get(variant, {}).items():
        d_det, d_tot, d_verdict, _ = defender_rate(rec)
        if d_det > 0:
            return False
        cv = clamav_verdict(rec)
        if cv == "detected":
            return False
        kv = kvrt_verdict(rec)
        if kv == "detected":
            return False
    if vt_kaspersky(vt.get(variant, [])) == "detected":
        return False
    if dynamic_rate(variant, dynamic)[0] > 0:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--vt", default=None)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    static, dynamic, controls = load_results(args.results)
    vt = vt_by_variant(args.vt)
    variants = sorted(set(static) | set(vt) | set(dynamic))
    os_set = sorted({o for v in static.values() for o in v})

    lines = ["# Launcher AV scan matrix", ""]
    # Defender functional controls
    lines.append("## Defender engine controls (per runner)")
    if controls:
        lines.append("| Runner | Functional | EICAR (positive) | benign (negative) | RealTime | Behavior |")
        lines.append("|---|---|---|---|---|---|")
        for osl, c in sorted(controls.items()):
            st = c.get("status") or {}
            lines.append(f"| {osl} | {c.get('functional')} | {c.get('eicar',{}).get('verdict')} | "
                         f"{c.get('benign',{}).get('verdict')} | {st.get('RealTimeProtectionEnabled')} | "
                         f"{st.get('BehaviorMonitorEnabled')} |")
    else:
        lines.append("_no Defender control data_")
    lines.append("")

    clam_cols = "".join(f" ClamAV/{o.split('-')[0]} |" for o in os_set)
    lines.append("## Variant x engine")
    lines.append(f"| Variant | Defender static (det/passes) | Defender dynamic (det/iters) |"
                 f"{clam_cols} KVRT | VT-Kaspersky | Clean? |")
    lines.append("|" + "---|" * (6 + len(os_set)))

    merged = {}
    for var in variants:
        d_det = d_tot = 0
        threats = []
        clam = {}
        kvrt = "-"
        for osl in os_set:
            rec = static.get(var, {}).get(osl)
            if not rec:
                continue
            if osl == "windows-latest":
                a, b, _, th = defender_rate(rec)
                d_det += a; d_tot += b; threats += th
                kvrt = kvrt_verdict(rec)
            clam[osl] = clamav_verdict(rec)
        dyn_det, dyn_tot = dynamic_rate(var, dynamic)
        kav = vt_kaspersky(vt.get(var, []))
        ok = is_clean_everywhere(var, static, dynamic, vt)
        clam_cells = "".join(f" {clam.get(o,'-')} |" for o in os_set)
        lines.append(f"| {var} | {d_det}/{d_tot} | {dyn_det}/{dyn_tot} |{clam_cells} "
                     f"{kvrt} | {kav} | {'YES' if ok else 'no'} |")
        merged[var] = {"defender_static_det": d_det, "defender_static_passes": d_tot,
                       "defender_threats": sorted(set(threats)),
                       "defender_dynamic_det": dyn_det, "defender_dynamic_iters": dyn_tot,
                       "clamav": clam, "kvrt": kvrt, "vt_kaspersky": kav, "clean": ok}

    md = "\n".join(lines) + "\n"
    open(args.out_md, "w").write(md)
    json.dump({"controls": controls, "variants": merged}, open(args.out_json, "w"), indent=2)
    print(md)

    defender_live = any(c.get("functional") for c in controls.values()) if controls else False
    v0 = merged.get("V0-baseline", {})
    v0_flagged = (v0.get("defender_static_det", 0) > 0 or v0.get("defender_dynamic_det", 0) > 0
                  or v0.get("kvrt") == "detected" or v0.get("vt_kaspersky") == "detected")
    clean_hardened = [v for v in variants if v != "V0-baseline"
                      and is_clean_everywhere(v, static, dynamic, vt)]
    print(f"Defender engine live on >=1 runner (EICAR control): {defender_live}")
    print(f"V0 control flagged by any real engine: {v0_flagged}")
    print(f"Clean hardened variants: {clean_hardened or 'NONE'}")


if __name__ == "__main__":
    main()
