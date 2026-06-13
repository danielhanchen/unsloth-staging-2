#!/usr/bin/env python3
"""Merge per-variant AV results (CI keyless engines + local VirusTotal) into a
variant x engine matrix, and apply the ship/no-ship gate.

Inputs:
  --results DIR   directory tree of <os>/<variant>.json from run-av-scan.*
  --vt FILE       optional local VirusTotal results json (from vt_scan.py)
  --out-md FILE   markdown matrix
  --out-json FILE merged json

Gate (exit nonzero if violated):
  * the V0 control must be flagged by at least one Kaspersky-family signal
    (KVRT detected, or VT Kaspersky malicious/suspicious) -- proves the repro;
  * at least one hardened variant (V1..V5) must be CLEAN on every engine
    (Defender, all ClamAV OSes, KVRT-or-inconclusive, VT-Kaspersky).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

CLEAN = "clean"


def load_ci(results_dir: str) -> dict:
    """variant -> {os -> {engine -> verdict-record}}"""
    data: dict = {}
    for path in glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True):
        try:
            rec = json.load(open(path))
        except Exception as e:
            print(f"[warn] skip {path}: {e}", file=sys.stderr)
            continue
        variant = rec.get("variant")
        osl = rec.get("os", "unknown")
        if not variant:
            continue
        data.setdefault(variant, {}).setdefault(osl, {})
        for eng, r in (rec.get("engines") or {}).items():
            data[variant][osl][eng] = r
    return data


def load_vt(vt_file: str | None) -> dict:
    """sha-keyed not needed; map by basename -> record. We attach VT to variants
    by matching the launch-studio.vbs/.ps1 path embedded in vt rec['path']."""
    out: dict = {}
    if not vt_file or not os.path.exists(vt_file):
        return out
    for rec in json.load(open(vt_file)):
        path = rec.get("path", "")
        # .../artifacts/<variant>/<file>
        parts = path.replace("\\", "/").split("/")
        variant = parts[-2] if len(parts) >= 2 else "?"
        out.setdefault(variant, []).append(rec)
    return out


def vt_kaspersky(records: list) -> tuple[str, dict]:
    """Return ('clean'|'detected'|'n/a', detail) aggregated over a variant's files."""
    verdict = "n/a"
    detail = {}
    for rec in records or []:
        eng = (rec.get("engines_of_interest") or {})
        kav = eng.get("Kaspersky")
        fname = os.path.basename(rec.get("path", ""))
        stats = rec.get("stats", {})
        detail[fname] = {
            "kaspersky": kav,
            "malicious": stats.get("malicious"),
            "suspicious": stats.get("suspicious"),
        }
        if kav and kav.lower() not in ("undetected", "clean", "none", "type-unsupported"):
            verdict = "detected"
        elif verdict == "n/a":
            verdict = "clean"
    return verdict, detail


def variant_clean_everywhere(variant: str, ci: dict, vt: dict) -> bool:
    oses = ci.get(variant, {})
    for osl, engines in oses.items():
        for eng, r in engines.items():
            v = (r or {}).get("verdict", "error")
            if eng == "kvrt" and v in ("inconclusive", "unavailable"):
                continue  # KVRT best-effort
            if eng == "clamav" and v == "unavailable":
                continue
            if eng == "defender" and v == "unavailable":
                continue
            if v != CLEAN:
                return False
    kv, _ = vt_kaspersky(vt.get(variant, []))
    if kv == "detected":
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--vt", default=None)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    ci = load_ci(args.results)
    vt = load_vt(args.vt)
    variants = sorted(set(ci) | set(vt))

    # Build matrix columns: defender, clamav-<os...>, kvrt, vt-kaspersky.
    os_set = sorted({o for v in ci.values() for o in v})
    rows = []
    merged = {}
    for var in variants:
        kv, kvdetail = vt_kaspersky(vt.get(var, []))
        cell = {"variant": var, "vt_kaspersky": kv, "vt_detail": kvdetail,
                "per_os": ci.get(var, {})}
        merged[var] = cell
        defender = clamavs = kvrt = "-"
        clam_by_os = {}
        for osl in os_set:
            eng = ci.get(var, {}).get(osl, {})
            if "defender" in eng:
                defender = eng["defender"].get("verdict", "-")
            if "clamav" in eng:
                clam_by_os[osl] = eng["clamav"].get("verdict", "-")
            if "kvrt" in eng:
                kvrt = eng["kvrt"].get("verdict", "-")
        rows.append((var, defender, clam_by_os, kvrt, kv,
                     variant_clean_everywhere(var, ci, vt)))

    # Markdown
    clam_cols = "".join(f" ClamAV/{o.split('-')[0]} |" for o in os_set)
    lines = [
        "# Launcher AV scan matrix",
        "",
        f"| Variant | Defender |{clam_cols} KVRT | VT-Kaspersky | Clean? |",
        "|" + "---|" * (5 + len(os_set)),
    ]
    for var, defv, clam, kvrt, kv, ok in rows:
        clam_cells = "".join(f" {clam.get(o,'-')} |" for o in os_set)
        lines.append(f"| {var} | {defv} |{clam_cells} {kvrt} | {kv} | "
                     f"{'YES' if ok else 'no'} |")
    md = "\n".join(lines) + "\n"
    open(args.out_md, "w").write(md)
    json.dump(merged, open(args.out_json, "w"), indent=2)
    print(md)

    # Gate
    v0_flagged = False
    v0 = ci.get("V0-baseline", {})
    for osl, eng in v0.items():
        if eng.get("kvrt", {}).get("verdict") == "detected":
            v0_flagged = True
    if vt_kaspersky(vt.get("V0-baseline", []))[0] == "detected":
        v0_flagged = True

    clean_hardened = [v for v in variants
                      if v != "V0-baseline" and variant_clean_everywhere(v, ci, vt)]

    print(f"\nV0 control flagged (Kaspersky family): {v0_flagged}")
    print(f"Clean hardened variants: {clean_hardened or 'NONE'}")
    if not v0_flagged:
        print("GATE WARN: V0 control not flagged -- engines may have missed the repro "
              "(could be signature drift). Review raw output.")
    if not clean_hardened:
        print("GATE FAIL: no hardened variant is clean everywhere.")
        sys.exit(1)
    print(f"GATE PASS: ship candidate(s): {clean_hardened}")


if __name__ == "__main__":
    main()
