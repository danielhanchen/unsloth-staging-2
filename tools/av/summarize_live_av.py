#!/usr/bin/env python3
"""Merge live behavioral AV results into a Markdown matrix.

Reads per-iteration Track A JSONs (iteration_*.json, Defender) and/or Track B
pass JSONs (pass_*.json, ClamAV), and writes a compact pass/fail matrix. A
detection is only counted when it is correlated with Studio (the EICAR control
is expected and excluded).
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def load(paths: list[str]) -> list[dict]:
    out = []
    for p in sorted(paths):
        try:
            out.append((p, json.load(open(p))))
        except Exception as e:
            out.append((p, {"_error": str(e)}))
    return out


def defender_rows(results) -> list[str]:
    rows = []
    for p, d in results:
        rows.append("| {it} | {ctrl} | {sc} | {sd} | {ev} | {ex} | {verdict} |".format(
            it=d.get("iteration", os.path.basename(p)),
            ctrl="yes" if d.get("control_valid") else "NO",
            sc="yes" if d.get("vbs_chain_ok") else "no",
            sd=len(d.get("studio_detections", []) or []),
            ev=len(d.get("hard_defender_events", []) or []),
            ex=len(d.get("added_exclusions", []) or []),
            verdict="CLEAN" if d.get("clean") else "FAIL",
        ))
    return rows


def clamav_rows(results) -> list[str]:
    rows = []
    for p, d in results:
        rows.append("| {it} | {ctrl} | {inf} | {img} | {trn} | {det} | {verdict} |".format(
            it=d.get("pass", os.path.basename(p)),
            ctrl="yes" if d.get("control_valid") else "NO",
            inf=d.get("inference", "?"),
            img=d.get("image", "?"),
            trn=d.get("training", "?"),
            det=d.get("clamav_detections", 0),
            verdict="CLEAN" if d.get("clean") else "FAIL",
        ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--engine", choices=["defender", "clamav"], default="defender")
    args = ap.parse_args()

    if args.engine == "defender":
        res = load(glob.glob(os.path.join(args.results, "**", "iteration_*.json"),
                             recursive=True))
        hdr = ("## Track A - Microsoft Defender (live behavioral)\n\n"
               "| Iter | EICAR control | shortcut->VBS | Studio detections | hard events | added exclusions | verdict |\n"
               "|---|---|---|---|---|---|---|\n")
        rows = defender_rows(res)
        clean = all(d.get("clean") for _, d in res) and len(res) > 0
    else:
        res = load(glob.glob(os.path.join(args.results, "**", "pass_*.json"),
                             recursive=True))
        hdr = ("## Track B - ClamAV (live behavioral + output scan)\n\n"
               "| Pass | EICAR control | inference | image | training | ClamAV detections | verdict |\n"
               "|---|---|---|---|---|---|---|\n")
        rows = clamav_rows(res)
        clean = all(d.get("clean") for _, d in res) and len(res) > 0

    body = hdr + "\n".join(rows) + "\n\n"
    body += f"**Overall: {'ALL CLEAN' if clean else 'FAILURES PRESENT'}** ({len(res)} runs)\n"
    with open(args.out_md, "w") as f:
        f.write(body)
    print(body)


if __name__ == "__main__":
    main()
