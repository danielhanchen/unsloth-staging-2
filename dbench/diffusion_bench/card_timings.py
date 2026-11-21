#!/usr/bin/env python3
"""Measured per-run overhead per card: what plan.py's VM-minute / CU estimate charges each fresh-process run on top
of its steady renders.

  python diffusion_bench/card_timings.py RUN_DIR [RUN_DIR ...]            # print the table
  python diffusion_bench/card_timings.py RUN_DIR ... --write              # also write it into the cards

For every ok timed record (purpose timing, trial / parity probes excluded) the overhead is load_s plus the first
render's excess over a steady one (compile, autotune, warm-up): ``load_s + max(0, first_render_s - steady)``, steady
= steady_s_median, else new_s_median. Grouped by (card family, format, GPU key of the record's target, stack), the
median goes into ``formats.<fmt>.est_load_s.<GPU>.<stack>`` (seconds, one decimal) with ``est_load_basis`` naming the
runs it came from. A group with fewer than 2 records is printed, never written. Budget (simulated) runs are skipped:
their load streams through a smaller card than the target has.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import plan as PL  # noqa: E402


def overhead(rec: dict):
    t = rec.get("timing") or {}
    load = t.get("load_s", rec.get("load_s"))
    first = t.get("first_render_s")
    steady = t.get("steady_s_median") or t.get("new_s_median") or rec.get("steady_s_median")
    if not isinstance(load, (int, float)):
        return None
    extra = max(0.0, float(first) - float(steady)) if isinstance(first, (int, float)) and \
        isinstance(steady, (int, float)) else 0.0
    return float(load) + extra


def key_of(rec: dict):
    cell = rec.get("cell") or {}
    o = cell.get("options") or {}
    arm = rec.get("arm") or o.get("arm") or {}
    tgt = (o.get("target") or {})
    if arm.get("purpose") not in (None, "timing") or o.get("parity_probe") or str(rec.get("tag", "")).startswith("trial__"):
        return None
    if tgt.get("simulated_budget_gib") or cell.get("budget_gib"):
        return None
    fam = cell.get("card") or cell.get("family") or rec.get("family")
    fmt = arm.get("format") or cell.get("variant")
    gpu = PL.est_key(tgt.get("name") or "")
    stack = arm.get("stack")
    if not (fam and fmt and gpu and stack):
        return None
    return fam, fmt, gpu, "blineui" if stack in ("bline", "blineui") else stack


def gather(dirs: list) -> dict:
    groups: dict = {}
    for d in dirs:
        for p in Path(d).rglob("record.json"):
            try:
                rec = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if rec.get("verdict") != "ok":
                continue
            k, v = key_of(rec), overhead(rec)
            if k and v is not None:
                groups.setdefault(k, {}).setdefault(rec.get("run_id") or str(p), (v, str(d)))
    return groups


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs = "+")
    ap.add_argument("--write", action = "store_true", help = "write est_load_s into models/<family>.json")
    a = ap.parse_args(argv)
    groups = gather(a.dirs)
    wrote: dict = {}
    print("| family | format | GPU | stack | n | median overhead s | range |")
    print("|---|---|---|---|---:|---:|---|")
    for (fam, fmt, gpu, stack), runs in sorted(groups.items()):
        vals = sorted(v for v, _ in runs.values())
        med = round(statistics.median(vals), 1)
        print(f"| {fam} | {fmt} | {gpu} | {stack} | {len(vals)} | {med} | {vals[0]:.1f}-{vals[-1]:.1f} |")
        if len(vals) >= 2:
            wrote.setdefault(fam, []).append((fmt, gpu, stack, med, sorted({d for _, d in runs.values()})))
    if not a.write:
        return 0
    for fam, rows in wrote.items():
        path = HERE / "models" / f"{fam}.json"
        if not path.exists():
            print(f"skip {fam}: no card")
            continue
        card = json.loads(path.read_text())
        for fmt, gpu, stack, med, srcs in rows:
            f = (card.get("formats") or {}).get(fmt)
            if f is None:
                continue
            f.setdefault("est_load_s", {}).setdefault(gpu, {})[stack] = med
            basis = f.setdefault("est_load_basis", [])
            line = (f"{gpu} {stack}: median of load_s + first-render excess over fresh-process timed runs in "
                    + ", ".join(Path(s).as_posix().split("/outputs/")[-1] for s in srcs))
            basis[:] = [b for b in basis if not b.startswith(f"{gpu} {stack}:")] + [line]
        path.write_text(json.dumps(card, indent = 1, ensure_ascii = True) + "\n")
        print(f"wrote {path.name}: {[(r[0], r[1], r[2], r[3]) for r in rows]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
