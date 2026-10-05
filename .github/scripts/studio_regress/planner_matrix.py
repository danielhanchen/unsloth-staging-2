"""`planner_matrix`: what Studio's image, video and MiniMax-H3 planners decide across a VRAM grid, base vs head.

One side (the switchboard `cmd`, run with that side's Studio python and source tree, CPU only, no weights):

    python studio_regress/planner_matrix.py --src <tree> --out <dir>      -> <dir>/plan.json

runs the three probes in planner_probe/ with cwd <src>/studio/backend over
VRAM_GIB x MEMORY (discrete, unified) x PRECISIONS (auto, int8, fp8, bf16); each cell records the precision
the load would hold, its placement (resident, model offload, group offload with the DiT resident or streamed,
sequential; H3: resident / streamed) and whether it is admitted or refused (with the message). A probe that
cannot run leaves the side incomplete: exit 2 (VOID).

compare_sides(before_dir, after_dir) (external.py, `compare = "planner_matrix"`) diffs the two plan.json:
every moved cell goes in a table, flagged
  REFUSED     admitted on base, refused on head
  SLOWER      both admitted, head's placement is a strictly slower tier (resident < model / group with the
              DiT resident < DiT streamed < sequential)
  BF16        head dropped a quantised denoiser to bf16 at the same or a slower placement
  ERR         the probe raised on one side (a renamed planner function, a crash)
Verdict PLAN_DIFF when any cell moved (never a regression: a PR may move cells on purpose), else SAME.
The full table is <after_dir>/diff.md (+ diff.json); summary.md gets the flagged rows first.

Standalone, two trees:  python studio_regress/planner_matrix.py --compare BEFORE_DIR AFTER_DIR
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBES = HERE / "planner_probe"
VRAM_GIB = [8, 12, 16, 20, 24, 32, 40, 48, 80, 96]
MEMORY = ["discrete", "unified"]
PRECISIONS = ["auto", "int8", "fp8", "bf16"]
RANK = {
    "resident": 0,
    "model": 1,
    "group:dit-resident": 1,
    "group:dit-streamed": 2,
    "streaming": 2,
    "streamed": 2,
    "sequential": 3,
}
SUMMARY_ROWS = 40


def grid():
    return {"vram_gib": VRAM_GIB, "memory": MEMORY, "precisions": PRECISIONS}


def run_side(
    src,
    out,
    python = sys.executable,
    timeout = 1800,
    log = print,
):
    backend = Path(src) / "studio" / "backend"
    out = Path(out)
    out.mkdir(parents = True, exist_ok = True)
    env = {
        **os.environ,
        "PLANNER_GRID": json.dumps(grid()),
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    jobs = {
        "image": [python, str(PROBES / "probe_image.py"), str(out / "image.json")],
        "h3": [python, str(PROBES / "probe_h3.py"), str(out / "h3.json")],
        "video": [
            python,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(PROBES / "probe_video.py"),
        ],
    }
    procs, cells, probes = {}, [], {}
    for name in jobs:  # a probe that crashes this time must not hand back the last run's cells
        (out / f"{name}.json").unlink(missing_ok = True)
    for name, cmd in jobs.items():  # the three probes are independent: run them side by side
        penv = {**env, "PLANNER_OUT": str(out / "video.json")} if name == "video" else env
        fh = open(out / f"{name}.log", "w")
        procs[name] = (
            subprocess.Popen(cmd, cwd = str(backend), env = penv, stdout = fh, stderr = subprocess.STDOUT),
            fh,
            time.time(),
        )
    for name, (p, fh, t0) in procs.items():
        try:
            rc = p.wait(timeout = timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = -9
        fh.close()
        path = out / f"{name}.json"
        rows = json.loads(path.read_text()) if path.exists() else None
        probes[name] = {
            "rc": rc,
            "cells": len(rows) if rows is not None else None,
            "s": round(time.time() - t0, 1),
        }
        log(
            f"{name}: exit {rc}, {probes[name]['cells']} cells, {probes[name]['s']}s ({out / f'{name}.log'})"
        )
        cells += rows or []
    (out / "plan.json").write_text(
        json.dumps({"grid": grid(), "probes": probes, "cells": cells}, indent = 1)
    )
    complete = all(v["cells"] for v in probes.values())
    return 0 if complete else 2


def key(c):
    return (c["engine"], c["family"], c["vram_gib"], c["memory"], c["request"])


def _quant(p):
    return bool(p) and p != "bf16"


def flags(b, a):
    out = []
    if "ERR" in (b.get("placement"), a.get("placement")):
        out.append("ERR")
    if b.get("admitted") and not a.get("admitted") and a.get("placement") != "ERR":
        out.append("REFUSED")
    rb, ra = RANK.get(b.get("placement")), RANK.get(a.get("placement"))
    if b.get("admitted") and a.get("admitted") and rb is not None and ra is not None and ra > rb:
        out.append("SLOWER")
    if (
        b.get("admitted")
        and a.get("admitted")
        and _quant(b.get("precision"))
        and a.get("precision") == "bf16"
        and (ra or 0) >= (rb or 0)
    ):
        out.append("BF16")
    return out


def _state(c):
    if c is None:
        return "-"
    if not c.get("admitted"):
        return f"{c.get('placement')}: {(c.get('reason') or '')[:70]}"
    return f"{c.get('precision')} / {c.get('placement')}"


def diff_cells(before, after):
    bmap, amap = {key(c): c for c in before}, {key(c): c for c in after}
    rows = []
    for k in sorted(
        set(bmap) | set(amap),
        key = lambda k: (k[0], k[1], k[3], k[2], PRECISIONS.index(k[4]) if k[4] in PRECISIONS else 9),
    ):
        b, a = bmap.get(k), amap.get(k)
        sig = (
            lambda c: None
            if c is None
            else (c.get("precision"), c.get("placement"), c.get("admitted"))
        )  # noqa: E731
        if sig(b) == sig(a):
            continue
        rows.append(
            {
                "engine": k[0],
                "family": k[1],
                "vram_gib": k[2],
                "memory": k[3],
                "request": k[4],
                "before": _state(b),
                "after": _state(a),
                "flags": flags(b or {}, a or {}),
            }
        )
    return rows


def table(rows):
    lines = [
        "| engine | family | GiB | memory | request | base | head | flags |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['engine']} | {r['family']} | {r['vram_gib']} | {r['memory']} | {r['request']} | "
            f"{r['before'].replace('|', '/')} | {r['after'].replace('|', '/')} | {' '.join(r['flags'])} |"
        )
    return "\n".join(lines)


def compare_sides(before_dir, after_dir):
    before = json.loads((Path(before_dir) / "plan.json").read_text())["cells"]
    after = json.loads((Path(after_dir) / "plan.json").read_text())["cells"]
    rows = diff_cells(before, after)
    flagged = [r for r in rows if r["flags"]]
    counts = {
        f: sum(1 for r in rows if f in r["flags"]) for f in ("REFUSED", "SLOWER", "BF16", "ERR")
    }
    (Path(after_dir) / "diff.json").write_text(
        json.dumps({"counts": counts, "rows": rows}, indent = 1)
    )
    (Path(after_dir) / "diff.md").write_text(
        f"# planner_matrix: {len(rows)} of {len(after)} cells moved\n\n"
        f"flags: {json.dumps(counts)}\n\n{table(rows)}\n"
    )
    if not rows:
        return {"verdict": "SAME", "note": f"{len(after)} planner cells, none moved"}
    shown = (flagged + [r for r in rows if not r["flags"]])[:SUMMARY_ROWS]
    more = (
        f"\n\n({len(rows) - len(shown)} more in {Path(after_dir) / 'diff.md'})"
        if len(rows) > len(shown)
        else ""
    )
    note = (
        f"{len(rows)} of {len(after)} planner cells moved; flagged: "
        + ", ".join(f"{v} {k}" for k, v in counts.items() if v)
        if flagged
        else f"{len(rows)} of {len(after)} planner cells moved, none flagged"
    )
    return {"verdict": "PLAN_DIFF", "note": note, "summary": table(shown) + more, "counts": counts}


def main(argv = None):
    p = argparse.ArgumentParser(description = __doc__.split("\n")[0])
    p.add_argument("--src", help = "source tree (the side's checkout)")
    p.add_argument("--out", help = "output dir for plan.json")
    p.add_argument(
        "--python", default = sys.executable, help = "interpreter with the tree's backend deps"
    )
    p.add_argument("--compare", nargs = 2, metavar = ("BEFORE_DIR", "AFTER_DIR"))
    a = p.parse_args(argv)
    if a.compare:
        res = compare_sides(*a.compare)
        print(f"{res['verdict']}: {res['note']}")
        if res.get("summary"):
            print(res["summary"])
        return 0
    if not (a.src and a.out):
        p.error("--src and --out (or --compare)")
    if not (Path(a.src) / "studio" / "backend").is_dir():
        print(f"no studio/backend under {a.src}", file = sys.stderr)
        return 2
    return run_side(a.src, a.out, python = a.python)


if __name__ == "__main__":
    sys.exit(main())
