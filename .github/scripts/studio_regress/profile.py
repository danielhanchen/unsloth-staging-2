"""Where a run's time went: per journey and per step, ranked.

    python -m studio_regress.profile OUT_ROOT [--side before|after] [--top 25] [--json]

Reads <root>/report.json timings (journey wall per side) and the step facts. Each step's time is
split into `action` (the step's own work), `capture` (settle + screenshot + DOM), and classified:

  timeout      the step ran out of its budget (_timeout) or died on a Playwright / asyncio timeout
  failed       failed for another reason (its time is still spent)
  model        a model load / reload / training / export step (llama-server spawn, trainer)
  ok           everything else

`overhead` per journey = journey wall - sum of its steps: login, the GPU warm, browser context,
teardown. Facts from runs before per-step splits were recorded only carry `_s` (all "action").
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MODEL_RX = re.compile(r"load|reload|reapply|train|export|complete|progress|generate|clip|run_ready|_suite|chat_ui|extra_ui")


def _facts(side_dir: Path):
    for f in sorted(side_dir.glob("*/*.facts.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if d.get("_status") in (None, "skipped_after_failure", "not_run_head_passed"):
            continue
        yield f.parent.name, d


def classify(journey, d):
    err = (d.get("_error") or "")
    if d.get("_timeout") or (d.get("_status") == "failed" and "Timeout" in err.split(":")[0]):
        return "timeout"
    if d.get("_status") != "ok":
        return "failed"
    if MODEL_RX.search(d.get("_step", "")):
        return "model"
    return "ok"


def profile(root: Path, side: str):
    root = Path(root)
    rep = {}
    try:
        rep = json.loads((root / "report.json").read_text())
    except (OSError, ValueError):
        pass
    walls = (rep.get("timings") or {}).get(side, {})
    rows, per_j = [], {}
    for j, d in _facts(root / side):
        total = float(d.get("_elapsed_s", d.get("_s")) or 0)
        cap = float(d.get("_capture_s") or 0)
        act = float(d.get("_action_s", total - cap) or 0)
        kind = classify(j, d)
        rows.append({"key": f"{j}/{d.get('_step')}", "total_s": round(total, 1), "action_s": round(act, 1),
                     "capture_s": round(cap, 1), "kind": kind, "status": d.get("_status"),
                     "budget_s": d.get("_budget_s"), "error": (d.get("_error") or "")[:90]})
        pj = per_j.setdefault(j, {"steps_s": 0.0, "action_s": 0.0, "capture_s": 0.0, "timeout_s": 0.0,
                                  "model_s": 0.0, "failed_s": 0.0})
        pj["steps_s"] += total
        pj["action_s"] += act
        pj["capture_s"] += cap
        if kind in ("timeout", "model", "failed"):
            pj[f"{kind}_s"] += total
    for j, pj in per_j.items():
        wall = walls.get(j)
        pj["wall_s"] = wall
        pj["overhead_s"] = round(wall - pj["steps_s"], 1) if isinstance(wall, (int, float)) else None
        for k in list(pj):
            if isinstance(pj[k], float):
                pj[k] = round(pj[k], 1)
    rows.sort(key=lambda r: -r["total_s"])
    return {"side": side, "journeys": per_j, "steps": rows}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root")
    p.add_argument("--side", default="before")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    res = profile(Path(a.root), a.side)
    if a.json:
        print(json.dumps(res, indent=1))
        return 0
    print(f"{'journey':22} {'wall':>7} {'steps':>7} {'action':>7} {'capture':>7} {'timeout':>7} {'model':>7} "
          f"{'failed':>7} {'overhead':>8}")
    for j, r in sorted(res["journeys"].items(), key=lambda kv: -(kv[1]["wall_s"] or kv[1]["steps_s"])):
        print(f"{j:22} {r['wall_s'] or '-':>7} {r['steps_s']:>7} {r['action_s']:>7} {r['capture_s']:>7} "
              f"{r['timeout_s']:>7} {r['model_s']:>7} {r['failed_s']:>7} {r['overhead_s'] if r['overhead_s'] is not None else '-':>8}")
    print()
    for r in res["steps"][:a.top]:
        print(f"{r['key']:45} {r['total_s']:>7} act {r['action_s']:>6} cap {r['capture_s']:>6} {r['kind']:8} {r['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
