"""Per-step timeout budgets, scaled by host load.

A step's budget is what it measurably needs, not a blanket 300 / 400 s:

    idle budget  = max(FLOOR_S, MULT x p95)   from the step's ok runs (budgets.json, n >= MIN_N)
                 = Step.timeout_s              when there is no measured row (new / rare steps)
    budget       = idle budget x factor()

factor() is the host's load per core, clamp(max(load1, load5) / ncpu, 1, CAP): an overloaded box
gets longer budgets instead of false failures, an idle one fails a wedged step fast. A slow step
already seen in this run (its time over its measured p50, `note_pace`) raises the factor too, since
load average misses I/O-bound slowness; still capped at CAP. The same
factor stretches the inner deadlines journeys poll with (scaled()) and the page's Playwright default
timeouts (engine.run_journey), so an inner wait never fires before the outer budget.

    STUDIO_REGRESS_LOAD_FACTOR=2.5     pin the factor (reproducible runs, tests)
    STUDIO_REGRESS_TIMEOUT_SCALE=2     extra multiplier on every budget (slow runner classes)

budgets.json is generated from facts files (each carries `_s` / `_elapsed_s`, the whole step
including its capture, so a p95 over it is conservative for the action alone):

    python -m studio_regress.timeouts refresh [ROOT ...]     # default outputs/ + temp/studio_regress/
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUDGETS_FILE = HERE / "budgets.json"
MULT = 3.0  # budget = 3 x p95 of the measured ok runs
FLOOR_S = 30.0  # never below this (one page load + capture on a busy box)
MIN_N = 5  # rows with fewer ok samples keep the declared Step.timeout_s
CAP = 3.0  # load factor ceiling
PLAYWRIGHT_DEFAULT_MS = 30_000

_BUDGETS: dict | None = None


def _ncpu() -> int:
    try:
        return len(os.sched_getaffinity(0)) or os.cpu_count() or 1
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def load_factor(loadavg = None, ncpu = None) -> float:
    """clamp(max(load1, load5) / ncpu, 1, CAP). 1.0 where there is no load average (Windows)."""
    env = os.environ.get("STUDIO_REGRESS_LOAD_FACTOR")
    if env:
        try:
            return max(1.0, min(CAP, float(env)))
        except ValueError:
            pass
    if loadavg is None:
        try:
            loadavg = os.getloadavg()
        except (AttributeError, OSError):
            return 1.0
    ncpu = ncpu or _ncpu()
    ratio = max(loadavg[0], loadavg[1] if len(loadavg) > 1 else 0.0) / max(1, ncpu)
    return round(max(1.0, min(CAP, ratio)), 2)


def extra_scale() -> float:
    try:
        return max(0.1, float(os.environ.get("STUDIO_REGRESS_TIMEOUT_SCALE", "1") or 1))
    except ValueError:
        return 1.0


# Observed pace: how much slower than its measured p50 a substantial step (p50 >= PACE_MIN_P50_S)
# just ran in this process. loadavg misses some of what slows a box down (page-cache thrash that
# makes a 270M llama-server start take 6 minutes instead of 40 s), so once a step has shown that,
# later budgets and inner deadlines stretch by it too, still capped at CAP.
PACE_MIN_P50_S = 5.0
_PACE = 1.0


def note_pace(key: str, action_s: float) -> float:
    """Record an ok step's action time against its measured p50; returns the current pace."""
    global _PACE
    row = budgets().get(key)
    if row and row.get("n", 0) >= MIN_N and (row.get("p50") or 0) >= PACE_MIN_P50_S:
        _PACE = max(_PACE, min(CAP, float(action_s) / float(row["p50"])))
    return _PACE


def reset_pace():
    global _PACE
    _PACE = 1.0


def factor() -> float:
    """What every budget and inner deadline is multiplied by right now:
    clamp(max(load factor, observed pace), 1, CAP) x STUDIO_REGRESS_TIMEOUT_SCALE."""
    return round(min(CAP, max(load_factor(), _PACE)) * extra_scale(), 2)


def scaled(seconds: float) -> float:
    """An inner deadline (poll loop, Playwright wait) stretched like the step budgets."""
    return float(seconds) * factor()


def scaled_ms(ms: float) -> float:
    return float(ms) * factor()


def remaining(
    state: dict | None,
    margin_s: float = 5.0,
    default: float | None = None,
) -> float | None:
    """Seconds left in the running step's budget minus `margin_s` (engine.run_journey sets
    state["_step_deadline"]), or `default` outside a step. Inner waits clamp to it so they end
    with their own diagnostic (or kill their subprocess) before the engine cancels the step."""
    import time

    dl = (state or {}).get("_step_deadline")
    if dl is None:
        return default
    return max(1.0, dl - time.monotonic() - margin_s)


def inner(
    seconds: float,
    state: dict | None = None,
    margin_s: float = 5.0,
) -> float:
    """An inner deadline: `seconds` stretched by factor(), never past the step budget."""
    s = scaled(seconds)
    left = remaining(state, margin_s)
    return s if left is None else min(s, left)


class unbudgeted:
    """`async with timeouts.unbudgeted(ctx.state):` time spent inside does not count against the
    running step's budget (engine._run_budgeted moves the deadline by it) and is recorded as the
    step's `_unbudgeted_s`. Only for shared prerequisites that are NOT the step's subject."""

    def __init__(self, state):
        self.state = state if state is not None else {}

    async def __aenter__(self):
        import time

        self.t0 = time.monotonic()
        self.state["_paused"] = self.state.get("_paused", 0) + 1
        return self

    async def __aexit__(self, *exc):
        import time

        spent = time.monotonic() - self.t0
        self.state["_paused"] -= 1
        if "_step_deadline" in self.state:
            self.state["_step_deadline"] += spent
        self.state["_unbudgeted_s"] = self.state.get("_unbudgeted_s", 0.0) + spent
        return False


def budgets() -> dict:
    global _BUDGETS
    if _BUDGETS is None:
        try:
            _BUDGETS = json.loads(BUDGETS_FILE.read_text()).get("steps", {})
        except (OSError, ValueError):
            _BUDGETS = {}
    return _BUDGETS


def idle_budget(journey: str, step) -> float:
    """Measured budget for `journey/step.id`, else the step's declared timeout."""
    row = budgets().get(f"{journey}/{step.id}")
    if row and row.get("n", 0) >= MIN_N and row.get("budget_s"):
        return float(row["budget_s"])
    return float(step.timeout_s)


def step_budget(
    journey: str,
    step,
    f: float | None = None,
) -> tuple[float, float, float]:
    """(budget_s, idle_s, factor) for one step."""
    f = factor() if f is None else f
    idle = idle_budget(journey, step)
    return round(idle * f, 1), idle, f


# ------------------------------------------------------------------ budgets.json generation
def _pct(vals, p):
    vals = sorted(vals)
    k = min(len(vals) - 1, max(0, int(math.ceil(p / 100 * len(vals))) - 1))
    return vals[k]


def collect(roots) -> dict:
    """{journey/step: [seconds of ok runs]} from every <journey>/<step>.facts.json under roots."""
    out: dict = {}
    for root in roots:
        for f in Path(root).rglob("*.facts.json"):
            if "/wt_" in str(f):
                continue
            try:
                d = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            s = d.get("_elapsed_s", d.get("_s"))
            if d.get("_status") != "ok" or not isinstance(s, (int, float)) or not d.get("_step"):
                continue
            out.setdefault(f"{f.parent.name}/{d['_step']}", []).append(float(s))
    return out


def build(samples: dict, note: str = "") -> dict:
    steps = {}
    for key in sorted(samples):
        v = samples[key]
        p95 = _pct(v, 95)
        steps[key] = {
            "n": len(v),
            "p50": round(_pct(v, 50), 2),
            "p95": round(p95, 2),
            "max": round(max(v), 2),
            "budget_s": round(max(FLOOR_S, MULT * p95), 1),
        }
    return {
        "_doc": f"budget_s = max({FLOOR_S:g}, {MULT:g} x p95) over ok runs; used when n >= {MIN_N}. "
        "Generated by `python -m studio_regress.timeouts refresh`. " + note,
        "steps": steps,
    }


def main(argv = None):
    import argparse

    ws = Path(os.environ.get("WORKSPACE") or HERE.parent.parent.parent.parent)
    p = argparse.ArgumentParser(description = __doc__.split("\n")[0])
    sub = p.add_subparsers(dest = "cmd", required = True)
    r = sub.add_parser("refresh", help = "recompute budgets.json from facts files")
    r.add_argument(
        "roots", nargs = "*", default = [str(ws / "outputs"), str(ws / "temp" / "studio_regress")]
    )
    r.add_argument("--note", default = "")
    r.add_argument("--out", default = str(BUDGETS_FILE))
    sub.add_parser("show", help = "print the budgets as they apply right now")
    a = p.parse_args(argv)
    if a.cmd == "refresh":
        data = build(collect(a.roots), a.note)
        Path(a.out).write_text(json.dumps(data, indent = 1) + "\n")
        print(f"{len(data['steps'])} steps -> {a.out}")
    else:
        f = factor()
        print(f"load factor {load_factor()} x scale {extra_scale()} = {f}")
        for k, row in sorted(budgets().items()):
            used = row["n"] >= MIN_N
            print(
                f"{k:55} n={row['n']:4} p95={row['p95']:8} idle={row['budget_s'] if used else '(declared)'!s:>9} "
                f"now={round(row['budget_s'] * f, 1) if used else '-'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
