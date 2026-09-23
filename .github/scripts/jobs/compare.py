#!/usr/bin/env python3
"""Base-vs-head verdict for two scripts/jobs metrics files.

    python compare.py BASE.json HEAD.json [--rtol 0.05] [--atol 1e-3]
                      [--perf-slow 0.20] [--perf-mem 0.15] [--perf-gate]
                      [--expect-fail-on-base CHECK] [--allow-version-diff]

Prints a metric table, any warnings, and one line `VERDICT <verdict> <reason>`.

| Verdict         | Meaning                                                              | Exit |
|-----------------|----------------------------------------------------------------------|------|
| NO_REGRESSION   | both arms valid, every check that passed on base passes on head,     | 0    |
|                 | per-step loss within atol + rtol*|base|                              |      |
| FIX_CONFIRMED   | --expect-fail-on-base CHECK fails on base and passes on head         | 0    |
| REGRESSION      | head fails a check base passes, head crashes, or head loss is higher | 1    |
|                 | than tolerance (perf too, with --perf-gate)                          |      |
| VOID            | the comparison proves nothing: base invalid or checkless, zero steps,| 2    |
|                 | missing metrics, different job/model/steps/seed, or differing core   |      |
|                 | versions (torch/transformers/trl/peft/vllm/mlx*)                     |      |
| NOT_RUN         | a file is missing or an arm was skipped                              | 2    |

Deliberately torch-free: it runs on any machine that has the two JSON files.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

# None == None, so a package absent on both arms (mlx on Linux, vllm on Mac) never differs.
CORE_PACKAGES = ("torch", "transformers", "trl", "peft", "vllm", "mlx", "mlx-lm", "mlx-vlm")
IDENTITY_KEYS = ("job", "model")
IDENTITY_CONFIG_KEYS = ("max_steps", "seed", "tiny")
# Jobs whose evidence is a summary, not per-optimizer-step records.
NO_STEP_JOBS = {"inference_smoke"}
# Sampled-rollout jobs: per-step loss differs run to run on identical code and seed (GRPO step 2
# measured 2.9e-05 vs 0.297), so loss deltas are advisory and the checks carry the verdict.
SAMPLED_LOSS_JOBS = {"grpo"}
EXIT = {"NO_REGRESSION": 0, "FIX_CONFIRMED": 0, "REGRESSION": 1, "VOID": 2, "NOT_RUN": 2}


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _load(path):
    p = Path(path)
    if not p.is_file():
        return None, f"{path} missing"
    try:
        return json.loads(p.read_text()), None
    except json.JSONDecodeError as e:
        return None, f"{path} is not valid JSON: {e}"


def _check_ok(d, name):
    c = (d.get("checks") or {}).get(name)
    return None if c is None else bool(c.get("ok"))


def _median(xs):
    xs = [x for x in xs if _finite(x)]
    return statistics.median(xs) if xs else None


def _steps(d):
    return [s for s in (d.get("steps") or []) if isinstance(s, dict)]


def _fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def compare(base, head, rtol=0.05, atol=1e-3, perf_slow=0.20, perf_mem=0.15, perf_gate=False,
            expect_fail_on_base=None, allow_version_diff=False):
    """Returns (verdict, reason, rows, warnings). rows = [(metric, base, head, status)]."""
    rows, warns = [], []

    def out(verdict, reason):
        return verdict, reason, rows, warns

    for name, d in (("base", base), ("head", head)):
        if d.get("skipped"):
            return out("NOT_RUN", f"{name} skipped: {d['skipped']}")

    # 1. Same experiment on both arms, or the comparison means nothing.
    for k in IDENTITY_KEYS:
        if base.get(k) != head.get(k):
            return out("VOID", f"{k} differs: {base.get(k)!r} vs {head.get(k)!r}")
    bc, hc = base.get("config") or {}, head.get("config") or {}
    for k in IDENTITY_CONFIG_KEYS:
        if bc.get(k) != hc.get(k):
            return out("VOID", f"config.{k} differs: {bc.get(k)!r} vs {hc.get(k)!r}")
    bv, hv = base.get("versions") or {}, head.get("versions") or {}
    drift = [f"{p} {bv.get(p)} vs {hv.get(p)}" for p in CORE_PACKAGES if bv.get(p) != hv.get(p)]
    if drift:
        if not allow_version_diff:
            return out("VOID", "core versions differ: " + "; ".join(drift))
        warns.append("core versions differ (allowed): " + "; ".join(drift))
    if base.get("backend") != head.get("backend"):
        return out("VOID", f"backend differs: {base.get('backend')} vs {head.get('backend')}")

    # 2. A fix claim: the named check must fail on base and pass on head.
    if expect_fail_on_base:
        b_ok, h_ok = _check_ok(base, expect_fail_on_base), _check_ok(head, expect_fail_on_base)
        rows.append((f"check:{expect_fail_on_base}", b_ok, h_ok, "expected fail->pass"))
        if head.get("error"):
            return out("REGRESSION", "head crashed:\n" + head["error"].strip().splitlines()[-1])
        if b_ok is None:
            return out("VOID", f"base never evaluated {expect_fail_on_base}"
                               + (" (base crashed first)" if base.get("error") else ""))
        if b_ok:
            return out("VOID", f"{expect_fail_on_base} passes on base, so the defect is not reproduced")
        if not h_ok:
            return out("REGRESSION", f"{expect_fail_on_base} still fails on head")
        others = [k for k, c in (head.get("checks") or {}).items()
                  if k != expect_fail_on_base and not c.get("ok") and _check_ok(base, k)]
        if others:
            return out("REGRESSION", "fixed but head now fails: " + ", ".join(others))
        return out("FIX_CONFIRMED", f"{expect_fail_on_base} fails on base, passes on head")

    # 3. Base must be a valid reference.
    if base.get("error"):
        return out("VOID", "base crashed: " + base["error"].strip().splitlines()[-1])
    failed_base = [k for k, c in (base.get("checks") or {}).items() if not c.get("ok")]
    if failed_base:
        return out("VOID", "base fails its own checks: " + ", ".join(failed_base))
    if not base.get("checks") or base.get("passed") is False:
        return out("VOID", "base recorded no checks or did not pass, so it is no reference")
    if head.get("error"):
        return out("REGRESSION", "head crashed: " + head["error"].strip().splitlines()[-1])

    # 4. Checks that held on base must hold on head.
    for k, c in (base.get("checks") or {}).items():
        h_ok = _check_ok(head, k)
        rows.append((f"check:{k}", True, h_ok, "ok" if h_ok else "FAIL"))
    lost = [k for k, c in (base.get("checks") or {}).items() if _check_ok(head, k) is not True]
    if lost:
        return out("REGRESSION", "check passes on base, not on head: " + ", ".join(lost))
    new_fail = [k for k, c in (head.get("checks") or {}).items()
                if k not in (base.get("checks") or {}) and not c.get("ok")]
    if new_fail:
        return out("REGRESSION", "head-only check fails: " + ", ".join(new_fail))

    # 5. Per-step metrics.
    bs, hs = _steps(base), _steps(head)
    needs_steps = base.get("job") not in NO_STEP_JOBS
    if needs_steps and (not bs or not hs):
        return out("VOID", f"zero steps recorded (base {len(bs)}, head {len(hs)})")
    if len(bs) != len(hs):
        return out("VOID", f"step count differs: base {len(bs)}, head {len(hs)}")
    worse, better = [], []
    for i, (b, h) in enumerate(zip(bs, hs), 1):
        bl, hl = b.get("loss"), h.get("loss")
        if needs_steps and (not _finite(bl) or hl is None):
            return out("VOID", f"step {i}: loss missing or non-finite (base {bl}, head {hl})")
        if needs_steps and not _finite(hl):  # NaN / inf where base was finite
            return out("REGRESSION", f"step {i}: head loss {hl}, base {bl:.6g}")
        if not (_finite(bl) and _finite(hl)):
            continue
        tol = atol + rtol * abs(bl)
        status = "ok"
        if hl - bl > tol:
            status = "HIGHER"
            worse.append((i, bl, hl))
        elif bl - hl > tol:
            status = "lower"
            better.append((i, bl, hl))
        rows.append((f"step {b.get('step', i)} loss", bl, hl, status))
        bg, hg = b.get("grad_norm"), h.get("grad_norm")
        if _finite(bg) or _finite(hg):
            rows.append((f"step {b.get('step', i)} grad_norm", bg, hg, ""))
    if better:
        warns.append(f"head loss lower than tolerance at {len(better)} step(s), first step "
                     f"{better[0][0]}: {better[0][1]:.6g} -> {better[0][2]:.6g} (behaviour changed)")

    # 6. Performance: advisory unless --perf-gate.
    perf_fail = []
    bt, ht = _median([s.get("time_ms") for s in bs]), _median([s.get("time_ms") for s in hs])
    if bt and ht:
        ratio = ht / bt - 1
        st = "SLOWER" if ratio > perf_slow else "ok"
        rows.append(("median step time_ms", bt, ht, f"{ratio:+.1%} {st}"))
        if st != "ok":
            perf_fail.append(f"step time {ratio:+.1%}")
    bsum, hsum = base.get("summary") or {}, head.get("summary") or {}
    bm = bsum.get("peak_mem_gb") or _median([s.get("peak_mem_gb") for s in bs])
    hm = hsum.get("peak_mem_gb") or _median([s.get("peak_mem_gb") for s in hs])
    if _finite(bm) and _finite(hm) and bm > 0:
        ratio = hm / bm - 1
        st = "MORE" if ratio > perf_mem else "ok"
        rows.append(("peak_mem_gb", bm, hm, f"{ratio:+.1%} {st}"))
        if st != "ok":
            perf_fail.append(f"peak memory {ratio:+.1%}")
    for k in sorted(set(bsum) | set(hsum)):
        if k in ("peak_mem_gb", "wall_s"):
            continue
        if _finite(bsum.get(k)) or _finite(hsum.get(k)):
            rows.append((f"summary.{k}", bsum.get(k), hsum.get(k), ""))
    if perf_fail:
        msg = "performance: " + ", ".join(perf_fail)
        if perf_gate:
            return out("REGRESSION", msg)
        warns.append(msg + " (advisory; --perf-gate to fail on it)")

    if worse and base.get("job") in SAMPLED_LOSS_JOBS:
        i, bl, hl = worse[0]
        warns.append(f"head loss higher at {len(worse)} step(s), first step {i}: {bl:.6g} -> {hl:.6g} "
                     f"(advisory: {base.get('job')} loss is sampling-dependent)")
        worse = []
    if worse:
        i, bl, hl = worse[0]
        return out("REGRESSION", f"head loss higher than tolerance at {len(worse)} step(s), "
                                 f"first step {i}: {bl:.6g} -> {hl:.6g}")
    n = len(bs)
    return out("NO_REGRESSION", f"{n} step(s) within atol={atol} rtol={rtol}; all base checks hold")


def render(rows, warns):
    lines = []
    if rows:
        w = max(len(r[0]) for r in rows)
        lines.append(f"{'metric':<{w}}  {'base':>12}  {'head':>12}  status")
        for m, b, h, st in rows:
            lines.append(f"{m:<{w}}  {_fmt(b):>12}  {_fmt(h):>12}  {st}")
    lines += [f"WARN {w}" for w in warns]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base")
    p.add_argument("head")
    p.add_argument("--rtol", type=float, default=0.05)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--perf-slow", type=float, default=0.20)
    p.add_argument("--perf-mem", type=float, default=0.15)
    p.add_argument("--perf-gate", action="store_true")
    p.add_argument("--expect-fail-on-base", metavar="CHECK")
    p.add_argument("--allow-version-diff", action="store_true")
    a = p.parse_args(argv)
    base, berr = _load(a.base)
    head, herr = _load(a.head)
    if berr or herr:
        verdict, reason, rows, warns = "NOT_RUN", "; ".join(e for e in (berr, herr) if e), [], []
    else:
        verdict, reason, rows, warns = compare(
            base, head, rtol=a.rtol, atol=a.atol, perf_slow=a.perf_slow, perf_mem=a.perf_mem,
            perf_gate=a.perf_gate, expect_fail_on_base=a.expect_fail_on_base,
            allow_version_diff=a.allow_version_diff)
    table = render(rows, warns)
    if table:
        print(table)
    print(f"VERDICT {verdict} {reason}", flush=True)
    return EXIT[verdict]


if __name__ == "__main__":
    sys.exit(main())
