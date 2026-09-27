"""`kind = "external"` switchboard targets: absolute pass / fail suites living elsewhere in the
scripts repo (e.g. scripts/diffusion_bench/).

`cmd` is an argv list run from the scripts repo root with placeholders {install} (that side's
Studio install home), {src} (the unsloth source checkout that install was built from), {side}
(before|after), {out} (<root>/<side>/<name>) and {gpu} (leased GPU index, "" when gpu_mem_gb is 0
or no GPU is visible). Head (after) runs first: exit 0 is SAME. Only on a head failure is the base
(before) run: base passes -> FAIL_HEAD, base fails too -> FAIL_BOTH (pre-existing, reported, not
blocking), unless a check that fails on head passes on base (results.json: FAIL_HEAD) or either
side did not complete (exit other than 0 / 1: VOID). Records join report.json "steps".

`known_failures = "results_json"`: a suite that labels checks it already knows fail on main
(diffusion_bench's KNOWN_STUDIO_FAILS, `"known"` on a results.json entry) passes a side whose
only FAILs are known ones, so a pre-existing Studio bug does not force a base run every time.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent   # the scripts dir
TAIL_LINES = 40
WS = Path(os.environ.get("WORKSPACE") or REPO_ROOT.parent.parent.parent)


def src_for(install):
    """Source checkout an install home was built from (cache.py: `.uidiff_src`, else the per-workspace
    temp/studio_regress/src/<sha> named by `.uidiff_sha`)."""
    from studio_regress import cache
    d = cache.source_dir(install)
    stamp = Path(install or "") / ".uidiff_sha"
    if d is None and install and stamp.exists():
        d = WS / "temp" / "studio_regress" / "src" / stamp.read_text().strip()
    return d if d is not None and d.is_dir() else None


def new_failures(results_json):
    """(new FAIL check names, known FAIL count) from a diffusion_bench-style results.json, or
    None when the file is missing or has no results list."""
    import json
    try:
        data = json.loads(Path(results_json).read_text())
    except (OSError, ValueError):
        return None
    rows = data.get("results") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return None
    fails = [r for r in rows if isinstance(r, dict) and r.get("status") == "FAIL"]
    new = [f"{r.get('surface', '')}:{r.get('check', '?')}".lstrip(":") for r in fails if not r.get("known")]
    return new, len(fails) - len(new)


def _argv(cmd, install, side, out, gpu):
    sub = {"install": str(install or ""), "src": str(src_for(install) or ""), "side": side,
           "out": str(out), "gpu": "" if gpu is None else str(gpu)}
    out = []
    for a in cmd:   # plain replace, not str.format: argv may carry literal braces (JSON, code)
        a = str(a)
        for k, v in sub.items():
            a = a.replace("{" + k + "}", v)
        out.append(a)
    return out


def wall_cap(target) -> float:
    """The suite's wall-clock cap: target timeout_s (default 7200 s) stretched by the host load
    factor (timeouts.factor), so a starved box does not turn a slow-but-passing suite into a
    timeout (-9, VOID) while an idle one keeps the declared cap."""
    from studio_regress import timeouts
    return float(target.get("timeout_s") or 7200) * timeouts.factor()


def run_side(target, side, install, root, env=None, runner=subprocess.run, lease=None):
    """Run one side; returns {"rc", "tail", "results_json", "s", "gpu"}."""
    out = Path(root).resolve() / side / target["name"]   # the cmd runs from REPO_ROOT
    out.mkdir(parents=True, exist_ok=True)
    gb = float(target.get("gpu_mem_gb") or 0)
    if lease is None:
        from studio_regress import gpu_pack
        lease = gpu_pack.lease
    t0 = time.time()
    with lease(gb, exclusive=bool(target.get("perf_sensitive")), what=f"external:{target['name']}") as (gpu, genv):
        argv = _argv(target["cmd"], install, side, out, gpu)
        # Output goes to a file, not a pipe: a suite that outlives a dead pipe reader dies on its
        # next write (SIGPIPE / BrokenPipeError) before it stops the Studio it launched.
        log = out / "external.log"
        try:
            with open(log, "w") as fh:
                p = runner(argv, cwd=str(REPO_ROOT), env={**(os.environ if env is None else env), **(genv or {})},
                           stdout=fh, stderr=subprocess.STDOUT, text=True,
                           timeout=wall_cap(target))
            rc, text = p.returncode, log.read_text(errors="replace")
        except subprocess.TimeoutExpired as e:
            rc, text = -9, log.read_text(errors="replace") + f"\ntimeout after {e.timeout}s"
        except OSError as e:
            rc, text = 127, f"{type(e).__name__}: {e}"
            log.write_text(text)
    res = out / "results.json"
    rec = {"rc": rc, "tail": "\n".join(text.splitlines()[-TAIL_LINES:]), "argv": argv, "gpu": gpu,
           "results_json": str(res) if res.exists() else None, "s": round(time.time() - t0, 1)}
    if target.get("known_failures") == "results_json" and rc == 1 and res.exists():
        nf = new_failures(res)
        if nf is not None:
            rec["new_failures"], rec["known_failures"] = nf
            if not nf[0]:
                rec["raw_rc"], rec["rc"] = rc, 0   # only known Studio failures
    return rec


def run_target(target, homes, root, env=None, runner=subprocess.run, lease=None):
    """Head first, base only on head failure. Returns a report step record."""
    base_env = {**os.environ, **(env or {})}
    head = run_side(target, "after", homes.get("after"), root, base_env, runner, lease)
    rec = {"key": f"{target['name']}/run", "journey": target["name"], "step": "run", "kind": "external",
           "pixels_changed": 0, "dom_delta": None, "facts_delta": {}, "png_before": None, "png_after": None,
           "after": head, "before": None, "note": ""}
    if head["rc"] == 0:
        known = f"; {head['known_failures']} known failures" if head.get("known_failures") else ""
        rec.update(verdict="SAME", status_before="not_run", status_after="ok",
                   note=f"head passed{known}; base not run")
        return rec
    before = homes.get("before")   # may install the base now (run.LazyHomes)
    base_env = {**os.environ, **(env or {})}   # again: that install can add a git safe.directory
    base = run_side(target, "before", before, root, base_env, runner, lease)
    rec["before"] = base
    if base["rc"] == 0:
        rec.update(verdict="FAIL_HEAD", status_before="ok", status_after="failed",
                   note=f"head exit {head['rc']}, base passes: regression")
    elif head["rc"] != 1 or base["rc"] != 1:
        # 2 (suite setup), 127 (no executable), -9 (timeout), a signal: that side proved nothing,
        # so the head failure is not shown to be pre-existing.
        rec.update(verdict="VOID", status_before="not_run", status_after="not_run",
                   note=f"head exit {head['rc']}, base exit {base['rc']}: suite did not complete")
    elif newly_failing(head, base):
        rec.update(verdict="FAIL_HEAD", status_before="failed", status_after="failed",
                   note=f"head exit 1, base exit 1, but base passes {', '.join(newly_failing(head, base)[:5])}")
    else:
        rec.update(verdict="FAIL_BOTH", status_before="failed", status_after="failed",
                   note=f"head exit {head['rc']}, base exit {base['rc']}: pre-existing")
    return rec


def _statuses(results_json):
    """{"surface:check": status} from a diffusion_bench-style results.json, or None."""
    import json
    try:
        rows = json.loads(Path(results_json).read_text()).get("results")
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    if not isinstance(rows, list):
        return None
    return {f"{r.get('surface', '')}:{r.get('check', '?')}".lstrip(":"): r.get("status")
            for r in rows if isinstance(r, dict)}


def newly_failing(head, base):
    """Checks that FAIL on head but PASS on base (both suites exited 1): a regression hidden behind
    two failing exit codes. [] when either side has no per-check results."""
    hs = _statuses(head.get("results_json")) if head.get("results_json") else None
    bs = _statuses(base.get("results_json")) if base.get("results_json") else None
    if hs is None or bs is None:
        return []
    return sorted(k for k, v in hs.items() if v == "FAIL" and bs.get(k) == "PASS")
