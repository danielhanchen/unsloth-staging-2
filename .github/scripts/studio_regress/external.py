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

Exit 75 (TEMPFAIL) from a side is a harness VOID: a worker that never started (run_edge.py). A head
TEMPFAIL does not run the base: nothing on head was tested.

Fairness: each side records gpu_state (gpu_pack.snapshot: free MiB, other tenants' MiB) at lease
time (recorded only; admission is the lease's job), and diffusion_bench checks record
gpu_free_mib_start plus the suite's own vram_mib_before (their sum = what other tenants left). A would-be FAIL_HEAD whose head failure
reads as a memory refusal / OOM or a hang (a check timeout) while head saw materially less free memory
than base (FAIR_MIN_MIB or FAIR_FRAC of base's) is VOID ("retryable"), not a regression: the other
tenants moved, not the PR. A retryable VOID is rerun by confirm.py, so a real hang still comes back
as FAIL_HEAD.

`compare = "<module>"`: both sides always run (no head-first short cut) and
studio_regress.<module>.compare_sides(before_out, after_out) returns the verdict (SAME or
PLAN_DIFF, e.g. planner_matrix); PLAN_DIFF is reported, never a regression.
"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # the scripts dir
TAIL_LINES = 40
WS = Path(os.environ.get("WORKSPACE") or REPO_ROOT.parent.parent.parent)
TEMPFAIL = 75
FAIR_MIN_MIB, FAIR_FRAC = 2048, 0.10
MEMORY_FAILURE = re.compile(
    r"out of memory|OutOfMemory|does not fit|not enough (?:GPU |video |device )?memory|"
    r"insufficient (?:GPU |V)?RAM|insufficient memory|shortfall|VRAM floor|"
    r"free (?:GPU )?memory|needs? (?:at least )?[0-9.]+ ?Gi?B|"
    r"timed out after|\(hang\)|harness hang",
    re.I,
)


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
    new = [
        f"{r.get('surface', '')}:{r.get('check', '?')}".lstrip(":")
        for r in fails
        if not r.get("known")
    ]
    return new, len(fails) - len(new)


def _argv(cmd, install, side, out, gpu):
    sub = {
        "install": str(install or ""),
        "src": str(src_for(install) or ""),
        "side": side,
        "out": str(out),
        "gpu": "" if gpu is None else str(gpu),
    }
    out = []
    for a in cmd:  # plain replace, not str.format: argv may carry literal braces (JSON, code)
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


def _gpu_state(gpu, genv, snapshot):
    """The lease-time snapshot, recorded for the fairness verdict; never waits (sleeping here would
    hold the lease while admitting nothing). Only for a real lease (genv pins a device)."""
    if gpu is None or not (genv or {}).get("CUDA_VISIBLE_DEVICES"):
        return None
    if snapshot is None:
        from studio_regress import gpu_pack
        snapshot = gpu_pack.snapshot
    return snapshot(gpu)


def run_suite(
    argv,
    cwd = None,
    env = None,
    stdout = None,
    stderr = None,
    text = True,
    timeout = None,
):
    """subprocess.run for a suite, except that a timeout stops the suite's whole process tree (SIGTERM,
    grace, SIGKILL; plat.stop_tree) before raising: killing only the suite left the Studio servers and
    workers it launched (own sessions) running after the switchboard moved on."""
    from studio_regress import plat

    p = subprocess.Popen(
        argv, cwd = cwd, env = env, stdout = stdout, stderr = stderr, text = text, preexec_fn = cpu_preexec(env)
    )
    try:
        return subprocess.CompletedProcess(argv, p.wait(timeout = timeout))
    except subprocess.TimeoutExpired:
        plat.stop_tree(p.pid, grace_s = 60, proc = p)
        raise


def side_gpu(side, env = None):
    """The GPU a side is pinned to, from $STUDIO_REGRESS_SIDE_GPUS ("before=0,after=1"), else None.
    Set by remote_driver.py on a two-GPU VM (Kaggle T4x2): base on GPU0, head on GPU1, like
    jobs/ab_remote.py --parallel-arms, instead of whichever card the lease would pick. A pin that is
    not a GPU index (a typo, a UUID) is ignored: the side falls back to the lease."""
    spec = (os.environ if env is None else env).get("STUDIO_REGRESS_SIDE_GPUS") or ""
    pins = dict(x.split("=", 1) for x in spec.replace(" ", "").split(",") if "=" in x)
    pin = pins.get(side) or ""
    return pin if pin.isdigit() else None


def side_cpus(side, env = None):
    """The CPU set a side is pinned to, from $STUDIO_REGRESS_SIDE_CPUS ("before=0,2;after=1,3"), else
    None. Set by remote_driver.py next to the GPU pins, so two Studios on one Kaggle T4x2 kernel do not
    compete for the same cores."""
    import cpu_split

    spec = (os.environ if env is None else env).get("STUDIO_REGRESS_SIDE_CPUS") or ""
    pins = dict(x.split("=", 1) for x in spec.replace(" ", "").split(";") if "=" in x)
    try:
        cpus = cpu_split.parse_list(pins.get(side) or "")
    except ValueError:
        return None  # malformed: unpinned, never a crashed arm
    return cpus or None


def side_cpu_env(side, env = None):
    """Env a pinned side's processes run with: the CPU set (read back by cpu_preexec in each Popen)
    and thread caps at its size; {} when the side is not pinned."""
    import cpu_split

    cpus = side_cpus(side, env)
    if not cpus:
        return {}
    return {"STUDIO_REGRESS_CPUS": cpu_split.fmt_list(cpus), **cpu_split.thread_env(len(cpus))}


def cpu_preexec(env):
    """preexec_fn pinning a child to env["STUDIO_REGRESS_CPUS"], or None (POSIX only, unset = no pin).
    Affinity is inherited, so the Studio server, its workers and llama-server stay on the set."""
    import cpu_split

    spec = (env or {}).get("STUDIO_REGRESS_CPUS")
    if not spec or os.name != "posix":
        return None
    return cpu_split.isolate(cpu_split.parse_list(spec))


def run_side(
    target,
    side,
    install,
    root,
    env = None,
    runner = None,
    lease = None,
    snapshot = None,
):
    """Run one side; returns {"rc", "tail", "results_json", "s", "gpu", "gpu_state"}."""
    runner = runner or run_suite
    out = Path(root).resolve() / side / target["name"]  # the cmd runs from REPO_ROOT
    out.mkdir(parents = True, exist_ok = True)
    gb = float(target.get("gpu_mem_gb") or 0)
    if lease is None:
        from studio_regress import gpu_pack
        lease = gpu_pack.lease
    t0 = time.time()
    with lease(
        gb, exclusive = bool(target.get("perf_sensitive")), what = f"external:{target['name']}"
    ) as (gpu, genv):
        pin = side_gpu(side)
        if pin is not None:
            gpu, genv = int(pin), {**(genv or {}), "CUDA_VISIBLE_DEVICES": pin}
        genv = {**(genv or {}), **side_cpu_env(side)}
        argv = _argv(target["cmd"], install, side, out, gpu)
        # Output goes to a file, not a pipe: a suite that outlives a dead pipe reader dies on its
        # next write (SIGPIPE / BrokenPipeError) before it stops the Studio it launched.
        log = out / "external.log"
        state = _gpu_state(gpu, genv, snapshot)
        try:
            with open(log, "w") as fh:
                p = runner(
                    argv,
                    cwd = str(REPO_ROOT),
                    env = {**(os.environ if env is None else env), **(genv or {})},
                    stdout = fh,
                    stderr = subprocess.STDOUT,
                    text = True,
                    timeout = wall_cap(target),
                )
            rc, text = p.returncode, log.read_text(errors = "replace")
        except subprocess.TimeoutExpired as e:
            rc, text = -9, log.read_text(errors = "replace") + f"\ntimeout after {e.timeout}s"
        except OSError as e:
            rc, text = 127, f"{type(e).__name__}: {e}"
            log.write_text(text)
    res = out / "results.json"
    rec = {
        "rc": rc,
        "tail": "\n".join(text.splitlines()[-TAIL_LINES:]),
        "argv": argv,
        "gpu": gpu,
        "cpus": (genv or {}).get("STUDIO_REGRESS_CPUS"),
        "gpu_state": state,
        "results_json": str(res) if res.exists() else None,
        "s": round(time.time() - t0, 1),
    }
    if target.get("known_failures") == "results_json" and rc == 1 and res.exists():
        nf = new_failures(res)
        if nf is not None:
            rec["new_failures"], rec["known_failures"] = nf
            if not nf[0]:
                rec["raw_rc"], rec["rc"] = rc, 0  # only known Studio failures
    return rec


def _record(target, head):
    return {
        "key": f"{target['name']}/run",
        "journey": target["name"],
        "step": "run",
        "kind": "external",
        "pixels_changed": 0,
        "dom_delta": None,
        "facts_delta": {},
        "png_before": None,
        "png_after": None,
        "after": head,
        "before": None,
        "note": "",
    }


def _rows(side):
    import json
    try:
        rows = json.loads(Path(side.get("results_json") or "").read_text()).get("results")
    except (OSError, ValueError, TypeError, AttributeError):
        return {}
    return {
        f"{r.get('surface', '')}:{r.get('check', '?')}".lstrip(":"): r
        for r in rows or []
        if isinstance(r, dict)
    }


def _free(rows, key, side):
    """What OTHER tenants left: the check's driver free MiB plus the suite's own model process
    (vram_mib_before), so a PR that grows its own VRAM and OOMs reads as its failure, not contention."""
    ev = (rows.get(key) or {}).get("evidence") or {}
    if ev.get("gpu_free_mib_start") is None:
        return (side.get("gpu_state") or {}).get("free_mib")
    return ev["gpu_free_mib_start"] + (ev.get("vram_mib_before") or 0)


def _unfair(head_free, base_free):
    return (
        head_free is not None
        and base_free is not None
        and base_free - head_free >= max(FAIR_MIN_MIB, FAIR_FRAC * base_free)
    )


def unfair_memory_failure(
    head,
    base,
    keys = None,
):
    """A reason string when EVERY failing head check (`keys`, default all new FAIL rows) is a memory
    refusal / OOM seen with materially less free GPU memory than base had for it (per check when the
    suite records it, else at lease time); None when any failure is fair or not about memory."""
    hr, br = _rows(head), _rows(base)
    failing = (
        list(keys)
        if keys is not None
        else [k for k, r in hr.items() if r.get("status") == "FAIL" and not r.get("known")]
    )
    whys = []
    for k in failing:
        r = hr.get(k) or {}
        text = " ".join(str(x) for x in (r.get("error"), *(r.get("failures") or [])))
        hf, bf = _free(hr, k, head), _free(br, k, base)
        m = MEMORY_FAILURE.search(text)
        if not (m and _unfair(hf, bf)):
            return None
        whys.append(f"{k}: head {hf} MiB free vs base {bf} MiB, failed on memory ({m[0]})")
    if whys:
        return "; ".join(whys[:3])
    hf = (head.get("gpu_state") or {}).get("free_mib")
    bf = (base.get("gpu_state") or {}).get("free_mib")
    m = MEMORY_FAILURE.search(head.get("tail") or "")
    if m and _unfair(hf, bf):
        return f"head {hf} MiB free vs base {bf} MiB at lease, failed on memory ({m[0]})"
    return None


def _void_unfair(rec, why):
    rec.update(
        verdict = "VOID",
        status_before = "ok",
        status_after = "failed",
        retryable = True,
        note = f"GPU contention, not the PR: {why}",
    )
    rec["fairness"] = why
    return rec


def run_compared(
    target,
    homes,
    root,
    env = None,
    runner = None,
    lease = None,
    snapshot = None,
):
    """`compare = "<module>"`: both sides run, the module's compare_sides decides."""
    base_env = {**os.environ, **(env or {})}
    head = run_side(target, "after", homes.get("after"), root, base_env, runner, lease, snapshot)
    before = homes.get("before")
    base = run_side(
        target, "before", before, root, {**os.environ, **(env or {})}, runner, lease, snapshot
    )
    rec = {**_record(target, head), "before": base}
    if head["rc"] != 0 or base["rc"] != 0:
        return (
            rec.update(
                verdict = "VOID",
                status_before = "ok" if base["rc"] == 0 else "failed",
                status_after = "ok" if head["rc"] == 0 else "failed",
                note = f"head exit {head['rc']}, base exit {base['rc']}: comparison not made",
            )
            or rec
        )
    mod = importlib.import_module(f"studio_regress.{target['compare']}")
    out = mod.compare_sides(
        Path(root).resolve() / "before" / target["name"],
        Path(root).resolve() / "after" / target["name"],
    )
    rec.update(
        verdict = out["verdict"],
        status_before = "ok",
        status_after = "ok",
        note = out.get("note", ""),
        compare = out.get("summary"),
    )
    return rec


def run_target(
    target,
    homes,
    root,
    env = None,
    runner = None,
    lease = None,
    snapshot = None,
):
    """Head first, base only on head failure. Returns a report step record."""
    if target.get("compare"):
        return run_compared(target, homes, root, env, runner, lease, snapshot)
    base_env = {**os.environ, **(env or {})}
    head = run_side(target, "after", homes.get("after"), root, base_env, runner, lease, snapshot)
    rec = _record(target, head)
    if head["rc"] == TEMPFAIL:
        return (
            rec.update(
                verdict = "VOID",
                status_before = "not_run",
                status_after = "not_run",
                retryable = True,
                note = f"head harness VOID (exit {TEMPFAIL}): {head['tail'].splitlines()[-1][:300] if head['tail'] else ''}; "
                "base not run",
            )
            or rec
        )
    if head["rc"] == 0:
        known = f"; {head['known_failures']} known failures" if head.get("known_failures") else ""
        rec.update(
            verdict = "SAME",
            status_before = "not_run",
            status_after = "ok",
            note = f"head passed{known}; base not run",
        )
        return rec
    before = homes.get("before")  # may install the base now (run.LazyHomes)
    base_env = {**os.environ, **(env or {})}  # again: that install can add a git safe.directory
    base = run_side(target, "before", before, root, base_env, runner, lease, snapshot)
    rec["before"] = base
    if base["rc"] == 0:
        rec.update(
            verdict = "FAIL_HEAD",
            status_before = "ok",
            status_after = "failed",
            note = f"head exit {head['rc']}, base passes: regression",
        )
        why = unfair_memory_failure(head, base)
        if why:
            _void_unfair(rec, why)
    elif head["rc"] != 1 or base["rc"] != 1:
        # 2 (suite setup), 127 (no executable), -9 (timeout), a signal: that side proved nothing,
        # so the head failure is not shown to be pre-existing.
        rec.update(
            verdict = "VOID",
            status_before = "not_run",
            status_after = "not_run",
            note = f"head exit {head['rc']}, base exit {base['rc']}: suite did not complete",
        )
    elif newly_failing(head, base):
        rec.update(
            verdict = "FAIL_HEAD",
            status_before = "failed",
            status_after = "failed",
            note = f"head exit 1, base exit 1, but base passes {', '.join(newly_failing(head, base)[:5])}",
        )
        why = unfair_memory_failure(head, base, keys = newly_failing(head, base))
        if why:
            _void_unfair(rec, why)
    else:
        rec.update(
            verdict = "FAIL_BOTH",
            status_before = "failed",
            status_after = "failed",
            note = f"head exit {head['rc']}, base exit {base['rc']}: pre-existing",
        )
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
    return {
        f"{r.get('surface', '')}:{r.get('check', '?')}".lstrip(":"): r.get("status")
        for r in rows
        if isinstance(r, dict)
    }


def failed_checks(rec):
    """The checks a FAIL_HEAD step blames: head's new (not known) FAILs that did not also FAIL on
    base, sorted. None when head wrote no per-check results (a crash, a non-diffusion_bench suite)."""
    head, base = rec.get("after") or {}, rec.get("before") or {}
    if not head.get("results_json") or not _statuses(head["results_json"]):
        return None
    bs = (_statuses(base["results_json"]) if base.get("results_json") else None) or {}
    return sorted(
        k
        for k, r in _rows(head).items()
        if r.get("status") == "FAIL" and not r.get("known") and bs.get(k) != "FAIL"
    )


def newly_failing(head, base):
    """Checks that FAIL on head but PASS on base (both suites exited 1): a regression hidden behind
    two failing exit codes. [] when either side has no per-check results."""
    hs = _statuses(head.get("results_json")) if head.get("results_json") else None
    bs = _statuses(base.get("results_json")) if base.get("results_json") else None
    if hs is None or bs is None:
        return []
    return sorted(k for k, v in hs.items() if v == "FAIL" and bs.get(k) == "PASS")
