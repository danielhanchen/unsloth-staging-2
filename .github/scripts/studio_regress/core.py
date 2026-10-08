"""Core switchboard targets: `kind = "job"` (jobs/*.py through jobs/ab.py) and `kind = "regression"`
(the tiny-model suite in scripts/regression/, base-vs-head in one run.py call).

Neither needs a Studio install. Both compare the PR merge base (before) with the PR head (after) in
ONE interpreter (--core-python), with the companion repo (unsloth-zoo for an unsloth PR and vice
versa) pinned to the same SHA on both sides:

  job         python jobs/ab.py --pr N --repo R --job "<stem>:<args>" --head-sha H --outdir D --json
              ab.py editable-installs each arm into the interpreter and restores it after, so job
              targets hold an EXCLUSIVE flock on that interpreter (one at a time per interpreter,
              across every tmux session).
  regression  python regression/run.py run --tier T --unsloth/--zoo <head trees>
              --base-unsloth/--base-zoo <base trees> --gpus <leased> --out D
              Trees are private detached worktrees keyed by SHA (temp/studio_regress/trees/), so
              nothing moves a shared wt_r<N>; the suite prepends them to PYTHONPATH (no install
              change), so it takes a SHARED flock on the interpreter.

Report steps use external.py's record shape, keyed "<target>/run":

  job (compare.py)   NO_REGRESSION -> SAME, FIX_CONFIRMED -> SAME (noted), REGRESSION -> FAIL_HEAD,
                     VOID / NOT_RUN / setup error -> VOID
  regression         exit 0 -> SAME, 1 (pass->fail or metric move) -> FAIL_HEAD,
                     2 infra / 3 invalid -> VOID. The A/B only flags pass -> fail, so exit 0 is
                     checked against the base side: main is kept green by registry `known`
                     entries, hence base failures mean the environment is broken (>= BROKEN_FRAC
                     of the cases that ran, or nothing passed: VOID) or a new failure on main
                     (a few: FAIL_BOTH, reported, not blocking).

A VOID step means the target proved nothing; run.py never reports such a run as clean.
Caps: every subprocess cap, and the regression suite's --deadline / --case-timeout, are the idle
values times timeouts.factor() (host load per core, 1..3): see _timeout / regression_limits.
GPU: gpu_pack leases exactly like external.py (perf_sensitive -> exclusive). A regression target
with `gpus = 2` leases two distinct GPUs so base and head run side by side.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
WS = Path(os.environ.get("WORKSPACE") or SCRIPTS.parent.parent.parent)
TREES = WS / "temp" / "studio_regress" / "trees"
LOCKS = WS / "temp" / "studio_regress" / "locks"
KINDS = ("job", "regression")
REPO_MODULE = {"unslothai/unsloth": "unsloth", "unslothai/unsloth-zoo": "unsloth_zoo"}
COMPANION_REPO = {
    "unslothai/unsloth": "unslothai/unsloth-zoo",
    "unslothai/unsloth-zoo": "unslothai/unsloth",
}
JOB_VERDICT = {"NO_REGRESSION": "SAME", "FIX_CONFIRMED": "SAME", "REGRESSION": "FAIL_HEAD"}
REGRESSION_VERDICT = {0: "SAME", 1: "FAIL_HEAD"}
BROKEN_FRAC = 0.2
GRACE_S = 120  # ab.py restores the interpreter on SIGTERM (a uv reinstall); give it time
TAIL_LINES = 40
for _p in (
    SCRIPTS / "jobs",
    SCRIPTS / "pr_review",
):  # ab.py, pr_review_status.py (appended: never shadow)
    if str(_p) not in sys.path:
        sys.path.append(str(_p))


def _log(msg):
    print(f"[studio_regress.core] {msg}", file = sys.stderr, flush = True)


def default_python():
    """Interpreter the Core targets run in: it must have torch / transformers / unsloth deps.
    studio_regress itself usually runs under the Playwright venv, which has none of them.
    $STUDIO_REGRESS_CORE_PYTHON, else temp/venv_core (a dedicated venv: ab.py swaps the package
    under test in and out of its interpreter, better not the workspace venv), else the active venv."""
    env = os.environ.get("STUDIO_REGRESS_CORE_PYTHON")
    if env:
        return env
    venv = os.environ.get("VIRTUAL_ENV")
    for cand in (
        [WS / "temp" / "venv_core" / "bin" / "python"]
        + ([Path(venv) / "bin" / "python"] if venv else [])
        + [WS / "bin" / "python"]
    ):
        if cand.exists():
            return str(cand)
    return sys.executable


def slug(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name).strip("_")


def job_spec(target):
    """'jobs/grpo.py --preset gpt_oss_tiny' (+ target args) -> 'grpo:--preset gpt_oss_tiny'."""
    parts = shlex.split(target["name"])
    stem = Path(parts[0]).stem
    args = parts[1:] + shlex.split(target.get("args") or "")
    return f"{stem}:{shlex.join(args)}" if args else stem


@contextlib.contextmanager
def interpreter_lock(python, exclusive):
    LOCKS.mkdir(parents = True, exist_ok = True)
    key = hashlib.sha1(os.path.realpath(python).encode()).hexdigest()[:12]
    from studio_regress import plat

    with plat.locked(LOCKS / f"python_{key}.lock", exclusive = exclusive):
        yield


@contextlib.contextmanager
def gpu_leases(
    target,
    n = 1,
    lease = None,
):
    """n distinct GPUs for the target (gpu_pack leases); yields [(gpu, env)] ([] without GPUs)."""
    if lease is None:
        from studio_regress import gpu_pack
        lease = gpu_pack.lease
    gb = float(target.get("gpu_mem_gb") or 0)
    got = []
    with contextlib.ExitStack() as stack:
        for _ in range(max(1, n)):
            kw = {"avoid": [g for g, _ in got]} if got else {}
            g, genv = stack.enter_context(
                lease(
                    gb,
                    exclusive = bool(target.get("perf_sensitive")),
                    what = f"{target['kind']}:{target['name']}",
                    **kw,
                )
            )
            if g is None:
                break
            got.append((g, genv))
        yield got


# ------------------------------------------------------------------ PR trees
def _git(*args, cwd = None):
    return subprocess.run(["git", *map(str, args)], cwd = cwd, capture_output = True, text = True)


def pr_shas(repo, pr, primary):
    """(base_sha = merge base with the PR's base branch, head_sha) for a PR; head fetched locally."""
    import pr_review_status as prs

    meta = (
        prs._gh_obj(["pr", "view", str(pr), "--repo", repo, "--json", "headRefOid,baseRefName"])
        or {}
    )
    head = meta.get("headRefOid")
    if not head:
        raise RuntimeError(f"{repo}#{pr}: could not resolve the PR head")
    base_ref = meta.get("baseRefName") or "main"
    _git("-C", primary, "fetch", "-q", "origin", base_ref, f"pull/{pr}/head")
    mb = _git("-C", primary, "merge-base", head, f"origin/{base_ref}").stdout.strip()
    if not mb:
        raise RuntimeError(f"{repo}#{pr}: no merge base between {head[:9]} and origin/{base_ref}")
    return mb, head


def tree(repo, sha, primary):
    """Private detached worktree of repo at sha, shared by every session (keyed by SHA, never moved)."""
    from ab import detached_worktree
    from studio_regress import gc, plat

    wt = TREES / f"{repo.replace('/', '__')}@{sha[:12]}"
    TREES.mkdir(parents = True, exist_ok = True)
    with plat.locked(TREES / f"{wt.name}.lock"):
        if not (
            (wt / ".git").exists() and _git("-C", wt, "rev-parse", "HEAD").stdout.strip() == sha
        ):
            detached_worktree(primary, wt, sha)
        gc.touch(wt)  # `studio_regress.py gc` reclaims trees unused for days
        return wt


def companion_sha(python, repo):
    """The companion SHA both sides use: ab.py's rule (installed git commit / editable HEAD, else
    upstream main), so job and regression targets of one run test the same pairing."""
    from ab import companion_plan
    return companion_plan(python, REPO_MODULE[repo], "auto")["sha"]


def regression_trees(
    repo,
    pr,
    python,
    companion = None,
):
    """{"unsloth", "zoo", "base_unsloth", "base_zoo", "base_sha", "head_sha", "companion_sha"}."""
    from ab import ensure_primary

    if repo not in REPO_MODULE:
        raise RuntimeError(f"{repo}: the regression suite covers {sorted(REPO_MODULE)} only")
    primary = ensure_primary(repo)
    base_sha, head_sha = pr_shas(repo, pr, primary)
    if base_sha == head_sha:
        raise RuntimeError(f"base and head are the same commit {head_sha[:12]}")
    comp_repo = COMPANION_REPO[repo]
    csha = companion or companion_sha(python, repo)
    if not csha:
        raise RuntimeError(f"could not pin a {comp_repo} SHA")
    comp = tree(comp_repo, csha, ensure_primary(comp_repo))
    head, base = tree(repo, head_sha, primary), tree(repo, base_sha, primary)
    own, other = ("unsloth", "zoo") if repo == "unslothai/unsloth" else ("zoo", "unsloth")
    return {
        own: head,
        f"base_{own}": base,
        other: comp,
        f"base_{other}": comp,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "companion_sha": csha,
        "companion_repo": comp_repo,
    }


# ------------------------------------------------------------------ records
def _record(
    target,
    verdict,
    note,
    after,
    before = None,
):
    return {
        "key": f"{target['name']}/run",
        "journey": target["name"],
        "step": "run",
        "kind": target["kind"],
        "pixels_changed": 0,
        "dom_delta": None,
        "facts_delta": {},
        "png_before": None,
        "png_after": None,
        "verdict": verdict,
        "status_before": {"VOID": "not_run", "FAIL_BOTH": "failed"}.get(verdict, "ran"),
        "status_after": {"SAME": "ok", "FAIL_HEAD": "failed", "FAIL_BOTH": "failed"}.get(
            verdict, "not_run"
        ),
        "after": after,
        "before": before,
        "note": note,
    }


def _tail(path):
    try:
        return "\n".join(Path(path).read_text(errors = "replace").splitlines()[-TAIL_LINES:])
    except OSError:
        return ""


def run_group(
    argv,
    *,
    timeout = None,
    **kw,
):
    """subprocess.run in a new session (a new process group on Windows); on timeout stop the whole
    tree (SIGTERM / CTRL_BREAK), then kill it after GRACE_S. subprocess.run alone kills only the
    direct child: ab.py's job (and the suite's case processes) would keep running on a GPU whose
    lease was just released, and ab.py would never restore the interpreter it swapped the package in."""
    from studio_regress import plat

    p = subprocess.Popen(argv, **plat.group_kwargs(), **kw)
    try:
        return subprocess.CompletedProcess(argv, p.wait(timeout = timeout))
    except subprocess.TimeoutExpired:
        plat.stop_tree(p.pid, grace_s = GRACE_S, proc = p)
        try:
            p.wait(timeout = 10)
        except subprocess.TimeoutExpired:
            pass
        raise


def _run_logged(argv, log, env, cwd, timeout, runner):
    with open(log, "w") as fh:
        try:
            p = runner(
                [str(a) for a in argv],
                cwd = str(cwd),
                env = env,
                stdout = fh,
                stderr = subprocess.STDOUT,
                text = True,
                timeout = timeout,
            )
            return p.returncode
        except subprocess.TimeoutExpired as e:
            fh.write(f"\ntimeout after {e.timeout}s\n")
            return -9
        except OSError as e:
            fh.write(f"{type(e).__name__}: {e}\n")
            return 127


# Wall-clock caps. Idle cap = the target's timeout_s, else max(floor, 6 x est_s) (est_s is warm
# seconds per side; 6x covers both arms / sides with headroom); then x timeouts.factor(), the host
# load per core (clamped 1..3), so a starved box stretches the cap instead of VOIDing a healthy run
# and an idle one gives up on a hung target sooner. Measured: a job's two arms take 43-115 s in
# total (dpo, moe_sft; outputs/studio_regress/core*_*/), regression smoke 190-870 s per side.
JOB_FLOOR_S = 1200
REGRESSION_FLOOR_S = 1800
REGRESSION_CASE_S = 420  # regression/run.py --case-timeout default, scaled like the rest
SUITE_MARGIN_S = 600  # collection + versions probe + the A/B compare, on top of the deadline


def _load_factor():
    from studio_regress import timeouts
    return timeouts.factor()


def _timeout(
    target,
    floor,
    factor = None,
):
    idle = float(target.get("timeout_s") or max(floor, 6 * float(target.get("est_s") or 0)))
    return round(idle * (_load_factor() if factor is None else factor), 1)


def regression_limits(target, factor = None):
    """(suite --deadline, --case-timeout, subprocess cap) for a regression target, load-scaled.
    Idle deadline = max(REGRESSION_FLOOR_S, 2 x est_s) (smoke: 2600 s against 865 s measured per
    side, where the suite's own default was a flat 3600 s); the outer cap adds SUITE_MARGIN_S so
    the suite's deadline ends first and it reports unfinished cases (EXIT_INFRA) instead of being
    killed mid-write. An explicit timeout_s is the (scaled) cap and bounds the deadline."""
    f = _load_factor() if factor is None else factor
    deadline = max(REGRESSION_FLOOR_S, 2 * float(target.get("est_s") or 0)) * f
    cap = deadline + SUITE_MARGIN_S * f
    if target.get("timeout_s"):
        cap = float(target["timeout_s"]) * f
        deadline = min(deadline, cap - SUITE_MARGIN_S * f)
    return max(60, int(deadline)), int(REGRESSION_CASE_S * f), round(cap, 1)


# ------------------------------------------------------------------ job
def run_job(
    target,
    pr,
    repo,
    root,
    python = None,
    env = None,
    runner = run_group,
    lease = None,
    head_sha = None,
    companion = None,
):
    """jobs/ab.py for one job target -> report step."""
    python = python or default_python()
    out = Path(root).resolve() / "core" / slug(target["name"])
    out.mkdir(parents = True, exist_ok = True)
    t0 = time.time()
    with gpu_leases(target, 1, lease) as got:
        gpu, genv = got[0] if got else (None, {})
        argv = [
            python,
            SCRIPTS / "jobs" / "ab.py",
            "--pr",
            pr,
            "--repo",
            repo,
            "--job",
            job_spec(target),
            "--python",
            python,
            "--outdir",
            out,
            "--json",
        ]
        if head_sha:
            argv += ["--head-sha", head_sha]
        if companion:  # the SHA the run's regression targets use too
            argv += ["--companion-sha", companion]
        if target.get("compare_args"):
            argv += ["--compare-args", target["compare_args"]]
        # ab.py writes JSON to stdout and progress to stderr: keep them apart.
        log, js = out / "ab.log", out / "ab.json"
        for stale in (
            js,
            out / "verdict.json",
        ):  # a crashed run must not read the last run's verdict
            stale.unlink(missing_ok = True)
        with interpreter_lock(python, exclusive = True):
            with open(log, "w") as err, open(js, "w") as so:
                try:
                    p = runner(
                        [str(a) for a in argv],
                        cwd = str(SCRIPTS),
                        stdout = so,
                        stderr = err,
                        text = True,
                        env = {**(os.environ if env is None else env), **(genv or {})},
                        timeout = _timeout(target, JOB_FLOOR_S),
                    )
                    rc = p.returncode
                except subprocess.TimeoutExpired as e:
                    err.write(f"\ntimeout after {e.timeout}s\n")
                    rc = -9
                except OSError as e:
                    err.write(f"{type(e).__name__}: {e}\n")
                    rc = 127
    res = {}
    for src in (js, out / "verdict.json"):
        try:
            res = json.loads(Path(src).read_text())
            break
        except (OSError, ValueError):
            continue
    v = res.get("verdict") or ("VOID" if rc else "NOT_RUN")
    rec = {
        "rc": rc,
        "verdict": v,
        "reason": res.get("reason", ""),
        "log": str(log),
        "tail": _tail(log),
        "outdir": str(out),
        "verdict_json": str(out / "verdict.json") if (out / "verdict.json").exists() else None,
        "gpu": gpu,
        "s": round(time.time() - t0, 1),
        "job": job_spec(target),
    }
    base = {"sha": (res.get("base") or {}).get("sha")}
    after = {**rec, "sha": (res.get("head") or {}).get("sha"), "companion": res.get("companion")}
    verdict = JOB_VERDICT.get(v, "VOID")
    if (verdict == "SAME") != (rc == 0):  # ab.py exits 0 exactly for NO_REGRESSION / FIX_CONFIRMED
        verdict = "VOID"
        res["reason"] = f"exit {rc} disagrees with verdict {v}" + (
            f"; {res['reason']}" if res.get("reason") else ""
        )
    note = f"ab.py {v}" + (f": {res['reason']}" if res.get("reason") else "") + f" (exit {rc})"
    return _record(target, verdict, note, after, base)


# ------------------------------------------------------------------ regression
def run_regression(
    target,
    pr,
    repo,
    root,
    python = None,
    env = None,
    runner = run_group,
    lease = None,
    trees = None,
    companion = None,
):
    """regression/run.py base-vs-head for one regression target -> report step."""
    python = python or default_python()
    out = Path(root).resolve() / "core" / slug(target["name"])
    out.mkdir(parents = True, exist_ok = True)
    t0 = time.time()
    try:
        trees = trees or regression_trees(repo, pr, python, companion)
    except Exception as e:  # noqa: BLE001 - any setup failure is a VOID step, never a crash
        return _record(
            target,
            "VOID",
            f"setup: {type(e).__name__}: {e}",
            {"rc": None, "s": round(time.time() - t0, 1), "outdir": str(out)},
        )
    tier = target.get("regression_tier") or "smoke"
    with gpu_leases(target, int(target.get("gpus") or 1), lease) as got:
        gpus = [g for g, _ in got]
        argv = [
            python,
            SCRIPTS / "regression" / "run.py",
            "run",
            "--tier",
            tier,
            "--unsloth",
            trees["unsloth"],
            "--zoo",
            trees["zoo"],
            "--base-unsloth",
            trees["base_unsloth"],
            "--base-zoo",
            trees["base_zoo"],
            "--python",
            python,
            "--out",
            out / "suite",
        ]
        argv += ["--gpus", ",".join(gpus)] if gpus else []
        extra = shlex.split(target.get("args") or "")
        deadline, case_s, cap = regression_limits(target)
        if not any(x == "--deadline" or x.startswith("--deadline=") for x in extra):
            argv += ["--deadline", str(deadline)]
        if not any(x == "--case-timeout" or x.startswith("--case-timeout=") for x in extra):
            argv += ["--case-timeout", str(case_s)]
        argv += extra
        e = {**(os.environ if env is None else env)}
        e.pop("CUDA_VISIBLE_DEVICES", None)  # run.py pins each case to a --gpus slot itself
        if not gpus:
            e["CUDA_VISIBLE_DEVICES"] = ""
        log = out / "regression.log"
        shutil.rmtree(out / "suite", ignore_errors = True)  # no stale report.json / compare.json
        with interpreter_lock(python, exclusive = False):
            rc = _run_logged(argv, log, e, SCRIPTS, cap, runner)
    cmp_md, cmp_js = out / "suite" / "compare.md", out / "suite" / "compare.json"
    counts = {}
    try:
        d = json.loads(cmp_js.read_text())
        counts = {k: len(d.get(k) or []) for k in ("regressions", "fixed", "metric_moves")}
    except (OSError, ValueError):
        pass
    rec = {
        "rc": rc,
        "log": str(log),
        "tail": _tail(log),
        "outdir": str(out / "suite"),
        "compare_md": str(cmp_md) if cmp_md.exists() else None,
        "counts": counts,
        "gpus": gpus,
        "tier": tier,
        "s": round(time.time() - t0, 1),
        "sha": trees.get("head_sha"),
        "companion": {"repo": trees.get("companion_repo"), "sha": trees.get("companion_sha")},
    }
    sides = {s: side_counts(out / "suite" / s) for s in ("base", "head")}
    rec["sides"] = sides
    verdict = REGRESSION_VERDICT.get(rc, "VOID")
    what = {0: "clean", 1: "regression or metric move", 2: "infra / timeout", 3: "invalid run"}.get(
        rc, "crashed"
    )
    note = f"regression {tier}: exit {rc} ({what})" + (f" {json.dumps(counts)}" if counts else "")
    if verdict == "SAME":
        verdict, why = base_health(sides.get("base"), sides.get("head"))
        note += f"; {why}" if why else ""
    return _record(target, verdict, note, rec, {"sha": trees.get("base_sha")})


def side_counts(side_dir):
    """{"passed", "failed", "error", ...} from one side's report.json, or None."""
    try:
        return json.loads((Path(side_dir) / "report.json").read_text()).get("counts") or None
    except (OSError, ValueError, AttributeError):
        return None


def base_health(base, head):
    """(verdict, why) for an A/B that flagged nothing: SAME only when the base side is green."""
    if not base or not head:
        return "VOID", "no per-side report.json: cannot tell a clean run from an empty one"
    bad = int(base.get("failed") or 0) + int(base.get("error") or 0)
    ran = bad + int(base.get("passed") or 0)
    if not int(head.get("passed") or 0) or not ran or bad >= BROKEN_FRAC * ran:
        return "VOID", (
            f"environment broken: base {bad}/{ran} failed, head passed {head.get('passed', 0)}; "
            "check the Core interpreter (run.py --core-python)"
        )
    if bad:
        return "FAIL_BOTH", f"{bad} case(s) fail on base too (pre-existing, not blocking)"
    return "SAME", ""


def run_target(target, pr, repo, root, **kw):
    if target["kind"] == "job":
        kw.pop("trees", None)
        if not kw.get("head_sha"):  # without it ab.py would move the shared wt_r<N>
            return _record(target, "VOID", "could not resolve the PR head SHA", {"rc": None})
        return run_job(target, pr, repo, root, **kw)
    if target["kind"] == "regression":
        kw.pop("head_sha", None)
        return run_regression(target, pr, repo, root, **kw)
    raise ValueError(f"{target['name']}: not a Core target ({target['kind']})")


def pr_head_sha(repo, pr):
    import pr_review_status as prs
    return (prs._gh_obj(["pr", "view", str(pr), "--repo", repo, "--json", "headRefOid"]) or {}).get(
        "headRefOid"
    )


def prepare(
    targets,
    pr,
    repo,
    python = None,
    log = _log,
):
    """(targets for `repo`, kw for run_one): the PR head SHA (job targets) and ONE companion SHA for
    the whole run, so every job and regression target tests one pairing (ab.py's `auto` falls back to
    upstream HEAD, which can move between targets)."""
    from studio_regress.selection import target_repos

    targets = [
        t
        for t in targets
        if repo in target_repos(t) or log(f"{t['name']}: not for {repo}, skipped")
    ]
    head_sha = pr_head_sha(repo, pr) if any(t["kind"] == "job" for t in targets) else None
    python = python or default_python()
    try:
        comp = companion_sha(python, repo) if targets and repo in REPO_MODULE else None
    except Exception as e:  # noqa: BLE001 - each target re-resolves (and VOIDs) on its own
        log(f"companion pin failed ({e}); each target resolves its own")
        comp = None
    return targets, {"python": python, "head_sha": head_sha, "companion": comp}


def run_one(
    t,
    pr,
    repo,
    root,
    kw,
    env = None,
    lease = None,
    log = _log,
):
    """One Core target -> report step; any crash is a VOID step (never a regression)."""
    t0 = time.time()
    try:
        step = run_target(t, pr, repo, root, env = env, lease = lease, **kw)
    except Exception as e:  # noqa: BLE001 - e.g. no GPU lease within wait_s: a crash exits 1 = "regression"
        step = _record(t, "VOID", f"{type(e).__name__}: {e}", {"rc": None})
    log(
        f"{t['kind']} {t['name']}: {step['verdict']} {round(time.time() - t0, 1)}s ({step['note']})"
    )
    return step


def run_all(
    targets,
    pr,
    repo,
    root,
    python = None,
    env = None,
    log = _log,
):
    """Core targets one after another (job targets serialise on the interpreter lock anyway).
    Job targets get --head-sha, so ab.py uses its private detached head worktree and never moves
    a shared wt_r<N> another review session may be using. Returns (steps, timings)."""
    steps, timings = [], {}
    targets, kw = prepare(targets, pr, repo, python, log)
    for t in targets:
        t0 = time.time()
        steps.append(run_one(t, pr, repo, root, kw, env = env, log = log))
        timings[t["name"]] = round(time.time() - t0, 1)
    return steps, timings
