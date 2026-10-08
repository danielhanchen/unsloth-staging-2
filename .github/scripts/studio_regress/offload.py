"""Switchboard offload: a job still waiting for a local GPU after OFFLOAD_S (20 min) runs on a
Colab / Kaggle machine that fits it, announced loudly; otherwise it keeps waiting locally.

Rules (scheduler.Scheduler calls `consider` once per admission pass for a job that has no local GPU,
or that only waits behind the one-at-a-time local Core slot):
  * only Core `kind = "job"` targets (jobs/*.py base vs head) can go: Studio journeys, isolated runs,
    diffusion (external) and `regression` targets need local installs / source trees, so they wait
    locally and the log says so once;
  * requirements come from the target: `gpu_mem_gb`, `dtypes` (or `dtype`, default bf16: T4 / Kaggle
    T4x2 run fp16 only, so they are offered only to a target listing "fp16"), `disk_gb` (default
    DEFAULT_DISK_GB; cloud_pool skips tiers last seen with less scratch, and the remote cell refuses
    before installing anything), `remote = false` opts out;
  * up to REMOTE_MAX ($STUDIO_REGRESS_REMOTE_MAX, 6) remote jobs per run at once, each on its own
    machine: cloud_pool.select_tier ranks a warm idle Colab worker of the account first, then the
    cheapest fitting tier (Kaggle T4x2 free quota first for fp16 targets). The pick is a preference:
    cloud_pool.dispatch falls through to the next fitting machine when a racing run took its slot;
  * a Kaggle T4x2 kernel runs one A/B job alone with base on GPU0 and head on GPU1 at once
    (jobs/ab_remote.py --parallel-arms auto), so its result returns when that job ends; a Colab VM
    (one GPU) runs base then head. A reused Colab worker starts each job clean (cloud_pool rollback);
  * compare.py runs here, the same rule as jobs/ab.py. Timings are that machine's, never B200: the
    step note says which (and "T4 vs T4, concurrent on one kernel" for parallel arms).
A remote run that never started (no slot, VM lost, disk short) is handed back to the local queue
(Requeue), never reported as a result. STUDIO_REGRESS_OFFLOAD=0 turns all of this off.
Note: gpu_queue.GPU_QUEUE_OFFLOAD_S (generic `gpu_queue.py run --offload auto`) should use the same
20 min; it is set in gpu_queue.py.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
OFFLOAD_S = float(os.environ.get("STUDIO_REGRESS_OFFLOAD_S") or 1200)
DEFAULT_DISK_GB = (
    25.0  # venv with torch + the head's dependencies (~12 GB), two trees, tiny models, caches
)
SETUP_S = 900  # remote venv + clone + two editable installs, on top of the two arms
BAR = "=" * 96
RETRY_AFTER_S = 300  # a requeued job is not offered remotely again before this
# Optional cap on remote jobs at once per run; 0 (default) = no cap: concurrency is bounded only by
# the real free slots in cloud_pool's host-wide ledger (Colab sessions per account / tier family,
# Kaggle kernels per token), quota and refusal benches.
REMOTE_MAX = int(os.environ.get("STUDIO_REGRESS_REMOTE_MAX") or 0)


class Requeue(Exception):
    """The remote run never produced a result: put the job back in the local queue."""

    def __init__(
        self,
        msg,
        res = None,
    ):
        super().__init__(msg)
        self.res = res


def enabled():
    return os.environ.get("STUDIO_REGRESS_OFFLOAD", "1") != "0"


def eligible(job):
    """(ok, why not) for a scheduler Job (Core; Studio jobs go through remote_studio.eligible)."""
    if job.kind != "core":
        return (
            False,
            "it needs a local Studio install (journeys / diffusion cannot run on Colab or Kaggle)",
        )
    t = job.arms[0].target or {}
    if t.get("kind") != "job":
        return False, "regression targets run on local source trees"
    if t.get("remote") is False:
        return False, "the target says remote = false"
    return True, ""


def requirements(target):
    dtypes = target.get("dtypes") or [target.get("dtype") or "bf16"]
    # fp16 is the widest choice (every tier has it); a job that cannot run in fp16 asks for bf16,
    # which keeps it off T4 / Kaggle T4x2
    dtype = "fp16" if "fp16" in dtypes else dtypes[0]
    est = float(target.get("est_s") or 600)
    return {
        "gb": float(target.get("gpu_mem_gb") or 8.0),
        "dtype": dtype,
        "dtypes": list(dtypes),
        "disk_gb": float(target.get("disk_gb") or DEFAULT_DISK_GB),
        "est_s": 2 * est + SETUP_S,
        "perf": bool(target.get("perf_sensitive")),
    }


def _mins(s):
    return f"{int(s // 60)}m {int(s % 60):02d}s"


def describe(m):
    who = m.get("account") or m.get("token_env") or "?"
    vm = f", VM {m['worker']}" if m.get("worker") else ""
    return f"{m['backend'].upper()} {m['tier']} ({who}{vm})"


class Offloader:
    def __init__(
        self,
        log = print,
        select = None,
        clock = time.time,
        on = None,
        max_remote = None,
        studio = False,
        capacity = None,
        booked = None,
    ):
        self.log, self.clock = log, clock
        self.on = enabled() if on is None else on
        self.max_remote = REMOTE_MAX if max_remote is None else max_remote
        self._select = select
        # Studio journeys / external suites remote (remote_studio.py): only for a two-sided PR run,
        # which run.py says; STUDIO_REGRESS_REMOTE_FORCE=1 offers them at once (tests, live checks)
        self.studio = studio
        self.force = os.environ.get("STUDIO_REGRESS_REMOTE_FORCE") == "1"
        self.batches = {}  # batch id -> remote_studio.Batch
        self._open = {}  # VRAM group -> the batch still taking targets
        # capacity(cand, pending) -> free slots on that machine now after this run's unbooked
        # picks `pending` {key: n} (None = unknown, no limit); booked() ->
        # job ids whose dispatch already holds a ledger slot. Default: cloud_pool's, unless a test
        # injected `select` alone, or offload is off (the remote bundle ships no cloud_pool).
        self._capacity, self._booked = capacity, booked
        if select is None and self.on:
            import cloud_pool
            self._capacity = capacity or cloud_pool.free_capacity
            self._booked = booked or cloud_pool.slot_jobs
        self.running = {}  # job id -> machine it was sent to
        self.machines = []  # every machine this run used, in order (report / tests)
        self._said = {}  # (job index, kind) -> once
        self._retry_at = {}  # job index -> earliest re-offer
        self._lock = threading.Lock()

    @property
    def busy(self):
        """Back-compat: the one running remote job id (or None / the first of several)."""
        return next(iter(self.running), None)

    def select(self, req):
        if self._select is not None:
            return self._select(req)
        import cloud_pool
        return cloud_pool.select_tier(
            req["gb"], req["dtype"], est_s = req["est_s"], perf = req["perf"], disk_gb = req["disk_gb"]
        )

    def _once(self, j, kind, msg):
        if (j.index, kind) not in self._said:
            self._said[(j.index, kind)] = True
            self.log(msg)

    def machines_in_use(self):
        """Distinct remote machines this run holds: one per Core job, one per Studio batch."""
        return len({m.get("batch") or jid for jid, m in self.running.items()})

    @staticmethod
    def _key(c):
        return (c.get("backend"), c.get("tier"), c.get("account") or c.get("token_env"))

    def _pick(self, cands):
        """The first candidate with a slot left once this run's own picks that have not booked their
        ledger slot yet are counted (they would all see the same free slot otherwise), or None."""
        if self._capacity is None:
            return cands[0] if cands else None
        try:
            booked = set(self._booked()) if self._booked else set()
        except Exception:  # noqa: BLE001
            booked = set()
        pending, seen = {}, set()
        for jid, m in self.running.items():
            one = m.get("batch") or jid  # a Studio batch is one machine for all its jobs
            if one in seen or m.get("job_id") in booked:
                continue
            seen.add(one)
            pending[self._key(m)] = pending.get(self._key(m), 0) + 1
        for c in cands:
            try:
                free = self._capacity(
                    c, pending
                )  # pending already subtracted (cross-token caps too)
            except Exception:  # noqa: BLE001 - unknown capacity: let dispatch's locked check decide
                free = None
            if free is None or free > 0:
                return c
        return None

    def consider(self, j, waited):
        """The machine to run `j` on now, or None (keep waiting for a local GPU)."""
        if (
            not self.on
            or (waited < OFFLOAD_S and not (self.force and j.kind != "core"))
            or self.clock() < self._retry_at.get(j.index, 0)
        ):
            return None
        if j.kind in ("journey", "external"):
            return self._consider_studio(j, waited)
        ok, why = eligible(j)
        if not ok:
            self._once(
                j,
                "local",
                f"SWITCHBOARD: {j.id} has waited {_mins(waited)} for a local GPU; {why}, "
                "so it keeps waiting for a local GPU.",
            )
            return None
        req = requirements(j.arms[0].target)
        with self._lock:
            if j.id in self.running:
                return None
            if self.max_remote and self.machines_in_use() >= self.max_remote:
                self._once(
                    j,
                    f"full:{len(self.running)}",
                    f"SWITCHBOARD: {j.id} waits: this run already has "
                    f"{len(self.running)} jobs on Colab/Kaggle (STUDIO_REGRESS_REMOTE_MAX={self.max_remote}); "
                    "still waiting for a local GPU or a remote slot.",
                )
                return None
            try:
                cands = self.select(req)
            except Exception as e:  # noqa: BLE001 - a broken pool never stops local waiting
                cands = []
                self.log(
                    f"SWITCHBOARD: remote selection failed ({type(e).__name__}: {e}); waiting locally"
                )
            if not cands:
                self._once(
                    j,
                    "nofit",
                    f"SWITCHBOARD: {j.id} has waited {_mins(waited)}; no Colab/Kaggle GPU fits "
                    f"{req['gb']:g} GB, {'/'.join(req['dtypes'])}, {req['disk_gb']:g} GB disk right now "
                    "(or every fitting one is busy / out of quota), so it keeps waiting for a local GPU.",
                )
                return None
            c = self._pick(cands)
            if c is None:
                self._once(
                    j,
                    "taken",
                    f"SWITCHBOARD: {j.id} waits: every fitting Colab/Kaggle slot is taken by "
                    "jobs being dispatched right now; it is offered again on the next pass.",
                )
                return None
            m = {k: c.get(k) for k in ("backend", "tier", "account", "token_env")}
            m["worker"] = None
            m["job_id"] = "sr-" + uuid.uuid4().hex[:8]  # remote_job uses it: booked() then sees it
            self.running[j.id] = m
            self.machines.append(dict(m, job = j.id))
            timing = " Timings from it are that card's, not B200." if req["perf"] else ""
            how = (
                "base on GPU0 and head on GPU1 at once"
                if c["backend"] == "kaggle"
                else "base then head on one GPU, clean slate"
            )
            self.log(
                f"{BAR}\nSWITCHBOARD OFFLOAD: {j.id} has waited {_mins(waited)} for a local GPU.\n"
                f"  It goes to {describe(m)} ({req['gb']:g} GB, {req['dtype']}, {req['disk_gb']:g} GB disk): "
                f"{c.get('reason', '')}\n"
                f"  {how}; {len(self.running)} remote job(s) in this run"
                f"{f' (cap {self.max_remote})' if self.max_remote else ''}.{timing}\n{BAR}"
            )
            return dict(m)

    def _consider_studio(self, j, waited):
        from studio_regress import remote_studio as rs

        if not self.studio:
            self._once(
                j,
                "local",
                f"SWITCHBOARD: {j.id} has waited {_mins(waited)}; Studio targets go remote "
                "only in a two-sided PR run, so it keeps waiting locally.",
            )
            return None
        ok, why = rs.eligible(j)
        if not ok:
            self._once(
                j,
                "local",
                f"SWITCHBOARD: {j.id} has waited {_mins(waited)}; {why}, so it keeps "
                "waiting locally.",
            )
            return None
        names, group = rs.target_names(j), rs.group_of(j.gb)
        with self._lock:
            if j.id in self.running:
                return None
            b = self._open.get(group)
            if b is not None and b.open():
                b.add(j.id, names, j.gb)
                self.running[j.id] = dict(b.machine, batch = b.id)
                self.log(
                    f"SWITCHBOARD OFFLOAD: {j.id} joins remote Studio batch {b.id} on {describe(b.machine)} "
                    f"({', '.join(b.targets)})."
                )
                return dict(b.machine, batch = b.id)
            if self.max_remote and self.machines_in_use() >= self.max_remote:
                self._once(
                    j,
                    f"full:{self.machines_in_use()}",
                    f"SWITCHBOARD: {j.id} waits: this run already "
                    f"holds {self.machines_in_use()} Colab/Kaggle machines (STUDIO_REGRESS_REMOTE_MAX="
                    f"{self.max_remote}).",
                )
                return None
            req = {
                "gb": max(float(j.gb or 0), 4.0),
                "dtype": "fp16",
                "dtypes": ["fp16"],
                "disk_gb": rs.DISK_GB,
                "est_s": rs.SETUP_S + 900,
                "perf": False,
            }
            try:
                if self._select is not None:
                    cands = self._select(req)
                else:
                    import cloud_pool
                    cands = cloud_pool.select_tier(
                        req["gb"],
                        "fp16",
                        ram_gb = rs.RAM_GB,
                        est_s = req["est_s"],
                        job_class = "eval",
                        disk_gb = rs.DISK_GB,
                    )
            except Exception as e:  # noqa: BLE001
                cands = []
                self.log(
                    f"SWITCHBOARD: remote selection failed ({type(e).__name__}: {e}); waiting locally"
                )
            if not cands:
                self._once(
                    j,
                    "nofit",
                    f"SWITCHBOARD: {j.id} has waited {_mins(waited)}; no Colab/Kaggle machine "
                    f"fits a {req['gb']:g} GB Studio batch with {rs.DISK_GB:g} GB disk right now, so it keeps "
                    "waiting locally.",
                )
                return None
            c = self._pick(cands)
            if c is None:
                self._once(
                    j,
                    "taken",
                    f"SWITCHBOARD: {j.id} waits: every fitting Colab/Kaggle slot is taken by "
                    "jobs being dispatched right now; it is offered again on the next pass.",
                )
                return None
            m = {k: c.get(k) for k in ("backend", "tier", "account", "token_env")}
            m["job_id"] = (
                "srs-" + uuid.uuid4().hex[:8]
            )  # the batch's remote job id (booked() sees it)
            b = rs.Batch("b" + uuid.uuid4().hex[:6], group, m, clock = self.clock)
            b.add(j.id, names, j.gb)
            self.batches[b.id] = b
            self._open[group] = b
            self.running[j.id] = dict(m, batch = b.id)
            self.machines.append(dict(m, job = j.id, batch = b.id))
            self.log(
                f"{BAR}\nSWITCHBOARD OFFLOAD: {j.id} has waited {_mins(waited)} for local capacity.\n"
                f"  Remote Studio batch {b.id} on {describe(m)}: {c.get('reason', '')}\n"
                f"  Studio targets waiting in the next {rs.BATCH_WINDOW_S:.0f}s join it; both sides install "
                f"there once, results come back with their evidence.\n{BAR}"
            )
            return dict(m, batch = b.id)

    def batch(self, bid):
        return self.batches.get(bid)

    def done(
        self,
        j,
        res = None,
        requeue = False,
    ):
        """After a remote attempt: free its place, and on a requeue hold the job back a while."""
        with self._lock:
            m = self.running.pop(j.id, None)
            if res and res.get("worker") and m is not None:
                m["worker"] = res["worker"]
                for rec in reversed(self.machines):  # the report's copy too
                    if rec.get("job") == j.id:
                        rec["worker"] = res["worker"]
                        break
            if requeue:
                self._retry_at[j.index] = self.clock() + RETRY_AFTER_S
                self.log(
                    f"SWITCHBOARD: {describe(m) if m else 'the remote machine'} did not run {j.id}"
                    f"{' (' + res.get('status') + ')' if res and res.get('status') else ''}; it waits "
                    "for a local GPU and may be offered remotely again later."
                )


# --------------------------------------------------------------------------- remote run of one job target
def merge_base(repo, pr, head_sha):
    import pr_review_status as prs

    meta = prs._gh_obj(["pr", "view", str(pr), "--repo", repo, "--json", "baseRefName"]) or {}
    base_ref = meta.get("baseRefName") or "main"
    cmp = prs._gh_obj(["api", f"repos/{repo}/compare/{base_ref}...{head_sha}"]) or {}
    return (cmp.get("merge_base_commit") or {}).get("sha")


def job_files():
    jobs = SCRIPTS / "jobs"
    out = [
        p
        for p in sorted(jobs.glob("*.py"))
        if not p.name.startswith("test_") and p.name not in ("ab.py", "ab_remote.py")
    ]
    extra = [SCRIPTS / n for n in ("tiny_models.py", "cpu_split.py")]
    return [str(p) for p in out + [p for p in extra if p.exists()]]


def remote_job(target, pr, repo, head_sha, base_sha, companion, req):
    from studio_regress import core

    comp_repo = core.COMPANION_REPO.get(repo, "")
    argv = [
        "--repo",
        repo,
        "--pr",
        str(pr),
        "--base-sha",
        base_sha,
        "--head-sha",
        head_sha,
        "--job",
        core.job_spec(target),
        "--outdir",
        "ab_out",
        "--parallel-arms",
        "auto",
    ]
    if target.get("remote_swap"):
        argv.append("--swap")
    if companion and comp_repo:
        argv += ["--companion-repo", comp_repo, "--companion-sha", companion]
    return {
        "id": req.get("job_id") or "sr-" + uuid.uuid4().hex[:8],
        "script": str(SCRIPTS / "jobs" / "ab_remote.py"),
        "argv": argv,
        "files": job_files(),
        "gb": req["gb"],
        "dtype": req["dtype"],
        "est_s": req["est_s"],
        "disk_gb": req["disk_gb"],
        "perf": req["perf"],
        "class": "train",
        "shared_models": True,
        "whole_kernel": True,  # a Kaggle T4x2 kernel to itself: base on GPU0, head on GPU1
        "artifacts": ["ab_out/*.json", "ab_out/*.log"],
        "timeout_s": int(req["est_s"] * 3 + 1800),
        "what": f"studio_regress pr{pr} {target['name']}"[:80],
    }


def compare_local(out, compare_args = ""):
    """compare.py on the two arm metrics, exactly as jobs/ab.py runs it -> (rc, verdict, reason, table)."""
    cmp = [
        sys.executable,
        str(SCRIPTS / "jobs" / "compare.py"),
        str(out / "base.json"),
        str(out / "head.json"),
        *shlex.split(compare_args or ""),
    ]
    r = subprocess.run(cmp, capture_output = True, text = True)
    text = r.stdout.rstrip()
    last = text.splitlines()[-1] if text else "VERDICT NOT_RUN compare.py printed nothing"
    parts = last.split(" ", 2)
    return (
        r.returncode,
        (parts[1] if len(parts) > 1 else "NOT_RUN"),
        (parts[2] if len(parts) > 2 else ""),
        text,
    )


def arms_note(remote):
    """'; base GPU0 CPUs 0-1 (2 threads), head GPU1 CPUs 2-3 (2 threads), affinity' for parallel arms,
    the sequential reason otherwise, '' when the remote recorded neither."""
    if remote.get("parallel_arms"):
        parts = [
            f"{arm} GPU{(remote.get(arm) or {}).get('gpu')} CPUs {(remote.get(arm) or {}).get('cpus')} "
            f"({(remote.get(arm) or {}).get('threads')} threads)"
            for arm in ("base", "head")
            if (remote.get(arm) or {}).get("cpus")
        ]
        method = (remote.get("cpu_isolation") or {}).get("method")
        return f"; {', '.join(parts)}, {method}" if parts else ""
    return f"; {remote['sequential_reason']}" if remote.get("sequential_reason") else ""


def run_remote_core(
    target,
    pr,
    repo,
    root,
    kw,
    machine,
    log = print,
    dispatch = None,
):
    """One Core job target on the pinned remote machine -> (report step shaped like core.run_job's,
    the cloud_pool result or None). Raises Requeue when the remote run never produced a result."""
    from studio_regress import core

    req = requirements(target)
    head_sha = kw.get("head_sha")
    # one directory per attempt: a requeued job never compares an earlier attempt's arm JSON
    out = (
        Path(root).resolve()
        / "core"
        / core.slug(target["name"])
        / "remote"
        / time.strftime("%Y%m%dT%H%M%S")
    )
    out.mkdir(parents = True, exist_ok = True)
    t0 = time.time()
    try:
        base_sha = merge_base(repo, pr, head_sha) if head_sha else None
    except Exception as e:  # noqa: BLE001
        base_sha, err = None, e
    else:
        err = None
    if not head_sha or not base_sha:
        return core._record(
            target,
            "VOID",
            f"remote: could not resolve the PR head / merge base ({err or 'no SHA'})",
            {"rc": None},
        ), None
    job = remote_job(
        target,
        pr,
        repo,
        head_sha,
        base_sha,
        kw.get("companion"),
        dict(req, job_id = machine.get("job_id")),
    )
    if dispatch is None:
        import cloud_pool
        dispatch = cloud_pool.dispatch
    res = dispatch(job, prefer = machine)
    if res.get("backend"):  # where it really ran (the preferred machine may have been taken)
        machine = dict(
            machine,
            **{k: res.get(k) for k in ("backend", "tier", "account", "token_env") if res.get(k)},
        )
    status = res.get("status")
    if status not in ("PASS", "FAIL"):
        raise Requeue(f"{describe(machine)}: {status}: {res.get('reason', '')}"[:400], res)
    arts = {Path(p).name: Path(p) for p in res.get("artifacts") or []}
    for name in ("base.json", "head.json", "base.log", "head.log", "remote.json"):
        if name in arts:
            (out / name).write_bytes(arts[name].read_bytes())
    remote = {}
    try:
        remote = json.loads((out / "remote.json").read_text())
    except (OSError, ValueError):
        pass
    where = (
        f"{describe(dict(machine, worker = res.get('worker') or machine.get('worker')))}"
        f"{' GPU ' + remote['gpu'] if remote.get('gpu') else ''}"
    )
    if not ((out / "base.json").exists() and (out / "head.json").exists()):
        rc, v, why, table = (
            2,
            "VOID",
            remote.get("reason") or f"remote run {status} without both arm metrics",
            "",
        )
    else:
        rc, v, why, table = compare_local(out, target.get("compare_args"))
    (out / "verdict.json").write_text(
        json.dumps(
            {"verdict": v, "reason": why, "table": table, "remote": remote, "where": where},
            indent = 2,
        )
    )
    verdict = core.JOB_VERDICT.get(v, "VOID")
    after = {
        "rc": rc,
        "verdict": v,
        "reason": why,
        "outdir": str(out),
        "log": str(out / "head.log"),
        "tail": core._tail(out / "head.log"),
        "s": round(time.time() - t0, 1),
        "job": core.job_spec(target),
        "sha": head_sha,
        "companion": kw.get("companion"),
        "gpu": f"remote:{machine['tier']}",
        "remote": {
            "where": where,
            "backend": machine["backend"],
            "tier": machine["tier"],
            "account": machine.get("account"),
            "token_env": machine.get("token_env"),
            "worker": res.get("worker"),
            "gpu_name": remote.get("gpu"),
            "wall_s": res.get("wall_s"),
            "cloud_outdir": res.get("outdir"),
        },
    }
    par = remote.get("parallel_arms")
    timing = ""
    if req["perf"]:
        timing = (
            f" (timings: {remote.get('gpu') or machine['tier']} vs the same card, base and head concurrent "
            "on one kernel, not B200)"
            if par
            else " (timings from that card, not B200)"
        )
    note = (
        f"REMOTE on {where}: compare.py {v}"
        + (f": {why}" if why else "")
        + timing
        + arms_note(remote)
    )
    log(f"SWITCHBOARD: {target['name']} finished on {where}: {v}")
    return core._record(target, verdict, note, after, {"sha": base_sha}), res
