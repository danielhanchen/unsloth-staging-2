"""Switchboard offload: a job still waiting for a local GPU after OFFLOAD_S (20 min) runs on ONE
Colab / Kaggle machine that fits it, announced loudly; otherwise it keeps waiting locally.

Rules (scheduler.Scheduler calls `consider` once per admission pass for a job that has no local GPU):
  * only Core `kind = "job"` targets (jobs/*.py base vs head) can go: Studio journeys, isolated runs,
    diffusion (external) and `regression` targets need local installs / source trees, so they wait
    locally and the log says so once;
  * requirements come from the target: `gpu_mem_gb`, `dtypes` (or `dtype`, default bf16: T4 / Kaggle
    T4x2 run fp16 only, so they are offered only to a target listing "fp16"), `disk_gb` (default
    DEFAULT_DISK_GB; cloud_pool skips tiers last seen with less scratch, and the remote cell refuses
    before installing anything), `remote = false` opts out;
  * one machine per run: the first offload pins (backend, tier, account / token, Colab worker); later
    offloaded jobs reuse it one at a time. Pinned machine busy, or no remote tier fits -> keep waiting
    for a local GPU. A pinned Colab VM that expired while idle is replaced by a fresh one of the same
    tier and account (announced);
  * a reused Colab worker starts each job clean (cloud_pool rollback: fresh venv, job dir removed,
    kernel restarted, GPU / pip freeze / disk verified, else quarantined); the HF model cache and the
    repo clone stay on the worker (shared_models) so a reused machine does not download them again;
  * the remote side (jobs/ab_remote.py) runs base then head on that one machine; compare.py runs here,
    the same rule as jobs/ab.py. Timings are that machine's, never B200: the step note says which.
A remote run that never started (no slot, VM lost, disk short) is handed back to the local queue
(Requeue), never reported as a result. STUDIO_REGRESS_OFFLOAD=0 turns all of this off.
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
    """(ok, why not) for a scheduler Job."""
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
    ):
        self.log, self.clock = log, clock
        self.on = enabled() if on is None else on
        self._select = select
        self.pin = None
        self.busy = None  # job id running on the pinned machine
        self._said = {}  # (job index, kind) -> once
        self._retry_at = {}  # job index -> earliest re-offer
        self._lock = threading.Lock()

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

    def _fits_pin(self, req):
        if self._select is not None:
            return True
        import cloud_pool

        t = cloud_pool.load_tiers().get(self.pin["tier"], {})
        return req["dtype"] in t.get("dtypes", []) and req["gb"] <= t.get("vram_gb", 0)

    def consider(self, j, waited):
        """The machine to run `j` on now, or None (keep waiting for a local GPU)."""
        if not self.on or waited < OFFLOAD_S or self.clock() < self._retry_at.get(j.index, 0):
            return None
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
            if self.busy:
                self._once(
                    j,
                    f"busy:{self.busy}",
                    f"SWITCHBOARD: {j.id} waits: the run's Colab/Kaggle machine "
                    f"{describe(self.pin)} is busy with {self.busy}; still waiting for a local GPU or that machine.",
                )
                return None
            if self.pin:
                if not self._fits_pin(req):
                    self._once(
                        j,
                        "pinfit",
                        f"SWITCHBOARD: {j.id} needs {req['gb']:g} GB {'/'.join(req['dtypes'])}, "
                        f"which the run's machine {describe(self.pin)} cannot give; still waiting for a local GPU.",
                    )
                    return None
                self.busy = j.id
                self.log(
                    f"{BAR}\nSWITCHBOARD OFFLOAD: {j.id} (waited {_mins(waited)} for a local GPU) now runs on "
                    f"the run's {describe(self.pin)}, "
                    f"{'a fresh kernel on the same token' if self.pin['backend'] == 'kaggle' else 'reused with a clean slate'}.\n{BAR}"
                )
                return dict(self.pin)
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
            c = cands[0]
            self.pin = {k: c.get(k) for k in ("backend", "tier", "account", "token_env")}
            self.pin["worker"] = None
            self.busy = j.id
            timing = " Timings from it are that card's, not B200." if req["perf"] else ""
            self.log(
                f"{BAR}\nSWITCHBOARD OFFLOAD: local GPUs have been busy for {_mins(waited)}.\n"
                f"  {j.id} ({req['gb']:g} GB, {req['dtype']}, {req['disk_gb']:g} GB disk) is going to "
                f"{describe(self.pin)}: {c.get('reason', '')}\n"
                f"  Every later offloaded job in this run reuses this same machine (one at a time, "
                f"{'a fresh kernel on this token each' if c['backend'] == 'kaggle' else 'clean slate each'}); "
                f"base and head run on it back to back.{timing}\n{BAR}"
            )
            return dict(self.pin)

    def done(
        self,
        j,
        res = None,
        requeue = False,
    ):
        """After a remote attempt: learn the Colab worker, free the machine, or drop a lost pin."""
        with self._lock:
            self.busy = None
            if res and res.get("worker") and self.pin:
                if self.pin.get("worker") and self.pin["worker"] != res["worker"]:
                    self.log(
                        f"SWITCHBOARD: the run's Colab VM {self.pin['worker']} had expired; continuing on "
                        f"{res['worker']} (same {self.pin['tier']}, same account)."
                    )
                self.pin["worker"] = res["worker"]
            if requeue:
                self._retry_at[j.index] = self.clock() + RETRY_AFTER_S
                if res and res.get("status") in ("PIN_UNAVAILABLE", "NO_REMOTE_FIT"):
                    self.log(
                        f"SWITCHBOARD: {describe(self.pin) if self.pin else 'the remote machine'} is no "
                        "longer available; the next offload picks a new machine."
                    )
                    self.pin = None


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
    tm = SCRIPTS / "tiny_models.py"
    return [str(p) for p in out + ([tm] if tm.exists() else [])]


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
    ]
    if companion and comp_repo:
        argv += ["--companion-repo", comp_repo, "--companion-sha", companion]
    return {
        "id": "sr-" + uuid.uuid4().hex[:8],
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
    out = Path(root).resolve() / "core" / core.slug(target["name"]) / "remote"
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
    job = remote_job(target, pr, repo, head_sha, base_sha, kw.get("companion"), req)
    if dispatch is None:
        import cloud_pool
        dispatch = cloud_pool.dispatch
    res = dispatch(job, pin = machine)
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
    note = (
        f"REMOTE on {where}: compare.py {v}"
        + (f": {why}" if why else "")
        + (" (timings from that card, not B200)" if req["perf"] else "")
    )
    log(f"SWITCHBOARD: {target['name']} finished on {where}: {v}")
    return core._record(target, verdict, note, after, {"sha": base_sha}), res
