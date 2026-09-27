"""Parallel scheduler (the default for a local two-sided `run`): (side x unit) arms plus Core /
external targets.

    python studio_regress.py run --pr N [--jobs 4] [--gpu-jobs 2] [--instance per_unit|shared] [--plan]

The legacy path (`--scheduler sequential` / `--sequential`) is untouched: before side, then after
side, journeys one after another on one Studio, then externals, then Core.

Instances. PerUnitInstance (default): a fresh Studio per arm, home state/<root>-iNN (equal length),
a port with the run's digit count, the literals masked glyph-exactly in captures and replaced in
DOM / facts (engine.instance_info); overlap_sides + private_fs, so pairs start together.
SharedInstance (`--instance shared`): today's one Studio per side, sides in turn (or, with
--parallel-sides, per-side paths: A/A timing only).

Units. The plan is a list of JOBS in a fixed order; a job is one or two ARMS (one per side):

  journey     a dependency CHAIN of Studio journeys (a connected component of switchboard `deps`,
              e.g. train_responses_only > export_reload): one instance per arm, so the dependant
              sees the adapter its dependency wrote in the same home. auth and crawl are chains of
              their own.
  head_first  a HEAD_FIRST journey (upstream_ci): the after arm first, then the before arm runs
              only the steps head did not pass (run.head_first_base_plan).
  isolated    the ISOLATED journeys (full_access_isolated): nested run.py in the isolation wrapper.
  external    one `kind = "external"` target (external.run_target: head first, base on failure).
  core        one Core target (core.run_target: base and head inside one call).

Constraints, all enforced here and unit-tested with fake instances (test_scheduler.py):

  deps         chains share an instance; a head_first before arm waits for its after arm.
  slots        arms whose instance slot is equal run one at a time and IN PLAN ORDER. The default
               SharedInstance (today's one-Studio-per-side) puts every Studio arm of a side on one
               slot, so auth still runs first on the fresh Studio and crawl last. An isolating
               instance returns no slot: every arm then gets its own Studio, does its own bootstrap
               login + password rotation (Handle.tokens), and auth gets an un-rotated one.
  pairing      when the instance lets sides overlap, both arms of a pair START TOGETHER (one job,
               one GPU for both), so timing-sensitive steps see the same host moment; a pair that
               must not overlap (perf_sensitive, a journey with OVERLAP_SIDES = False unless the
               instance is private_fs, AFTER_REPLAYS_BEFORE (crawl: before first), head_first,
               isolated) runs its arms BACK TO BACK in one worker under one lease. Otherwise
               (SharedInstance) sides are side-major, exactly as today. private_fs instances also
               start the longest units first.
  perf         perf_sensitive jobs take an EXCLUSIVE gpu_pack lease; both arms on that device.
  GPU          LeaseBook: host-wide gpu_pack leases, one holder entry per (pid, GPU) carrying the
               summed GB of this process's jobs there (old-format holder keys, so older sessions
               still read the lease files). Journey arms lease only when the instance pins a GPU.
  concurrency  running weight <= min(--jobs / $STUDIO_REGRESS_JOBS, auto_jobs()), re-read at every
               admission from os.getloadavg(), nproc and MemAvailable; never below 1. gpu-tier
               Studio arms: at most --gpu-jobs (2) at once.
  browsers     ONE Chromium per (run, side) (SideBrowser, on the side's own event-loop thread), never
               shared across runs, workspaces or sides; a fresh BrowserContext per (side, journey),
               closed when it ends; a crash VOIDs only that side's in-flight units.

Failures: an arm that raises or times out is recorded (status error / timeout) and only its own
steps turn VOID (void_missing_facts -> diff.UNIT_VOID); every other job still runs. Ctrl-C or
SIGTERM: running arms are cancelled, the instance is closed (engine.stop on every Studio it
launched) and any process this run spawned that is still alive is stopped (plat.stop_tree).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from studio_regress import gpu_pack  # noqa: E402

SIDES = ("before", "after")
STUDIO_KINDS = ("journey", "head_first")       # arms that drive a Studio from the instance
LAUNCH_S = 300                                  # launch + first login allowance per arm
MIN_ARM_TIMEOUT_S = 1800
WATCHDOG_GRACE_S = 300                          # past its own timeout before the scheduler steps in
ABANDON_S = 60                                  # after cancelling, before giving up on the thread
STARVE_S = 120                                  # a cap-blocked job waiting this long reserves its room
DEFAULT_MAX_JOBS = 8


def _log(msg):
    print(f"[studio_regress.scheduler] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ concurrency cap
def mem_available_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def auto_jobs(nproc=None, load=None, mem_gb=None, max_jobs=None):
    """Concurrent arms this host can take now. Each arm is a Studio + Chromium (or a suite
    process): ~CPU_PER_UNIT cores of burst and ~GB_PER_UNIT of RAM. The host is shared and
    oversubscribed by design (much of its load is I/O wait), so CPU room is nproc * OVERSUB - load1.
    Env: STUDIO_REGRESS_CPU_PER_UNIT (8), STUDIO_REGRESS_GB_PER_UNIT (6), STUDIO_REGRESS_OVERSUB (3),
    STUDIO_REGRESS_MAX_AUTO_JOBS (8)."""
    env = os.environ
    nproc = nproc or os.cpu_count() or 1
    if load is None:
        try:
            load = os.getloadavg()[0]
        except (OSError, AttributeError):
            load = 0.0
    mem_gb = mem_available_gb() if mem_gb is None else mem_gb
    cpu_per = float(env.get("STUDIO_REGRESS_CPU_PER_UNIT") or 8)
    gb_per = float(env.get("STUDIO_REGRESS_GB_PER_UNIT") or 6)
    oversub = float(env.get("STUDIO_REGRESS_OVERSUB") or 3)
    max_jobs = max_jobs or int(env.get("STUDIO_REGRESS_MAX_AUTO_JOBS") or DEFAULT_MAX_JOBS)
    cpu = int((nproc * oversub - load) // cpu_per)
    mem = int((mem_gb - 16) // gb_per) if mem_gb is not None else max_jobs
    return max(1, min(cpu, mem, max_jobs))


def make_cap(jobs=None, auto=None):
    """cap() -> current limit: min(hard cap, auto); --jobs / $STUDIO_REGRESS_JOBS is the hard cap.
    auto defaults to auto_jobs, looked up per call (at load 700 it is 1: the run goes serial)."""
    hard = jobs or int(os.environ.get("STUDIO_REGRESS_JOBS") or 0) or None

    def cap():
        a = (auto or auto_jobs)()
        return max(1, min(hard, a) if hard else a)
    cap.hard = hard
    return cap


# ------------------------------------------------------------------ plan
@dataclasses.dataclass
class Arm:
    """One side of one unit. `names`: journeys (chain order); `target`: external / core target."""
    id: str
    pair: str
    kind: str
    side: str                  # before | after | both (external / core compare inside one call)
    names: tuple = ()
    target: Optional[dict] = None
    gb: float = 0.0
    perf: bool = False
    est_s: float = 0.0
    timeout_s: Optional[float] = None
    session: Optional[str] = None      # SharedInstance: arms with one session share one Studio
    studio_env: dict = dataclasses.field(default_factory=dict)
    deps: tuple = ()           # arm ids that must finish first


@dataclasses.dataclass
class Job:
    """What the scheduler admits: arms run `together` (one thread each, started at once),
    in `sequence` (one thread, back to back) or `single`."""
    index: int
    pair: str
    kind: str
    mode: str
    arms: list
    slots: tuple = ()
    deps: tuple = ()           # job indexes
    gb: float = 0.0            # per arm
    gpus: int = 1              # distinct GPUs (core regression `gpus = 2`)
    perf: bool = False
    weight: int = 1
    lease_gpu: bool = False
    gpu_tier: bool = False     # a Studio job with a tier = "gpu" journey (--gpu-jobs caps these)

    @property
    def id(self):
        return self.pair if self.mode != "single" else self.arms[0].id

    def lease_gb(self):
        return self.gb * (len(self.arms) if self.mode == "together" else 1)


@dataclasses.dataclass
class Plan:
    jobs: list
    mode: str                  # side_major | paired
    notes: list = dataclasses.field(default_factory=list)

    def arms(self):
        return [a for j in self.jobs for a in j.arms]


def chains(names, data):
    """Connected components of switchboard deps over `names` (already in run.order()), each in
    that order, listed by first member."""
    deps = {t["name"]: [d for d in (t.get("deps") or []) if d in names] for t in data.get("target", [])}
    parent = {n: n for n in names}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    for n in names:
        for d in deps.get(n, []):
            parent[find(n)] = find(d)
    groups = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    out, seen = [], set()
    for n in names:
        r = find(n)
        if r not in seen:
            seen.add(r)
            out.append(tuple(groups[r]))
    return out


def _target(data, name):
    return next((t for t in data.get("target", []) if t["name"] == name), {})


def _arm_timeout(data, names):
    est = sum(float(_target(data, n).get("est_s") or 60) for n in names)
    return max(MIN_ARM_TIMEOUT_S, 6 * est) + LAUNCH_S


class Caps:
    """What the scheduler needs to know about an instance implementation (see IsolatedInstance)."""

    def __init__(self, overlap_sides=False, pins_gpu=False, private_fs=False, slot=None):
        self.overlap_sides, self.pins_gpu, self.private_fs = overlap_sides, pins_gpu, private_fs
        self.slot = slot or (lambda side, arm: None)


def build_plan(names, journeys, data, sides, *, iso=(), hf=(), externals=(), cores=(),
               skip_before=(), instance=None, core_slot="core-python"):
    """Pure: the job list for this run. `names`: Studio journeys in run.order() (iso / hf included
    or not, both work). `skip_before`: journeys whose before arm is not run (reused base side).
    `instance`: an IsolatedInstance or Caps."""
    caps = instance if instance is not None else Caps()
    iso, hf = [n for n in names if n in iso], [n for n in names if n in hf]
    shared = [n for n in names if n not in iso and n not in hf]
    paired = bool(caps.overlap_sides) and len(sides) == 2
    mods = {n: journeys[n][1] if n in journeys else None for n in names}

    def env_of(ns):
        env = {}
        for n in ns:
            env.update(getattr(mods.get(n), "STUDIO_ENV", {}) or {})
        return env

    def gb_of(ns):
        return max([float(_target(data, n).get("gpu_mem_gb") or 0) for n in ns] or [0.0])

    def perf_of(ns):
        return any(bool(_target(data, n).get("perf_sensitive")) for n in ns)

    def arm(kind, pair, side, ns, session=None, deps=()):
        return Arm(id=f"{side}:{pair}", pair=pair, kind=kind, side=side, names=tuple(ns), gb=gb_of(ns),
                   perf=perf_of(ns), est_s=sum(float(_target(data, n).get("est_s") or 60) for n in ns),
                   timeout_s=_arm_timeout(data, ns), session=session or side, studio_env=env_of(ns), deps=deps)

    units = []      # (pair, kind, [arms], mode)
    notes = []
    if paired:
        for ch in chains([n for n in names if n not in iso], data):
            pair = ">".join(ch)
            if any(n in hf for n in ch):
                a = arm("head_first", pair, "after", ch)
                b = arm("head_first", pair, "before", ch, deps=(a.id,))
                units.append((pair, "head_first", [a, b], "sequence"))
                continue
            arms_ = [arm("journey", pair, s, ch) for s in sides if not (s == "before" and set(ch) <= set(skip_before))]
            overlap_ok = caps.private_fs or all(getattr(mods.get(n), "OVERLAP_SIDES", True) for n in ch)
            # the after crawl replays the control list the before crawl saved (coverage._load_route):
            # before first, never together
            replays = any(getattr(mods.get(n), "AFTER_REPLAYS_BEFORE", False) for n in ch)
            mode = "single" if len(arms_) == 1 else \
                "sequence" if (perf_of(ch) or not overlap_ok or replays) else "together"
            units.append((pair, "journey", arms_, mode))
        if iso:
            pair = "isolated:" + ",".join(iso)
            arms_ = [arm("isolated", pair, s, iso) for s in sides if not (s == "before" and set(iso) <= set(skip_before))]
            units.append((pair, "isolated", arms_, "sequence" if len(arms_) > 1 else "single"))
    else:
        # side-major (today's order): every arm of a side, then the other side, then head-first bases
        hf_after = {}
        for s in sides:
            for ch in chains([n for n in names if n not in iso], data):
                pair = ">".join(ch)
                if any(n in hf for n in ch):
                    if s == "after":
                        a = arm("head_first", pair, "after", ch)
                        hf_after[pair] = a.id
                        units.append((a.id, "head_first", [a], "single"))
                    elif "after" not in sides:     # a before-only run has no head to go first
                        units.append((f"before:{pair}", "head_first",
                                      [arm("head_first", pair, "before", ch)], "single"))
                    continue
                if s == "before" and set(ch) <= set(skip_before):
                    continue
                units.append((f"{s}:{pair}", "journey", [arm("journey", pair, s, ch)], "single"))
            if iso and not (s == "before" and set(iso) <= set(skip_before)):
                pair = "isolated:" + ",".join(iso)
                units.append((f"{s}:{pair}", "isolated", [arm("isolated", pair, s, iso)], "single"))
        for pair, aid in hf_after.items():
            if "before" in sides:
                units.append((f"before:{pair}", "head_first",
                              [arm("head_first", pair, "before", pair.split(">"), session="before#hf", deps=(aid,))],
                              "single"))
    if paired and caps.private_fs:
        # no shared Studio, so no auth-first / crawl-last order to keep: longest first (a
        # head_first chain runs its two arms back to back, so it counts twice)
        units.sort(key=lambda u: -(max if u[3] == "together" else sum)(a.est_s for a in u[2]))
        notes.append("per-unit Studios: longest units first")
    for t in externals:
        a = Arm(id=f"both:{t['name']}", pair=t["name"], kind="external", side="both", target=t,
                gb=float(t.get("gpu_mem_gb") or 0), perf=bool(t.get("perf_sensitive")),
                est_s=float(t.get("est_s") or 600))
        units.append((t["name"], "external", [a], "single"))
    for t in cores:
        a = Arm(id=f"both:{t['name']}", pair=t["name"], kind="core", side="both", target=t,
                gb=float(t.get("gpu_mem_gb") or 0), perf=bool(t.get("perf_sensitive")),
                est_s=float(t.get("est_s") or 600))
        units.append((t["name"], "core", [a], "single"))

    jobs, by_arm = [], {}
    for i, (pair, kind, arms_, mode) in enumerate(units):
        if not arms_:
            continue
        slots = []
        for a in arms_:
            s = caps.slot(a.side, a) if kind in STUDIO_KINDS + ("isolated",) else None
            if s and s not in slots:
                slots.append(s)
        gpus = 1
        if kind == "core":
            if a.target.get("kind") == "job" and core_slot:
                slots.append(core_slot)    # ab.py swaps the package in the interpreter: one at a time
            gpus = max(1, int(a.target.get("gpus") or 1)) if a.target.get("kind") == "regression" else 1
        gb = max(a.gb for a in arms_)
        lease_gpu = gb > 0 and (kind in ("external", "core") or bool(caps.pins_gpu))
        weight = len(arms_) if mode == "together" else gpus
        gpu_tier = kind in STUDIO_KINDS + ("isolated",) and any(
            getattr(journeys.get(n, (None,))[0], "tier", None) == "gpu" for a in arms_ for n in a.names)
        j = Job(index=len(jobs), pair=pair, kind=kind, mode=mode, arms=arms_, slots=tuple(slots), gb=gb,
                gpus=gpus, perf=any(a.perf for a in arms_), weight=weight, lease_gpu=lease_gpu,
                gpu_tier=gpu_tier)
        for a in arms_:
            by_arm[a.id] = j.index
        jobs.append(j)
    for j in jobs:
        j.deps = tuple(sorted({by_arm[d] for a in j.arms for d in a.deps if d in by_arm and by_arm[d] != j.index}))
    if not paired and caps.overlap_sides is False and len(sides) == 2 and shared:
        notes.append("sides run one after the other (the instance cannot run both sides at once)")
    return Plan(jobs=jobs, mode="paired" if paired else "side_major", notes=notes)


def format_plan(plan, cap=None, instance_name="shared"):
    lines = [f"schedule: {plan.mode}, instance {instance_name}"
             + (f", concurrency cap now {cap}" if cap is not None else "")]
    lines += [f"  note: {n}" for n in plan.notes]
    for j in plan.jobs:
        arms = " + ".join(f"{a.side}[{','.join(a.names) or (a.target or {}).get('name', '')}]" for a in j.arms)
        extra = []
        if j.slots:
            extra.append("slot " + ",".join(j.slots))
        if j.deps:
            extra.append("after #" + ",".join(map(str, j.deps)))
        if j.gpu_tier:
            extra.append("gpu tier")
        if j.lease_gpu:
            extra.append(f"gpu {j.lease_gb():g} GB" + (f" x{j.gpus}" if j.gpus > 1 else "")
                         + (" EXCLUSIVE" if j.perf else ""))
        elif j.gb:
            extra.append(f"gpu {j.gb:g} GB (not pinned)")
        lines.append(f"  #{j.index:<3} {j.kind:10} {j.mode:8} w{j.weight} {arms}"
                     + (f"  ({'; '.join(extra)})" if extra else ""))
    return "\n".join(lines)


# ------------------------------------------------------------------ GPU leases
class LeaseBook:
    """This process's gpu_pack leases. gpu_pack keys a holder by PID, so two jobs of one process on
    one GPU would overwrite each other's entry; here the process holds ONE entry per GPU whose gb is
    the sum of its jobs there (exclusive when a perf job holds it, which then admits nothing else)."""

    def __init__(self, gpus=None, lock_dir=None, query=None, headroom=gpu_pack.HEADROOM_GB, pid=None,
                 ttl_s=30.0):
        self.gpus = list(gpus) if gpus is not None else gpu_pack.visible_gpus()
        self.lock_dir, self.headroom = lock_dir, headroom
        self.query = query or query_gpus_slow
        self.pid = pid or os.getpid()
        self.held = {}          # gpu -> [(token, gb, exclusive, what)]
        self._n = 0
        self._lock = threading.Lock()
        self.ttl_s = ttl_s
        self._info, self._info_t, self._refreshing = None, 0.0, False

    def info(self, ids):
        """nvidia-smi view, cached: on this host one query took 56 s under load, and the admission
        loop must not stall on it. The first call blocks; later ones return the cache and refresh it
        in the background. A failed query (all zeros) never replaces a good reading."""
        now = time.time()
        if self._info is None:
            self._refresh(list(self.gpus))
        elif now - self._info_t > self.ttl_s and not self._refreshing:
            self._refreshing = True
            threading.Thread(target=self._refresh, args=(list(self.gpus),), daemon=True).start()
        return {g: v for g, v in (self._info or {}).items() if g in set(ids)}

    def _refresh(self, ids):
        try:
            got = self.query(ids) or {}
        except Exception:  # noqa: BLE001
            got = {}
        good = {g: v for g, v in got.items() if v.get("total_gb")}
        if good or self._info is None:
            self._info = {**(self._info or {}), **(good or got)}
            self._info_t = time.time()
        self._refreshing = False

    def why(self):
        """One line for a waiting job: blind (no reading) or each GPU's free GB and lease count."""
        info = self._info or {}
        if not any(v.get("total_gb") for v in info.values()):
            return "no nvidia-smi reading" + (f" ({QUERY_ERROR['error']})" if QUERY_ERROR else "")
        parts = []
        for g in self.gpus:
            try:
                n = len(gpu_pack.leases(g, lock_dir=self.lock_dir))
            except OSError:
                n = "?"
            parts.append(f"GPU {g}: {info.get(g, {}).get('free_gb', 0):.1f} GB free, {n} lease(s)")
        return ", ".join(parts)

    def _total(self, g):
        return sum(x[1] for x in self.held.get(g, []))

    def _write(self, g):
        hs = self.held.get(g) or []
        if not hs:
            gpu_pack.release(g, pid=self.pid, lock_dir=self.lock_dir)
            return True
        what = ",".join(sorted({x[3] for x in hs}))[:200]
        return gpu_pack.try_acquire(g, self._total(g), exclusive=any(x[2] for x in hs), what=what, pid=self.pid,
                                    lock_dir=self.lock_dir)

    def try_take(self, gb, exclusive=False, what="", avoid=()):
        """(token, gpu, total_gb) or None when nothing fits now. No visible GPU / gb <= 0:
        (None, None, 0) -- run unpinned, like gpu_pack.lease."""
        cands = [g for g in self.gpus if g not in set(map(str, avoid))]
        if not cands or gb <= 0:
            return (None, None, 0.0)
        info = self.info(cands)
        with self._lock:
            for g in sorted(cands, key=lambda g: -info.get(g, {}).get("free_gb", 0)):
                mine = self.held.get(g, [])
                if mine and (exclusive or any(x[2] for x in mine)):
                    continue
                inf = info.get(g, {"total_gb": 0.0, "free_gb": 0.0})
                total = self._total(g) + gb
                # nvidia-smi's free already excludes what our running jobs allocated: add it back
                if gpu_pack.try_acquire(g, total, exclusive=exclusive, what=what, pid=self.pid,
                                        total_gb=inf["total_gb"], free_gb=inf["free_gb"] + self._total(g),
                                        headroom=self.headroom, lock_dir=self.lock_dir):
                    self._n += 1
                    self.held.setdefault(g, []).append((self._n, gb, exclusive, what))
                    self._write(g)
                    return (self._n, g, inf["total_gb"])
        return None

    def give(self, token, gpu):
        if token is None:
            return
        with self._lock:
            self.held[gpu] = [x for x in self.held.get(gpu, []) if x[0] != token]
            if not self._write(gpu):
                _log(f"gpu{gpu}: could not shrink this process's lease (kept larger until the next change)")
            if not self.held[gpu]:
                self.held.pop(gpu)

    def release_all(self):
        with self._lock:
            for g in list(self.held):
                self.held[g] = []
                self._write(g)
                self.held.pop(g)

    @contextlib.contextmanager
    def lease(self, gb, exclusive=False, what="", wait_s=3600, poll_s=15, env=None, avoid=()):
        """gpu_pack.lease's signature on top of this book (external.py / core.py accept it)."""
        deadline = time.time() + wait_s
        while True:
            got = self.try_take(gb, exclusive, what, avoid)
            if got is not None:
                break
            if time.time() > deadline:
                raise TimeoutError(f"no GPU in {self.gpus} could take {gb} GB (exclusive={exclusive})")
            time.sleep(poll_s)
        token, g, total = got
        try:
            yield g, (gpu_pack.env_for(g, gb, total) if g is not None else {})
        finally:
            self.give(token, g)


QUERY_ERROR = {}    # the last failed nvidia-smi reading, for the scheduler's waiting message
_SHIM_MARKER = b"nvidia-smi shim: one real query per"     # scripts/nvsmi_cache/nvidia-smi docstring


def _real_nvidia_smi():
    """First nvidia-smi on PATH that is not a copy of the caching shim, or None."""
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = os.path.join(d, "nvidia-smi") if d else ""
        if not p or not (os.path.isfile(p) and os.access(p, os.X_OK)):
            continue
        try:
            with open(p, "rb") as f:
                if _SHIM_MARKER in f.read(4096):
                    continue
        except OSError:
            continue
        return p
    return None


def query_gpus_slow(ids, timeout=180):
    """gpu_pack.query_gpus with a patient timeout (its 20 s is too short at load 300+). A shim on
    PATH that fails or never answers (a wedged cache lock, #406) is retried once against the real
    binary: a blind reading counts as 0 GB free, and admits no GPU job for the whole gpu_wait_s."""
    import shutil
    import subprocess
    if not ids:
        return {}
    first = shutil.which("nvidia-smi") or "nvidia-smi"
    real = _real_nvidia_smi()
    tries = [first] + ([real] if real and os.path.realpath(real) != os.path.realpath(first) else [])
    for exe in tries:
        try:
            out = subprocess.run([exe, "--query-gpu=index,memory.total,memory.free",
                                  "--format=csv,noheader,nounits", "-i", ",".join(ids)],
                                 capture_output=True, text=True, timeout=timeout, check=True).stdout
            res = {}
            for line in out.strip().splitlines():
                idx, tot, free = [x.strip() for x in line.split(",")]
                res[idx] = {"total_gb": float(tot) / 1024, "free_gb": float(free) / 1024}
            QUERY_ERROR.clear()
            return res
        except Exception as e:  # noqa: BLE001
            QUERY_ERROR.update(error=f"{exe}: {type(e).__name__}: {(str(e).splitlines() or [''])[0][:160]}",
                               at=time.time())
    return {i: {"total_gb": 0.0, "free_gb": 0.0} for i in ids}


class FixedLease:
    """A `lease` callable handing out the GPUs the scheduler already leased for a job: the first
    grant not in use and not in `avoid`. An external target leasing for head and then for base gets
    the same GPU both times (back to back on one device); a regression target asking for a second,
    distinct GPU while holding the first gets the next grant."""

    def __init__(self, grants):
        self.grants = list(grants)     # [(gpu, env)]
        self._busy = set()
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def __call__(self, gb, exclusive=False, what="", avoid=(), **_kw):
        avoid = set(map(str, avoid))
        with self._lock:
            i = next((k for k, (g, _e) in enumerate(self.grants) if k not in self._busy and str(g) not in avoid), None)
            if i is not None:
                self._busy.add(i)
        if i is None:
            yield None, {}
            return
        try:
            yield self.grants[i]
        finally:
            with self._lock:
                self._busy.discard(i)


# ------------------------------------------------------------------ instances
@dataclasses.dataclass
class Handle:
    """A running Studio an arm drives. tokens(): fresh access tokens (the first call rotates the
    bootstrap password: never before the auth journey ran); stop(): this arm is done with it."""
    base_url: str
    home: Optional[str] = None
    bootstrap_password: Optional[str] = None
    studio_log: Optional[str] = None
    install_home: Optional[str] = None
    password: Optional[str] = None
    tokens_fn: Optional[Callable] = None
    stop_fn: Optional[Callable] = None
    # engine.instance_info of a per-unit Studio (token / port the captures neutralise), else None
    instance: Optional[dict] = None

    def tokens(self):
        return self.tokens_fn()

    def stop(self):
        if self.stop_fn is not None:
            fn, self.stop_fn = self.stop_fn, None
            fn()


class TokenState:
    """The tokens of ONE Studio, shared by every arm / journey that drives it (thread-safe):
    engine.fresh_tokens under a lock, i.e. bootstrap rotation on first use and a re-login once the
    token is engine.TOKEN_MAX_AGE_S (40 min) old. The sequential path (run.run_side) uses it too."""

    def __init__(self, base_url, home, password, max_age_s=None, log=None):
        self.base_url, self.home, self.password, self.max_age_s = base_url, home, password, max_age_s
        self.tokens, self.at = None, 0.0
        self.log = log or _log
        self._lock = threading.Lock()

    @property
    def rotated(self):
        return self.tokens is not None

    def __call__(self):
        from studio_regress import engine
        with self._lock:
            self.tokens, self.at = engine.fresh_tokens(self.tokens, self.at, self.base_url, self.home,
                                                       self.password, max_age_s=self.max_age_s, log=self.log)
            return self.tokens


class IsolatedInstance:
    """Plug-in point for the isolation mechanism. The scheduler calls, per Studio arm:

        start(side, unit, env) -> Handle   (base_url, tokens(), home, bootstrap_password, stop())
        finish(side, unit)                 always, after the arm (also when it failed / skipped)
        close()                            once at the end and on Ctrl-C: tear EVERYTHING down;
                                           idempotent and safe from another thread

    `unit` is a scheduler.Arm: .names (journeys, in order), .session, .studio_env (union of the
    journeys' STUDIO_ENV), .kind. An isolating implementation starts a fresh Studio per arm (fresh
    state home, its own bootstrap password) and stops it in Handle.stop(). `env` carries the GPU
    lease (CUDA_VISIBLE_DEVICES, PYTORCH_CUDA_ALLOC_CONF, UNSLOTH_REGRESS_MEM_FRACTION) when
    pins_gpu is True, else {}.

    Capabilities: overlap_sides (both arms of a pair may run at once: needs identical UI-visible
    paths / ports on both sides), pins_gpu (start() applies env's CUDA_VISIBLE_DEVICES, so journey
    arms get gpu_pack leases), private_fs (journeys with fixed side-invariant paths, OVERLAP_SIDES =
    False, may still overlap). slot(side, unit): arms returning the same string never overlap and
    run in plan order; None = no constraint."""

    name = "abstract"
    overlap_sides = False
    pins_gpu = False
    private_fs = False

    def prepare(self, plan):
        pass

    def slot(self, side, unit):
        # the nested isolated run.py uses <state>/<root.name> for both sides
        return "nested-state" if unit.kind == "isolated" else None

    def start(self, side, unit, env=None) -> Handle:
        raise NotImplementedError

    def finish(self, side, unit):
        pass

    def close(self):
        pass


class SharedInstance(IsolatedInstance):
    """Today's model: ONE fresh-state Studio per side (session), every journey of the side in plan
    order on it, stopped after the session's last arm. Same state path and port on both sides (they
    show in the UI), so the sides cannot overlap. parallel_sides=True gives each side its own path
    and port instead ("-b" / "-a", equal length): both sides run at once, but path / port text in
    shots differs between sides, so use it for A/A timing and plumbing checks, not for verdicts."""

    name = "shared"
    pins_gpu = False

    def __init__(self, root, homes, password, parallel_sides=False, online=False, state_base=None,
                 launch=None, stop=None, ports=None, log=_log):
        from studio_regress import engine
        self.root, self.homes, self.password = Path(root), homes, password
        self.overlap_sides = bool(parallel_sides)
        self.online = online
        self.state_base = Path(state_base or engine.WS / "temp" / "studio_regress" / "state")
        self._launch = launch or engine.launch
        self._stop = stop or engine.stop
        self._ports = ports
        self.log = log
        self.sessions = {}       # key -> dict(home, port, base_url, bootstrap, tokens, error, live)
        self.remaining = {}
        self.env = {}
        self._lock = threading.RLock()
        self._closed = False

    def prepare(self, plan):
        for a in plan.arms():
            if a.kind in STUDIO_KINDS:
                self.remaining[a.session] = self.remaining.get(a.session, 0) + 1
                self.env.setdefault(a.session, {}).update(a.studio_env)

    def slot(self, side, unit):
        if unit.kind == "isolated":
            return "studio" if not self.overlap_sides else "nested-state"
        return f"studio:{side}" if self.overlap_sides else "studio"

    def _paths(self, side):
        if not self.overlap_sides:
            return self.state_base / self.root.name, self.root.name
        tag = f"{self.root.name}-{side[0]}"
        return self.state_base / tag, tag

    def _pick_port(self, seed):
        if self._ports is not None:
            return self._ports(seed)
        from studio_regress import engine
        return engine.PORTS.take(seed=seed)

    def start(self, side, unit, env=None):
        from studio_regress import engine
        key = unit.session or side
        with self._lock:
            if self._closed:
                raise RuntimeError("instance closed")
            s = self.sessions.get(key)
            if s is None:
                if not self.overlap_sides:      # one path + port: never two sessions alive
                    for k, o in list(self.sessions.items()):
                        if o.get("live"):
                            self._stop_session(k)
                s = self.sessions[key] = {"side": side, "live": False, "error": None}
                dest, seed = self._paths(side)
                s["home"] = str(engine.state_home(self.homes[side], dest))
                s["log"] = str(self.root / "logs" / f"studio_{side}{'' if key == side else '_hf'}.log")
                extra = dict(self.env.get(key, {}))
                extra.update(env or {})
                if self.online:
                    extra["HF_HUB_OFFLINE"] = "0"
                t0 = time.time()
                port = self._pick_port(seed)
                try:
                    inst = self._launch(Path(s["home"]), port, Path(s["log"]), extra)
                except Exception as e:  # noqa: BLE001 - every arm of this session reports it, no relaunch
                    s["error"] = f"{type(e).__name__}: {e}"
                    self._release(port)
                    raise
                if inst.port != port:
                    self._release(port)
                s.update(live=True, port=inst.port, base_url=f"http://127.0.0.1:{inst.port}",
                         bootstrap=inst.bootstrap_password, launch_s=round(time.time() - t0, 1))
                s["tokens"] = TokenState(s["base_url"], s["home"], self.password + side[:1])
                self.log(f"{side}: Studio up on :{inst.port} for session {key} ({s['launch_s']}s)")
            if s["error"]:
                raise RuntimeError(f"Studio for {key} did not start: {s['error']}")
            if not s["live"]:
                raise RuntimeError(f"Studio for {key} was already stopped")
            return Handle(base_url=s["base_url"], home=s["home"],
                          bootstrap_password=None if s["tokens"].rotated else s["bootstrap"],
                          studio_log=s["log"], install_home=str(self.homes[side] or ""),
                          password=self.password + side[:1], tokens_fn=s["tokens"])

    def finish(self, side, unit):
        key = unit.session or side
        with self._lock:
            self.remaining[key] = self.remaining.get(key, 1) - 1
            if self.remaining[key] <= 0 and key in self.sessions:
                self._stop_session(key)

    def _release(self, port):
        if self._ports is None:
            from studio_regress import engine
            engine.PORTS.release(port)

    def _stop_session(self, key):
        s = self.sessions.get(key) or {}
        if s.get("live"):
            s["live"] = False
            try:
                self._stop(s["port"], home=s["home"])
            except Exception as e:  # noqa: BLE001
                self.log(f"stop {key} :{s.get('port')}: {e}")
            self._release(s["port"])

    def close(self):
        with self._lock:
            self._closed = True
            for k in list(self.sessions):
                self._stop_session(k)


class PerUnitInstance(IsolatedInstance):
    """The default instance: a fresh Studio per ARM (side x journey chain), so both sides of a pair
    run at once and no two units share a home, a port, a database or a settings file.

    Home: state/<root>-iNN (engine.instance_token; every home of a run has the same length) and a
    port from engine.PORTS with the same digit count as every other port of the run (never 8888).
    Both reach the UI (auth 401 reset hint, Settings > Logs paths, window.location.origin in the
    agent / API panels), so the Handle carries engine.instance_info: fixture.capture masks exactly
    the "NN" / port-digit glyphs, and the DOM snapshot and facts replace them with fixed text
    (engine.canon_text). Journeys that make extra homes (cli: <home>_cli) hang them off this home.

    private_fs: nothing a journey writes is shared between arms, so journeys with OVERLAP_SIDES =
    False (cli: its extra home used to be state/<root>_cli on both sides) may overlap their sides.
    The nested isolated run keeps its own single state path (slot "nested-state": sides in turn).
    Each arm rotates its own bootstrap password on first tokens(); an auth arm sees it unrotated.
    Homes are deleted when their arm ends unless STUDIO_REGRESS_KEEP_STATE=1 (logs stay in
    <root>/logs/studio_<side>_<unit>.log)."""

    name = "per_unit"
    overlap_sides = True
    private_fs = True
    pins_gpu = False

    def __init__(self, root, homes, password, parallel_sides=False, online=False, state_base=None,
                 launch=None, stop=None, ports=None, keep_state=None, log=_log):
        from studio_regress import engine
        self.root, self.homes, self.password = Path(root), homes, password
        self.online = online
        self.state_base = Path(state_base or engine.WS / "temp" / "studio_regress" / "state")
        self._launch = launch or engine.launch
        self._stop = stop or engine.stop
        self.ports = ports or engine.PORTS
        self.keep_state = (os.environ.get("STUDIO_REGRESS_KEEP_STATE", "") not in ("", "0")
                           if keep_state is None else keep_state)
        self.log = log
        self.live = {}           # arm id -> {"port", "home", "idx"}
        self.used = set()        # instance indexes in use
        self.started = 0
        self._lock = threading.RLock()
        self._closed = False

    def _take_idx(self):
        from studio_regress import engine
        with self._lock:
            idx = next((i for i in range(engine.MAX_INSTANCES) if i not in self.used), None)
            if idx is None:
                raise RuntimeError(f"more than {engine.MAX_INSTANCES} Studios alive in one run")
            self.used.add(idx)
            return idx

    def start(self, side, unit, env=None):
        from studio_regress import engine
        with self._lock:
            if self._closed:
                raise RuntimeError("instance closed")
        idx = self._take_idx()
        home = self.state_base / engine.instance_token(self.root.name, idx)
        tag = unit.pair.replace(">", "+").replace(":", "_").replace("/", "_")[:80]
        log_path = self.root / "logs" / f"studio_{side}_{tag}.log"
        extra = dict(unit.studio_env or {})
        extra.update(env or {})
        if self.online:
            extra["HF_HUB_OFFLINE"] = "0"
        port = None
        try:
            engine.state_home(self.homes[side], home)
            port = self.ports.take()
            t0 = time.time()
            inst = self._launch(home, port, log_path, extra, pick=self.ports.take)
        except BaseException:
            if port is not None:
                self.ports.release(port)
            self._drop_home(home, idx)
            raise
        if inst.port != port:
            self.ports.release(port)
        port = inst.port
        with self._lock:
            self.live[unit.id] = {"port": port, "home": home, "idx": idx}
            self.started += 1
            closed = self._closed
        if closed:          # close() ran while this Studio was starting: it must not outlive the run
            self._stop_arm(unit.id)
            raise RuntimeError("instance closed")
        base_url = f"http://127.0.0.1:{port}"
        self.log(f"{side}: Studio i{idx:02d} up on :{port} for {unit.pair} ({time.time() - t0:.1f}s)")
        return Handle(base_url=base_url, home=str(home), bootstrap_password=inst.bootstrap_password,
                      studio_log=str(log_path), install_home=str(self.homes[side] or ""),
                      password=self.password + side[:1],
                      tokens_fn=TokenState(base_url, str(home), self.password + side[:1], log=self.log),
                      stop_fn=lambda aid=unit.id: self._stop_arm(aid),
                      instance=engine.instance_info(self.root.name, idx, port))

    def _drop_home(self, home, idx):
        import shutil
        if not self.keep_state:
            for d in [home] + sorted(home.parent.glob(home.name + "_*")):   # + journeys' extra homes
                shutil.rmtree(d, ignore_errors=True)
        with self._lock:
            self.used.discard(idx)

    def _stop_arm(self, aid):
        with self._lock:
            rec = self.live.pop(aid, None)
        if rec is None:
            return
        try:
            self._stop(rec["port"], home=str(rec["home"]))
        except Exception as e:  # noqa: BLE001
            self.log(f"stop {aid} :{rec['port']}: {e}")
        self.ports.release(rec["port"])
        self._drop_home(rec["home"], rec["idx"])

    def finish(self, side, unit):
        self._stop_arm(unit.id)     # normally Handle.stop() already did; a no-op then

    def close(self):
        with self._lock:
            self._closed = True
            ids = list(self.live)
        for aid in ids:
            self._stop_arm(aid)


INSTANCES = {"per_unit": PerUnitInstance, "shared": SharedInstance}
DEFAULT_INSTANCE = "per_unit"


def load_instance(spec, **kw):
    """`per_unit` (default), `shared`, or `package.module:Factory` (an IsolatedInstance), built with kw."""
    spec = spec or DEFAULT_INSTANCE
    if spec in INSTANCES:
        return INSTANCES[spec](**kw)
    import importlib
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr or "Instance")(**kw)


# ------------------------------------------------------------------ browsers
class BrowserCrashed(RuntimeError):
    """The side's Chromium died while this unit was in flight (at: epoch of the crash)."""

    def __init__(self, side, at):
        super().__init__(f"{side} browser crashed at {time.strftime('%H:%M:%S', time.localtime(at))}")
        self.side, self.at = side, at


async def _launch_chromium():
    from playwright.async_api import async_playwright
    pw = await async_playwright().start()
    browser = await pw.chromium.launch()

    async def closer():
        with contextlib.suppress(Exception):
            await browser.close()
        with contextlib.suppress(Exception):
            await pw.stop()
    return browser, closer


class SideBrowser:
    """ONE Playwright Chromium per (run, side), never shared across runs, workspaces or sides, and
    no browser server. Playwright objects belong to the event loop that made them, so the browser
    lives on this side's own loop thread and every unit of the side runs its coroutine there
    (run()); each unit opens its own BrowserContext per journey and closes it when done
    (run.drive_journeys). If the browser dies, the units in flight raise BrowserCrashed (only
    they turn VOID) and the next unit gets a fresh browser."""

    def __init__(self, side, launch=None, log=_log, warm=None):
        import asyncio
        self.side, self.log = side, log
        self._launch = launch or _launch_chromium
        # fixture.warm_browser for real Chromium (the default launcher); injected launchers (tests)
        # pass their own or none
        if warm is None and launch is None:
            from studio_regress import fixture
            warm = fixture.warm_browser
        self._warm, self.warm_s = warm, []
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name=f"sr-browser-{side}")
        self._thread.start()
        self._alock = asyncio.Lock()
        self.browser, self._closer = None, None
        self.launches, self.crashes = 0, []
        self._closing = False

    async def _ensure(self):
        async with self._alock:
            if self.browser is not None and self.browser.is_connected():
                return self.browser
            if self._closer is not None:
                await self._closer()
            self.browser, self._closer = await self._launch()
            self.launches += 1
            b = self.browser
            b.on("disconnected", lambda *_a: self._disconnected(b))
            if self._warm is not None:   # the first frame's cost, paid here and not inside a step
                with contextlib.suppress(Exception):
                    self.warm_s.append(await self._warm(b))
            return b

    def _disconnected(self, b):
        if not self._closing and b is self.browser:
            self.crashes.append(time.time())
            self.log(f"{self.side}: browser disconnected (crash #{len(self.crashes)})")

    def run(self, factory, timeout=None, on_cancel=None):
        """factory(browser) -> coroutine, run on this side's loop; blocks the calling (arm) thread."""
        import asyncio
        if self._closing:
            raise RuntimeError(f"{self.side} browser closed")

        async def go():
            browser = await self._ensure()
            return await asyncio.wait_for(factory(browser), timeout)
        n0 = len(self.crashes)
        fut = asyncio.run_coroutine_threadsafe(go(), self._loop)
        if on_cancel is not None:
            on_cancel(fut.cancel)
        try:
            out = fut.result()
        except BaseException:
            if len(self.crashes) > n0:
                raise BrowserCrashed(self.side, self.crashes[n0]) from None
            raise
        if len(self.crashes) > n0:
            raise BrowserCrashed(self.side, self.crashes[n0])
        return out

    def close(self, timeout=30):
        import asyncio
        if self._closing:
            return
        self._closing = True

        async def shut():
            if self._closer is not None:
                await self._closer()
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(shut(), self._loop).result(timeout)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


class SideBrowsers(dict):
    """side -> SideBrowser, created on first use; close() closes them all."""

    def __init__(self, launch=None, log=_log):
        super().__init__()
        self._launch, self._log = launch, log
        self._lock = threading.Lock()

    def get_side(self, side):
        with self._lock:
            if side not in self:
                self[side] = SideBrowser(side, launch=self._launch, log=self._log)
            return self[side]

    def close(self):
        for b in list(self.values()):
            b.close()


# ------------------------------------------------------------------ execution
@dataclasses.dataclass
class Result:
    status: str = "pending"      # ok | error | timeout | cancelled | skipped
    error: str = ""
    t0: float = 0.0
    t1: float = 0.0
    gpu: Optional[str] = None
    value: object = None

    def secs(self):
        return round(self.t1 - self.t0, 1) if self.t1 else None


class UnitCtx:
    """Handed to the executor: cancellation plumbing and the arm's deadline."""

    def __init__(self, arm, env, lease=None, deadline=None):
        self.arm, self.env, self.lease, self.deadline = arm, env, lease, deadline
        self.cancelled = threading.Event()
        self.reason = None             # "timeout" (watchdog) | "cancelled" (run torn down)
        self.cancel_at = None
        self._hooks = []
        self._lock = threading.Lock()

    def on_cancel(self, fn):
        with self._lock:
            if self.cancelled.is_set():
                fn()
            else:
                self._hooks.append(fn)

    def cancel(self, reason="cancelled"):
        with self._lock:
            self.reason = self.reason or reason
            self.cancelled.set()
            hooks, self._hooks = self._hooks, []
        for fn in hooks:
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass

    def remaining(self):
        return None if self.deadline is None else max(1.0, self.deadline - time.time())


def _status_of(e):
    import asyncio
    import subprocess
    if isinstance(e, (TimeoutError, asyncio.TimeoutError, subprocess.TimeoutExpired)):
        return "timeout"
    return "error"


def _live(pids):
    """pids still running; a killed child we never waited for is a zombie, not alive."""
    from studio_regress import plat
    out = set()
    for p in pids:
        try:
            if Path(f"/proc/{p}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z":
                out.add(p)
        except (OSError, IndexError):
            if not Path("/proc/self").is_dir() and plat.pid_alive(p):
                out.add(p)
    return out


def _reap(pids):
    for p in pids:
        with contextlib.suppress(OSError, ChildProcessError, AttributeError):
            os.waitpid(p, os.WNOHANG)


def sweep_children(grace_s=10):
    """Stop every process this one spawned that is still alive (Studios, Chromium, suites):
    SIGTERM, up to grace_s, then SIGKILL; direct children are reaped."""
    from studio_regress import plat
    me = os.getpid()
    left = sorted(_live(plat.descendants(me) - {me}))
    if not left:
        return []
    _log(f"stopping {len(left)} leftover child processes: {left[:12]}")
    if plat._is_windows():
        plat.stop_tree(None, grace_s=grace_s, extra=left)
        return left
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in _live(left):
            with contextlib.suppress(OSError):
                os.kill(p, sig)
        deadline = time.time() + (grace_s if sig == signal.SIGTERM else 5)
        while time.time() < deadline:
            _reap(left)
            if not _live(left):
                return left
            time.sleep(0.2)
    return left


class Scheduler:
    """Admit jobs in plan order (Core / external first among the ready ones: they are long and
    independent), subject to deps, slot FIFO, the concurrency cap and GPU leases."""

    def __init__(self, plan, execute, *, instance=None, book=None, cap=None, log=_log, poll_s=2.0,
                 gpu_wait_s=3600.0, sweep=sweep_children, clock=time.time, watchdog_grace_s=WATCHDOG_GRACE_S,
                 abandon_s=ABANDON_S, starve_s=STARVE_S, closers=(), gpu_jobs=None):
        self.plan, self.execute = plan, execute
        self.instance = instance or IsolatedInstance()
        self.book = book or LeaseBook(gpus=[])
        self.cap = cap or make_cap()
        self.log, self.poll_s, self.gpu_wait_s = log, poll_s, gpu_wait_s
        self.sweep, self.clock = sweep, clock
        self.watchdog_grace_s, self.abandon_s, self.starve_s = watchdog_grace_s, abandon_s, starve_s
        # --gpu-jobs: at most this many gpu-tier Studio arms at once (None: no limit). Their Studios
        # load models on the visible GPUs unpinned, so this bounds GPU memory and contention.
        self.gpu_jobs = gpu_jobs
        self.max_gpu_weight = 0
        self._cap_blocked = {}
        self.closers = list(closers)    # e.g. SideBrowsers.close: after the instance, before the sweep
        self.results = {a.id: Result() for a in plan.arms()}
        self.events = []                # (t, "start" | "end", job index): tests, report
        self.max_weight = 0
        self._wake = threading.Event()
        self._ctx = {}                  # arm id -> UnitCtx while running
        self._threads = {}              # job index -> [threads]
        self._grants = {}               # job index -> [(token, gpu)]
        self._pending_since = {}
        self._said = {}                 # job index -> last waiting message time
        self._abandoned = set()
        self._lock = threading.Lock()

    # -------------------------------------------------------------- readiness
    def _ready(self, j, pending, running):
        unfinished = pending | running
        if any(d in unfinished for d in j.deps):
            return False
        for s in j.slots:
            if any(o in unfinished and o < j.index and s in self.plan.jobs[o].slots for o in unfinished):
                return False
        return True

    def _take_gpus(self, j):
        """[(token, gpu, env_per_arm)] or None (wait)."""
        if not j.lease_gpu:
            return []
        got = []
        for _ in range(j.gpus):
            r = self.book.try_take(j.lease_gb(), exclusive=j.perf, what=f"studio_regress:{j.pair}"[:80],
                                   avoid=[g for _t, g, _e in got if g is not None])
            if r is None:
                for t, g, _e in got:
                    self.book.give(t, g)
                return None
            token, g, total = r
            got.append((token, g, gpu_pack.env_for(g, j.gb, total) if g is not None else {}))
        return got

    # -------------------------------------------------------------- run
    def run(self):
        jobs = {j.index: j for j in self.plan.jobs}
        pending, running, done = set(jobs), set(), set()
        self.instance.prepare(self.plan)
        old = None
        if threading.current_thread() is threading.main_thread():
            def _term(signum, frame):
                raise KeyboardInterrupt(f"signal {signum}")
            with contextlib.suppress(ValueError):
                old = signal.signal(signal.SIGTERM, _term)
        ok = False
        try:
            while pending or running:
                self._admit(jobs, pending, running)
                self._wake.wait(self.poll_s)
                self._wake.clear()
                for i in list(running):
                    if self._job_done(i):
                        running.discard(i)
                        done.add(i)
                        self._release(i)
                        self.events.append((self.clock(), "end", i))
                self._watchdog(jobs, running)
            ok = True
        finally:
            if old is not None:
                with contextlib.suppress(ValueError):
                    signal.signal(signal.SIGTERM, old)
            if not ok:
                self._teardown(jobs, running)
            else:
                self.instance.close()
                self._close_others()
                self.book.release_all()
        return self.results

    def _admit(self, jobs, pending, running):
        order = sorted(pending, key=lambda i: (0 if jobs[i].kind in ("core", "external") else 1, i))
        # The job the cap has kept waiting longest (past starve_s) holds its room: nothing else starts
        # unless it leaves that much free, so a two-arm pair is not overtaken forever by single arms.
        now = self.clock()
        self._cap_blocked = {i: t for i, t in self._cap_blocked.items() if i in pending}
        starving = sorted((t, i) for i, t in self._cap_blocked.items() if now - t > self.starve_s)
        hold = starving[0][1] if starving else None
        for i in order:
            j = jobs[i]
            if not self._ready(j, pending, running):
                continue
            in_use = sum(jobs[r].weight for r in running)
            if j.gpu_tier and self.gpu_jobs:
                g_use = sum(jobs[r].weight for r in running if jobs[r].gpu_tier)
                if g_use and g_use + j.weight > self.gpu_jobs:
                    continue
            cap = self.cap()
            reserve = jobs[hold].weight if hold is not None and hold != i else 0
            # idle: the first ready job always starts (weight > cap included), unless one is held
            if (running or reserve) and in_use + j.weight + reserve > cap:
                if in_use + j.weight > cap:
                    self._cap_blocked.setdefault(i, now)
                continue
            self._cap_blocked.pop(i, None)
            grants = self._take_gpus(j)
            if grants is None:
                t = self._pending_since.setdefault(i, self.clock())
                if self.clock() - self._said.get(i, t - 60) >= 60 or self.clock() - t > self.gpu_wait_s:
                    self._said[i] = self.clock()   # at most once a minute, not silent for gpu_wait_s
                    self.log(f"{j.id} waiting {self.clock() - t:.0f}s for {j.lease_gb():g} GB"
                             f"{' exclusive' if j.perf else ''}: {getattr(self.book, 'why', lambda: '?')()}")
                if self.clock() - t > self.gpu_wait_s:
                    pending.discard(i)
                    for a in j.arms:
                        self._set(a.id, status="error", error=f"no GPU could take {j.lease_gb():g} GB "
                                  f"within {self.gpu_wait_s:.0f}s", t0=self.clock(), t1=self.clock())
                        self._finish_instance(a)
                continue
            pending.discard(i)
            running.add(i)
            self._grants[i] = [(t, g) for t, g, _e in grants]
            self.events.append((self.clock(), "start", i))
            self.max_weight = max(self.max_weight, in_use + j.weight)
            if j.gpu_tier:
                self.max_gpu_weight = max(self.max_gpu_weight, j.weight + sum(
                    jobs[r].weight for r in running if r != i and jobs[r].gpu_tier))
            self._start(j, grants)

    def _set(self, aid, **kw):
        with self._lock:
            r = self.results[aid]
            for k, v in kw.items():
                setattr(r, k, v)

    def _finish_instance(self, arm):
        if arm.kind in STUDIO_KINDS:
            try:
                self.instance.finish(arm.side, arm)
            except Exception as e:  # noqa: BLE001
                self.log(f"{arm.id}: instance finish: {e}")

    def _run_arm(self, arm, env, lease, gpu):
        if arm.id in self._abandoned:
            return
        dl = self.clock() + arm.timeout_s if arm.timeout_s else None
        ctx = UnitCtx(arm, env, lease=lease, deadline=dl)
        self._ctx[arm.id] = ctx
        self._set(arm.id, status="running", t0=self.clock(), gpu=gpu)
        try:
            val = self.execute(arm, ctx)
            if arm.id not in self._abandoned:
                self._set(arm.id, status="skipped" if val == "skipped" else "ok", value=val)
        except BaseException as e:  # noqa: BLE001 - one arm never takes the run down
            st = ctx.reason if ctx.cancelled.is_set() and ctx.reason else _status_of(e)
            if arm.id not in self._abandoned:
                self._set(arm.id, status=st, error=f"{type(e).__name__}: {e}"[:800])
            self.log(f"{arm.id}: {st}: {type(e).__name__}: {e}")
        finally:
            if arm.id not in self._abandoned:   # the watchdog already closed the books on it
                self._set(arm.id, t1=self.clock())
                self._ctx.pop(arm.id, None)
                self._finish_instance(arm)
            self._wake.set()

    def _start(self, j, grants):
        gpu_env = [e for _t, _g, e in grants]
        gpus = [g for _t, g, _e in grants]
        env0 = gpu_env[0] if gpu_env else {}
        lease = FixedLease([(g, e) for g, e in zip(gpus, gpu_env)]) if j.kind in ("external", "core") and gpus else None
        g0 = gpus[0] if gpus else None
        if j.mode == "together":
            ths = [threading.Thread(target=self._run_arm, args=(a, env0, lease, g0), daemon=True,
                                    name=f"sr-{a.id}") for a in j.arms]
        else:
            def seq(arms=j.arms):
                for a in arms:
                    if self._cancelled_run:
                        self._set(a.id, status="cancelled", error="run cancelled", t0=self.clock(), t1=self.clock())
                        self._finish_instance(a)
                        continue
                    self._run_arm(a, env0, lease, g0)
            ths = [threading.Thread(target=seq, daemon=True, name=f"sr-{j.id}")]
        self._threads[j.index] = ths
        for t in ths:
            t.start()

    _cancelled_run = False

    def _job_done(self, i):
        if all(not t.is_alive() for t in self._threads.get(i, [])):
            return True
        # a thread the watchdog gave up on is still alive: the job is over once every arm is settled
        arms = self.plan.jobs[i].arms
        return any(a.id in self._abandoned for a in arms) and all(
            self.results[a.id].status not in ("pending", "running") for a in arms)

    def _release(self, i):
        for t, g in self._grants.pop(i, []):
            self.book.give(t, g)

    def _watchdog(self, jobs, running):
        now = self.clock()
        for i in list(running):
            for a in jobs[i].arms:
                r, ctx = self.results[a.id], self._ctx.get(a.id)
                if ctx is None or r.status != "running" or ctx.deadline is None:
                    continue
                over = now - ctx.deadline
                if over > self.watchdog_grace_s and not ctx.cancelled.is_set():
                    self.log(f"{a.id}: {over:.0f}s past its {a.timeout_s:.0f}s budget: cancelling")
                    ctx.cancel_at = now
                    ctx.cancel("timeout")
                elif ctx.cancelled.is_set() and now - (ctx.cancel_at or now) > self.abandon_s:
                    # The thread ignores cancellation: give up on it (it is a daemon thread) and on
                    # the arms queued behind it in the same worker, so the run still finishes.
                    self.log(f"{a.id}: did not stop after cancel: abandoned (timeout)")
                    self._abandon(a, f"over budget {a.timeout_s:.0f}s; abandoned", "timeout", now)
                    later = False
                    for b in jobs[i].arms:
                        later = later or b.id == a.id
                        if later and b.id != a.id and self.results[b.id].status in ("pending", "running"):
                            self._abandon(b, f"not run: {a.id} was abandoned", "cancelled", now)

    def _abandon(self, arm, error, status, now):
        self._abandoned.add(arm.id)
        self._set(arm.id, status=status, error=error, t1=now, t0=self.results[arm.id].t0 or now)
        ctx = self._ctx.pop(arm.id, None)
        if ctx is not None:
            ctx.cancel(status)
        self._finish_instance(arm)

    def _close_others(self):
        for fn in self.closers:
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                self.log(f"close: {e}")

    def _teardown(self, jobs, running):
        self.log(f"teardown: cancelling {len(running)} running jobs")
        self._cancelled_run = True
        for ctx in list(self._ctx.values()):
            ctx.cancel("cancelled")
        try:
            self.instance.close()
        except Exception as e:  # noqa: BLE001
            self.log(f"instance close: {e}")
        deadline = time.time() + 30
        for i in running:
            for t in self._threads.get(i, []):
                t.join(timeout=max(0.1, deadline - time.time()))
        for aid, r in self.results.items():
            if r.status in ("pending", "running"):
                r.status, r.error = "cancelled", r.error or "run cancelled"
        self._close_others()
        try:
            self.sweep()
        except Exception as e:  # noqa: BLE001
            self.log(f"sweep: {e}")
        self.book.release_all()


# ------------------------------------------------------------------ evidence for failed arms
def void_missing_facts(root, side, journeys, names, reason, status="unit_void"):
    """Write a `unit_void` facts file for every step of `names` that has none on `side`
    (diff.compare_step turns it into VOID with the reason), so a crashed arm is VOID for its own
    steps only and never shows up as a one-sided DIVERGED UI change. Returns the count written."""
    n = 0
    for name in names:
        if name not in journeys:
            continue
        d = Path(root) / side / name
        for st in journeys[name][0].steps:
            f = d / f"{st.id}.facts.json"
            if f.exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"_step": st.id, "_status": status, "_error": reason[:500]}, indent=1))
            n += 1
    return n


def void_facts_since(root, side, journeys, names, reason, since):
    """A browser crash makes the steps it interrupted look `failed`; those written at or after the
    crash (and not ok) become unit_void. Steps that passed before it stay valid evidence."""
    n = 0
    for name in names:
        if name not in journeys:
            continue
        for st in journeys[name][0].steps:
            f = Path(root) / side / name / f"{st.id}.facts.json"
            try:
                facts = json.loads(f.read_text())
                if facts.get("_status") == "ok" or f.stat().st_mtime < since - 1:
                    continue
            except (OSError, ValueError):
                continue
            f.write_text(json.dumps({"_step": st.id, "_status": "unit_void", "_error": reason[:500],
                                     "_was": facts.get("_status")}, indent=1))
            n += 1
    return n


def summary(plan, results, t0=None, t1=None):
    """Plan-ordered, completion-order-independent record of what ran where (report.json)."""
    rows = []
    for j in plan.jobs:
        for a in j.arms:
            r = results.get(a.id) or Result()
            rows.append({"job": j.index, "arm": a.id, "kind": a.kind, "mode": j.mode, "status": r.status,
                         "s": r.secs(), "gpu": r.gpu, "error": r.error or None,
                         "start": round(r.t0 - t0, 1) if (t0 and r.t0) else None})
    out = {"mode": plan.mode, "units": rows}
    if t0 and t1:
        out["wall_s"] = round(t1 - t0, 1)
    return out
