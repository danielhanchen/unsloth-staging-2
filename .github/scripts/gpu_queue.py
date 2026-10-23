"""Host-wide GPU queue: VRAM-packed shared leases, exclusive leases for timing, predicted waits, offload.

    python gpu_queue.py run --gb 12 -- python train.py            # shared: packed by VRAM
    python gpu_queue.py run --gb 40 --exclusive --est-min 30 -- python bench.py   # timing: one remote card
    python gpu_queue.py run --gb 40 --exclusive --b200 --est-min 30 -- python bench.py  # timing: local B200
    python gpu_queue.py run --gb 8 --dtype bf16 --offload auto --est-min 20 -- python check.py
    python gpu_queue.py status | reclaim
    python gpu_queue.py estimate --model unsloth/Llama-3.2-1B-Instruct --load 4bit --mode lora

One view for every user and workspace: <lock dir>/gpu<N>.json (flock'd holders, read-modify-written
under gpu<N>.lock) plus <lock dir>/switchboard/{queue/*.json, mem_cache.json}. The lock dir is the
0777 shared dir studio_regress already uses, so studio_regress units, gpu_pool_runner.sh jobs and
ad-hoc `run`s all see each other. The launcher's per-workspace CUDA_VISIBLE_DEVICES mask is only a
home preference: a session with any GPU leases from every host GPU ($GPU_QUEUE_GPUS / $STUDIO_REGRESS_GPUS:
`all` default, `mask`, or a list).

--no-local-gpu sessions (launcher: UNSLOTH_NO_LOCAL_GPU=1, empty mask): `run` keeps a job local only
when it is small and short (gb <= SMALL_GB, --est-min <= SMALL_MIN, not --exclusive); anything else
goes to Colab / Kaggle at once (offload force, no local lease) and a job that cannot ship exits 3.

Timing (--exclusive) needs --est-min and runs on Colab / Kaggle the same way (every arm on one remote
card) unless --b200 asks for a local B200; at most MAX_EXCL (GPU_QUEUE_MAX_EXCL, 2; 0 = no cap) cards
are exclusive host-wide at once, so whole local cards stay in the shared pool.

Holder = {"gb", "exclusive", "what", "t", "start", "boot", "est_end", "user", "ws"} keyed "<pid>" (one
per process, old callers) or "<pid>:<n>" (one per job). A holder whose pid is gone, whose boot id
differs, or whose /proc start time differs (PID reuse) is reclaimed on the next touch. A corrupt
gpu<N>.json is renamed aside and the GPU takes nothing for QUARANTINE_S (its holders are unknown).

Admission (shared): room = min(total - leased, free - pending) - headroom, where pending is what
holders granted in the last PENDING_S have reserved but not yet allocated. Placement spreads (the
least busy card over the last 5 s, then the most room) so local GPUs all get used. A waiting exclusive
ticket keeps its card: one with a drain target through drained(), one without (the MAX_EXCL cap holds
it) through protected_cards(), an empty card shared work takes only when nothing else fits. Many
shared jobs still pack onto one card by VRAM when the others are full. Exclusive: no other holder, and foreign (unleased) use either
under FOREIGN_GB or idle: at most EXCL_FOREIGN_FRAC of the card AND utilization at most FOREIGN_UTIL_PCT
over the last 5 s (gpu_sample's window, not one instantaneous reading). foreign_blocks() is the one
rule; every exclusive path (acquire, gpu_pool_lease, studio_regress LeaseBook) applies it.

Queue: a waiter writes switchboard/queue/<id>.json. Priority is submit time, exclusive tickets
EXCL_PRIORITY_S earlier. An exclusive ticket (or a shared one waiting over STARVE_S) names a drain GPU
(soonest to fit it); tickets behind it place nothing new there, so big and exclusive jobs are not
starved by a stream of small ones. predict_wait() simulates the same policy. Backfill: a shared job
with its own --est-min may still take a drained GPU when it fits now and, stretched by
BACKFILL_MARGIN, ends before that GPU frees for the drainer (drain_ready_at), so the drainer starts
no later. A request from inside a running lease (a script collecting GPUs one at a time) keeps the
queue position of the lease that started it, and never drains a GPU its own ancestor holds: a lease
waiting on its child must not sit idle behind newer tickets. A unit placed under a parent's lease on
the same GPU (UnitLeases: regression/run.py inside the switchboard) skips that GPU's drain entirely.

Offload (local first): only after really waiting OFFLOAD_MIN_WAIT_S with >= OFFLOAD_PRED_S still
predicted, or after OFFLOAD_S regardless, `--offload auto` starts a remote attempt (cloud_pool.dispatch)
in the background while the local ticket stays queued. A local GPU freeing first cancels the attempt
if it has not started running remotely; once it runs remotely it wins (never twice). A failed attempt
(credits, capacity, no fit) keeps the job waiting locally, re-trying remote every REMOTE_RETRY_S.
"""

from __future__ import annotations

import argparse
import contextlib
import getpass
import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from studio_regress import plat  # noqa: E402
import gpu_sample  # noqa: E402
import nvsmi  # noqa: E402

HEADROOM_GB = 4.0
PENDING_S = 120.0  # a fresh grant may not have allocated yet: count its full reservation
QUARANTINE_S = 300.0
FOREIGN_GB = float(os.environ.get("GPU_QUEUE_FOREIGN_GB", "1.0"))
# Idle CUDA contexts (a loaded-but-idle model, a notebook kernel, a pytest between tests) hold GBs on
# most cards of a shared host; memory alone would starve every exclusive request. Foreign memory blocks
# timing work while the card was busy in the last 5 s (max utilization above FOREIGN_UTIL_PCT), when
# utilization is unknown, or when it holds more than EXCL_FOREIGN_FRAC of the card.
FOREIGN_UTIL_PCT = float(os.environ.get("GPU_QUEUE_FOREIGN_UTIL_PCT", "5"))
EXCL_FOREIGN_FRAC = float(os.environ.get("GPU_QUEUE_EXCL_FOREIGN_FRAC", "0.25"))
FOREIGN_IGNORE_GB = (
    float(os.environ.get("GPU_QUEUE_FOREIGN_IGNORE_MIB", "1024")) / 1024
)  # per process
# Local first: remote only after really waiting. Offload once waited >= OFFLOAD_MIN_WAIT_S with a
# predicted remaining wait >= OFFLOAD_PRED_S, or unconditionally at waited >= OFFLOAD_S; never on a
# prediction alone at t=0. A failed remote attempt is retried at most every REMOTE_RETRY_S.
OFFLOAD_S = float(
    os.environ.get("GPU_QUEUE_OFFLOAD_S", "1200")
)  # 20 min, as studio_regress offload
OFFLOAD_MIN_WAIT_S = float(os.environ.get("GPU_QUEUE_OFFLOAD_MIN_WAIT_S", "600"))
OFFLOAD_PRED_S = float(os.environ.get("GPU_QUEUE_OFFLOAD_PRED_S", "1200"))
REMOTE_RETRY_S = float(os.environ.get("GPU_QUEUE_REMOTE_RETRY_S", "600"))
# A Colab worker already running (BUSY shared with room, or IDLE) costs no new VM: a shared job may
# join it sooner, after OFFLOAD_WARM_S waited with >= OFFLOAD_WARM_PRED_S still predicted.
OFFLOAD_WARM_S = float(os.environ.get("GPU_QUEUE_OFFLOAD_WARM_S", "300"))
OFFLOAD_WARM_PRED_S = float(os.environ.get("GPU_QUEUE_OFFLOAD_WARM_PRED_S", "600"))
WARM_CACHE_S = 15.0
EXCL_PRIORITY_S = 600.0
# Exclusive leases host-wide at once (GPU_QUEUE_MAX_EXCL; 0 = no cap): an exclusive lease holds a
# whole card whatever its job uses (one Flux timing run held 179 GB for a 22 GB peak), so at most
# this many cards are ever out of the shared pool.
MAX_EXCL = int(os.environ.get("GPU_QUEUE_MAX_EXCL", "2") or 0)
STARVE_S = 600.0
# --no-local-gpu sessions: what may still take a local GPU (quick accuracy checks)
SMALL_GB = float(os.environ.get("GPU_QUEUE_SMALL_GB", "16"))
SMALL_MIN = float(os.environ.get("GPU_QUEUE_SMALL_MIN", "15"))
EXIT_REMOTE_ONLY = 3


def no_local_gpu(env = None):
    return (os.environ if env is None else env).get("UNSLOTH_NO_LOCAL_GPU", "") == "1"


def local_refusal(gb, exclusive, est_s):
    """Why a --no-local-gpu session may not run this job locally ('' = small enough to stay)."""
    if exclusive:
        return "exclusive (timing) work"
    if gb > SMALL_GB:
        return f"{gb:g} GB > {SMALL_GB:g} GB (GPU_QUEUE_SMALL_GB)"
    if est_s and est_s > SMALL_MIN * 60:
        return f"~{est_s / 60:.0f} min > {SMALL_MIN:g} min (GPU_QUEUE_SMALL_MIN)"
    return ""


OVERDUE_S = 300.0  # a holder past its est_end is assumed to finish this much later
BACKFILL_MARGIN = (
    1.25  # a backfilled job's est_s, stretched, must end before the drainer's GPU frees
)
DEFAULT_EST_S = {False: 1800.0, True: 3600.0}  # unknown duration, by exclusive
# Observed admission: a shared lease running >= OBSERVED_AFTER_S whose measured peak (published by its
# `gpu_queue.py run` sampler) stayed under OBSERVED_FRAC of its reservation counts as
# max(peak * 1.25, peak + 2 GB) toward the card's leased total. The physical free-memory check still
# gates every grant. A lease that grows past that size counts in full again for the rest of its life.
# GPU_QUEUE_OBSERVED_ADMIT=0 turns it off.
OBSERVED_ADMIT = os.environ.get("GPU_QUEUE_OBSERVED_ADMIT", "1") != "0"
# 2 min, not 10: most jobs end inside 10 min, so their reservation (a guess, typically 16-32 GB for
# a job peaking at 1-5 GB) was counted in full for their whole life and cards sat at ~12% use
# (2026-10-16: 827 GB reserved, ~101 GB measured across 8 B200s).
OBSERVED_AFTER_S = float(os.environ.get("GPU_QUEUE_OBSERVED_AFTER_S", "120"))
OBSERVED_FRAC = float(os.environ.get("GPU_QUEUE_OBSERVED_FRAC", "0.5"))
OBSERVED_STALE_S = 120.0  # a peak not refreshed this long (sampler gone) counts the lease in full
PUBLISH_S = 30.0  # the sampler refreshes its lease's peak at least this often

_SHARED_LOCK_DIR = Path("/mnt/disks/unslothai/shared/studio-regress-locks")
_FALLBACK_LOCK_DIR = Path(tempfile.gettempdir()) / "unsloth-studio-regress-locks"
QUERY_ERROR = {}  # the last nvidia-smi failure, for waiting messages
_HOST_GPUS = []


def _log(msg):
    print(f"[gpu_queue] {msg}", file = sys.stderr, flush = True)


# ------------------------------------------------------------------ paths
def resolve_lock_dir(shared = None, fallback = None):
    env = os.environ.get("GPU_QUEUE_LOCK_DIR") or os.environ.get("STUDIO_REGRESS_LOCK_DIR")
    if env:
        return Path(env)
    shared = Path(shared or _SHARED_LOCK_DIR)
    fallback = Path(fallback or _FALLBACK_LOCK_DIR)
    # The existing shared dir first: it is 0777 but its parent belongs to whoever made it, so testing
    # only the parent sent every other user to /tmp and split the leases.
    if shared.is_dir():
        return shared if os.access(shared, os.W_OK | os.X_OK) else fallback
    return shared if os.access(shared.parent, os.W_OK) else fallback


def lock_dir():
    return resolve_lock_dir()


def state_dir(ld = None):
    d = Path(ld or lock_dir()) / "switchboard"
    _mkdir_shared(d)
    return d


def hf_read_token():
    """Read-only public-repo HF token for downloads (never logged)."""
    tok = os.environ.get("SWITCHBOARD_HF_READ_TOKEN")
    if tok:
        return tok.strip()
    try:
        return (Path(lock_dir()) / "switchboard" / "hf_read_token").read_text().strip() or None
    except OSError:
        return None


def _open_shared(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        os.fchmod(fd, 0o666)
    except (OSError, AttributeError):
        pass  # another user's file: already 0666
    return os.fdopen(fd, "r+")


def _mkdir_shared(d):
    # 0777, NOT sticky: in a sticky dir one user could not replace another user's gpu<N>.json
    d = Path(d)
    d.mkdir(parents = True, exist_ok = True)
    try:
        os.chmod(d, 0o777)
    except OSError:
        pass


def _write_shared(
    path,
    obj,
    indent = 1,
):
    tmp = path.with_suffix(f".tmp{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(json.dumps(obj, indent = indent, separators = None if indent else (",", ":")))
    try:
        os.chmod(tmp, 0o666)
    except OSError:
        pass
    tmp.replace(path)


# ------------------------------------------------------------------ identity
_BOOT = []


def boot_id():
    if not _BOOT:
        try:
            _BOOT.append(Path("/proc/sys/kernel/random/boot_id").read_text().strip())
        except OSError:
            _BOOT.append("")
    return _BOOT[0]


def proc_start(pid):
    """/proc/<pid>/stat starttime (clock ticks since boot), None when unreadable."""
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text()
        return int(raw[raw.rindex(")") + 2 :].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def owner(pid = None):
    pid = int(pid or os.getpid())
    return {"start": proc_start(pid), "boot": boot_id()}


def _key_pid(key):
    try:
        return int(str(key).split(":")[0])
    except ValueError:
        return 0


def holder_alive(key, h):
    pid = _key_pid(key)
    if not plat.pid_alive(pid):
        return False
    if h.get("boot") and boot_id() and h["boot"] != boot_id():
        return False
    if h.get("start") is not None:
        cur = proc_start(pid)
        if cur is not None and cur != h["start"]:
            return False  # PID reused by another process
    return True


# ------------------------------------------------------------------ GPUs
def visible_gpus(env = None):
    env = os.environ if env is None else env
    raw = (env.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not raw or raw in ("-1", "none", "NoDevFiles"):
        return []
    return [g.strip() for g in raw.split(",") if g.strip()]


def host_gpus(cache = None):
    """Every GPU index the host-wide reading lists (cached after the first good answer), [] when it cannot say."""
    cache = _HOST_GPUS if cache is None else cache
    if not cache:
        st = gpu_sample.read(max_age = 600)
        ids = sorted(st.get("gpus", {}), key = lambda g: int(g) if g.isdigit() else 0)
        if not ids:
            return []
        cache.extend(ids)
    return list(cache)


def pool_gpus(env = None, host = None):
    """GPUs this process may lease: every host GPU for a session with a GPU mask, none for a CPU one
    (CUDA_VISIBLE_DEVICES set but empty / -1). Unset means CUDA sees every GPU, so the whole host
    (e.g. a plain login shell of another unix user): running such a job unpinned would bypass the queue."""
    env = os.environ if env is None else env
    mine = visible_gpus(env)
    mode = (env.get("GPU_QUEUE_GPUS") or env.get("STUDIO_REGRESS_GPUS") or "all").strip()
    if "CUDA_VISIBLE_DEVICES" not in env or (no_local_gpu(env) and not mine):
        mine = (host or host_gpus)()  # --no-local-gpu: no mask, yet small jobs lease host GPUs
    if not mine or mode == "mask":
        return mine
    if mode != "all":
        return [g.strip() for g in mode.split(",") if g.strip()]
    return (host or host_gpus)() or mine


def query_gpus(ids):
    """{id: {"total_gb", "free_gb", "util", "util_max_5s", "util_mean_5s"}} from the host-wide reading
    (gpu_sample: one refresher for every caller). A reading too old to admit on (gpu_sample.admissible)
    reads as 0 GB free -- blind, never a guess -- and is recorded in QUERY_ERROR; util is None when
    the backend does not report it."""
    if not ids:
        return {}
    st = gpu_sample.read()
    if not gpu_sample.admissible(st):
        QUERY_ERROR.update(error = st.get("error") or "no GPU reading", at = time.time())
        return {i: {"total_gb": 0.0, "free_gb": 0.0} for i in ids}
    if st.get("stale"):
        QUERY_ERROR.update(
            error = f"{st.get('error')} (using a {gpu_sample.age(st):.0f}s old reading)",
            at = time.time(),
        )
    else:
        QUERY_ERROR.clear()
    res = {}
    for i in ids:
        d = st["gpus"].get(str(i))
        if d is None:
            continue
        res[str(i)] = {
            "total_gb": d["total_gb"],
            "free_gb": d["free_gb"],
            "util": d.get("util_now"),
            "util_max_5s": d.get("util_max_5s"),
            "util_mean_5s": d.get("util_mean_5s"),
        }
    return res


def foreign_blocks(s):
    """True when foreign (unleased) GPU use rules out timing work on this card: over FOREIGN_GB and
    either over EXCL_FOREIGN_FRAC of the card or busy in the 5 s window (util_max_5s, else the one
    reading `util`) or of unknown utilization."""
    foreign = float(s.get("foreign_gb") or 0.0)
    if foreign <= FOREIGN_GB:
        return False
    total = float(s.get("total_gb") or 0.0)
    if total and foreign > EXCL_FOREIGN_FRAC * total:
        return True
    util = s.get("util_max_5s", s.get("util"))
    return util is None or util > FOREIGN_UTIL_PCT


def foreign_at(s):
    """What shared the card when an exclusive lease was granted (recorded on the lease)."""
    return {
        "gb": round(float(s.get("foreign_gb") or 0.0), 2),
        "util_max_5s": s.get("util_max_5s", s.get("util")),
        "util_mean_5s": s.get("util_mean_5s"),
    }


def exclusive_state(
    gpus,
    ld = None,
    info = None,
    apps = None,
):
    """{gpu: state} for exclusive admission: gather_state with the process view, so foreign use is
    what runs outside every holder's process tree. The one assessment every exclusive path uses."""
    info = info if info is not None else query_gpus(gpus)
    return gather_state(gpus, ld, info = info, apps = apps if apps is not None else query_apps())[
        "gpus"
    ]


def query_apps():
    """{gpu index: [(pid, used_gb)]} of compute processes from the host-wide reading; {} when it cannot say.
    A card with no listed process is left out: an empty list is also what a failed process query
    gives, so gather_state counts its unleased use as foreign instead of trusting the silence."""
    st = gpu_sample.read()
    if not gpu_sample.admissible(st):
        return {}
    return {
        g: [(int(p[0]), float(p[2])) for p in d["procs"]]
        for g, d in st["gpus"].items()
        if d.get("procs")
    }


# ------------------------------------------------------------------ leases
@contextlib.contextmanager
def _locked(gpu, ld = None):
    ld = Path(ld or lock_dir())
    _mkdir_shared(ld)
    with _open_shared(ld / f"gpu{gpu}.lock") as fh:
        plat.lock(fh)
        try:
            path = ld / f"gpu{gpu}.json"
            state = {}
            if path.exists():
                try:
                    state = json.loads(path.read_text() or "{}")
                    if not isinstance(state, dict) or not isinstance(
                        state.get("holders", {}), dict
                    ):
                        raise ValueError("not a lease file")
                except ValueError:
                    # unknown holders: never read as empty. Set aside and refuse admission for a while;
                    # live free memory still covers whoever is on the card meanwhile.
                    with contextlib.suppress(OSError):
                        path.replace(ld / f"gpu{gpu}.corrupt.{int(time.time())}")
                    state = {"quarantined_until": time.time() + QUARANTINE_S}
                    _log(
                        f"gpu{gpu}.json was corrupt; set aside, GPU {gpu} quarantined {QUARANTINE_S:.0f}s"
                    )
            state.setdefault("holders", {})
            state["holders"] = {k: h for k, h in state["holders"].items() if holder_alive(k, h)}
            yield state
            _write_shared(path, state)
        finally:
            plat.unlock(fh)


def leases(gpu, lock_dir = None):
    with _locked(gpu, lock_dir) as st:
        return dict(st["holders"])


def quarantined(st, now = None):
    return (st.get("quarantined_until") or 0) > (now or time.time())


def _pending_gb(
    holders,
    used_by = None,
    now = None,
):
    """Reserved but (likely) not yet allocated: young grants without an observed usage count in full."""
    now = now or time.time()
    used_by = used_by or {}
    tot = 0.0
    for k, h in holders.items():
        if k in used_by:
            tot += max(float(h["gb"]) - used_by[k], 0.0)
        elif now - float(h.get("t", 0)) < PENDING_S:
            tot += float(h["gb"])
    return tot


def admit_gb(h, now = None):
    """What lease `h` counts toward its card's leased total: its reservation, or, for a shared lease
    old enough whose fresh measured peak stayed under OBSERVED_FRAC of it, that peak plus margin."""
    gb = float(h.get("gb", 0))
    if (
        not OBSERVED_ADMIT
        or h.get("exclusive")
        or h.get("no_discount")
        or h.get("discount_revoked")
        or h.get("peak_gb") is None
        or h.get("peak_t") is None
    ):
        return gb
    now = now or time.time()
    peak = float(h["peak_gb"])
    if (
        now - float(h["peak_t"]) > OBSERVED_STALE_S
        or now - float(h.get("t", now)) < OBSERVED_AFTER_S
    ):
        return gb
    if peak >= OBSERVED_FRAC * gb:
        return gb
    return min(gb, max(peak * 1.25, peak + 2.0))


def publish_peak(
    gpu,
    key,
    peak_gb,
    lock_dir = None,
    now = None,
):
    """Record the lease's measured group peak. A lease admitted others at a discounted size and has
    since grown past it: revoke the discount for good (counts in full; no new admissions until it fits)."""
    now = now or time.time()
    with _locked(gpu, lock_dir) as st:
        h = st["holders"].get(str(key))
        if h is None:
            return None
        before = admit_gb(h, now)
        h["peak_gb"] = round(float(peak_gb), 2)
        h["peak_t"] = round(now, 1)
        if before < float(h["gb"]) and float(peak_gb) > before:
            h["discount_revoked"] = True
            _log(
                f"lease {key} on GPU {gpu}: peak {peak_gb:.1f} GB grew past its admitted {before:.1f} GB; "
                f"counting the full {float(h['gb']):g} GB again"
            )
        return dict(h)


def exclusive_elsewhere(
    gpu,
    lock_dir = None,
    key = None,
):
    """Cards other than `gpu` held exclusively by a live lease other than `key` (one multi-GPU job
    takes all its cards under one key, so it counts once against the cap, not against itself).
    Read without their locks: the caller holds exclusive.lock, which every new exclusive grant takes."""
    n = 0
    for f in Path(lock_dir or globals()["lock_dir"]()).glob("gpu*.json"):
        g = f.stem[3:]
        if not g.isdigit() or g == str(gpu):
            continue
        try:
            st = json.loads(f.read_text() or "{}")
        except (OSError, ValueError):
            continue
        n += any(
            isinstance(h, dict) and h.get("exclusive") and k != key and holder_alive(k, h)
            for k, h in (st.get("holders") or {}).items()
        )
    return n


def try_acquire(
    gpu,
    gb,
    exclusive = False,
    what = "",
    pid = None,
    total_gb = None,
    free_gb = None,
    headroom = HEADROOM_GB,
    lock_dir = None,
    key = None,
    est_s = None,
    used_by = None,
    foreign_gb = None,
    foreign = None,
    backfill = False,
    queued_at = None,
    nested = False,
    discount = True,
):
    """try_acquire_card under the host-wide cap: a NEW exclusive lease waits while MAX_EXCL other
    cards are exclusive (exclusive.lock serialises those grants, so two cannot both pass)."""
    kw = dict(
        exclusive = exclusive,
        what = what,
        pid = pid,
        total_gb = total_gb,
        free_gb = free_gb,
        headroom = headroom,
        lock_dir = lock_dir,
        key = key,
        est_s = est_s,
        used_by = used_by,
        foreign_gb = foreign_gb,
        foreign = foreign,
        backfill = backfill,
        queued_at = queued_at,
        nested = nested,
        discount = discount,
    )
    if not exclusive or not MAX_EXCL:
        return try_acquire_card(gpu, gb, **kw)
    ld = Path(lock_dir or globals()["lock_dir"]())
    _mkdir_shared(ld)
    with _open_shared(ld / "exclusive.lock") as fh:
        plat.lock(fh)
        try:
            if exclusive_elsewhere(gpu, ld, key) >= MAX_EXCL:
                return False
            return try_acquire_card(gpu, gb, **kw)
        finally:
            plat.unlock(fh)


def try_acquire_card(
    gpu,
    gb,
    exclusive = False,
    what = "",
    pid = None,
    total_gb = None,
    free_gb = None,
    headroom = HEADROOM_GB,
    lock_dir = None,
    key = None,
    est_s = None,
    used_by = None,
    foreign_gb = None,
    foreign = None,
    backfill = False,
    queued_at = None,
    nested = False,
    discount = True,
):
    """Reserve `gb` on `gpu` under `key` (default the pid); False if it does not fit / exclusivity
    conflicts / the GPU is quarantined. Re-acquiring an existing key replaces its entry (resize).
    Exclusive: `foreign` (a gather_state entry) is judged by foreign_blocks() and recorded on the lease
    as foreign_at_start; the older `foreign_gb` (blocking above FOREIGN_GB) still works. `backfill`:
    the caller checked (backfill_ok) that this shared job ends before a drained GPU frees for its
    drainer. `queued_at`: the ticket's submit time, which a request nested in this lease inherits.
    `nested`: an ancestor of the caller already holds `gpu`, so no drain of it can finish before this
    request does; waiting on the drain would only idle the parent's lease.
    `discount` False: the lease always counts its full reservation (see admit_gb)."""
    pid = int(pid or os.getpid())
    key = str(key or pid)
    now = time.time()
    with _locked(gpu, lock_dir) as st:
        if quarantined(st, now):
            return False
        prev = st["holders"].get(key)
        # A GPU a waiting ticket drains takes no NEW or GROWING shared work from anyone else, whichever
        # tool asks (studio_regress, gpu_pool_runner, kld/regression runners): otherwise only
        # gpu_queue.acquire honoured the drain and timing work starved behind the others.
        if (
            not exclusive
            and not backfill
            and not nested
            and (prev is None or gb > float(prev.get("gb", 0)))
            and drained(gpu, lock_dir, pid, before = queued_at)
        ):
            return False
        holders = {k: h for k, h in st["holders"].items() if k != key}
        if exclusive and holders:
            return False
        if exclusive and foreign_gb is not None and foreign_gb > FOREIGN_GB:
            return False
        if exclusive and foreign is not None and foreign_blocks(foreign):
            return False
        if any(h.get("exclusive") for h in holders.values()):
            return False
        leased = sum(admit_gb(h, now) for h in holders.values())
        if total_gb is not None:
            room = total_gb - leased - headroom
            if free_gb is not None:
                room = min(room, free_gb - _pending_gb(holders, used_by, now) - headroom)
            if gb > room:
                return False
        h = {
            "gb": gb,
            "exclusive": bool(exclusive),
            "what": what,
            "t": now,
            "user": _user(),
            "ws": os.path.basename(os.environ.get("WORKSPACE", "")),
            **owner(pid),
        }
        if est_s:
            h["est_end"] = now + float(est_s)
        if queued_at is not None:
            h["t_submit"] = float(queued_at)
        if backfill:
            h["backfill"] = True
        if nested:
            h["nested"] = True
        if not discount:
            h["no_discount"] = True
        elif prev and prev.get("discount_revoked"):
            h["discount_revoked"] = (
                True  # a resize restarts the observation; a revoked one stays so
            )
        if exclusive and foreign is not None:
            h["foreign_at_start"] = foreign_at(foreign)
        elif exclusive and prev and prev.get("exclusive") and "foreign_at_start" in prev:
            h["foreign_at_start"] = prev["foreign_at_start"]  # a resize keeps the admission record
        st["holders"][key] = h
        return True


def drained(
    gpu,
    ld = None,
    pid = None,
    before = None,
):
    """A live ticket of another process is draining `gpu` (see pick_drain). `before`: the caller's
    own queue priority; a drainer queued behind it does not hold it back."""
    pid = int(pid or os.getpid())
    try:
        return any(
            t.get("drain") is not None
            and str(t["drain"]) == str(gpu)
            and int(t.get("pid") or 0) != pid
            and (before is None or priority(t) < float(before))
            for t in tickets(ld)
        )
    except OSError:
        return False


def release(
    gpu,
    pid = None,
    lock_dir = None,
    key = None,
):
    with _locked(gpu, lock_dir) as st:
        st["holders"].pop(str(key or pid or os.getpid()), None)


_TMP_LEFTOVER = re.compile(r"\.tmp\d+\.\d+$")
TMP_STALE_S = 3600.0


def reclaim(gpus, lock_dir = None):
    for g in gpus:
        with _locked(g, lock_dir):
            pass
    for _t in tickets(lock_dir):  # read_tickets drops dead owners' tickets
        pass
    # a writer killed between write and rename leaves <name>.tmp<pid>.<tid>; a live one renames in ms
    ld = Path(lock_dir) if lock_dir else resolve_lock_dir()
    now = time.time()
    for d in (ld, ld / "switchboard", ld / "switchboard" / "queue"):
        with contextlib.suppress(OSError):
            for f in d.iterdir():
                with contextlib.suppress(OSError):
                    if _TMP_LEFTOVER.search(f.name) and now - f.stat().st_mtime > TMP_STALE_S:
                        f.unlink()


def _user():
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        return str(os.getuid()) if hasattr(os, "getuid") else "?"


def env_for(gpu, gb, total_gb):
    frac = min(0.95, max(0.05, (gb + 1.0) / total_gb)) if total_gb else 0.95
    return {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "UNSLOTH_REGRESS_MEM_FRACTION": f"{frac:.3f}",
    }


# ------------------------------------------------------------------ pure placement (planning)
def plan(
    items,
    gpus_info,
    existing = None,
    headroom = HEADROOM_GB,
):
    """items: [{"name", "gb", "perf": bool}] -> {name: gpu | None}. Perf first, each on an idle GPU;
    the rest best-fit decreasing. existing: {gpu: {"leased_gb", "exclusive", "holders": n}}."""
    existing = existing or {}
    room, busy = {}, {}
    for g, inf in gpus_info.items():
        ex = existing.get(g, {})
        cap = inf["total_gb"] - ex.get("leased_gb", 0.0)
        room[g] = min(cap, inf.get("free_gb", cap)) - headroom
        busy[g] = ex.get("holders", 0) > 0 or ex.get("exclusive", False)
    placed, exclusive_taken = {}, {g for g in room if existing.get(g, {}).get("exclusive")}
    for it in [i for i in items if i.get("perf")]:
        g = next(
            (
                g
                for g in sorted(room, key = lambda g: -room[g])
                if not busy[g] and g not in exclusive_taken and room[g] >= it["gb"]
            ),
            None,
        )
        placed[it["name"]] = g
        if g is not None:
            exclusive_taken.add(g)
    for it in sorted([i for i in items if not i.get("perf")], key = lambda i: -i["gb"]):
        g = next(
            (
                g
                for g in sorted(room, key = lambda g: room[g])  # tightest fit first
                if g not in exclusive_taken and room[g] >= it["gb"]
            ),
            None,
        )
        placed[it["name"]] = g
        if g is not None:
            room[g] -= it["gb"]
    return placed


# ------------------------------------------------------------------ queue tickets
def _queue_dir(ld = None):
    d = state_dir(ld) / "queue"
    _mkdir_shared(d)
    return d


def priority(t):
    return float(t["t_submit"]) - (EXCL_PRIORITY_S if t.get("exclusive") else 0.0)


def tickets(ld = None):
    """Live waiting tickets, highest priority first; dead owners' tickets are removed."""
    out = []
    for p in _queue_dir(ld).glob("*.json"):
        try:
            t = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if not holder_alive(t.get("pid", 0), t):
            with contextlib.suppress(OSError):
                p.unlink()
            continue
        t["_path"] = str(p)
        out.append(t)
    return sorted(out, key = priority)


def write_ticket(t, ld = None):
    t = {k: v for k, v in t.items() if not k.startswith("_")}
    _write_shared(_queue_dir(ld) / f"{t['id']}.json", t)


def drop_ticket(tid, ld = None):
    with contextlib.suppress(OSError):
        (_queue_dir(ld) / f"{tid}.json").unlink()


def drained_by_ahead(me, all_tickets):
    """GPUs a higher-priority ticket is draining."""
    mp = priority(me)
    return {
        t["drain"]
        for t in all_tickets
        if t.get("id") != me.get("id") and t.get("drain") is not None and priority(t) < mp
    }


# ------------------------------------------------------------------ wait prediction
def _holder_end(h, now):
    end = h.get("est_end")
    if end is None:
        end = float(h.get("t", now)) + DEFAULT_EST_S[bool(h.get("exclusive"))]
    return end if end > now else now + OVERDUE_S


def _fits(g, gb, exclusive, gs, running, headroom):
    inf = gs[g]
    mine = [r for r in running if r["gpu"] == g]
    if any(r["exclusive"] for r in mine):
        return False
    if exclusive:
        return (
            not mine
            and not foreign_blocks(inf)
            and inf["total_gb"] - inf.get("foreign_gb", 0.0) - headroom >= gb
        )
    used = sum(r["gb"] for r in mine) + inf.get("foreign_gb", 0.0)
    return inf["total_gb"] - used - headroom >= gb


def simulate(
    state,
    me,
    now = None,
    headroom = HEADROOM_GB,
    horizon = 64,
):
    """Predicted start (epoch s) of ticket `me` or math.inf.

    state = {"gpus": {g: {"total_gb", "foreign_gb", "holders": {key: holder}}}, "tickets": [ahead...]}.
    Holders release at est_end; tickets ahead (priority order) then `me` are placed greedily at each
    release event, shared best-fit, exclusive on an empty GPU, drain GPUs of tickets ahead skipped."""
    now = now or time.time()
    gs = state.get("gpus", {})
    if not gs:
        return math.inf
    running = [
        {
            "gpu": g,
            "gb": admit_gb(h, now),
            "exclusive": bool(h.get("exclusive")),
            "end": _holder_end(h, now),
        }
        for g, inf in gs.items()
        for h in inf.get("holders", {}).values()
    ]
    queue = [dict(t) for t in sorted(state.get("tickets", []), key = priority)] + [dict(me, _me = True)]
    t = now
    for _ in range(horizon + len(queue)):
        running = [r for r in running if r["end"] > t]
        drained = set()
        for q in list(queue):
            cands = [
                g
                for g in gs
                if g not in drained
                and _fits(g, float(q["gb"]), bool(q.get("exclusive")), gs, running, headroom)
            ]
            if cands:
                used = {g: sum(r["gb"] for r in running if r["gpu"] == g) for g in cands}
                room = {
                    g: gs[g]["total_gb"] - used[g] - gs[g].get("foreign_gb", 0.0) for g in cands
                }
                if q.get("exclusive"):
                    g = min(cands, key = lambda g: (-room[g], g))
                else:  # admission's rule: spread, an empty card kept per drain-less timing ticket
                    busy = {r["gpu"] for r in running}
                    protect = protected_cards(
                        [x for x in queue if x is not q],
                        {
                            c: gs[c]["total_gb"] - gs[c].get("foreign_gb", 0.0) - headroom
                            for c in gs
                            if c not in busy and not foreign_blocks(gs[c])
                        },
                        q,
                    )
                    g = min(
                        cands,
                        key = lambda g: shared_order(
                            g, room[g], gs[g].get("util_mean_5s"), True, g in protect
                        ),
                    )
                if q.get("_me"):
                    return t
                running.append(
                    {
                        "gpu": g,
                        "gb": float(q["gb"]),
                        "exclusive": bool(q.get("exclusive")),
                        "end": t + float(q.get("est_s") or DEFAULT_EST_S[bool(q.get("exclusive"))]),
                    }
                )
                queue.remove(q)
            elif q.get("drain") is not None:
                drained.add(q["drain"])
        if not running:
            return math.inf  # nothing will ever free: does not fit any GPU
        t = min(r["end"] for r in running)
    return math.inf


def unleased_gb(inf, holders):
    """Memory in use on a card beyond its leases (the foreign estimate when no process is listed)."""
    used = float(inf.get("total_gb") or 0.0) - float(inf.get("free_gb") or 0.0)
    return max(0.0, used - sum(float(h.get("gb", 0)) for h in holders.values()))


def gather_state(
    gpus,
    ld = None,
    info = None,
    apps = None,
    now = None,
):
    """Snapshot for simulate(): holders per GPU, foreign usage (compute pids outside every holder's
    process tree), and the waiting tickets."""
    info = info if info is not None else query_gpus(gpus)
    apps = apps if apps is not None else {}
    table = plat.process_table() if apps else {}
    out = {}
    for g in gpus:
        hs = leases(g, ld)
        inf = info.get(g, {"total_gb": 0.0, "free_gb": 0.0})
        mine = set()
        for k in hs:
            mine |= plat.descendants(_key_pid(k), table) if table else {_key_pid(k)}
        foreign = sum(u for p, u in apps.get(g, []) if p not in mine and u >= FOREIGN_IGNORE_GB)
        if not apps.get(
            g
        ):  # no process view of this card: whatever is used beyond the leases is foreign
            foreign = unleased_gb(inf, hs)
        out[g] = {
            "total_gb": inf["total_gb"],
            "free_gb": inf.get("free_gb", 0.0),
            "foreign_gb": foreign,
            "util": inf.get("util"),
            "util_max_5s": inf.get("util_max_5s", inf.get("util")),
            "util_mean_5s": inf.get("util_mean_5s"),
            "holders": hs,
        }
    return {"gpus": out, "tickets": tickets(ld)}


def predict_wait(
    gb,
    exclusive = False,
    gpus = None,
    now = None,
    state = None,
    est_s = None,
    me = None,
):
    """Seconds until a new request (or ticket `me`) would start locally, math.inf when never."""
    now = now or time.time()
    if state is None:
        state = gather_state(gpus if gpus is not None else pool_gpus())
    me = me or {"id": "_new", "gb": gb, "exclusive": exclusive, "est_s": est_s, "t_submit": now}
    ahead = [
        t
        for t in state.get("tickets", [])
        if t.get("id") != me.get("id") and priority(t) < priority(me)
    ]
    start = simulate({**state, "tickets": ahead}, me, now = now)
    return max(0.0, start - now)


def drain_ready_at(
    inf,
    t,
    now = None,
):
    """When GPU `inf` (a gather_state entry) frees enough for ticket `t`, from its holders' ends."""
    now = now or time.time()
    hs = inf.get("holders", {}).values()
    if t.get("exclusive"):
        return max([_holder_end(h, now) for h in hs], default = now)
    need = float(t["gb"]) - (
        inf["total_gb"]
        - HEADROOM_GB
        - inf.get("foreign_gb", 0.0)
        - sum(admit_gb(h, now) for h in hs)
    )
    ready = now
    for h in sorted(hs, key = lambda h: _holder_end(h, now)):
        if need <= 0:
            break
        need -= admit_gb(h, now)
        ready = _holder_end(h, now)
    return ready


def backfill_ok(
    g,
    inf,
    me,
    ahead,
    now = None,
):
    """`me` (shared, with its own est_s) may take GPU `g` that tickets in `ahead` drain: it ends,
    stretched by BACKFILL_MARGIN, before `g` frees for the earliest of them, so none starts later."""
    now = now or time.time()
    if me.get("exclusive") or not me.get("est_s"):
        return False
    drainers = [t for t in ahead if str(t.get("drain")) == str(g)]
    if not drainers:
        return True
    ready = min(drain_ready_at(inf, t, now) for t in drainers)
    return now + float(me["est_s"]) * BACKFILL_MARGIN <= ready


def pick_drain(
    state,
    me,
    now = None,
    avoid = (),
):
    """The GPU that frees soonest for `me` among those no ticket ahead is draining and `avoid`
    (GPUs `me`'s own ancestors hold) leaves out (None: none fits)."""
    now = now or time.time()
    taken = drained_by_ahead(me, state.get("tickets", [])) | {str(g) for g in avoid}
    best = None
    for g, inf in state.get("gpus", {}).items():
        if g in taken or inf["total_gb"] - HEADROOM_GB - inf.get("foreign_gb", 0.0) < float(
            me["gb"]
        ):
            continue
        if me.get("exclusive") and foreign_blocks(inf):
            continue
        t = drain_ready_at(inf, me, now)
        if best is None or (t, g) < best:
            best = (t, g)
    return best[1] if best else None


def ancestor_pids(pid = None, table = None):
    """Pids of every ancestor of `pid` (this process when None), init excluded."""
    pid = int(pid or os.getpid())
    table = plat.process_table() if table is None else table
    chain, p = set(), table.get(pid)
    while p and p not in chain and p != 1:
        chain.add(p)
        p = table.get(p)
    return chain


def ancestor_leases(
    gpus,
    ld = None,
    pid = None,
    table = None,
):
    """{gpu: holder} for leases held by an ancestor of `pid` (this process when None): a request
    made from inside a running lease."""
    chain = ancestor_pids(pid, table)
    out = {}
    for g in gpus:
        for k, h in leases(g, ld).items():
            if _key_pid(k) in chain:
                out[g] = h
    return out


# ------------------------------------------------------------------ acquire
_SEQ = [0]
GRANT = (
    threading.local()
)  # .last: {gpu, key, lock_dir} of the lease acquire() is yielding in this thread
_SEQ_LOCK = threading.Lock()


def _next_key():
    with _SEQ_LOCK:
        _SEQ[0] += 1
        return f"{os.getpid()}:{_SEQ[0]}"


PORTABLE_ERR = "not portable: the command must run a .py script or .ipynb (or pass --remote-cmd)"


def remote_job_from_cmd(
    cmd,
    gb,
    dtype = None,
    est_s = None,
    perf = False,
    job_class = "train",
    what = "",
    sig = None,
    whole_kernel = False,
    multi_gpu = False,
):
    """cloud_pool.dispatch job for `cmd` ([python [-u]] x.py args | x.ipynb), None when not portable.
    `share`: may join a running Colab worker beside other jobs (never for perf / whole-machine work)."""
    cmd = list(cmd)
    if cmd and (os.path.basename(cmd[0]).startswith("python") or cmd[0] == sys.executable):
        cmd = cmd[1:]
        while cmd and cmd[0] in ("-u", "-B", "-O"):
            cmd = cmd[1:]
    if not cmd:
        return None
    job = {
        "gb": float(gb),
        "perf": bool(perf),
        "class": {"generate": "decode"}.get(job_class, job_class),
        "what": what,
        "sig": sig,
        "share": not (perf or multi_gpu),
    }  # whole_kernel: Kaggle only
    if whole_kernel:
        job["whole_kernel"] = True
    if multi_gpu:
        job["multi_gpu"] = True
    if os.path.isfile(cmd[0]) and cmd[0].endswith(".ipynb"):
        job["notebook"] = os.path.abspath(cmd[0])
    elif os.path.isfile(cmd[0]) and cmd[0].endswith(".py"):
        job["script"], job["argv"] = os.path.abspath(cmd[0]), cmd[1:]
    else:
        # Any other command: a generated wrapper execs it remotely; local files it names travel with
        # it (by basename, next to the wrapper). A binary the remote image lacks fails there, visibly.
        files = [a for a in cmd if os.path.isfile(a)]
        remote = [os.path.basename(a) if os.path.isfile(a) else a for a in cmd]
        pre = ""
        if cmd[0] in files:  # a local executable: shipped files lose their mode bits
            remote[0] = "./" + remote[0]
            pre = f"os.chmod({remote[0]!r}, 0o755)\n"
        d = Path(tempfile.mkdtemp(prefix = "sbcmd_"))
        w = d / "sb_cmd.py"
        w.write_text(
            "import os, subprocess, sys\n"
            + pre
            + f"sys.exit(subprocess.call({remote!r} + sys.argv[1:]))\n"
        )
        job["script"], job["argv"], job["files"] = str(w), [], [os.path.abspath(f) for f in files]
    if dtype:
        job["dtype"] = dtype
    if est_s:
        job["est_s"] = float(est_s)
    return job


REMOTE_DONE = (
    "PASS",
    "FAIL",
)  # anything else (NO_REMOTE_FIT, INFRA, MANUAL, CANCELLED): stay local


def _dispatch(remote_job, gate = None):
    import cloud_pool  # noqa: F401  lazy: optional, and imports notebook_cloud_run
    return cloud_pool.dispatch(remote_job, gate = gate)


def should_offload(
    waited,
    wait,
    warm = False,
):
    """Local first: never at t=0 on a prediction alone. `wait` = predicted remaining (inf = unknown).
    warm: a running Colab worker can take the job now, so the earlier OFFLOAD_WARM_S rule applies."""
    if waited >= OFFLOAD_S:
        return True
    if warm and waited >= OFFLOAD_WARM_S and not math.isinf(wait) and wait >= OFFLOAD_WARM_PRED_S:
        return True
    return waited >= OFFLOAD_MIN_WAIT_S and not math.isinf(wait) and wait >= OFFLOAD_PRED_S


_WARM_CACHE = {}


def _read_workers():
    """The Colab pool's worker records, read without its lock (a snapshot is enough for a hint)."""
    try:
        import cloud_pool
        st = json.loads(Path(cloud_pool.workers_path()).read_text() or "{}")
    except Exception:  # noqa: BLE001 - no pool, no file, mid-write
        return None, {}
    ws = st.get("workers") if isinstance(st, dict) else None
    return cloud_pool, (ws if isinstance(ws, dict) else {})


def warm_worker_fits(job, now = None):
    """True when a Colab worker that is already running could take this shared job now: a BUSY
    shared worker with room (cloud_pool.joinable_workers) or an IDLE one whose tier fits. Cached
    WARM_CACHE_S per (gb, ram, dtype); False for a job that may not share."""
    if not job or not job.get("share") or os.environ.get("COLAB_SHARE", "1") == "0":
        return False
    now = time.time() if now is None else now
    gb, ram = float(job.get("gb") or 0), float(job.get("ram_gb") or 0)
    dtype = job.get("dtype")
    key = (gb, ram, dtype)
    hit = _WARM_CACHE.get(key)
    if hit and now - hit[0] < WARM_CACHE_S:
        return hit[1]
    ok = False
    cp, ws = _read_workers()
    if cp is not None and ws:
        try:
            tiers = cp.load_tiers()
        except Exception:  # noqa: BLE001
            tiers = {}
        norm = getattr(cp, "_norm_dtype", None)  # dispatch's default (bf16) and spellings (half)
        dtype = norm(dtype) if norm else (dtype or "bf16")
        ready = getattr(cp, "account_ready", None)
        join = getattr(cp, "joinable_workers", None)
        pairs = set()
        for w in ws.values():
            if not isinstance(w, dict):
                continue
            t = tiers.get(w.get("tier")) or {}
            if dtype not in (t.get("dtypes") or ()):
                continue
            try:
                if ready is not None and not ready(w.get("account")):
                    continue  # this unix user is not logged in to that account
            except Exception:  # noqa: BLE001
                continue
            if w.get("state") == "IDLE" and not w.get("needs_clean"):
                ok = ok or (
                    float(t.get("vram_gb") or 0) >= gb + 1.5
                    and (not ram or float(t.get("ram_gb") or 0) >= ram + 2)
                )
            elif w.get("state") == "BUSY" and w.get("mode") == "shared":
                pairs.add((w.get("tier"), w.get("account")))
        if not ok and join is not None:
            for tier, account in sorted(pairs, key = str):
                try:
                    if join(tier, account, gb, ram):
                        ok = True
                        break
                except Exception:  # noqa: BLE001 - an older / newer cloud_pool: no warm hint
                    break
    _WARM_CACHE[key] = (now, ok)
    return ok


class RemoteAttempt(threading.Thread):
    """cloud_pool.dispatch in the background with a cancel gate, so the local ticket keeps queueing."""

    def __init__(self, remote_job, what):
        super().__init__(daemon = True)
        self.remote_job, self.what, self.result = remote_job, what, None
        self.cancel, self.started_ev = threading.Event(), threading.Event()
        self.start()

    # cloud_pool.Gate protocol
    @property
    def started(self):
        return self.started_ev

    def cancelled(self):
        return self.cancel.is_set() and not self.started_ev.is_set()

    def run(self):
        self.result = _try_remote(self.remote_job, self.what, gate = self)

    def done(self):
        return not self.is_alive()


def exclusive_waiting(tks, me = None):
    """An exclusive ticket other than `me` is queued (placement itself uses protected_cards)."""
    mid = (me or {}).get("id")
    return any(t.get("exclusive") and t.get("id") != mid for t in tks)


def protected_cards(
    tks,
    empty,
    me = None,
):
    """Empty cards kept for timing work: one per exclusive ticket other than `me` that waits with no
    drain target (held back by the MAX_EXCL cap, or a studio_regress perf job before its drain), each
    the highest-numbered card in `empty` ({card with no lease timing work could use: usable GB}) that
    fits it. A ticket with a drain target is protected by drained() on that card. Best fit on every
    card used to do this job and stacked a whole burst onto one card while the rest idled."""
    mid = (me or {}).get("id")
    free = sorted(empty, key = lambda g: (len(str(g)), str(g)), reverse = True)
    out = set()
    for t in tks:
        if t.get("exclusive") and t.get("id") != mid and t.get("drain") is None:
            g = next(
                (g for g in free if g not in out and empty[g] >= float(t.get("gb") or 0)), None
            )
            if g is not None:
                out.add(g)
    return out


def shared_order(
    g,
    room,
    util_mean,
    spread = True,
    protected = False,
):
    """Sort key for shared placement: spread = least busy, then most room; else best fit. A protected
    card (protected_cards) comes last, tightest first: used only when nothing else fits."""
    if protected:
        return (1, room, g)
    return (0, util_mean or 0.0, -room, g) if spread else (0, room, g)


def _attempt(
    gpus,
    gb,
    exclusive,
    what,
    key,
    est_s,
    me,
    ld,
    query = query_gpus,
    apps_fn = query_apps,
    discount = True,
):
    """One placement pass: (gpu, total_gb) or None."""
    info = query(gpus)
    apps = apps_fn() if exclusive else {}
    st = gather_state(gpus, ld, info = info, apps = apps)
    skip = drained_by_ahead(me, st["tickets"])
    ahead = [t for t in st["tickets"] if t.get("id") != me.get("id") and priority(t) < priority(me)]
    now = time.time()
    backfill = set()
    cands = []
    for g in gpus:
        if g in skip:
            if not backfill_ok(g, st["gpus"][g], me, ahead, now):
                continue
            backfill.add(g)
        s = st["gpus"][g]
        room = (
            min(s["total_gb"] - sum(admit_gb(h, now) for h in s["holders"].values()), s["free_gb"])
            - HEADROOM_GB
        )
        if exclusive and s["holders"]:
            continue
        if room >= gb:
            cands.append((room, g))
    # exclusive: the emptiest card; shared: spread, keeping one empty card per waiting timing ticket
    # that has no drain target (protected_cards) for last
    protect = (
        set()
        if exclusive
        else protected_cards(
            st["tickets"],
            {
                g: min(st["gpus"][g]["total_gb"], st["gpus"][g]["free_gb"]) - HEADROOM_GB
                for g in gpus
                if not st["gpus"][g]["holders"] and not foreign_blocks(st["gpus"][g])
            },
            me,
        )
    )
    cands.sort(
        key = lambda x: (-x[0], x[1])
        if exclusive
        else shared_order(x[1], x[0], st["gpus"][x[1]].get("util_mean_5s"), True, x[1] in protect)
    )
    for _room, g in cands:
        s = st["gpus"][g]
        if try_acquire(
            g,
            gb,
            exclusive,
            what,
            total_gb = s["total_gb"],
            free_gb = s["free_gb"],
            lock_dir = ld,
            key = key,
            est_s = est_s,
            foreign = s if exclusive else None,
            backfill = g in backfill,
            queued_at = me.get("t_submit"),
            discount = discount,
        ):
            if g in backfill:
                _log(
                    f"{what or 'job'}: backfilled onto GPU {g} (ends before its drainer can start)"
                )
            return g, s["total_gb"]
    return None


@contextlib.contextmanager
def acquire(
    gb,
    exclusive = False,
    what = "",
    est_s = None,
    dtype = None,
    wait_s = 3600,
    poll_s = 15,
    env = None,
    avoid = (),
    offload = "never",
    remote_job = None,
    lock_dir = None,
    query = None,
    apps_fn = None,
    gpus = None,
    discount = True,
):
    """Block until a host GPU can take `gb` (exclusive if asked); yield (gpu, env). No GPU pool or
    gb <= 0: (None, {}), run unpinned. With offload auto (portable remote_job given) or force, a
    predicted local wait >= OFFLOAD_S dispatches to cloud_pool and yields ("remote", {"result": r})."""
    ld = lock_dir
    gpus = [g for g in (pool_gpus(env) if gpus is None else gpus) if g not in set(map(str, avoid))]
    if (not gpus or gb <= 0) and offload != "force":
        yield None, {}
        return
    query = query or query_gpus
    apps_fn = apps_fn or query_apps
    key = _next_key()
    started = time.time()
    deadline = started + wait_s
    me = {
        "id": uuid.uuid4().hex[:12],
        "pid": os.getpid(),
        **owner(),
        "gb": gb,
        "exclusive": bool(exclusive),
        "dtype": dtype,
        "est_s": est_s,
        "t_submit": started,
        "what": what,
        "user": _user(),
        "ws": os.path.basename(os.environ.get("WORKSPACE", "")),
    }
    try:
        parents = ancestor_leases(gpus, ld)
    except Exception:  # noqa: BLE001 -- a missing process view only loses the inheritance
        parents = {}
    if parents:
        # Nested in a running lease: that lease waits on this request, so it keeps the lease's place.
        me["t_submit"] = min(
            [started] + [float(h.get("t_submit", h.get("t", started))) for h in parents.values()]
        )
        me["nested_in"] = sorted(parents)
    ticketed, said, hinted = False, 0.0, False
    remote, next_remote = None, 0.0  # next_remote: earliest time a new remote attempt may start
    warm_retry = 0.0  # a failed warm-only attempt backs off the warm path alone
    remote_was_warm = False
    local_only = offload == "never" or not remote_job
    try:
        while True:
            now = time.time()
            waited = now - started
            if remote is not None and remote.done():
                r, remote = remote.result, None
                if r is not None:
                    if ticketed:
                        drop_ticket(me["id"], ld)
                        ticketed = False
                    yield "remote", {"result": r}
                    return
                if remote_was_warm:
                    warm_retry = (
                        now + REMOTE_RETRY_S
                    )  # the normal offload rule still applies meanwhile
                else:
                    next_remote = now + REMOTE_RETRY_S
                _log(
                    f"{what or 'job'}: remote attempt did not run it; waiting locally, "
                    + (
                        "no running Colab worker had room"
                        if remote_was_warm
                        else f"remote again in {REMOTE_RETRY_S / 60:.0f} min"
                    )
                )
                if offload == "force" and (not gpus or gb <= 0):
                    yield None, {}  # forced offload failed and there is nothing to wait for
                    return
            running_remote = remote is not None and remote.started_ev.is_set()
            got = (
                _attempt(gpus, gb, exclusive, what, key, est_s, me, ld, query, apps_fn, discount)
                if gpus and gb > 0 and not running_remote
                else None
            )
            if got is not None:
                g, total = got
                if remote is not None:
                    remote.cancel.set()
                    remote.join()  # bounded: cloud_pool stops a not-yet-running attempt
                    if remote.started_ev.is_set() and remote.result is not None:
                        release(g, lock_dir = ld, key = key)  # it ran remotely after all: remote wins
                        if ticketed:
                            drop_ticket(me["id"], ld)
                            ticketed = False
                        yield "remote", {"result": remote.result}
                        return
                    remote = None
                if ticketed:
                    drop_ticket(me["id"], ld)
                    ticketed = False
                GRANT.last = {"gpu": g, "key": key, "lock_dir": ld}
                try:
                    yield g, env_for(g, gb, total)
                finally:
                    GRANT.last = None
                    release(g, lock_dir = ld, key = key)
                return
            wait, st = math.inf, None
            if gpus and gb > 0:
                try:
                    st = gather_state(gpus, ld, info = query(gpus))
                    wait = predict_wait(gb, exclusive, now = now, state = st, est_s = est_s, me = me)
                    if exclusive and MAX_EXCL and exclusive_elsewhere("", ld) >= MAX_EXCL:
                        me["drain"] = (
                            None  # the cap holds it back: draining a card now only idles it
                        )
                    elif exclusive or waited > STARVE_S:
                        me["drain"] = pick_drain(st, me, now, avoid = parents)
                except Exception as e:  # noqa: BLE001
                    st, wait = None, math.inf
                    _log(f"wait prediction failed: {type(e).__name__}: {e}")
                if not running_remote:
                    write_ticket(
                        {**me, "predicted_wait_s": None if math.isinf(wait) else round(wait)}, ld
                    )
                    ticketed = True
                elif ticketed:  # running remotely: stop holding a place in the local queue
                    drop_ticket(me["id"], ld)
                    ticketed = False
            warm = (
                remote is None
                and not local_only
                and offload != "force"
                and not exclusive
                and now >= next_remote
                and now >= warm_retry
                and waited >= OFFLOAD_WARM_S
                and waited < OFFLOAD_S
                and bool(remote_job)
                and warm_worker_fits(remote_job, now)
            )
            if (
                remote is None
                and not local_only
                and now >= next_remote
                and (offload == "force" or should_offload(waited, wait, warm))
            ):
                remote = RemoteAttempt(
                    dict(remote_job, warm_only = True) if warm else remote_job, what
                )
                remote_was_warm = warm
                _log(
                    f"{what or 'job'}: waited {waited / 60:.0f} min, predicted "
                    f"{'unknown' if math.isinf(wait) else f'~{wait / 60:.0f} min'} more: "
                    + ("joining a running Colab worker" if warm else "trying Colab / Kaggle")
                    + " while staying in the local queue"
                )
            if offload == "auto" and not remote_job and not hinted and waited >= OFFLOAD_S:
                _log(f"{what or 'job'}: LOCAL_ONLY (no portable remote job); keeps waiting")
            if not hinted and waited >= OFFLOAD_S:
                hinted = True
                _log(offload_hint(waited))
            if (
                now - said >= 60 and st is not None
            ):  # at most once a minute, not silently for an hour
                said = now
                why = (
                    f"GPU reading failing ({QUERY_ERROR['error']})"
                    if QUERY_ERROR
                    else ", ".join(
                        f"GPU {g}: {st['gpus'][g]['free_gb']:.1f} GB free, "
                        f"{len(st['gpus'][g]['holders'])} lease(s)"
                        for g in gpus
                    )
                )
                eta = "never (fits no GPU)" if math.isinf(wait) else f"~{wait / 60:.0f} min"
                _log(
                    f"{what or 'unit'} waiting {waited:.0f}s for {gb} GB{' exclusive' if exclusive else ''}"
                    f" (predicted start {eta}){', remote attempt in flight' if remote else ''}: {why}"
                )
            if now > deadline and remote is None:
                raise TimeoutError(
                    f"no GPU in {gpus} could take {gb} GB (exclusive={exclusive})"
                    + (f"; GPU reading failing: {QUERY_ERROR['error']}" if QUERY_ERROR else "")
                    + "\n"
                    + offload_hint(waited)
                )
            time.sleep(poll_s)
    finally:
        if ticketed:
            drop_ticket(me["id"], ld)
        if remote is not None and remote.is_alive():
            remote.cancel.set()  # an attempt not yet running remotely is abandoned with us


def _try_remote(
    remote_job,
    what,
    gate = None,
):
    if not remote_job:
        return None
    try:
        r = _dispatch(remote_job, gate) if gate is not None else _dispatch(remote_job)
        if not isinstance(r, dict) or r.get("status") not in REMOTE_DONE:
            _log(
                f"{what or 'job'}: offload did not run it ({(r or {}).get('status')}: "
                f"{str((r or {}).get('reason', ''))[:160]}); keeps waiting locally"
            )
            return None
        return r
    except ImportError as e:
        _log(f"{what or 'job'}: offload unavailable ({e}); keeps waiting")
    except Exception as e:  # noqa: BLE001
        _log(f"{what or 'job'}: offload failed ({type(e).__name__}: {e}); keeps waiting")
    return None


OFFLOAD_HINT = """[gpu_queue] OFFLOAD: host GPUs backed up. `gpu_queue.py run --offload auto` dispatches portable jobs itself; else run on rented hardware:
  Colab   notebook_cloud_run.py --backend colab --gpu T4|L4|A100|A100-HM|G4 (<=3 parallel per tier):
          T4 16GB fp16 only (no bf16); L4 24GB FP8; A100 40GB; A100-HM 80GB (scarce); G4 RTX PRO 6000 FP8 + NVFP4
  Kaggle  --backend kaggle --gpu T4x2: 2x T4 (2 scripts via CUDA_VISIBLE_DEVICES, or multi-GPU), 2 runs/account;
          KAGGLE_API_TOKEN 60 h/wk, _2 45 h/wk, _3 60 h/wk, billed on notebook wall time
  AMD CI  amd_ci_workflow: Strix Halo gfx1151, Linux + Windows, ~4 queued, unlimited
  Quota first: python notebook_cloud_run.py --quota"""


def offload_hint(waited_s):
    return f"{OFFLOAD_HINT}\n  (waited {waited_s / 60:.0f} min)"


# ------------------------------------------------------------------ VRAM estimate + measured peaks
LOAD_BYTES = {
    "4bit": 0.6,
    "nvfp4": 0.6,
    "8bit": 1.1,
    "fp8": 1.1,
    "16bit": 2.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "32bit": 4.0,
}
CTX_GB = 1.5
_PARAMS = {}


def model_params_b(model):
    """Parameter count (billions) from the Hub's safetensors metadata; no weight download."""
    if model in _PARAMS:
        return _PARAMS[model]
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model, token = hf_read_token(), expand = ["safetensors"])
        total = getattr(getattr(info, "safetensors", None), "total", None)
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            f"could not get the parameter count of {model} ({type(e).__name__}); "
            f"pass params_b / --params-b"
        ) from e
    if not total:
        raise ValueError(f"{model} has no safetensors metadata; pass params_b / --params-b")
    _PARAMS[model] = total / 1e9
    return _PARAMS[model]


def estimate_peak_gb(
    params_b,
    load = "16bit",
    mode = "infer",
    seq = 2048,
    batch = 1,
):
    if load not in LOAD_BYTES:
        raise ValueError(f"load must be one of {sorted(LOAD_BYTES)}")
    w = params_b * LOAD_BYTES[load]
    tokens = max(1, seq) * max(1, batch)
    if mode == "full":
        body = params_b * 16.0 + 0.05 * params_b * tokens / 4096  # weights+grads+Adam fp32 states
    elif mode == "lora":
        body = w * 1.3 + 0.1 * params_b * tokens / 4096  # adapters, activations
    elif mode == "infer":
        body = w * 1.1 + 0.05 * params_b * tokens / 4096  # KV cache
    else:
        raise ValueError("mode must be infer | lora | full")
    return body + CTX_GB


def reservation_gb(peak_gb):
    return round(peak_gb + max(2.0, 0.25 * peak_gb), 1)


def estimate_gb(
    params_b = None,
    model = None,
    load = "16bit",
    mode = "infer",
    seq = 2048,
    batch = 1,
):
    """Reservation GB = peak + max(2 GB, 25%): a deliberately dumb prior; MemCache overrides it."""
    if params_b is None:
        if not model:
            raise ValueError("pass params_b or model")
        params_b = model_params_b(model)
    return reservation_gb(estimate_peak_gb(float(params_b), load, mode, seq, batch))


class MemCache:
    """Host-wide measured peaks: switchboard/mem_cache.json {sig: {"peak_gb", "runtime_s", "arch",
    "n", "lower": [...], "t"}}. A higher peak raises immediately; 3 lower runs in a row lower it to
    their max; an OOM run is a floor and raises the estimate to 1.5x what it had reached.
    Every write prunes: one-off entries after ONE_OFF_DAYS, anything after MAX_AGE_DAYS, and the
    oldest beyond MAX_KEYS (the file is read and rewritten under one host-wide lock per job)."""

    LOWER_AFTER = 3
    TRUSTED_N = 3  # runs before a measured peak overrides an explicit --gb
    ONE_OFF_DAYS = 14
    MAX_AGE_DAYS = 60
    MAX_KEYS = 6000

    def __init__(self, path = None):
        self.path = Path(path) if path else state_dir() / "mem_cache.json"

    @contextlib.contextmanager
    def _edit(self):
        _mkdir_shared(self.path.parent)
        with _open_shared(self.path.with_suffix(".lock")) as fh:
            plat.lock(fh)
            try:
                try:
                    data = json.loads(self.path.read_text()) if self.path.exists() else {}
                except ValueError:
                    data = {}
                yield data
                self.prune(data)
                _write_shared(self.path, data, indent = None)
            finally:
                plat.unlock(fh)

    @classmethod
    def prune(
        cls,
        data,
        now = None,
    ):
        now = now or time.time()
        for k in [
            k
            for k, e in data.items()
            if not isinstance(e, dict)
            or now - float(e.get("t") or 0) > cls.MAX_AGE_DAYS * 86400
            or (e.get("n", 0) <= 1 and now - float(e.get("t") or 0) > cls.ONE_OFF_DAYS * 86400)
        ]:
            del data[k]
        if len(data) > cls.MAX_KEYS:
            for k in sorted(data, key = lambda k: float(data[k].get("t") or 0))[
                : len(data) - cls.MAX_KEYS
            ]:
                del data[k]
        return data

    def lookup(
        self,
        sig,
        legacy = None,
    ):
        """The entry for `sig`, else for `legacy` (the full-argv key used before default_sig was
        normalised), so a command measured under the old key keeps its peak."""
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None
        return data.get(sig) or (data.get(legacy) if legacy else None)

    def reservation(
        self,
        sig,
        legacy = None,
    ):
        e = self.lookup(sig, legacy)
        return reservation_gb(e["peak_gb"]) if e and e.get("peak_gb") else None

    def record(
        self,
        sig,
        peak_gb,
        runtime_s = None,
        arch = None,
        oom = False,
        cmd = None,
    ):
        with self._edit() as data:
            e = data.setdefault(sig, {"peak_gb": 0.0, "n": 0, "lower": []})
            if cmd and "cmd" not in e:
                e["cmd"] = cmd[:200]  # what the key stands for, for humans
            e["n"] = e.get("n", 0) + 1
            e["t"] = time.time()
            if arch:
                e["arch"] = arch
            if oom:
                e["peak_gb"] = round(max(e.get("peak_gb", 0.0), float(peak_gb)) * 1.5, 2)
                e["lower"] = []
            elif float(peak_gb) >= e.get("peak_gb", 0.0):
                e["peak_gb"] = round(float(peak_gb), 2)
                e["lower"] = []
            else:
                e.setdefault("lower", []).append(round(float(peak_gb), 2))
                if len(e["lower"]) >= self.LOWER_AFTER:
                    e["peak_gb"] = max(e["lower"])
                    e["lower"] = []
            if runtime_s is not None and not oom:
                rs = (e.get("runtimes") or [])[-4:] + [round(float(runtime_s), 1)]
                e["runtimes"] = rs
                e["runtime_s"] = sorted(rs)[len(rs) // 2]
            return dict(e)


# ------------------------------------------------------------------ run
def _gpu_name(gpu):
    return (gpu_sample.read(max_age = 600).get("gpus", {}).get(str(gpu)) or {}).get("name")


class PeakSampler(threading.Thread):
    """Peak VRAM (GB) of a process tree on one GPU, from nvidia-smi compute apps every `every` s."""

    def __init__(
        self,
        root_pid,
        gpu,
        every = 5.0,
        apps_fn = query_apps,
        on_peak = None,
    ):
        super().__init__(daemon = True)
        self.root, self.gpu, self.every, self.apps_fn = root_pid, str(gpu), every, apps_fn
        self.on_peak = on_peak  # called with the group peak: on a 0.5 GB rise, else every PUBLISH_S
        self._pub = (None, 0.0)
        self.peak = 0.0
        self.seen = set()  # every pid of the tree seen: orphans re-parent once the root exits
        self._stop = threading.Event()

    def sample(self):
        tree = plat.descendants(self.root)
        self.seen |= tree
        used = sum(u for p, u in self.apps_fn().get(self.gpu, []) if p in tree)
        self.peak = max(self.peak, used)
        return used

    def publish(self, now = None):
        now = now or time.time()
        last, t = self._pub
        if self.on_peak and (last is None or self.peak >= last + 0.5 or now - t >= PUBLISH_S):
            self.on_peak(self.peak)
            self._pub = (self.peak, now)

    def run(self):
        while not self._stop.wait(self.every):
            with contextlib.suppress(Exception):
                self.sample()
                self.publish()

    def stop(self):
        self._stop.set()


def legacy_sig(cmd):
    """The pre-normalisation key: a hash of the exact argv (every output path and timestamp in it)."""
    return "cmd:" + hashlib.sha1(" ".join(cmd).encode()).hexdigest()[:16]


# Flags whose value names where a run writes, not what it computes. Matched whole (plus a
# -dir / -path / -file / ... suffix), so --dirichlet or --outlier never lose their value; a value
# starting with "-" is the next flag. Often-boolean names (--json, --save) are left out.
_OUT_FLAG = re.compile(
    r"(?<![\w-])(--?(?:out|output|outdir|outfile|log|logs|logdir|logfile|dir|root|run[-_]root|save[-_](?:dir|path|to)|"
    r"report|tmpdir|workdir|work[-_]dir|results|dest|cache[-_]dir|port|home)"
    r"(?:[-_](?:dir|path|file|root|json|prefix|name))?)(?:=|\s+)(?!-)(\"[^\"]*\"|'[^']*'|[^\s'\"]+)"
)
_WS_PREFIX = re.compile(r"(?:/[^\s'\"=:]*)?/workspace_\d+(?=/|\b)")
_SCRATCH_PATH = re.compile(
    r"(?<![\w.-])(?:\$WS/|/)?(?:[^\s'\"=:]*/)?(?:temp|tmp|outputs|logs|async_outputs|runs)/"
    r"((?:[^\s'\"=:/]+/)*)([^\s'\"=:/]*)"
)
_TIMESTAMP = re.compile(
    r"(?<!\d)20\d{6}(?:[_T-]?\d{4,6})?(?!\d)|\d{4}-\d{2}-\d{2}[T_ ]\d{2}[:-]?\d{2}(?:[:-]?\d{2})?"
)
_HEX_ID = re.compile(
    r"(?<![0-9A-Za-z])(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{8,}(?![0-9A-Za-z])"
)
_PYTHON = re.compile(r"(?<![\w/.-])(?:\S*/)?python(?:\d+(?:\.\d+)?)?(?![\w.-])")


def _scratch_tail(m):
    dirs = [d for d in m.group(1).split("/") if d]
    return "<tmp>/" + "/".join(dirs[-2:] + ([m.group(2)] if m.group(2) else []))


def normalized_cmd(cmd):
    """The command as a workload, not a run: output / log / dir flag values, timestamps, hex run
    ids, which workspace (and venv) it ran from and per-run scratch dirs are blanked; the script,
    model ids, sizes, batch / seq and every other argument stay. Works inside `bash -c '...'`."""
    s = " ".join(cmd)
    s = _OUT_FLAG.sub(lambda m: m.group(1) + "=<v>", s)
    s = _WS_PREFIX.sub("$WS", s)
    s = s.replace(str(Path.home()) + "/", "~/")
    s = _TIMESTAMP.sub("<ts>", s)
    s = _HEX_ID.sub("<id>", s)
    s = _SCRATCH_PATH.sub(_scratch_tail, s)
    s = _PYTHON.sub("python", s)
    return " ".join(s.split())


def default_sig(cmd):
    """Measured-peak key: the normalised command, so reruns of one workload share their peak."""
    return "sig2:" + hashlib.sha1(normalized_cmd(cmd).encode()).hexdigest()[:16]


def _run_child(
    cmd,
    env,
    gpu,
    gb,
    sig,
    est_s,
    label = None,
    lease = None,
):
    t0 = time.time()
    child = subprocess.Popen(cmd, env = env, **plat.group_kwargs())
    on_peak = None
    if lease:

        def on_peak(peak):
            publish_peak(lease["gpu"], lease["key"], peak, lease.get("lock_dir"))

    sampler = PeakSampler(child.pid, gpu, on_peak = on_peak) if gpu not in (None, "remote") else None
    if sampler:
        sampler.start()

    def _fwd(signum, _frame):
        plat.signal_tree(child.pid, hard = signum == getattr(signal, "SIGKILL", None))

    old = {}
    if threading.current_thread() is threading.main_thread():
        for s in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(ValueError, OSError):
                old[s] = signal.signal(s, _fwd)
    try:
        rc = child.wait()
        # the lease covers the whole group: wait for stragglers the child left on the GPU
        left = plat.stop_tree(
            child.pid, grace_s = 10, extra = sampler.seen if sampler else (), proc = child
        )
        if left:
            _log(f"{len(left)} process(es) of the job survived the stop")
    finally:
        for s, h in old.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(s, h)
        if sampler:
            sampler.stop()
            with contextlib.suppress(Exception):
                sampler.sample()
    if sampler and sampler.peak > 0:
        oom = rc != 0 and sampler.peak >= 0.9 * gb
        with contextlib.suppress(OSError):
            MemCache().record(
                sig, sampler.peak, time.time() - t0, _gpu_name(gpu), oom = oom, cmd = label
            )
        _log(f"peak {sampler.peak:.1f} GB of {gb:g} reserved ({sig})")
    return rc


def default_offload(
    cmd,
    remote_cmd = None,
    exclusive = False,
    perf_remote_ok = False,
):
    """auto for a portable job (a python script / notebook plus the files it names), else never.
    Exclusive timing work kept local (--b200) waits for a B200 unless perf_remote_ok."""
    if exclusive and not perf_remote_ok:
        return "never"
    c = shlex.split(remote_cmd) if remote_cmd else list(cmd)
    if c and (os.path.basename(c[0]).startswith("python") or c[0] == sys.executable):
        c = [x for x in c[1:] if x not in ("-u", "-B", "-O")]
    return "auto" if c and os.path.isfile(c[0]) and c[0].endswith((".py", ".ipynb")) else "never"


_IMPORT_PROBE = r"""
import importlib.util, json, sys
names, paths = json.loads(sys.argv[1])
sys.path[:0] = paths
out = []
for n in names:
    try:
        if importlib.util.find_spec(n) is None:
            out.append(n)
    except (ImportError, ValueError):
        out.append(n)
print(json.dumps(out))
"""


def _script_of(cmd):
    """(python, script.py) for `python [-flags] x.py ...`, else None (`-m mod`, `-c`, other programs)."""
    if not cmd or not (os.path.basename(cmd[0]).startswith("python") or cmd[0] == sys.executable):
        return None
    i = 1
    while i < len(cmd) and cmd[i].startswith("-"):
        if cmd[i] in ("-m", "-c"):
            return None
        i += 2 if cmd[i] in ("-X", "-W") else 1
    return (
        (cmd[0], cmd[i])
        if i < len(cmd) and cmd[i].endswith(".py") and os.path.isfile(cmd[i])
        else None
    )


def script_imports(path):
    """({module-level imports}, {imports inside functions}) of a script, top-level package names, minus
    imports under `try` (optional by construction) and relative imports."""
    import ast

    tree = ast.parse(Path(path).read_text(encoding = "utf-8", errors = "replace"))
    top, nested = set(), set()

    def walk(node, in_def, in_try):
        for ch in ast.iter_child_nodes(node):
            d = in_def or isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            t = (
                in_try
                or isinstance(ch, ast.Try)
                or (hasattr(ast, "TryStar") and isinstance(ch, ast.TryStar))
            )
            names = []
            if isinstance(ch, ast.Import):
                names = [al.name.split(".")[0] for al in ch.names]
            elif isinstance(ch, ast.ImportFrom) and not ch.level and ch.module:
                names = [ch.module.split(".")[0]]
            if names and not in_try:
                (nested if in_def else top).update(names)
            walk(ch, d, t)

    walk(tree, False, False)
    return top, nested - top


def import_preflight(cmd):
    """Missing (module-level, in-function) imports of a `python x.py` command, checked with that
    interpreter, so a job missing a package fails now instead of after its wait in the queue.
    None when the command is not a python script or the probe itself cannot run."""
    ps = _script_of(cmd)
    if not ps:
        return None
    py, script = ps
    try:
        top, nested = script_imports(script)
    except (OSError, SyntaxError, ValueError):
        return None
    if not top and not nested:
        return [], []
    d = Path(script).resolve().parent
    paths = [
        str(d),
        str(d.parent),
        str(d.parent.parent),
    ]  # scripts add their own / parent dirs to sys.path
    try:
        r = subprocess.run(
            [py, "-c", _IMPORT_PROBE, json.dumps([sorted(top | nested), paths])],
            capture_output = True,
            text = True,
            timeout = 60,
        )
        missing = set(json.loads(r.stdout.strip().splitlines()[-1])) if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        missing = None
    if missing is None:
        return None
    return sorted(missing & top), sorted(missing & nested)


def cmd_run(a):
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        _log("nothing to run: gpu_queue.py run --gb N -- cmd ...")
        return 2
    if not a.skip_import_check and os.environ.get("GPU_QUEUE_IMPORT_CHECK", "1") != "0":
        pre = import_preflight(cmd)
        if pre:
            hard, soft = pre
            if soft:
                _log(
                    f"note: {cmd[0]} cannot import {', '.join(soft)} (used inside functions of the script); "
                    f"fine if that path does not run"
                )
            if hard:
                _log(
                    f"{cmd[0]} cannot import {', '.join(hard)}, which the script imports at module level, so it "
                    f"would fail as soon as it got a GPU. Install them into that interpreter first "
                    f"(uv pip install --python {cmd[0]} ...), or pass --skip-import-check."
                )
                return 2
    sig = a.sig or default_sig(cmd)
    legacy = None if a.sig else legacy_sig(cmd)
    measured = MemCache().lookup(sig, legacy) or {}
    gb = a.gb
    if gb is None:
        gb = reservation_gb(measured["peak_gb"]) if measured.get("peak_gb") else None
        if gb is None and (a.model or a.params_b):
            gb = estimate_gb(a.params_b, a.model, a.load, a.mode, a.seq, a.batch)
        if gb is None:
            _log(
                "no --gb and no measured peak for this command: pass --gb (or --model / --params-b)"
            )
            return 2
    elif (
        not a.gb_exact
        and measured.get("peak_gb")
        and measured.get("n", 0) >= MemCache.TRUSTED_N
        and abs(reservation_gb(measured["peak_gb"]) - gb) >= 1.0
    ):
        # a guess passed as --gb loses to what this workload measured over several runs, both ways:
        # an over-sized guess blocks packing, an under-sized one lets a neighbour in to OOM it
        new = reservation_gb(measured["peak_gb"])
        _log(
            f"--gb {gb:g} -> {new:g}: measured peak {measured['peak_gb']:g} GB over {measured['n']} runs "
            f"(--gb-exact keeps --gb)"
        )
        gb = new
    est_s = a.est_min * 60 if a.est_min else measured.get("runtime_s")
    offload = a.offload or default_offload(cmd, a.remote_cmd, a.exclusive, a.perf_remote_ok)
    if a.exclusive and not a.est_min:
        _log("--exclusive holds a whole card: pass --est-min M (how long it holds it)")
        return 2
    refusal = local_refusal(gb, a.exclusive, est_s) if no_local_gpu() else ""
    if refusal:
        offload = "force"  # remote only: no local lease, all arms on one remote card for timing
        _log(f"--no-local-gpu session: {refusal}, so Colab / Kaggle only")
    elif a.exclusive and not a.b200:
        # Timing compares arms on one card; a remote card does that as well as a B200 and keeps a whole
        # local card out of the shared pool. --b200: numbers that must come from this host's B200s.
        refusal = "timing work (--exclusive) without --b200"
        offload = "force"
        _log(
            f"{refusal} runs on Colab / Kaggle, every arm on one remote card; --b200 keeps it on a local B200"
        )
    remote_job = None
    if offload != "never":
        rcmd = shlex.split(a.remote_cmd) if a.remote_cmd else cmd
        remote_job = remote_job_from_cmd(
            rcmd, gb, a.dtype, est_s, a.exclusive, a.job_class, a.what or " ".join(cmd)[:80], sig
        )
        if remote_job is None and refusal:
            _log(
                f"{PORTABLE_ERR}, and {refusal} does not run locally: ship it as a script / notebook "
                f"(notebook_cloud_run.py), use amd_ci_workflow, "
                + (
                    "or shrink it to a small check"
                    if no_local_gpu()
                    else "or pass --b200 if the numbers must come from a local B200"
                )
            )
            return EXIT_REMOTE_ONLY
        if remote_job is None:
            _log(f"--offload {offload}: {PORTABLE_ERR}; local only")
    env = nvsmi.path_env()  # the job and its shells reach the nvidia-smi shim, whatever PATH we got
    if not env.get("HF_TOKEN"):
        tok = hf_read_token()
        if tok:
            env["HF_TOKEN"] = tok
    # --gb-exact asks for exact sizing: such a lease counts in full unless --observed-admit opts it in
    discount = not a.gb_exact or a.observed_admit
    with acquire(
        gb,
        exclusive = a.exclusive,
        what = a.what or " ".join(cmd)[:80],
        est_s = est_s,
        dtype = a.dtype,
        wait_s = a.wait_min * 60,
        offload = offload,
        remote_job = remote_job,
        discount = discount,
        gpus = [] if refusal else None,
    ) as (gpu, genv):
        if refusal and gpu != "remote":
            _log(
                f"{refusal}: the Colab / Kaggle attempt did not run it (credits, capacity); retry later, ship it "
                "via notebook_cloud_run.py / amd_ci_workflow"
                + ("" if no_local_gpu() else ", or pass --b200")
            )
            return EXIT_REMOTE_ONLY
        if gpu == "remote":
            r = genv.get("result") or {}
            print(json.dumps(r, indent = 1, default = str))
            return 0 if r.get("status") == "PASS" else 1
        env.update(genv)
        if gpu is not None:
            _log(
                f"GPU {gpu}: {gb:g} GB{' exclusive' if a.exclusive else ''} for {shlex.join(cmd)[:120]}"
            )
        return _run_child(
            cmd,
            env,
            gpu,
            gb,
            sig,
            est_s,
            label = None if a.sig else normalized_cmd(cmd),
            lease = getattr(GRANT, "last", None),
        )


def cmd_status(_a):
    gpus = pool_gpus() or host_gpus()
    now = time.time()
    st = gather_state(gpus, apps = query_apps())
    for g in gpus:
        s = st["gpus"][g]
        print(
            f"GPU {g}: {s['free_gb']:.1f}/{s['total_gb']:.1f} GB free, foreign {s['foreign_gb']:.1f} GB"
        )
        for k, h in s["holders"].items():
            left = f", ~{(h['est_end'] - now) / 60:.0f} min left" if h.get("est_end") else ""
            adm = admit_gb(h, now)
            obs = (
                f" (counts {adm:.1f}: peak {float(h['peak_gb']):.1f})"
                if adm < float(h["gb"])
                else " (grew: full)"
                if h.get("discount_revoked")
                else ""
            )
            print(
                f"  {k:<14} {float(h['gb']):6.1f} GB{obs}{' EXCL' if h.get('exclusive') else ''}"
                f" {h.get('user', '?')}/{h.get('ws', '')} {h.get('what', '')[:60]}"
                f" ({(now - float(h.get('t', now))) / 60:.0f} min{left})"
            )
    tk = st["tickets"]
    print(f"queue: {len(tk)} waiting")
    for t in tk:
        w = predict_wait(t["gb"], t.get("exclusive"), now = now, state = st, me = t)
        eta = "never" if math.isinf(w) else f"~{w / 60:.0f} min"
        print(
            f"  {t['id']} {float(t['gb']):6.1f} GB{' EXCL' if t.get('exclusive') else ''}"
            f" {t.get('user', '?')}/{t.get('ws', '')} waited {(now - t['t_submit']) / 60:.0f} min,"
            f" start {eta}{', draining GPU ' + str(t['drain']) if t.get('drain') is not None else ''}"
            f" {t.get('what', '')[:50]}"
        )
    return 0


def cmd_estimate(a):
    try:
        peak = estimate_peak_gb(
            float(a.params_b) if a.params_b else model_params_b(a.model),
            a.load,
            a.mode,
            a.seq,
            a.batch,
        )
    except ValueError as e:
        _log(str(e))
        return 2
    print(json.dumps({"peak_gb": round(peak, 1), "reserve_gb": reservation_gb(peak)}))
    return 0


def cmd_reclaim(_a):
    gpus = pool_gpus() or host_gpus()
    reclaim(gpus)
    for g in gpus:
        print(f"gpu{g}: {json.dumps(leases(g))}")
    return 0


def _est_args(p):
    p.add_argument("--model", help = "HF repo id (parameter count from Hub metadata)")
    p.add_argument("--params-b", type = float, help = "parameter count in billions")
    p.add_argument("--load", default = "16bit", choices = sorted(LOAD_BYTES))
    p.add_argument("--mode", default = "infer", choices = ["infer", "lora", "full"])
    p.add_argument("--seq", type = int, default = 2048)
    p.add_argument("--batch", type = int, default = 1)


def main(argv = None):
    p = argparse.ArgumentParser(description = "Host-wide GPU queue (see module doc)")
    sub = p.add_subparsers(dest = "cmd_name", required = True)
    r = sub.add_parser("run", help = "lease a GPU (or offload) and run a command")
    r.add_argument(
        "--gb", type = float, help = "VRAM to reserve (default: measured peak for --sig, else estimate)"
    )
    r.add_argument(
        "--exclusive",
        action = "store_true",
        help = "timing work: one card to itself; needs --est-min; runs on Colab / Kaggle (one remote "
        "card for every arm) unless --b200",
    )
    r.add_argument(
        "--b200",
        action = "store_true",
        help = "with --exclusive: time on a local B200 (Blackwell kernels, B200 throughput claims); at "
        "most GPU_QUEUE_MAX_EXCL (2) cards are exclusive host-wide",
    )
    r.add_argument(
        "--dtype", choices = ["fp16", "bf16", "fp8", "nvfp4", "fp32"], help = "needed compute dtype"
    )
    r.add_argument("--est-min", type = float, help = "expected runtime (min), for wait prediction")
    r.add_argument("--wait-min", type = float, default = 24 * 60, help = "give up after this long")
    r.add_argument(
        "--offload",
        choices = ["auto", "never", "force"],
        default = None,
        help = "default: auto for a portable job (python x.py / x.ipynb), never for other "
        "commands and for --exclusive --b200 timing work (unless --perf-remote-ok)",
    )
    r.add_argument(
        "--perf-remote-ok",
        action = "store_true",
        help = "with --b200: may still go remote after a long local wait (timings then from that GPU)",
    )
    r.add_argument("--remote-cmd", help = "command to run remotely instead of the local one")
    r.add_argument(
        "--job-class", default = "train", choices = ["train", "eval", "decode", "generate", "export"]
    )
    r.add_argument("--what", default = "")
    r.add_argument(
        "--sig", help = "measured-peak cache key (default: hash of the normalised command)"
    )
    r.add_argument(
        "--gb-exact",
        action = "store_true",
        help = "reserve exactly --gb even when this command's measured peak (3+ runs) says otherwise; "
        "the lease then also always counts in full (no observed admission)",
    )
    r.add_argument(
        "--observed-admit",
        action = "store_true",
        help = "with --gb-exact: still let the lease count at its measured peak once it has run "
        "GPU_QUEUE_OBSERVED_AFTER_S under half its reservation",
    )
    r.add_argument(
        "--skip-import-check",
        action = "store_true",
        help = "queue even when the script's module-level imports do not resolve (GPU_QUEUE_IMPORT_CHECK=0)",
    )
    _est_args(r)
    r.add_argument("cmd", nargs = argparse.REMAINDER)
    sub.add_parser("status", help = "leases, queue and predicted waits")
    sub.add_parser("reclaim", help = "drop leases / tickets of dead processes")
    e = sub.add_parser("estimate", help = "VRAM reservation estimate")
    _est_args(e)
    a = p.parse_args(argv)
    return {"run": cmd_run, "status": cmd_status, "estimate": cmd_estimate, "reclaim": cmd_reclaim}[
        a.cmd_name
    ](a)


if __name__ == "__main__":
    raise SystemExit(main())
