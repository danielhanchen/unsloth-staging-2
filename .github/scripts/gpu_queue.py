"""Host-wide GPU queue: VRAM-packed shared leases, exclusive leases for timing, predicted waits, offload.

    python gpu_queue.py run --gb 12 -- python train.py            # shared: packed by VRAM
    python gpu_queue.py run --gb 40 --exclusive --est-min 30 -- python bench.py   # timing: GPU to itself
    python gpu_queue.py run --gb 8 --dtype bf16 --offload auto --est-min 20 -- python check.py
    python gpu_queue.py status | reclaim
    python gpu_queue.py estimate --model unsloth/Llama-3.2-1B-Instruct --load 4bit --mode lora

One view for every user and workspace: <lock dir>/gpu<N>.json (flock'd holders, read-modify-written
under gpu<N>.lock) plus <lock dir>/switchboard/{queue/*.json, mem_cache.json}. The lock dir is the
0777 shared dir studio_regress already uses, so studio_regress units, gpu_pool_runner.sh jobs and
ad-hoc `run`s all see each other. The launcher's per-workspace CUDA_VISIBLE_DEVICES mask is only a
home preference: a session with any GPU leases from every host GPU ($GPU_QUEUE_GPUS / $STUDIO_REGRESS_GPUS:
`all` default, `mask`, or a list).

Holder = {"gb", "exclusive", "what", "t", "start", "boot", "est_end", "user", "ws"} keyed "<pid>" (one
per process, old callers) or "<pid>:<n>" (one per job). A holder whose pid is gone, whose boot id
differs, or whose /proc start time differs (PID reuse) is reclaimed on the next touch. A corrupt
gpu<N>.json is renamed aside and the GPU takes nothing for QUARANTINE_S (its holders are unknown).

Admission (shared): room = min(total - leased, free - pending) - headroom, where pending is what
holders granted in the last PENDING_S have reserved but not yet allocated. Best fit (tightest room),
so empty GPUs stay free for timing work. Exclusive: no other holder and no foreign compute process
over FOREIGN_GB on the card (a co-tenant moves step time).

Queue: a waiter writes switchboard/queue/<id>.json. Priority is submit time, exclusive tickets
EXCL_PRIORITY_S earlier. An exclusive ticket (or a shared one waiting over STARVE_S) names a drain GPU
(soonest to fit it); tickets behind it place nothing new there, so big and exclusive jobs are not
starved by a stream of small ones. predict_wait() simulates the same policy.

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

HEADROOM_GB = 4.0
PENDING_S = 120.0  # a fresh grant may not have allocated yet: count its full reservation
QUARANTINE_S = 300.0
FOREIGN_GB = float(os.environ.get("GPU_QUEUE_FOREIGN_GB", "1.0"))
# Idle CUDA contexts (a loaded-but-idle model, a notebook kernel) hold GBs on most cards of a shared
# host; memory alone would starve every exclusive request. Foreign memory blocks timing work only
# while the card is also busy (utilization above this), or when utilization is unknown.
FOREIGN_UTIL_PCT = float(os.environ.get("GPU_QUEUE_FOREIGN_UTIL_PCT", "10"))
FOREIGN_IGNORE_GB = (
    float(os.environ.get("GPU_QUEUE_FOREIGN_IGNORE_MIB", "1024")) / 1024
)  # per process
# Local first: remote only after really waiting. Offload once waited >= OFFLOAD_MIN_WAIT_S with a
# predicted remaining wait >= OFFLOAD_PRED_S, or unconditionally at waited >= OFFLOAD_S; never on a
# prediction alone at t=0. A failed remote attempt is retried at most every REMOTE_RETRY_S.
OFFLOAD_S = float(os.environ.get("GPU_QUEUE_OFFLOAD_S", "1800"))
OFFLOAD_MIN_WAIT_S = float(os.environ.get("GPU_QUEUE_OFFLOAD_MIN_WAIT_S", "600"))
OFFLOAD_PRED_S = float(os.environ.get("GPU_QUEUE_OFFLOAD_PRED_S", "1200"))
REMOTE_RETRY_S = float(os.environ.get("GPU_QUEUE_REMOTE_RETRY_S", "600"))
EXCL_PRIORITY_S = 600.0
STARVE_S = 600.0
OVERDUE_S = 300.0  # a holder past its est_end is assumed to finish this much later
DEFAULT_EST_S = {False: 1800.0, True: 3600.0}  # unknown duration, by exclusive

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


def _write_shared(path, obj):
    tmp = path.with_suffix(f".tmp{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(json.dumps(obj, indent = 1))
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
    """Every GPU index nvidia-smi lists (cached after the first good answer), [] when it cannot say."""
    cache = _HOST_GPUS if cache is None else cache
    if not cache:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output = True,
                text = True,
                timeout = 60,
                check = True,
            ).stdout
            cache.extend(ln.strip() for ln in out.splitlines() if ln.strip().isdigit())
        except Exception:  # noqa: BLE001
            return []
    return list(cache)


def pool_gpus(env = None, host = None):
    """GPUs this process may lease: every host GPU for a session with a GPU mask, none for a CPU one
    (CUDA_VISIBLE_DEVICES set but empty / -1). Unset means CUDA sees every GPU, so the whole host
    (e.g. a plain login shell of another unix user): running such a job unpinned would bypass the queue."""
    env = os.environ if env is None else env
    mine = visible_gpus(env)
    mode = (env.get("GPU_QUEUE_GPUS") or env.get("STUDIO_REGRESS_GPUS") or "all").strip()
    if "CUDA_VISIBLE_DEVICES" not in env:
        mine = (host or host_gpus)()
    if not mine or mode == "mask":
        return mine
    if mode != "all":
        return [g.strip() for g in mode.split(",") if g.strip()]
    return (host or host_gpus)() or mine


def query_gpus(ids):
    """{id: {"total_gb", "free_gb", "util"}}. A failed query reads as 0 GB free and is recorded in
    QUERY_ERROR; util is None when nvidia-smi does not report it."""
    if not ids:
        return {}
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                ",".join(ids),
            ],
            capture_output = True,
            text = True,
            timeout = 20,
            check = True,
        ).stdout
        QUERY_ERROR.clear()
    except Exception as e:  # noqa: BLE001
        QUERY_ERROR.update(
            error = f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}",
            at = time.time(),
        )
        return {i: {"total_gb": 0.0, "free_gb": 0.0} for i in ids}
    res = {}
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        util = None
        if len(parts) > 3:
            with contextlib.suppress(ValueError):
                util = float(parts[3])
        res[parts[0]] = {
            "total_gb": float(parts[1]) / 1024,
            "free_gb": float(parts[2]) / 1024,
            "util": util,
        }
    return res


def foreign_blocks(s):
    """True when foreign (unleased) GPU use rules out timing work on this card (see FOREIGN_UTIL_PCT)."""
    if s.get("foreign_gb", 0.0) <= FOREIGN_GB:
        return False
    util = s.get("util")
    return util is None or util > FOREIGN_UTIL_PCT


def query_apps():
    """{gpu index: [(pid, used_gb)]} of compute processes; {} when nvidia-smi cannot say."""
    try:
        idx = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output = True,
            text = True,
            timeout = 20,
            check = True,
        ).stdout
        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output = True,
            text = True,
            timeout = 20,
            check = True,
        ).stdout
    except Exception:  # noqa: BLE001
        return {}
    by_uuid = {}
    for ln in idx.strip().splitlines():
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) == 2:
            by_uuid[parts[1]] = parts[0]
    res = {}
    for ln in apps.strip().splitlines():
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) != 3 or parts[0] not in by_uuid:
            continue
        try:
            res.setdefault(by_uuid[parts[0]], []).append((int(parts[1]), float(parts[2]) / 1024))
        except ValueError:
            continue
    return res


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
):
    """Reserve `gb` on `gpu` under `key` (default the pid); False if it does not fit / exclusivity
    conflicts / the GPU is quarantined. Re-acquiring an existing key replaces its entry (resize)."""
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
            and (prev is None or gb > float(prev.get("gb", 0)))
            and drained(gpu, lock_dir, pid)
        ):
            return False
        holders = {k: h for k, h in st["holders"].items() if k != key}
        if exclusive and holders:
            return False
        if exclusive and foreign_gb is not None and foreign_gb > FOREIGN_GB:
            return False
        if any(h.get("exclusive") for h in holders.values()):
            return False
        leased = sum(float(h["gb"]) for h in holders.values())
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
        st["holders"][key] = h
        return True


def drained(
    gpu,
    ld = None,
    pid = None,
):
    """A live ticket of another process is draining `gpu` (see pick_drain)."""
    pid = int(pid or os.getpid())
    try:
        return any(
            t.get("drain") is not None
            and str(t["drain"]) == str(gpu)
            and int(t.get("pid") or 0) != pid
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


def reclaim(gpus, lock_dir = None):
    for g in gpus:
        with _locked(g, lock_dir):
            pass
    for _t in tickets(lock_dir):  # read_tickets drops dead owners' tickets
        pass


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
        return not mine and not foreign_blocks(inf) and inf["total_gb"] - headroom >= gb
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
            "gb": float(h["gb"]),
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
                g = min(cands, key = lambda g: (gs[g]["total_gb"] - used[g], g))  # best fit
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
        if not apps:  # no process view: whatever is used beyond the leases is foreign
            foreign = max(
                0.0,
                inf["total_gb"]
                - inf.get("free_gb", 0.0)
                - sum(float(h["gb"]) for h in hs.values()),
            )
        out[g] = {
            "total_gb": inf["total_gb"],
            "free_gb": inf.get("free_gb", 0.0),
            "foreign_gb": foreign,
            "util": inf.get("util"),
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


def pick_drain(
    state,
    me,
    now = None,
):
    """The GPU that frees soonest for `me` among those no ticket ahead is draining (None: none fits)."""
    now = now or time.time()
    taken = drained_by_ahead(me, state.get("tickets", []))
    best = None
    for g, inf in state.get("gpus", {}).items():
        if g in taken or inf["total_gb"] - HEADROOM_GB - inf.get("foreign_gb", 0.0) < float(
            me["gb"]
        ):
            continue
        if me.get("exclusive") and foreign_blocks(inf):
            continue
        hs = inf.get("holders", {}).values()
        if me.get("exclusive"):
            t = max([_holder_end(h, now) for h in hs], default = now)
        else:
            need = float(me["gb"]) - (
                inf["total_gb"]
                - HEADROOM_GB
                - inf.get("foreign_gb", 0.0)
                - sum(float(h["gb"]) for h in hs)
            )
            t = now
            for h in sorted(hs, key = lambda h: _holder_end(h, now)):
                if need <= 0:
                    break
                need -= float(h["gb"])
                t = _holder_end(h, now)
        if best is None or (t, g) < best:
            best = (t, g)
    return best[1] if best else None


# ------------------------------------------------------------------ acquire
_SEQ = [0]
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
):
    """cloud_pool.dispatch job for `cmd` ([python [-u]] x.py args | x.ipynb), None when not portable."""
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
    }
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


def should_offload(waited, wait):
    """Local first: never at t=0 on a prediction alone. `wait` = predicted remaining (inf = unknown)."""
    if waited >= OFFLOAD_S:
        return True
    return waited >= OFFLOAD_MIN_WAIT_S and not math.isinf(wait) and wait >= OFFLOAD_PRED_S


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
):
    """One placement pass: (gpu, total_gb) or None."""
    info = query(gpus)
    apps = apps_fn() if exclusive else {}
    st = gather_state(gpus, ld, info = info, apps = apps)
    skip = drained_by_ahead(me, st["tickets"])
    cands = []
    for g in gpus:
        if g in skip:
            continue
        s = st["gpus"][g]
        room = (
            min(s["total_gb"] - sum(float(h["gb"]) for h in s["holders"].values()), s["free_gb"])
            - HEADROOM_GB
        )
        if exclusive and s["holders"]:
            continue
        if room >= gb:
            cands.append((room, g))
    # shared: tightest fit keeps empty GPUs for exclusive work; exclusive: the emptiest card
    cands.sort(key = lambda x: (-x[0], x[1]) if exclusive else (x[0], x[1]))
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
            foreign_gb = (s["foreign_gb"] if foreign_blocks(s) else 0.0) if exclusive else None,
        ):
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
    ticketed, said, hinted = False, 0.0, False
    remote, next_remote = None, 0.0  # next_remote: earliest time a new remote attempt may start
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
                next_remote = now + REMOTE_RETRY_S
                _log(
                    f"{what or 'job'}: remote attempt did not run it; waiting locally, remote again in "
                    f"{REMOTE_RETRY_S / 60:.0f} min"
                )
                if offload == "force" and (not gpus or gb <= 0):
                    yield None, {}  # forced offload failed and there is nothing to wait for
                    return
            running_remote = remote is not None and remote.started_ev.is_set()
            got = (
                _attempt(gpus, gb, exclusive, what, key, est_s, me, ld, query, apps_fn)
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
                try:
                    yield g, env_for(g, gb, total)
                finally:
                    release(g, lock_dir = ld, key = key)
                return
            wait, st = math.inf, None
            if gpus and gb > 0:
                try:
                    st = gather_state(gpus, ld, info = query(gpus))
                    wait = predict_wait(gb, exclusive, now = now, state = st, est_s = est_s, me = me)
                    if exclusive or waited > STARVE_S:
                        me["drain"] = pick_drain(st, me, now)
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
            if (
                remote is None
                and not local_only
                and now >= next_remote
                and (offload == "force" or should_offload(waited, wait))
            ):
                remote = RemoteAttempt(remote_job, what)
                _log(
                    f"{what or 'job'}: waited {waited / 60:.0f} min, predicted "
                    f"{'unknown' if math.isinf(wait) else f'~{wait / 60:.0f} min'} more: "
                    "trying Colab / Kaggle while staying in the local queue"
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
                    f"nvidia-smi query failing ({QUERY_ERROR['error']})"
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
                    + (f"; nvidia-smi failing: {QUERY_ERROR['error']}" if QUERY_ERROR else "")
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
          KAGGLE_API_TOKEN 60 h/wk, _2 45 h/wk, billed on notebook wall time
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
    their max; an OOM run is a floor and raises the estimate to 1.5x what it had reached."""

    LOWER_AFTER = 3

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
                _write_shared(self.path, data)
            finally:
                plat.unlock(fh)

    def lookup(self, sig):
        try:
            return json.loads(self.path.read_text()).get(sig)
        except (OSError, ValueError):
            return None

    def reservation(self, sig):
        e = self.lookup(sig)
        return reservation_gb(e["peak_gb"]) if e and e.get("peak_gb") else None

    def record(
        self,
        sig,
        peak_gb,
        runtime_s = None,
        arch = None,
        oom = False,
    ):
        with self._edit() as data:
            e = data.setdefault(sig, {"peak_gb": 0.0, "n": 0, "lower": []})
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
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", str(gpu)],
            capture_output = True,
            text = True,
            timeout = 20,
            check = True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


class PeakSampler(threading.Thread):
    """Peak VRAM (GB) of a process tree on one GPU, from nvidia-smi compute apps every `every` s."""

    def __init__(
        self,
        root_pid,
        gpu,
        every = 5.0,
        apps_fn = query_apps,
    ):
        super().__init__(daemon = True)
        self.root, self.gpu, self.every, self.apps_fn = root_pid, str(gpu), every, apps_fn
        self.peak = 0.0
        self.seen = set()  # every pid of the tree seen: orphans re-parent once the root exits
        self._stop = threading.Event()

    def sample(self):
        tree = plat.descendants(self.root)
        self.seen |= tree
        used = sum(u for p, u in self.apps_fn().get(self.gpu, []) if p in tree)
        self.peak = max(self.peak, used)
        return used

    def run(self):
        while not self._stop.wait(self.every):
            with contextlib.suppress(Exception):
                self.sample()

    def stop(self):
        self._stop.set()


def default_sig(cmd):
    return "cmd:" + hashlib.sha1(" ".join(cmd).encode()).hexdigest()[:16]


def _run_child(cmd, env, gpu, gb, sig, est_s):
    t0 = time.time()
    child = subprocess.Popen(cmd, env = env, **plat.group_kwargs())
    sampler = PeakSampler(child.pid, gpu) if gpu not in (None, "remote") else None
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
            MemCache().record(sig, sampler.peak, time.time() - t0, _gpu_name(gpu), oom = oom)
        _log(f"peak {sampler.peak:.1f} GB of {gb:g} reserved ({sig})")
    return rc


def default_offload(
    cmd,
    remote_cmd = None,
    exclusive = False,
    perf_remote_ok = False,
):
    """auto for a portable job (a python script / notebook plus the files it names), else never.
    Exclusive timing work stays local (B200 numbers) unless perf_remote_ok."""
    if exclusive and not perf_remote_ok:
        return "never"
    c = shlex.split(remote_cmd) if remote_cmd else list(cmd)
    if c and (os.path.basename(c[0]).startswith("python") or c[0] == sys.executable):
        c = [x for x in c[1:] if x not in ("-u", "-B", "-O")]
    return "auto" if c and os.path.isfile(c[0]) and c[0].endswith((".py", ".ipynb")) else "never"


def cmd_run(a):
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        _log("nothing to run: gpu_queue.py run --gb N -- cmd ...")
        return 2
    sig = a.sig or default_sig(cmd)
    gb = a.gb
    if gb is None:
        gb = MemCache().reservation(sig)
        if gb is None and (a.model or a.params_b):
            gb = estimate_gb(a.params_b, a.model, a.load, a.mode, a.seq, a.batch)
        if gb is None:
            _log(
                "no --gb and no measured peak for this command: pass --gb (or --model / --params-b)"
            )
            return 2
    est_s = a.est_min * 60 if a.est_min else (MemCache().lookup(sig) or {}).get("runtime_s")
    offload = a.offload or default_offload(cmd, a.remote_cmd, a.exclusive, a.perf_remote_ok)
    remote_job = None
    if offload != "never":
        rcmd = shlex.split(a.remote_cmd) if a.remote_cmd else cmd
        remote_job = remote_job_from_cmd(
            rcmd, gb, a.dtype, est_s, a.exclusive, a.job_class, a.what or " ".join(cmd)[:80], sig
        )
        if remote_job is None:
            _log(f"--offload {offload}: {PORTABLE_ERR}; local only")
    env = dict(os.environ)
    if not env.get("HF_TOKEN"):
        tok = hf_read_token()
        if tok:
            env["HF_TOKEN"] = tok
    with acquire(
        gb,
        exclusive = a.exclusive,
        what = a.what or " ".join(cmd)[:80],
        est_s = est_s,
        dtype = a.dtype,
        wait_s = a.wait_min * 60,
        offload = offload,
        remote_job = remote_job,
    ) as (gpu, genv):
        if gpu == "remote":
            r = genv.get("result") or {}
            print(json.dumps(r, indent = 1, default = str))
            return 0 if r.get("status") == "PASS" else 1
        env.update(genv)
        if gpu is not None:
            _log(
                f"GPU {gpu}: {gb:g} GB{' exclusive' if a.exclusive else ''} for {shlex.join(cmd)[:120]}"
            )
        return _run_child(cmd, env, gpu, gb, sig, est_s)


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
            print(
                f"  {k:<14} {float(h['gb']):6.1f} GB{' EXCL' if h.get('exclusive') else ''}"
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
    r.add_argument("--exclusive", action = "store_true", help = "timing work: GPU to itself")
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
        "commands and for --exclusive timing work (unless --perf-remote-ok)",
    )
    r.add_argument(
        "--perf-remote-ok",
        action = "store_true",
        help = "an --exclusive timing job may run remotely (timings then from that GPU, not B200)",
    )
    r.add_argument("--remote-cmd", help = "command to run remotely instead of the local one")
    r.add_argument(
        "--job-class", default = "train", choices = ["train", "eval", "decode", "generate", "export"]
    )
    r.add_argument("--what", default = "")
    r.add_argument("--sig", help = "measured-peak cache key (default: hash of the command)")
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
