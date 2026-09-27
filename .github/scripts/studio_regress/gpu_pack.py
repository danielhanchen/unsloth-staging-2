"""Memory-aware GPU packing + cross-session leases.

    python studio_regress.py gpu-pack --plan items.json          # print placement
    python studio_regress.py gpu-pack --status                    # current leases per GPU
    python studio_regress.py gpu-pack --reclaim                   # drop leases of dead PIDs

Only devices in $CUDA_VISIBLE_DEVICES are ever used (read at runtime, never widened).

Leases: <lock dir>/gpu<N>.json (host-wide, see lock_dir_default) = {"holders": {"<pid>": {"gb": x, "exclusive": b,
"what": str, "t": epoch}}}, read-modify-written under a file lock (plat.lock) on gpu<N>.lock, so every
tmux review session on the box shares one view. A holder whose PID is gone is reclaimed.

Placement (non-perf items): first-fit decreasing by gpu_mem_gb against
min(live free MiB from nvidia-smi, total - leased) - headroom. Perf-sensitive items need a GPU
with NO other holder and take an exclusive lease (both arms back to back there).
Each placed process gets env_for(): CUDA_VISIBLE_DEVICES=<one>, PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True and UNSLOTH_REGRESS_MEM_FRACTION (jobs/_common and the Studio
launch apply it via torch.cuda.set_per_process_memory_fraction), so an overrunning item OOMs
itself, not its neighbours.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from studio_regress import plat

HEADROOM_GB = 4.0
# Leases must be HOST-wide: every tmux review session runs in its own workspace_N, and the four unix
# users share the GPUs, so a per-workspace dir would give each session an empty view. Same layout as
# gh_budget.py: the 1777 shared parent, else /tmp; $STUDIO_REGRESS_LOCK_DIR overrides (tests).
_SHARED_LOCK_DIR = Path("/mnt/disks/unslothai/shared/studio-regress-locks")
_FALLBACK_LOCK_DIR = Path(tempfile.gettempdir()) / "unsloth-studio-regress-locks"


def lock_dir_default():
    env = os.environ.get("STUDIO_REGRESS_LOCK_DIR")
    if env:
        return Path(env)
    # The existing shared dir first: it is 0777, but its parent is writable only by the user who
    # made it, so testing only the parent sent every other user to /tmp and split the leases.
    if _SHARED_LOCK_DIR.is_dir():
        return _SHARED_LOCK_DIR if os.access(_SHARED_LOCK_DIR, os.W_OK | os.X_OK) else _FALLBACK_LOCK_DIR
    return _SHARED_LOCK_DIR if os.access(_SHARED_LOCK_DIR.parent, os.W_OK) else _FALLBACK_LOCK_DIR


def _open_shared(path):
    """Open (creating) a lease file every user can rewrite."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        os.fchmod(fd, 0o666)
    except (OSError, AttributeError):   # no os.fchmod on Windows before 3.13
        pass   # created by another user; its mode is already 0666
    return os.fdopen(fd, "r+")


def _mkdir_shared(d):
    # 0777 but NOT sticky: in a sticky dir one user could not replace another user's gpu<N>.json.
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o777)
    except OSError:
        pass


def visible_gpus(env=None):
    env = os.environ if env is None else env
    raw = (env.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not raw or raw in ("-1", "none", "NoDevFiles"):
        return []
    return [g.strip() for g in raw.split(",") if g.strip()]


QUERY_ERROR = {}   # the last nvidia-smi failure, for lease()'s waiting message


def query_gpus(ids):
    """{id: {"total_gb", "free_gb"}} for the given physical ids via nvidia-smi.

    A failed query reads as 0 GB free (never place work on an unknown GPU) and is recorded in
    QUERY_ERROR, so a caller that keeps waiting can say it is blind rather than that GPUs are full."""
    if not ids:
        return {}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.free",
                              "--format=csv,noheader,nounits", "-i", ",".join(ids)],
                             capture_output=True, text=True, timeout=20, check=True).stdout
        QUERY_ERROR.clear()
    except Exception as e:
        QUERY_ERROR.update(error=f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}",
                           at=time.time())
        return {i: {"total_gb": 0.0, "free_gb": 0.0} for i in ids}
    res = {}
    for line in out.strip().splitlines():
        idx, tot, free = [x.strip() for x in line.split(",")]
        res[idx] = {"total_gb": float(tot) / 1024, "free_gb": float(free) / 1024}
    return res


def _pid_alive(pid):
    return plat.pid_alive(pid)   # never os.kill(pid, 0): on Windows that terminates the process


@contextlib.contextmanager
def _locked(gpu, lock_dir=None):
    lock_dir = Path(lock_dir or lock_dir_default())
    _mkdir_shared(lock_dir)
    with _open_shared(lock_dir / f"gpu{gpu}.lock") as fh:
        plat.lock(fh)
        path = lock_dir / f"gpu{gpu}.json"
        try:
            state = json.loads(path.read_text()) if path.exists() else {}
        except ValueError:
            state = {}
        state.setdefault("holders", {})
        # stale-PID reclaim on every touch
        state["holders"] = {p: h for p, h in state["holders"].items() if _pid_alive(p)}
        yield state
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(state, indent=1))
        os.chmod(tmp, 0o666)
        tmp.replace(path)
        plat.unlock(fh)


def leases(gpu, lock_dir=None):
    with _locked(gpu, lock_dir) as st:
        return dict(st["holders"])


def try_acquire(gpu, gb, exclusive=False, what="", pid=None, total_gb=None, free_gb=None,
                headroom=HEADROOM_GB, lock_dir=None):
    """Reserve `gb` on `gpu` for `pid`; False if it does not fit / exclusivity conflicts."""
    pid = str(pid or os.getpid())
    with _locked(gpu, lock_dir) as st:
        holders = {p: h for p, h in st["holders"].items() if p != pid}
        if exclusive and holders:
            return False
        if any(h.get("exclusive") for h in holders.values()):
            return False
        leased = sum(h["gb"] for h in holders.values())
        if total_gb is not None:
            room = total_gb - leased - headroom
            if free_gb is not None:
                room = min(room, free_gb - headroom)
            if gb > room:
                return False
        st["holders"][pid] = {"gb": gb, "exclusive": bool(exclusive), "what": what, "t": time.time()}
        return True


def release(gpu, pid=None, lock_dir=None):
    with _locked(gpu, lock_dir) as st:
        st["holders"].pop(str(pid or os.getpid()), None)


def reclaim(gpus, lock_dir=None):
    for g in gpus:
        with _locked(g, lock_dir):
            pass


def plan(items, gpus_info, existing=None, headroom=HEADROOM_GB):
    """Pure placement. items: [{"name", "gb", "perf": bool}] -> {name: gpu | None}.

    existing: {gpu: {"leased_gb", "exclusive": bool, "holders": n}} from current leases."""
    existing = existing or {}
    room, busy = {}, {}
    for g, inf in gpus_info.items():
        ex = existing.get(g, {})
        cap = inf["total_gb"] - ex.get("leased_gb", 0.0)
        room[g] = min(cap, inf.get("free_gb", cap)) - headroom
        busy[g] = ex.get("holders", 0) > 0 or ex.get("exclusive", False)
    # A GPU someone already holds exclusively takes nothing else (try_acquire refuses it too).
    placed, exclusive_taken = {}, {g for g in room if existing.get(g, {}).get("exclusive")}
    # perf-sensitive first: each needs an idle GPU to itself
    for it in [i for i in items if i.get("perf")]:
        g = next((g for g in sorted(room, key=lambda g: -room[g])
                  if not busy[g] and g not in exclusive_taken and room[g] >= it["gb"]), None)
        placed[it["name"]] = g
        if g is not None:
            exclusive_taken.add(g)
    for it in sorted([i for i in items if not i.get("perf")], key=lambda i: -i["gb"]):
        g = next((g for g in sorted(room, key=lambda g: room[g])  # tightest fit first
                  if g not in exclusive_taken and room[g] >= it["gb"]), None)
        placed[it["name"]] = g
        if g is not None:
            room[g] -= it["gb"]
    return placed


def items_from_targets(targets):
    """Switchboard targets of any kind (journey / job / external) -> plan() items; CPU-only
    targets (gpu_mem_gb 0) need no placement and are left out."""
    return [{"name": t["name"], "gb": float(t["gpu_mem_gb"]), "perf": bool(t.get("perf_sensitive"))}
            for t in targets if float(t.get("gpu_mem_gb") or 0) > 0]


def env_for(gpu, gb, total_gb):
    frac = min(0.95, max(0.05, (gb + 1.0) / total_gb)) if total_gb else 0.95
    return {"CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "UNSLOTH_REGRESS_MEM_FRACTION": f"{frac:.3f}"}


@contextlib.contextmanager
def lease(gb, exclusive=False, what="", wait_s=3600, poll_s=15, env=None, avoid=()):
    """Block until some visible GPU can take `gb` (exclusive if asked); yield (gpu, env).
    `avoid`: GPUs this process already leased (a target taking two distinct GPUs)."""
    gpus = [g for g in visible_gpus(env) if g not in set(map(str, avoid))]
    if not gpus or gb <= 0:
        yield None, {}
        return
    deadline = time.time() + wait_s
    started, said = time.time(), 0.0
    while True:
        info = query_gpus(gpus)
        for g in sorted(gpus, key=lambda g: -info.get(g, {}).get("free_gb", 0)):
            inf = info.get(g, {"total_gb": 0, "free_gb": 0})
            if try_acquire(g, gb, exclusive, what, total_gb=inf["total_gb"], free_gb=inf["free_gb"]):
                try:
                    yield g, env_for(g, gb, inf["total_gb"])
                finally:
                    release(g)
                return
        if time.time() - said >= 60:   # say why at most once a minute, not silently for an hour
            said = time.time()
            why = (f"nvidia-smi query failing ({QUERY_ERROR['error']})" if QUERY_ERROR else
                   ", ".join(f"GPU {g}: {info.get(g, {}).get('free_gb', 0):.1f} GB free, "
                             f"{len(leases(g))} lease(s)" for g in gpus))
            print(f"[gpu_pack] {what or 'unit'} waiting {time.time() - started:.0f}s for {gb} GB"
                  f"{' exclusive' if exclusive else ''}: {why}", file=sys.stderr, flush=True)
        if time.time() > deadline:
            raise TimeoutError(f"no GPU in {gpus} could take {gb} GB (exclusive={exclusive})"
                               + (f"; nvidia-smi failing: {QUERY_ERROR['error']}" if QUERY_ERROR else ""))
        time.sleep(poll_s)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Memory-aware GPU packing and leases")
    p.add_argument("--plan", help="JSON list of {name, gb, perf}")
    p.add_argument("--status", action="store_true")
    p.add_argument("--reclaim", action="store_true")
    a = p.parse_args(argv)
    gpus = visible_gpus()
    if a.reclaim or a.status:
        reclaim(gpus)
        for g in gpus:
            print(f"gpu{g}: {json.dumps(leases(g))}")
        return 0
    if a.plan:
        items = json.loads(Path(a.plan).read_text())
        info = query_gpus(gpus)
        existing = {}
        for g in gpus:
            hs = leases(g)
            existing[g] = {"leased_gb": sum(h["gb"] for h in hs.values()), "holders": len(hs),
                           "exclusive": any(h.get("exclusive") for h in hs.values())}
        print(json.dumps(plan(items, info, existing), indent=1))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
