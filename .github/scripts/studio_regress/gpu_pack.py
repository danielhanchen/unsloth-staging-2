"""Memory-aware GPU packing + cross-session leases (studio_regress's view of scripts/gpu_queue.py).

    python studio_regress.py gpu-pack --plan items.json          # print placement
    python studio_regress.py gpu-pack --status                    # current leases per GPU
    python studio_regress.py gpu-pack --reclaim                   # drop leases of dead PIDs

The lease store, host-wide queue, wait prediction and offload live in scripts/gpu_queue.py, shared
with gpu_pool_runner.sh and ad-hoc `gpu_queue.py run` jobs, so every user and workspace sees one
view. This module keeps the names studio_regress (scheduler, external, core, tests) already uses.

Pool: pool_gpus(). A session with a GPU mask may lease ANY host GPU ($STUDIO_REGRESS_GPUS: `all`
default, `mask` = only $CUDA_VISIBLE_DEVICES, or a list); a session with no mask stays GPU-free.

Leases: <lock dir>/gpu<N>.json = {"holders": {"<pid>[:<n>]": {"gb", "exclusive", "what", "t", "start",
"boot", ...}}}, read-modify-written under a file lock on gpu<N>.lock. A holder whose PID is gone (or
was reused: /proc start time differs) is reclaimed.

Placement (non-perf items): best fit by gpu_mem_gb against min(live free, total - leased) - headroom.
Perf-sensitive items need a GPU with NO other holder and take an exclusive lease. Each placed
process gets env_for(): CUDA_VISIBLE_DEVICES=<one>, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
and UNSLOTH_REGRESS_MEM_FRACTION (jobs/_common applies it via torch.cuda.set_per_process_memory_fraction).
"""

from __future__ import annotations

import contextlib
import json
import subprocess  # noqa: F401  tests patch gp.subprocess.run (the shared module)
import sys
import tempfile
import time  # noqa: F401  tests patch gp.time.sleep (the shared module)
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import gpu_queue as gq  # noqa: E402

HEADROOM_GB = gq.HEADROOM_GB
# Leases must be HOST-wide: every tmux review session runs in its own workspace_N and the unix users
# share the GPUs. $STUDIO_REGRESS_LOCK_DIR (or $GPU_QUEUE_LOCK_DIR) overrides (tests).
_SHARED_LOCK_DIR = Path("/mnt/disks/unslothai/shared/studio-regress-locks")
_FALLBACK_LOCK_DIR = Path(tempfile.gettempdir()) / "unsloth-studio-regress-locks"
_HOST_GPUS = []
QUERY_ERROR = gq.QUERY_ERROR
OFFLOAD_AFTER_S = gq.OFFLOAD_S
OFFLOAD_HINT = gq.OFFLOAD_HINT
offload_hint = gq.offload_hint
visible_gpus = gq.visible_gpus
query_gpus = gq.query_gpus
plan = gq.plan
env_for = gq.env_for
_open_shared = gq._open_shared
_mkdir_shared = gq._mkdir_shared


def lock_dir_default():
    return gq.resolve_lock_dir(_SHARED_LOCK_DIR, _FALLBACK_LOCK_DIR)


def host_gpus():
    return gq.host_gpus(_HOST_GPUS)


def pool_gpus(env = None):
    return gq.pool_gpus(env, host = host_gpus)


def snapshot(gpu, timeout = 60):
    """What other tenants left on one physical GPU right now: {"gpu", "free_mib", "total_mib",
    "foreign_mib", "procs", "t"}. Taken at lease time, before the leased work starts, so every compute
    process on the device is someone else's. None when nvidia-smi cannot answer."""
    try:
        q = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(gpu),
                ],
                capture_output = True,
                text = True,
                timeout = timeout,
                check = True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        _idx, uuid, tot, free = [x.strip() for x in q.split(",")]
        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output = True,
            text = True,
            timeout = timeout,
        ).stdout
    except Exception:  # noqa: BLE001
        return None
    used = []
    for line in apps.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3 and parts[0] == uuid and parts[2].isdigit():
            used.append(int(parts[2]))
    return {
        "gpu": str(gpu),
        "free_mib": int(float(free)),
        "total_mib": int(float(tot)),
        "foreign_mib": sum(used),
        "procs": len(used),
        "t": round(time.time(), 1),
    }


def _pid_alive(pid):
    return gq.plat.pid_alive(pid)


def leases(gpu, lock_dir = None):
    return gq.leases(gpu, lock_dir or lock_dir_default())


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
):
    """Reserve `gb` on `gpu` for `pid`; False if it does not fit / exclusivity conflicts."""
    return gq.try_acquire(
        gpu,
        gb,
        exclusive,
        what,
        pid = pid,
        total_gb = total_gb,
        free_gb = free_gb,
        headroom = headroom,
        lock_dir = lock_dir or lock_dir_default(),
    )


def release(
    gpu,
    pid = None,
    lock_dir = None,
):
    gq.release(gpu, pid = pid, lock_dir = lock_dir or lock_dir_default())


def reclaim(gpus, lock_dir = None):
    gq.reclaim(gpus, lock_dir or lock_dir_default())


def items_from_targets(targets):
    """Switchboard targets of any kind (journey / job / external) -> plan() items; CPU-only
    targets (gpu_mem_gb 0) need no placement and are left out."""
    return [
        {"name": t["name"], "gb": float(t["gpu_mem_gb"]), "perf": bool(t.get("perf_sensitive"))}
        for t in targets
        if float(t.get("gpu_mem_gb") or 0) > 0
    ]


@contextlib.contextmanager
def lease(
    gb,
    exclusive = False,
    what = "",
    wait_s = 3600,
    poll_s = 15,
    env = None,
    avoid = (),
):
    """Block until some pool GPU can take `gb` (exclusive if asked); yield (gpu, env).
    `avoid`: GPUs this process already leased (a target taking two distinct GPUs)."""
    with gq.acquire(
        gb,
        exclusive = exclusive,
        what = what,
        wait_s = wait_s,
        poll_s = poll_s,
        env = env if env is not None else None,
        avoid = avoid,
        lock_dir = lock_dir_default(),
        query = query_gpus,
        gpus = pool_gpus(env),
    ) as got:
        yield got


def main(argv = None):
    import argparse

    p = argparse.ArgumentParser(description = "Memory-aware GPU packing and leases")
    p.add_argument("--plan", help = "JSON list of {name, gb, perf}")
    p.add_argument("--status", action = "store_true")
    p.add_argument("--reclaim", action = "store_true")
    a = p.parse_args(argv)
    gpus = pool_gpus()
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
            existing[g] = {
                "leased_gb": sum(h["gb"] for h in hs.values()),
                "holders": len(hs),
                "exclusive": any(h.get("exclusive") for h in hs.values()),
            }
        print(json.dumps(plan(items, info, existing), indent = 1))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
