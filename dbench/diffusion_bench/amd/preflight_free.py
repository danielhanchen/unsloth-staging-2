#!/usr/bin/env python3
"""Fail a gfx1151 job early, and loudly, when another tenant already holds part of the GPU pool.

v6 r2 (run 37196109008) landed on a runner where something outside the job held ~31 GiB of the 62.5 GiB pool before
the first cell: every Studio cell was refused at load and every BlineUI cell OOMed at VAE decode, which read as a
lever failure. This waits up to --timeout-s for the pool (VRAM + GTT, from amdgpu sysfs) to have --min-free-gib free,
then exits 0; otherwise it prints the reading plus the processes holding /dev/kfd or /dev/dri and exits 3.

  python diffusion_bench/amd/preflight_free.py --min-free-gib 54 --timeout-s 1800 --out out/preflight.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common as C  # noqa: E402


def holders() -> list:
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            fds = [os.readlink(f) for f in (d / "fd").iterdir()]
        except Exception:  # noqa: BLE001 - other users' processes are unreadable
            continue
        if any(t.startswith(("/dev/kfd", "/dev/dri/renderD")) for t in fds):
            try:
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors = "replace")[:200]
            except Exception:  # noqa: BLE001
                cmd = "?"
            out.append({"pid": int(d.name), "cmd": cmd})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free-gib", type = float, required = True)
    ap.add_argument("--timeout-s", type = float, default = 1800)
    ap.add_argument("--poll-s", type = float, default = 30)
    ap.add_argument("--out", type = Path, default = None)
    args = ap.parse_args()
    need = args.min_free_gib * 1024
    deadline = time.time() + args.timeout_s
    while True:
        q = C.gpu_query(C.visible_gpu()) or {}
        free = (q.get("total_mib") or 0) - (q.get("used_mib") or 0)
        rec = {"free_gib": round(free / 1024, 2), "total_gib": round((q.get("total_mib") or 0) / 1024, 2),
               "min_free_gib": args.min_free_gib, "reading": q}
        if q and free >= need:
            rec["verdict"] = "ok"
            break
        if time.time() > deadline:
            rec["verdict"] = "occupied" if q else "no_reading"
            rec["holders"] = holders()
            break
        print(f"preflight: {rec['free_gib']} GiB free of {rec['total_gib']}, need {args.min_free_gib}; waiting",
              flush = True)
        time.sleep(args.poll_s)
    print(json.dumps(rec, indent = 1, default = str))
    if args.out:
        args.out.parent.mkdir(parents = True, exist_ok = True)
        args.out.write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")
    if rec["verdict"] != "ok":
        print(f"PREFLIGHT FAIL: another tenant holds the GPU pool ({rec['free_gib']} GiB free, need "
              f"{args.min_free_gib}). Harness/host fault, not a result: rerun on a clean runner.", flush = True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
