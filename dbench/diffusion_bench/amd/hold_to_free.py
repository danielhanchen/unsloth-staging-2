#!/usr/bin/env python3
"""Fixture for amd_ci/lib/differential.py --fixture: hold device memory in a separate process so that about
--leave-free-gib stays free for the whole differential, whatever the runner already had in use. Reproduces the
state a gfx1151 runner was in when Studio refused Z-Image-Turbo (31 GB of the pool free). Memory is TOUCHED, not
merely reserved, so the driver's free figure moves. Prints one JSON line with "status": "READY", then waits to be
terminated (or --max-seconds)."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time

GIB = 1024 ** 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leave-free-gib", type = float, required = True)
    ap.add_argument("--max-seconds", type = int, default = 14400)
    args = ap.parse_args()
    import torch

    if not torch.cuda.is_available():
        print(json.dumps({"status": "ERROR", "error": "no cuda/hip device"}), flush = True)
        return 2
    free0, total = torch.cuda.mem_get_info()
    want = max(0, int(free0 - args.leave_free_gib * GIB))
    chunks, held = [], 0
    step = 512 * 1024 * 1024
    try:
        while held < want:
            n = min(step, want - held)
            t = torch.empty(n, dtype = torch.uint8, device = "cuda")
            t.fill_(1)
            chunks.append(t)
            held += n
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"[:300],
                          "held_gib": round(held / GIB, 2)}), flush = True)
        return 3
    free1, _ = torch.cuda.mem_get_info()
    print(json.dumps({"status": "READY", "held_gib": round(held / GIB, 2), "free_before_gib": round(free0 / GIB, 2),
                      "free_after_gib": round(free1 / GIB, 2), "total_gib": round(total / GIB, 2)}), flush = True)
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
    deadline = time.time() + args.max_seconds
    while time.time() < deadline:
        time.sleep(5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
