#!/usr/bin/env python3
"""Hold VRAM on one GPU so that only --free-mib remains free (as NVML, else torch, reports it), print
``READY <free_mib>``, then sleep until terminated. Started by budget.Ballast with CUDA_VISIBLE_DEVICES=<device> and
CUDA_DEVICE_ORDER=PCI_BUS_ID; --device is the NVML index or UUID of the same card."""

import argparse
import signal
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--free-mib", type = int, required = True)
ap.add_argument("--device", default = "0")
ap.add_argument("--parent-pid", type = int, default = None, help = "exit when this process is gone (no orphan)")
a = ap.parse_args()

import torch  # noqa: E402

free_fn = None
try:
    import pynvml

    pynvml.nvmlInit()
    h = (pynvml.nvmlDeviceGetHandleByUUID(a.device) if str(a.device).startswith("GPU-")
         else pynvml.nvmlDeviceGetHandleByIndex(int(a.device)))
    free_fn = lambda: pynvml.nvmlDeviceGetMemoryInfo(h).free / 2**20  # noqa: E731
except Exception:  # noqa: BLE001 - AMD / no NVML: torch's view of the device
    free_fn = lambda: torch.cuda.mem_get_info()[0] / 2**20  # noqa: E731

torch.cuda.init()
torch.empty(1, device = "cuda")
held = []
while True:
    over = free_fn() - a.free_mib
    if over <= 32:
        break
    size = min(1024, over - 16) if over > 1024 else max(over - 16, 1)
    held.append(torch.empty(int(size * 2**20), dtype = torch.uint8, device = "cuda"))
torch.cuda.synchronize()
print(f"READY {free_fn():.0f}", flush = True)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
parent = a.parent_pid


def parent_alive() -> bool:
    if parent is None:
        return True
    try:
        import os

        if os.name == "nt":
            import psutil

            return psutil.pid_exists(parent)
        os.kill(parent, 0)
        return os.getppid() == parent
    except Exception:  # noqa: BLE001 - gone (or not ours any more): release the VRAM
        return False


# a SIGKILLed / OOM-killed run_cell never stops us: poll the parent and release the card when it is gone
while parent_alive():
    time.sleep(1)
sys.exit(0)
