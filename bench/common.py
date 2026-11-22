"""Shared harness bits: physical-footprint sampler (proc_pid_rusage, the ledger `footprint` reports), engine load."""

import ctypes
import hashlib
import os
import subprocess
import threading
import time

GIB = 1024**3
_RUSAGE_INFO_V4 = 4
# uint64 slots after the 16-byte uuid in struct rusage_info_v4
_RSS = 6
_PHYS = 7
_LIFETIME_MAX = 28
_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")


def _raw():
    buf = (ctypes.c_uint64 * 64)()
    if _libc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(buf)) != 0:
        return None
    return buf  # uuid = first 2 uint64


def rusage():
    buf = _raw()
    if buf is None:
        return None, None
    return int(buf[2 + _PHYS]), int(buf[2 + _LIFETIME_MAX])


def rss_bytes():
    buf = _raw()
    return None if buf is None else int(buf[2 + _RSS])


class Sampler:
    """Samples phys_footprint every 20 ms; phase() returns and resets the max seen since the last call."""

    def __init__(self):
        self.peak = 0
        self.lock = threading.Lock()
        self.stop = False
        threading.Thread(target = self._run, daemon = True).start()

    def _run(self):
        while not self.stop:
            cur, _ = rusage()
            if cur:
                with self.lock:
                    self.peak = max(self.peak, cur)
            time.sleep(0.02)

    def phase(self):
        cur, lifetime = rusage()
        with self.lock:
            peak, self.peak = max(self.peak, cur or 0), cur or 0
        return {
            "phys_peak_gib": peak / GIB,
            "phys_now_gib": (cur or 0) / GIB,
            "phys_lifetime_max_gib": (lifetime or 0) / GIB,
            "rss_now_gib": (rss_bytes() or 0) / GIB,
        }


def footprint_tool():
    try:
        out = subprocess.run(["footprint", "-p", str(os.getpid())], capture_output = True, text = True, timeout = 60)
        return [l for l in out.stdout.splitlines() if "Footprint" in l or "phys_footprint" in l][:4]
    except Exception as exc:  # noqa: BLE001
        return [f"footprint error: {exc}"]


def pixel_sha(img):
    import numpy as np

    arr = np.asarray(img.convert("RGB"))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def load(backend, repo, gguf, base_repo = None, log = print, **kw):
    backend.begin_load(repo, gguf_filename = gguf, model_kind = "gguf", base_repo = base_repo, **kw)
    last = None
    while True:
        p = backend.load_progress()
        key = (p["phase"], round(p["fraction"], 1))
        if key != last:
            log(p)
            last = key
        if p["phase"] in ("ready", "error"):
            return p
        time.sleep(0.5)
