"""Device and host telemetry for one cell: a 20 ms NVML sampler, GPU identity, a cooldown gate and a foreign-VRAM
idle wait. Ported from the outside tester's mon.py / run_r5b.sh, generalised to NVML-less hosts.

Sources, in order:
  fake   DBENCH_FAKE_GPU='{"name": ..., "used_mib": ..., "sm_mhz": ...}' (selftests: the real code path, a fake card)
  nvml   pynvml (nvidia-ml-py): device used memory, per-process VRAM, SM clock, temperature, power, PCIe link
  smi    common.gpu_query (nvidia-smi shim / amd-smi / rocm-smi) every 250 ms: memory; on NVIDIA also SM clock,
         temperature and power from nvidia-smi (a driver venv without nvidia-ml-py, e.g. a tester's Studio venv)
  none   CPU / no device: host RSS only
A source that cannot answer leaves its fields None, which reports show as UNAVAILABLE.

Sampler(device, root_pids=lambda: [pid], exclude_pids=lambda: []) records:
  samples  (t, device_used_mib, own_vram_mib, tree_rss_mib, self_rss_mib)
  clocks   (t, sm_mhz, temp_c, power_w, power_limit_w, throttle_mask): power_limit_w is the ENFORCED power cap
           (NVML nvmlDeviceGetEnforcedPowerLimit / nvidia-smi enforced.power.limit), throttle_mask the clock event
           (throttle) reasons bitmask (nvmlDeviceGetCurrentClocksEventReasons / clocks_event_reasons.active), None
           when unreadable: the reporter's equal-power and below-cap clock rules
  ``own`` = VRAM of the process tree under the root pids (the run_cell process, its servers, short-lived sd-cli
  children) minus excluded pids (the VRAM ballast); the tree is re-walked every sample, so a child living a few
  hundred ms is still counted.
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from typing import Callable, Optional

_NVML: dict = {}


def _fake() -> Optional[dict]:
    raw = os.environ.get("DBENCH_FAKE_GPU")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return None


def nvml():
    """The pynvml module, initialised once, or None (not installed, no driver, CPU host)."""
    if "mod" not in _NVML:
        mod = None
        try:
            import pynvml

            pynvml.nvmlInit()
            mod = pynvml
        except Exception:  # noqa: BLE001 - no NVML is a normal CPU / AMD / Mac host
            mod = None
        _NVML["mod"] = mod
    return _NVML["mod"]


def handle(device: Optional[str]):
    """NVML handle for a mask entry: a UUID (GPU-...) or an index in PCI bus order (run_cell sets
    CUDA_DEVICE_ORDER=PCI_BUS_ID so torch and NVML agree on indices)."""
    nv = nvml()
    if nv is None or device is None:
        return None
    try:
        if str(device).startswith(("GPU-", "MIG-")):
            return nv.nvmlDeviceGetHandleByUUID(str(device))
        return nv.nvmlDeviceGetHandleByIndex(int(device))
    except Exception:  # noqa: BLE001
        return None


def _s(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def identity(device: Optional[str]) -> dict:
    """{name, uuid, driver, pcie_gen, pcie_width, pcie_gen_max, pcie_width_max, cc, total_mib, source}."""
    fake = _fake()
    if fake is not None:
        return {"name": fake.get("name", "fake"), "uuid": fake.get("uuid", "GPU-fake"), "driver": fake.get("driver"),
                "pcie_gen": fake.get("pcie_gen"), "pcie_width": fake.get("pcie_width"), "cc": fake.get("cc"),
                "total_mib": fake.get("total_mib"), "source": "fake"}
    h = handle(device)
    if h is None:
        return {"source": "none"}
    nv = nvml()
    out = {"source": "nvml"}
    for key, fn in (("name", lambda: _s(nv.nvmlDeviceGetName(h))), ("uuid", lambda: _s(nv.nvmlDeviceGetUUID(h))),
                    ("driver", lambda: _s(nv.nvmlSystemGetDriverVersion())),
                    ("pcie_gen", lambda: nv.nvmlDeviceGetCurrPcieLinkGeneration(h)),
                    ("pcie_width", lambda: nv.nvmlDeviceGetCurrPcieLinkWidth(h)),
                    ("pcie_gen_max", lambda: nv.nvmlDeviceGetMaxPcieLinkGeneration(h)),
                    ("pcie_width_max", lambda: nv.nvmlDeviceGetMaxPcieLinkWidth(h)),
                    ("cc", lambda: "%d.%d" % nv.nvmlDeviceGetCudaComputeCapability(h)),
                    ("total_mib", lambda: int(nv.nvmlDeviceGetMemoryInfo(h).total / 2**20))):
        try:
            out[key] = fn()
        except Exception:  # noqa: BLE001
            out[key] = None
    return out


def compute_procs(device: Optional[str]) -> Optional[dict]:
    """{pid: used_mib} of compute processes on the device, or None when this host cannot say."""
    fake = _fake()
    if fake is not None:
        return {int(k): int(v) for k, v in (fake.get("procs") or {}).items()}
    h = handle(device)
    if h is None:
        return None
    nv = nvml()
    out: dict = {}
    for fn in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
        try:
            for p in getattr(nv, fn)(h):
                used = getattr(p, "usedGpuMemory", None)
                if used is None:  # WSL2 / WDDM: NVML cannot attribute memory; unknown, never counted as 0
                    continue
                out[int(p.pid)] = out.get(int(p.pid), 0) + int(used / 2**20)
        except Exception:  # noqa: BLE001
            pass
    return out


def device_used_mib(device: Optional[str]) -> Optional[int]:
    fake = _fake()
    if fake is not None:
        return fake.get("used_mib")
    h = handle(device)
    if h is not None:
        try:
            return int(nvml().nvmlDeviceGetMemoryInfo(h).used / 2**20)
        except Exception:  # noqa: BLE001
            return None
    import common as C

    q = C.gpu_query(device) if device is not None else {}
    return q.get("used_mib") if q else None


# NVML clock event (throttle) reason bits that mean the device itself held the clock down (nvml.h
# nvmlClocksEventReason*): software power cap, hardware slowdown, software / hardware thermal, power brake
THROTTLE_BITS = {"sw_power_cap": 0x4, "hw_slowdown": 0x8, "sw_thermal": 0x20, "hw_thermal": 0x40,
                 "hw_power_brake": 0x80}
THROTTLE_LIMITING = sum(THROTTLE_BITS.values())
_SMI_FIELDS: list = []  # the nvidia-smi query that works on this driver (with or without the reasons field)


def throttle_names(mask: Optional[int]) -> list:
    return [] if mask is None else [n for n, b in THROTTLE_BITS.items() if mask & b]


def smi_clocks(device: Optional[str]) -> Optional[tuple]:
    """(sm_mhz, temp_c, power_w, power_limit_w, throttle_mask) of an NVIDIA device from nvidia-smi, for a driver venv
    without NVML; None elsewhere. A field the card does not report ([N/A]) is None; a driver that rejects the
    reasons field is queried without it (throttle_mask None)."""
    import common as C

    if device is None or C.gpu_vendor() != "nvidia":
        return None

    def num(x: str) -> Optional[float]:
        try:
            return float(x)
        except ValueError:
            return None

    base = "clocks.sm,temperature.gpu,power.draw,enforced.power.limit"
    for fields in (_SMI_FIELDS or [base + ",clocks_event_reasons.active", base]):
        try:
            res = C.smi(["-i", device, f"--query-gpu={fields}", "--format=csv,noheader,nounits"], timeout = 10)
            parts = [x.strip() for x in res.stdout.strip().splitlines()[0].split(",")]
        except Exception:  # noqa: BLE001 - sampling is best effort
            continue
        parts += [""] * (5 - len(parts))
        sm, temp, power, cap = (num(x) for x in parts[:4])
        if sm is None:  # a rejected field prints an error line, not numbers: try the query without it
            continue
        if not _SMI_FIELDS:
            _SMI_FIELDS.append(fields)
        try:
            mask = int(parts[4], 16)
        except ValueError:
            mask = None
        return (sm, temp, power, cap, mask)
    return None


def temp_c(device: Optional[str]) -> Optional[int]:
    fake = _fake()
    if fake is not None:
        return fake.get("temp_c")
    h = handle(device)
    if h is None:  # no NVML in the driver venv: nvidia-smi, or the cooldown gate is "unavailable" on every cell
        c = smi_clocks(device)
        return int(c[1]) if c is not None and c[1] is not None else None
    try:
        return int(nvml().nvmlDeviceGetTemperature(h, nvml().NVML_TEMPERATURE_GPU))
    except Exception:  # noqa: BLE001
        return None


def tree_pids(roots: list) -> set:
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return set(roots)
    pids = set()
    for pid in roots:
        try:
            p = psutil.Process(pid)
            pids.add(p.pid)
            pids.update(c.pid for c in p.children(recursive = True))
        except Exception:  # noqa: BLE001
            continue
    return pids


def tree_rss_mib(pids: set) -> Optional[float]:
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return None
    tot = 0
    for pid in pids:
        try:
            tot += psutil.Process(pid).memory_info().rss
        except Exception:  # noqa: BLE001
            continue
    return tot / 2**20


def foreign(device: Optional[str], own_pids: set) -> dict:
    """{procs, mib, device_used_mib}: compute processes on the device outside our tree, and what they hold."""
    procs = compute_procs(device)
    used = device_used_mib(device)
    if procs is None:
        return {"procs": None, "mib": None, "device_used_mib": used}
    other = {p: m for p, m in procs.items() if p not in own_pids}
    return {"procs": len(other), "pids": sorted(other)[:16], "mib": sum(other.values()), "device_used_mib": used}


# ---------------------------------------------------------------------------------------------- gates
def cooldown(device: Optional[str], target_c: float, timeout_s: float = 900, poll_s: float = 5, log=print) -> dict:
    """Wait until the GPU is at or below ``target_c`` before a timed cell. {gate: ok|timeout|unavailable, ...}."""
    t0 = time.time()
    t = temp_c(device)
    if t is None:
        return {"gate": "unavailable", "target_c": target_c}
    while t is not None and t > target_c and time.time() - t0 < timeout_s:
        time.sleep(poll_s)
        t = temp_c(device)
    gate = "ok" if t is not None and t <= target_c else "timeout"
    if gate == "timeout":
        log(f"cooldown TIMEOUT: {t} C > {target_c} C after {timeout_s:.0f} s; running anyway, flagged")
    return {"gate": gate, "temp_c": t, "target_c": target_c, "waited_s": round(time.time() - t0, 1)}


def foreign_idle_wait(device: Optional[str], own_pids: set, max_foreign_mib: int = 1024, hold_s: float = 10,
                      timeout_s: float = 1800, poll_s: float = 2, log=print) -> dict:
    """Wait until other tenants hold at most ``max_foreign_mib`` of the device for ``hold_s`` in a row (the tester's
    run_r5b.sh rule). With no per-process accounting the device-wide used figure stands in. {gate, foreign...}."""
    t0 = time.time()
    quiet_since = None
    while True:
        f = foreign(device, own_pids)
        held = f["mib"] if f["mib"] is not None else f["device_used_mib"]
        if held is None:
            return {"gate": "unavailable", **f}
        now = time.time()
        if held <= max_foreign_mib:
            quiet_since = quiet_since or now
            if now - quiet_since >= hold_s:
                return {"gate": "ok", "waited_s": round(now - t0, 1), "max_foreign_mib": max_foreign_mib, **f}
        else:
            quiet_since = None
        if now - t0 > timeout_s:
            log(f"foreign-VRAM wait TIMEOUT: {held} MiB held by {f['procs']} other process(es); flagged")
            return {"gate": "timeout", "waited_s": round(now - t0, 1), "max_foreign_mib": max_foreign_mib, **f}
        time.sleep(poll_s)


# ---------------------------------------------------------------------------------------------- sampler
class Sampler(threading.Thread):
    """Device memory, per-tree VRAM and RSS, clocks / temperature / power, every ``interval`` s (20 ms with NVML,
    250 ms through SMI). Windows over [t0, t1] give per-request peaks; ``baseline_mib`` is the pre-load reading."""

    def __init__(self, device: Optional[str], interval: float = 0.02,
                 root_pids: Optional[Callable[[], list]] = None, exclude_pids: Optional[Callable[[], list]] = None,
                 own_vram_fallback: Optional[Callable[[], Optional[float]]] = None):
        super().__init__(daemon = True)
        self.device = device
        self.fake = _fake()
        self.h = handle(device) if self.fake is None else None
        self.source = "fake" if self.fake is not None else "nvml" if self.h is not None else (
            "smi" if device is not None and device_used_mib(device) is not None else "none")
        self.interval = interval if self.source in ("nvml", "fake") else max(interval, 0.25)
        self.root_pids = root_pids or (lambda: [os.getpid()])
        self.exclude_pids = exclude_pids or (lambda: [])
        self.own_vram_fallback = own_vram_fallback
        self.fallback_hits = 0  # samples whose own-tree VRAM came from the backend (e.g. the Metal driver)
        self.samples: list = []
        self.clk: list = []
        self.baseline_mib = self._used()
        # not "_stop": threading.Thread has its own _stop() method, which Python 3.12's join() calls
        self._stop_ev = threading.Event()
        self._lock = threading.Lock()

    def _used(self) -> Optional[float]:
        if self.fake is not None:
            return self.fake.get("used_mib")
        if self.h is not None:
            try:
                return nvml().nvmlDeviceGetMemoryInfo(self.h).used / 2**20
            except Exception:  # noqa: BLE001
                return None
        return device_used_mib(self.device) if self.source == "smi" else None

    def _own(self, pids: set) -> Optional[float]:
        if self.fake is not None:
            return self.fake.get("own_mib", self.fake.get("used_mib"))
        procs = compute_procs(self.device) if self.h is not None else None
        mine = {p: m for p, m in (procs or {}).items() if p in pids and m is not None}
        if not mine:
            # NVML lists none of our pids (containers: other pid namespace; no attribution): unknown, not 0.0
            got = self.own_vram_fallback() if self.own_vram_fallback else None
            if got is not None:
                self.fallback_hits += 1
            return got
        return float(sum(mine.values()))

    def _clocks(self):
        if self.fake is not None:
            if self.fake.get("sm_mhz") is None:
                return None
            return (self.fake.get("sm_mhz"), self.fake.get("temp_c"), self.fake.get("power_w"),
                    self.fake.get("power_limit_w"), self.fake.get("throttle_mask"))
        if self.h is None:
            return smi_clocks(self.device) if self.source == "smi" else None
        nv = nvml()
        try:
            got = (nv.nvmlDeviceGetClockInfo(self.h, nv.NVML_CLOCK_SM),
                   nv.nvmlDeviceGetTemperature(self.h, nv.NVML_TEMPERATURE_GPU),
                   nv.nvmlDeviceGetPowerUsage(self.h) / 1000)
        except Exception:  # noqa: BLE001
            return None
        try:  # the enforced cap (the governor's real limit; a card without power management raises)
            cap = nv.nvmlDeviceGetEnforcedPowerLimit(self.h) / 1000
        except Exception:  # noqa: BLE001
            cap = None
        mask = None
        for fn in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
            try:
                mask = int(getattr(nv, fn)(self.h))
                break
            except Exception:  # noqa: BLE001 - older bindings / drivers: the other name, else unreadable
                continue
        return (*got, cap, mask)

    def run(self) -> None:
        n = 0
        own = rss = self_rss = None
        pids: set = set()
        tree_every = max(1, int(round(0.5 / self.interval)))  # the /proc tree walk (costly on a busy host) ~2 Hz
        while not self._stop_ev.is_set():
            t = time.time()
            if n % 5 == 0:  # clocks first: a short cell must not lose its first clock sample behind the tree walk
                c = self._clocks()
                if c is not None:
                    with self._lock:
                        self.clk.append((t, *c))
            used = self._used()
            if n % tree_every == 0:
                pids = tree_pids(list(self.root_pids()) + list(self.extra_roots())) - set(self.exclude_pids())
            if n % 2 == 0:  # the per-process VRAM / RSS over the cached tree every 40 ms; device memory every 20 ms
                own = self._own(pids)
                rss = tree_rss_mib(pids)
                self_rss = tree_rss_mib({os.getpid()})
            with self._lock:
                self.samples.append((t, used, own, rss, self_rss))
            n += 1
            self._stop_ev.wait(self.interval)

    def extra_roots(self) -> list:
        return []

    def stop(self) -> None:
        self._stop_ev.set()
        if self.is_alive():
            self.join(timeout = 5)

    # -------------------------------------------------------------------------------------- windows
    def _win(self, t0: float, t1: float) -> list:
        with self._lock:
            return [s for s in self.samples if t0 <= s[0] <= t1]

    def window(self, t0: float = 0.0, t1: float = float("inf")) -> dict:
        """Peaks over [t0, t1]: device used, over the pre-load baseline, own tree VRAM, tree RSS, server RSS."""
        s = self._win(t0, t1)

        def mx(i):
            vals = [x[i] for x in s if x[i] is not None]
            return max(vals) if vals else None

        dev, own, rss = mx(1), mx(2), mx(3)
        server = [x[3] - x[4] for x in s if x[3] is not None and x[4] is not None]
        return {"device_mib": dev, "over_base_mib": (dev - self.baseline_mib) if dev is not None and
                self.baseline_mib is not None else None, "own_mib": own, "rss_tree_mib": rss,
                "rss_server_mib": max(server) if server else None, "n": len(s)}

    def clocks(self, t0: float = 0.0, t1: float = float("inf")) -> dict:
        """Mean SM clock (idle readings under 500 MHz dropped), max temperature, mean power draw, and the enforced
        power cap (its lowest and highest reading: a cap changed mid-run shows as min != max) over [t0, t1]."""
        with self._lock:
            c = [x for x in self.clk if t0 <= x[0] <= t1 and x[1] is not None and x[1] > 500]
        if not c:
            return {"sm_mhz_mean": None, "temp_c_max": None, "power_w_mean": None, "power_limit_w": None,
                    "power_limit_w_min": None, "power_limit_w_max": None, "throttle_reasons": None,
                    "throttle_samples": None, "samples": 0}
        temps = [x[2] for x in c if x[2] is not None]
        power = [x[3] for x in c if x[3] is not None]
        caps = [x[4] for x in c if len(x) > 4 and x[4] is not None]
        masks = [x[5] for x in c if len(x) > 5]
        # the throttle reasons seen in the window: OR of every sample's limiting bits, each reason's sample count;
        # None when ANY sample could not read them (an unread sample cannot prove "no throttle")
        unread = not masks or any(m is None for m in masks)
        throttle = None if unread else {
            "mask": sum(b for b in THROTTLE_BITS.values() if any(m & b for m in masks)),
            "counts": {n: sum(1 for m in masks if m & b) for n, b in THROTTLE_BITS.items() if any(m & b for m in masks)},
            "n": len(masks)}
        return {"sm_mhz_mean": round(statistics.mean(x[1] for x in c), 1),
                "temp_c_max": max(temps) if temps else None,
                "power_w_mean": round(statistics.mean(power), 1) if power else None,
                "power_limit_w": round(statistics.mean(caps), 1) if caps else None,
                "power_limit_w_min": round(min(caps), 1) if caps else None,
                "power_limit_w_max": round(max(caps), 1) if caps else None,
                "throttle_reasons": None if throttle is None else throttle_names(throttle["mask"]),
                "throttle_samples": throttle, "samples": len(c)}

    def timeline(self, every_s: float = 1.0) -> list:
        """(t, device_mib, own_mib, rss_tree_mib) thinned to one sample per ``every_s``: memory forensics."""
        out, last = [], -1e18
        with self._lock:
            for s in self.samples:
                if s[0] - last >= every_s:
                    out.append([round(s[0], 2)] + [None if v is None else round(v) for v in s[1:4]])
                    last = s[0]
        return out
