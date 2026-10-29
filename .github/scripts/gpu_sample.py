#!/usr/bin/env python3
"""One host-wide GPU reading, shared by every caller: gpu_queue, studio_regress, the dashboard.

    python gpu_sample.py            # print the current state (refreshing it when stale)
    python gpu_sample.py --json

Dozens of sessions polling nvidia-smi at once is what wedges it (each call queues in the driver and
takes tens of seconds). Here a caller reads <lock dir>/gpu_state.json; only when it is older than
`max_age` does ONE process refresh it (non-blocking flock on gpu_state.lock), and everyone else waits
for that answer (bounded) instead of asking the driver too.

Backend: NVML in process (nvidia-ml-py / pynvml): memory, the driver's own utilization samples
(nvmlDeviceGetSamples: ~14 s of history at ~5 Hz, one call, no sleeping), running compute processes.
Without NVML, one nvidia-smi call (the real binary, not the nvsmi_cache shim) with a 20 s budget.
A failed refresh never reads as "0 GB free": the last good reading is kept with stale=True and the
error, and `admissible()` says whether it is still fresh enough to admit work on.

State: {"t", "source": "nvml"|"nvidia-smi", "stale", "error", "gpus": {idx: {"index", "uuid", "name",
"total_gb", "free_gb", "util_now", "util_max_5s", "util_mean_5s", "procs": [[pid, uid, gb]]}},
"history": {idx: [[t, util], ...]}} (last HISTORY_N points, so a single-sample nvidia-smi reading
still has a window).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

MAX_AGE_S = float(os.environ.get("GPU_SAMPLE_MAX_AGE_S", "3"))
WAIT_S = float(os.environ.get("GPU_SAMPLE_WAIT_S", "30"))  # how long a reader waits on a refresher
ADMIT_MAX_AGE_S = float(os.environ.get("GPU_SAMPLE_ADMIT_MAX_AGE_S", "60"))
ERROR_RETRY_S = float(os.environ.get("GPU_SAMPLE_ERROR_RETRY_S", "10"))  # after a failed refresh
# one refresher serves every caller, so it may wait longer than any one caller would: under driver
# contention a single nvidia-smi took 20-30 s, and a shorter cut-off left no reading at all
SMI_TIMEOUT_S = float(os.environ.get("GPU_SAMPLE_SMI_TIMEOUT_S", "45"))
NVML_TIMEOUT_S = float(os.environ.get("GPU_SAMPLE_NVML_TIMEOUT_S", "10"))
NVML_HUNG_SKIP_S = 300.0  # after an NVML call hung, use nvidia-smi this long
WINDOW_S = 5.0
HISTORY_N = 10
STATE_NAME, LOCK_NAME = "gpu_state.json", "gpu_state.lock"
_SHIM_MARKER = b"nvidia-smi shim: one real query per"
_NVML = {}
_NVML_LOCK = threading.Lock()


def default_dir():
    """$GPU_SAMPLE_DIR, else the gpu_queue lock dir (host-wide)."""
    if os.environ.get("GPU_SAMPLE_DIR"):
        return Path(os.environ["GPU_SAMPLE_DIR"])
    import gpu_queue  # lazy: gpu_queue imports this module

    return Path(gpu_queue.lock_dir())


def _plat():
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    from studio_regress import plat  # POSIX flock / Windows msvcrt

    return plat


def state_path(ld = None):
    return Path(ld or default_dir()) / STATE_NAME


def _open_shared(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    with contextlib.suppress(OSError):
        os.fchmod(fd, 0o666)
    return os.fdopen(fd, "r+")


def _mkdir_shared(d):
    """Create `d` 0777 like gpu_queue's lock dir: every unix user refreshes the one reading, which
    means writing its temp file and replacing gpu_state.json in this directory."""
    d = Path(d)
    if not d.is_dir():
        d.mkdir(parents = True, exist_ok = True)
        with contextlib.suppress(OSError):
            os.chmod(d, 0o777)


def load(ld = None):
    try:
        st = json.loads(state_path(ld).read_text())
        return st if isinstance(st, dict) and isinstance(st.get("gpus"), dict) else None
    except (OSError, ValueError):
        return None


def _write(ld, st):
    path = state_path(ld)
    _mkdir_shared(path.parent)
    tmp = path.with_suffix(f".tmp{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(json.dumps(st, separators = (",", ":")))
    with contextlib.suppress(OSError):
        os.chmod(tmp, 0o666)
    tmp.replace(path)


def proc_uid(pid):
    try:
        for ln in Path(f"/proc/{int(pid)}/status").read_text().splitlines():
            if ln.startswith("Uid:"):
                return int(ln.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


# ------------------------------------------------------------------ backends
def _import_pynvml():
    """pynvml from this interpreter, else from the active venv's site-packages (the nvsmi_cache shim
    runs under /usr/bin/python3, the workspace venv has nvidia-ml-py; pynvml is pure Python)."""
    try:
        import pynvml
        return pynvml
    except ImportError:
        import glob
        for base in (os.environ.get("GPU_SAMPLE_PYNVML_DIR"), os.environ.get("VIRTUAL_ENV")):
            for d in (
                [base]
                if base and os.path.isfile(os.path.join(base, "pynvml.py"))
                else sorted(
                    glob.glob(os.path.join(base, "lib", "python3*", "site-packages"))
                    if base
                    else []
                )
            ):
                if os.path.isfile(os.path.join(d, "pynvml.py")):
                    sys.path.append(d)
                    import pynvml
                    return pynvml
        raise


def _nvml():
    with _NVML_LOCK:
        # a failed init is retried after a minute: one transient driver error must not pin a
        # long-running process (the switchboard runs for hours) to the nvidia-smi fallback
        if "mod" not in _NVML or (_NVML["mod"] is None and time.time() - _NVML.get("t", 0) > 60):
            try:
                pynvml = _import_pynvml()
                pynvml.nvmlInit()
                _NVML["mod"] = pynvml
            except Exception as e:  # noqa: BLE001 - no NVML: the nvidia-smi fallback answers
                _NVML["mod"], _NVML["error"], _NVML["t"] = (
                    None,
                    f"{type(e).__name__}: {e}",
                    time.time(),
                )
        return _NVML["mod"]


def sample_nvml(now = None):
    """{idx: gpu dict} from NVML, or raises. util_* from the driver's own sample buffer."""
    N = _nvml()
    if N is None:
        raise RuntimeError(f"NVML unavailable ({_NVML.get('error', '?')})")
    now = now or time.time()
    out = {}
    for i in range(N.nvmlDeviceGetCount()):
        h = N.nvmlDeviceGetHandleByIndex(i)
        mem = N.nvmlDeviceGetMemoryInfo(h)
        util_now = float(N.nvmlDeviceGetUtilizationRates(h).gpu)
        win = []
        with contextlib.suppress(Exception):
            _typ, samples = N.nvmlDeviceGetSamples(h, N.NVML_GPU_UTILIZATION_SAMPLES, 0)
            vals = [(s.timeStamp / 1e6, float(s.sampleValue.uiVal)) for s in samples]
            if vals:
                last = max(t for t, _v in vals)
                win = [v for t, v in vals if t >= last - WINDOW_S]
        procs, raw = [], None
        with contextlib.suppress(Exception):
            raw = list(N.nvmlDeviceGetComputeRunningProcesses(h))
            for p in raw:
                procs.append(
                    [int(p.pid), proc_uid(p.pid), round((p.usedGpuMemory or 0) / 2**30, 3)]
                )
        name, uuid = N.nvmlDeviceGetName(h), N.nvmlDeviceGetUUID(h)
        out[str(i)] = {
            "index": str(i),
            "uuid": uuid.decode() if isinstance(uuid, bytes) else uuid,
            "name": name.decode() if isinstance(name, bytes) else name,
            "total_gb": round(mem.total / 2**30, 3),
            "free_gb": round(mem.free / 2**30, 3),
            "util_now": util_now,
            "_win": win,
            "procs": procs,
        }
        with contextlib.suppress(Exception):
            out[str(i)]["smi"], apps = smi_fields(N, h, i, out[str(i)], now, raw or ())
            if raw is not None:  # unread is not "no processes": the shim then asks the driver
                out[str(i)]["apps"] = apps
    return out


# Fields that do not change while the driver is loaded: reused from the previous reading for
# STATIC_REUSE_S instead of asked again (nvidia-smi shim answers; nvsmi_cache/nvidia-smi).
SMI_STATIC = (
    "name",
    "uuid",
    "pci.bus_id",
    "driver_version",
    "compute_cap",
    "serial",
    "vbios_version",
    "clocks.max.sm",
    "clocks.max.mem",
    "power.limit",
    "persistence_mode",
)
STATIC_REUSE_S = 600.0
NA, NO_DATA = "[N/A]", "[No data]"
_PREV = {}  # the previous state, set by refresh() for smi_fields' static reuse


def _s(v):
    return v.decode() if isinstance(v, bytes) else v


def _nv(fn, *a):
    """fn(*a), or nvidia-smi's own placeholder when NVML cannot say."""
    try:
        return fn(*a)
    except Exception:  # noqa: BLE001 - nvidia-smi's csv says [N/A] for unsupported too (fan.speed on B200)
        return NA


def smi_fields(
    N,
    h,
    i,
    g,
    now,
    raw_procs = (),
):
    """(fields, apps) in nvidia-smi's own units for the shim: {field: value} with MiB / % / W / MHz /
    C numbers or strings, and [{pid, process_name, used_memory}] for compute apps."""
    prev = (((_PREV.get("state") or {}).get("gpus") or {}).get(str(i)) or {}).get("smi") or {}
    static_ok = (
        prev
        and prev.get("uuid") == g["uuid"]
        and now - float(prev.get("_static_t") or 0) < STATIC_REUSE_S
    )
    f = {k: prev[k] for k in SMI_STATIC if static_ok and k in prev}
    f["_static_t"] = prev.get("_static_t") if static_ok else round(now, 3)
    if not static_ok:
        cc = _nv(N.nvmlDeviceGetCudaComputeCapability, h)
        pci = _nv(N.nvmlDeviceGetPciInfo, h)
        lim = _nv(N.nvmlDeviceGetPowerManagementLimit, h)
        pm = _nv(N.nvmlDeviceGetPersistenceMode, h)
        f.update(
            {
                "name": g["name"],
                "uuid": g["uuid"],
                "pci.bus_id": _s(pci.busId) if hasattr(pci, "busId") else pci,
                "driver_version": _s(_nv(N.nvmlSystemGetDriverVersion)),
                "compute_cap": "%d.%d" % tuple(cc) if isinstance(cc, tuple) else cc,
                "serial": _s(_nv(N.nvmlDeviceGetSerial, h)),
                "vbios_version": _s(_nv(N.nvmlDeviceGetVbiosVersion, h)),
                "clocks.max.sm": _nv(N.nvmlDeviceGetMaxClockInfo, h, N.NVML_CLOCK_SM),
                "clocks.max.mem": _nv(N.nvmlDeviceGetMaxClockInfo, h, N.NVML_CLOCK_MEM),
                "power.limit": lim / 1000 if isinstance(lim, int) else lim,
                "persistence_mode": ("Enabled" if pm else "Disabled")
                if isinstance(pm, int)
                else pm,
            }
        )
    try:  # v2 (what nvidia-smi reports): used excludes the driver's reserved memory
        mem = N.nvmlDeviceGetMemoryInfo(h, version = N.nvmlMemory_v2)
        reserved = mem.reserved // 2**20
    except Exception:  # noqa: BLE001 - an older pynvml / driver: v1, no reserved figure
        mem, reserved = N.nvmlDeviceGetMemoryInfo(h), NA
    util = _nv(N.nvmlDeviceGetUtilizationRates, h)
    pst = _nv(N.nvmlDeviceGetPerformanceState, h)
    draw = _nv(N.nvmlDeviceGetPowerUsage, h)
    f.update(
        {
            "index": str(i),
            "memory.total": mem.total // 2**20,
            "memory.used": mem.used // 2**20,
            "memory.free": mem.free // 2**20,
            "memory.reserved": reserved,
            "utilization.gpu": util.gpu if hasattr(util, "gpu") else util,
            "utilization.memory": util.memory if hasattr(util, "memory") else util,
            "temperature.gpu": _nv(N.nvmlDeviceGetTemperature, h, N.NVML_TEMPERATURE_GPU),
            "power.draw": draw / 1000 if isinstance(draw, int) else draw,
            "clocks.sm": _nv(N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_SM),
            "clocks.mem": _nv(N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_MEM),
            "pstate": "P%d" % pst if isinstance(pst, int) else pst,
            "fan.speed": _nv(N.nvmlDeviceGetFanSpeed, h),
        }
    )
    apps = []
    for p in raw_procs:
        name = _nv(N.nvmlSystemGetProcessName, p.pid)
        name = NO_DATA if name in (NA, None, b"", "") else name
        used = p.usedGpuMemory
        apps.append(
            {
                "pid": int(p.pid),
                "process_name": _s(name),
                "used_memory": used // 2**20 if isinstance(used, int) else NA,
            }
        )
    return f, apps


def real_smi():
    """First nvidia-smi on PATH that is not a copy of the nvsmi_cache shim (or None)."""
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = os.path.join(d, "nvidia-smi") if d else ""
        try:
            if p and os.access(p, os.X_OK) and os.path.isfile(p):
                with open(p, "rb") as fh:
                    if _SHIM_MARKER in fh.read(4096):
                        continue
                return p
        except OSError:
            continue
    return None


def _smi_rows(
    exe,
    query,
    budget,
    check = False,
):
    r = subprocess.run(
        [exe, query, "--format=csv,noheader,nounits"],
        capture_output = True,
        text = True,
        timeout = max(1.0, budget),
        check = check,
    )
    return [[x.strip() for x in ln.split(",")] for ln in (r.stdout or "").strip().splitlines()]


# nvidia-smi fallback in the shim's field names (nvsmi_cache/nvidia-smi formats any subset): the
# fields that change every refresh in one call, the static ones (SMI_STATIC) in a second call only
# every STATIC_REUSE_S. Under driver contention each extra field is more driver work queued on the
# same lock: the full superset did not finish in 45 s where the dynamic set took 15-20 s.
SMI_QUERY = (
    "index",
    "uuid",
    "name",
    "memory.total",
    "memory.free",
    "utilization.gpu",
    "memory.used",
    "memory.reserved",
    "utilization.memory",
    "temperature.gpu",
    "power.draw",
    "clocks.sm",
    "clocks.mem",
    "pstate",
)
SMI_STATIC_QUERY = (
    "index",
    "pci.bus_id",
    "driver_version",
    "compute_cap",
    "serial",
    "vbios_version",
    "clocks.max.sm",
    "clocks.max.mem",
    "power.limit",
    "persistence_mode",
    "fan.speed",
)
SMI_STRINGS = {
    "uuid",
    "name",
    "pstate",
    "persistence_mode",
    "compute_cap",
    "pci.bus_id",
    "driver_version",
    "serial",
    "vbios_version",
    "index",
}


def _smi_value(field, v):
    if field in SMI_STRINGS or v.startswith("["):
        return v
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return v


def sample_smi(now = None, timeout = SMI_TIMEOUT_S):
    """The same reading from nvidia-smi: one --query-gpu of the changing fields (memory, utilization,
    uuid, name and the shim's dynamic `smi` fields; the old minimal query if this driver refuses a
    field), the static fields at most every STATIC_REUSE_S, then compute apps (best effort), all
    inside one `timeout` budget."""
    exe = real_smi() or shutil.which("nvidia-smi") or "nvidia-smi"
    now = now or time.time()
    end = time.time() + timeout
    out = {}
    try:
        rows = [
            p
            for p in _smi_rows(exe, "--query-gpu=" + ",".join(SMI_QUERY), timeout, check = True)
            if len(p) == len(SMI_QUERY)
        ]
    except subprocess.CalledProcessError:
        rows = None  # an older driver without one of the fields: memory + utilization only
    if rows:
        prev = (_PREV.get("state") or {}).get("gpus") or {}
        static = {}
        for k, g in prev.items():
            ps = g.get("smi") or {}
            if now - float(ps.get("_static_t") or 0) < STATIC_REUSE_S and "pci.bus_id" in ps:
                static[k] = {f: ps[f] for f in SMI_STATIC_QUERY if f in ps}
                static[k]["_static_t"], static[k]["uuid"] = ps["_static_t"], ps.get("uuid")
        if len(static) < len(rows):
            static = {}
            with contextlib.suppress(
                Exception
            ):  # best effort: without them only static queries fall through
                for q in _smi_rows(
                    exe, "--query-gpu=" + ",".join(SMI_STATIC_QUERY), end - time.time(), check = True
                ):
                    if len(q) == len(SMI_STATIC_QUERY):
                        static[q[0]] = {f: _smi_value(f, v) for f, v in zip(SMI_STATIC_QUERY, q)}
                        static[q[0]]["_static_t"] = round(now, 3)
        for p in rows:
            smi = {f: _smi_value(f, v) for f, v in zip(SMI_QUERY, p)}
            st_ = static.get(p[0]) or {}
            if st_.get("uuid") in (None, smi["uuid"]):
                smi.update({k: v for k, v in st_.items() if k != "uuid"})
            util = (
                smi["utilization.gpu"] if isinstance(smi["utilization.gpu"], (int, float)) else None
            )
            out[p[0]] = {
                "index": p[0],
                "uuid": smi["uuid"],
                "name": smi["name"],
                "total_gb": round(float(smi["memory.total"]) / 1024, 3),
                "free_gb": round(float(smi["memory.free"]) / 1024, 3),
                "util_now": util,
                "_win": [],
                "procs": [],
                "smi": smi,
            }
    else:
        for p in _smi_rows(
            exe,
            "--query-gpu=index,memory.total,memory.free,utilization.gpu",
            end - time.time(),
            check = True,
        ):
            if len(p) < 3:
                continue
            util = None
            with contextlib.suppress(ValueError, IndexError):
                util = float(p[3])
            out[p[0]] = {
                "index": p[0],
                "uuid": None,
                "name": None,
                "total_gb": round(float(p[1]) / 1024, 3),
                "free_gb": round(float(p[2]) / 1024, 3),
                "util_now": util,
                "_win": [],
                "procs": [],
            }
    if not out:
        raise RuntimeError("nvidia-smi listed no GPUs")
    by_uuid = {g["uuid"]: k for k, g in out.items() if g["uuid"]}
    with contextlib.suppress(Exception):
        if not by_uuid:
            for p in _smi_rows(exe, "--query-gpu=index,uuid,name", end - time.time()):
                if len(p) >= 2 and p[0] in out:
                    out[p[0]]["uuid"], by_uuid[p[1]] = p[1], p[0]
                    if len(p) >= 3:
                        out[p[0]]["name"] = p[2]
        apps_ok = all("smi" in g for g in out.values())
        q = (
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory"
            if apps_ok
            else "--query-compute-apps=gpu_uuid,pid,used_memory"
        )
        rows_apps = _smi_rows(exe, q, end - time.time(), check = True)
        if apps_ok:  # answered: an empty list now means "no processes", not "not asked"
            for g in out.values():
                g["apps"] = []
        for p in rows_apps:
            if len(p) < 3 or p[0] not in by_uuid:
                continue
            g = out[by_uuid[p[0]]]
            with contextlib.suppress(ValueError):
                used = p[-1]
                g["procs"].append([int(p[1]), proc_uid(p[1]), round(float(used) / 1024, 3)])
                if apps_ok:  # a process name with commas was split: join the middle back
                    g["apps"].append(
                        {
                            "pid": int(p[1]),
                            "process_name": ", ".join(p[2:-1]),
                            "used_memory": int(used) if used.isdigit() else used,
                        }
                    )
    return out


def backend():
    """$GPU_SAMPLE_BACKEND: auto (NVML, else nvidia-smi), nvml, or smi (tests with a fake nvidia-smi)."""
    return (os.environ.get("GPU_SAMPLE_BACKEND") or "auto").strip().lower()


def _bounded(fn, timeout, *args):
    """fn(*args) in a daemon thread; TimeoutError when it has not returned within `timeout` (a driver
    call wedged in the kernel cannot be interrupted, so the thread is abandoned, never joined)."""
    box = {}

    def run():
        try:
            box["v"] = fn(*args)
        except BaseException as e:  # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target = run, daemon = True, name = "gpu_sample-nvml")
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"no answer in {timeout:.0f}s")
    if "e" in box:
        raise box["e"]
    return box["v"]


def sample(now = None, sampler_nvml = None):
    """(gpus, source). NVML first (bounded by NVML_TIMEOUT_S; a hung call skips NVML for
    NVML_HUNG_SKIP_S), else one nvidia-smi call. Raises when neither answers."""
    mode, nvml_err = backend(), ""
    hung = time.time() - _NVML.get("hung_t", 0) < NVML_HUNG_SKIP_S
    if mode == "nvml" or (mode == "auto" and not hung):
        try:
            return _bounded(sampler_nvml or sample_nvml, NVML_TIMEOUT_S, now), "nvml"
        except Exception as e:  # noqa: BLE001
            if isinstance(e, TimeoutError):
                _NVML["hung_t"] = time.time()
            nvml_err = f"NVML {type(e).__name__}: {str(e)[:120]}; "
            if mode == "nvml":
                raise RuntimeError(nvml_err.rstrip("; ")) from e
    try:
        return sample_smi(now), "nvidia-smi"
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"{nvml_err}{type(e).__name__}: {(str(e).splitlines() or [''])[0][:160]}"
        ) from e


# ------------------------------------------------------------------ merge / window
def _merge(prev, gpus, source, now):
    hist = dict((prev or {}).get("history") or {})
    for g, d in gpus.items():
        pts = [p for p in hist.get(g, []) if now - p[0] <= 60]
        if d.get("util_now") is not None:
            pts.append([round(now, 2), d["util_now"]])
        hist[g] = pts[-HISTORY_N:]
        win = d.pop("_win", [])
        if not win:  # no driver buffer (nvidia-smi): our own recent points
            win = [u for t, u in hist[g] if now - t <= WINDOW_S + 1]
        d["util_max_5s"] = max(win) if win else d.get("util_now")
        d["util_mean_5s"] = round(sum(win) / len(win), 1) if win else d.get("util_now")
    return {
        "t": round(now, 3),
        "source": source,
        "stale": False,
        "error": None,
        "by": os.getpid(),
        "gpus": gpus,
        "history": hist,
    }


def refresh(
    ld = None,
    sampler = None,
    now = None,
):
    """Sample now and write the state (caller holds the refresh lock)."""
    prev = load(ld)
    now = now or time.time()
    _PREV["state"] = prev
    try:
        gpus, source = (sampler or sample)(now)
    except Exception as e:  # noqa: BLE001
        st = dict(prev or {"gpus": {}, "history": {}, "t": 0, "source": None})
        msg = str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}"
        st.update(stale = True, error = msg[:300], error_t = round(now, 3))
        _write(ld, st)
        return st
    st = _merge(prev, gpus, source, now)
    _write(ld, st)
    return st


def read(
    max_age = MAX_AGE_S,
    ld = None,
    wait_s = WAIT_S,
    sampler = None,
):
    """The shared state, refreshed by at most one process host-wide when older than max_age."""
    ld = Path(ld or default_dir())
    st = load(ld)
    now = time.time()
    if st and not st.get("stale") and max_age > 0 and now - st.get("t", 0) < max_age:
        return st
    # a failed refresh is not retried by every caller in turn: a wedged nvidia-smi would then be asked
    # back to back for as long as it stays wedged; the stale reading (and its error) serves meanwhile
    if (
        st
        and st.get("stale")
        and max_age > 0
        and now - float(st.get("error_t") or 0) < ERROR_RETRY_S
    ):
        return st
    with contextlib.suppress(OSError):
        _mkdir_shared(ld)
    try:
        fh = _open_shared(ld / LOCK_NAME)
    except OSError:
        return st or {
            "gpus": {},
            "history": {},
            "t": 0,
            "stale": True,
            "error": "lock dir unwritable",
        }
    plat = _plat()
    with fh:
        if not plat.lock(fh, blocking = False):
            # someone is refreshing: wait for their answer, never ask the driver too
            deadline = time.time() + wait_s
            prev_t = (st or {}).get("t", 0)
            while time.time() < deadline:
                time.sleep(0.05)
                cur = load(ld)
                if cur and (cur.get("t", 0) > prev_t or cur.get("error_t", 0) > now):
                    return cur
                if plat.lock(fh, blocking = False):
                    break  # the refresher died / finished without a write: our turn
            else:
                cur = load(ld) or st or {"gpus": {}, "history": {}, "t": 0}
                return {
                    **cur,
                    "stale": True,
                    "error": cur.get("error") or f"refresh in progress over {wait_s:.0f}s",
                }
        try:
            cur = load(ld)  # a refresher that finished between our load and our lock
            if (
                cur
                and not cur.get("stale")
                and max_age > 0
                and time.time() - cur.get("t", 0) < max_age
            ):
                return cur
            return refresh(ld, sampler)
        finally:
            plat.unlock(fh)


def age(st, now = None):
    return (now or time.time()) - float((st or {}).get("t") or 0)


def admissible(st, now = None):
    """Whether the reading is recent enough to admit work on (a stale one is only for display)."""
    return bool(st and st.get("gpus")) and age(st, now) <= ADMIT_MAX_AGE_S


def main(argv = None):
    p = argparse.ArgumentParser(description = "Host-wide GPU reading (see module doc)")
    p.add_argument("--json", action = "store_true")
    p.add_argument("--max-age", type = float, default = MAX_AGE_S)
    p.add_argument(
        "--wait", type = float, default = WAIT_S, help = "seconds to wait on another refresher"
    )
    p.add_argument("--quiet", action = "store_true", help = "refresh only, print nothing")
    a = p.parse_args(argv)
    st = read(a.max_age, wait_s = a.wait)
    if a.quiet:
        return 0
    if a.json:
        print(json.dumps(st, indent = 1))
        return 0
    print(
        f"source {st.get('source')}, {age(st):.1f}s old{', STALE: ' + str(st.get('error')) if st.get('stale') else ''}"
    )
    for g, d in sorted(
        st.get("gpus", {}).items(), key = lambda kv: int(kv[0]) if kv[0].isdigit() else 0
    ):
        print(
            f"GPU {g}: {d['free_gb']:.1f}/{d['total_gb']:.1f} GB free, util now {d.get('util_now')} "
            f"max5s {d.get('util_max_5s')} mean5s {d.get('util_mean_5s')}, {len(d.get('procs') or [])} procs"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
