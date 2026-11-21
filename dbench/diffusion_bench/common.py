"""Shared plumbing for diffusion_bench: prompts, cell specs, GPU sampling and gating, host memory,
environment fingerprint, media saving and the record file every backend writes.

Nothing here imports torch at module level, so the matrix driver and the scorer can run from any
Python (the render itself always runs in the cell's own venv, one cell per process).
"""

from __future__ import annotations

import base64
import io
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

PKG = Path(__file__).resolve().parent
_NVSMI: list = []


def smi(args, timeout: float = 20):
    """nvidia-smi `args` through the host's shim (scripts/nvsmi.py, loaded by path so nothing joins
    sys.path), else the plain binary (a VM or venv without the helper)."""
    if not _NVSMI:
        mod, p = None, PKG.parent / "nvsmi.py"
        if p.exists():
            import importlib.util
            try:
                spec = importlib.util.spec_from_file_location("nvsmi", p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            except Exception:  # noqa: BLE001 - a broken helper must not break a bench
                mod = None
        _NVSMI.append(mod)
    if _NVSMI[0] is not None:
        return _NVSMI[0].run(list(args), timeout = timeout)
    return subprocess.run(["nvidia-smi", *args], capture_output = True, text = True, timeout = timeout)
WS = Path(os.environ.get("WORKSPACE") or os.getcwd()).resolve()
RECORD = "record.json"

# Defaults for a cell. A spec's "defaults" block and each cell override these, in that order.
CELL_DEFAULTS: dict = {
    "kind": "image",  # image | video
    "width": 1024,
    "height": 1024,
    "steps": 25,
    "guidance": 1.0,
    "negative_prompt": None,
    "frames": None,  # video only
    "fps": None,
    "prompts": "image_24",  # a name under prompts/ or a path to a json/txt file
    "ids": None,  # explicit prompt ids instead of the first n
    "n": 24,
    "warmup": 1,
    "short_steps": 5,  # s/step = (median long - median short) / (steps - short_steps)
    "short_n": 3,
    "save_frames_every": 1,  # video: keep every k-th frame in frames.npz for scoring
    "options": {},
    "env": {},
}


# ---------------------------------------------------------------------------------------------- specs
# what every per-VM export of a plan carries (targets.common.vm_slice, matrix.plan_spec, the Mac and tester plan
# copies): baseline file downloads / builds, the Studio prefetch repos, the VM's comparisons (parity gate) and the
# revisions. One helper so no target's export drops one of them.
PLAN_CARRY_KEYS = ("downloads", "prefetch", "comparisons", "revisions")


def plan_carry(plan: dict, vm: dict) -> dict:
    """The PLAN_CARRY_KEYS of ``plan`` for one of its VMs (comparisons restricted to the VM's own)."""
    mine = set(vm.get("comparisons") or [])
    comps = [c for c in plan.get("comparisons") or [] if c.get("id") in mine]
    prefetch = dict(plan.get("prefetch") or {})
    if any("prefetch" in c for c in plan.get("comparisons") or []):
        # the Studio repos this VM's comparisons load: a multi-VM plan's other families would otherwise be
        # snapshot-downloaded on every VM before its first slot (and outlive the cache janitor)
        keep = {k for c in comps for k in c.get("prefetch") or []}
        prefetch = {k: v for k, v in prefetch.items() if k in keep}
    return {"downloads": plan.get("downloads") or {}, "prefetch": prefetch, "comparisons": comps,
            "revisions": dict(plan.get("revisions") or {})}


def merge_cell(defaults: dict, cell: dict) -> dict:
    out = {**CELL_DEFAULTS, **(defaults or {}), **cell}
    out["options"] = {**CELL_DEFAULTS["options"], **(defaults or {}).get("options", {}), **cell.get("options", {})}
    out["env"] = {**(defaults or {}).get("env", {}), **cell.get("env", {})}
    return out


def load_spec(path: str | Path) -> dict:
    spec = json.loads(Path(path).read_text(encoding = "utf-8"))
    spec.setdefault("name", Path(path).stem)
    spec.setdefault("defaults", {})
    spec.setdefault("cells", [])
    tags = [c["tag"] for c in spec["cells"]]
    dup = {t for t in tags if tags.count(t) > 1}
    if dup:
        raise ValueError(f"duplicate cell tags in {path}: {sorted(dup)}")
    resolve_model_aliases(spec)
    return spec


def model_env_key(alias: str) -> str:
    """DBENCH_MODEL_<ALIAS>: the per-host override of a spec model alias (a local dir or another repo id)."""
    return "DBENCH_MODEL_" + "".join(ch if ch.isalnum() else "_" for ch in alias).upper()


def resolve_model(value: Any, models: dict) -> tuple:
    """``"@alias"`` -> (path or repo id, provenance). Order: $DBENCH_MODEL_<ALIAS>, the alias's "local" path when it
    exists on this host, then its "repo" id. Anything not starting with "@" is returned unchanged."""
    if not (isinstance(value, str) and value.startswith("@")):
        return value, None
    alias = value[1:]
    if alias not in models:
        raise KeyError(f"model alias {value!r} is not in the spec's \"models\" table: {sorted(models)}")
    entry = models[alias] or {}
    env = os.environ.get(model_env_key(alias))
    if env:
        return expand_env(env), {"alias": alias, "source": "env", "repo": entry.get("repo")}
    local = entry.get("local")
    if local and Path(expand_env(local)).exists():
        return expand_env(local), {"alias": alias, "source": "local", "repo": entry.get("repo")}
    if entry.get("repo"):
        return entry["repo"], {"alias": alias, "source": "repo", "revision": entry.get("revision")}
    raise FileNotFoundError(f"model alias {value!r}: no {model_env_key(alias)}, no local path, no repo id")


def resolve_model_aliases(spec: dict) -> dict:
    """Replace "@alias" in "model" and in string-valued options with the host's path or repo id, in place. The
    spec's "models" table maps alias -> {"repo", "local", "revision"}; the resolved source is kept in
    cell["model_source"] so a record says whether it ran from a local copy or a repo id."""
    models = spec.get("models") or {}
    if not models:
        return spec
    for block in [spec.get("defaults") or {}] + list(spec.get("cells") or []):
        if "model" in block:
            block["model"], src = resolve_model(block["model"], models)
            if src:
                block["model_source"] = src
        for key, val in list((block.get("options") or {}).items()):
            if isinstance(val, str) and val.startswith("@") and val[1:] in models:
                block["options"][key] = resolve_model(val, models)[0]
    return spec


def expand_env(value: Any) -> Any:
    """``$WORKSPACE`` / ``~`` / env vars inside strings, recursively."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value.replace("$WORKSPACE", str(WS))))
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------------------------- prompts
def load_prompts(name: str, n: Optional[int] = None, ids: Optional[Iterable[str]] = None) -> list[dict]:
    """Rows of {id, prompt, seed, split}. ``name`` is a file under prompts/ (without .json) or a path.
    A .txt file gives one prompt per line with seeds 20260920+i."""
    path = Path(expand_env(name))
    if not path.exists():
        path = PKG / "prompts" / f"{name}.json"
    if path.suffix == ".txt":
        lines = [ln.strip() for ln in path.read_text(encoding = "utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
        rows = [{"id": f"p{i:02d}", "prompt": p, "seed": 20260920 + i, "split": "eval"} for i, p in enumerate(lines)]
    else:
        data = json.loads(path.read_text(encoding = "utf-8"))
        rows = data["rows"] if isinstance(data, dict) else data
    if ids:
        want = list(ids)
        by_id = {r["id"]: r for r in rows}
        missing = [i for i in want if i not in by_id]
        if missing:
            raise KeyError(f"prompt ids not in {path.name}: {missing}")
        return [by_id[i] for i in want]
    return rows[:n] if n else rows


# ---------------------------------------------------------------------------------------------- GPU
def gpu_vendor() -> str:
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if shutil.which("amd-smi") or shutil.which("rocm-smi"):
        return "amd"
    return "none"


def visible_gpu() -> Optional[str]:
    """The first device of the process's mask (index or UUID), as the SMI tools take it. None when the mask
    is set but empty (a CPU-only run), so nothing samples a device the process cannot see."""
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        raw = os.environ.get(key)
        if raw is not None:
            raw = raw.strip()
            return raw.split(",")[0].strip() if raw and raw != "-1" else None
    return "0"


def gpu_query(device: Optional[str] = None) -> dict:
    """{used_mib, total_mib, util, name} for one device, or {} when there is no SMI to ask."""
    device = device or visible_gpu()
    vendor = gpu_vendor()
    if device is None:
        return {}
    try:
        if vendor == "nvidia":
            out = smi(["-i", device, "--query-gpu=memory.used,memory.total,utilization.gpu,name",
                       "--format=csv,noheader,nounits"], timeout = 10).stdout.strip().splitlines()[0]
            used, total, util, name = [x.strip() for x in out.split(",", 3)]
            return {"used_mib": int(used), "total_mib": int(total), "util": int(util), "name": name}
        if vendor == "amd" and shutil.which("amd-smi"):
            raw = subprocess.run(["amd-smi", "metric", "-g", device, "--json"], capture_output = True,
                                 text = True, timeout = 10).stdout
            data = json.loads(raw)
            data = data[0] if isinstance(data, list) else data
            mem = data.get("mem_usage") or data.get("memory_usage") or {}
            used = mem.get("used_vram", {}).get("value") if isinstance(mem.get("used_vram"), dict) else mem.get("used_vram")
            total = mem.get("total_vram", {}).get("value") if isinstance(mem.get("total_vram"), dict) else mem.get("total_vram")
            util = (data.get("usage") or {}).get("gfx_activity", {})
            util = util.get("value") if isinstance(util, dict) else util
            return {"used_mib": int(used or 0), "total_mib": int(total or 0), "util": int(util or 0), "name": "amd"}
        if vendor == "amd":
            raw = subprocess.run(["rocm-smi", "-d", device, "--showmeminfo", "vram", "--showuse", "--json"],
                                 capture_output = True, text = True, timeout = 10).stdout
            card = next(iter(json.loads(raw).values()))
            used = int(card.get("VRAM Total Used Memory (B)", 0)) >> 20
            total = int(card.get("VRAM Total Memory (B)", 0)) >> 20
            return {"used_mib": used, "total_mib": total, "util": int(card.get("GPU use (%)", 0)), "name": "amd"}
    except Exception:  # noqa: BLE001 - sampling is best effort
        return {}
    return {}


def gpu_compute_cap(device: Optional[str] = None) -> Optional[tuple]:
    """(major, minor) of an NVIDIA device, or None (other vendors, no SMI, unparsable)."""
    device = device or visible_gpu()
    if device is None or gpu_vendor() != "nvidia":
        return None
    try:
        out = smi(["-i", device, "--query-gpu=compute_cap", "--format=csv,noheader"], timeout = 10).stdout
        major, minor = out.strip().splitlines()[0].split(".")
        return int(major), int(minor)
    except Exception:  # noqa: BLE001 - unknown: the cell runs and judges for itself
        return None


class GpuSampler(threading.Thread):
    """Peak device memory as the driver sees it (every process on the device, allocator caches included),
    sampled every ``interval`` s. The only fair VRAM number across frameworks that manage memory themselves
    (BlineUI, sd.cpp); pair it with torch's own peak where the backend is in process."""

    def __init__(self, interval: float = 0.25, device: Optional[str] = None):
        super().__init__(daemon = True)
        self.extra_pids = lambda: []  # set by the runner once the backend exists
        self.interval, self.device = interval, device or visible_gpu()
        self.peak_mib = 0
        self.own_peak_mib = 0
        self.baseline_mib = gpu_query(self.device).get("used_mib", 0)
        self._stop_ev = threading.Event()  # not _stop: Thread._stop() is called by join() on Python 3.12

    def _own_mib(self) -> int:
        """VRAM held by this process and its children (servers a backend launched), NVIDIA only. On a shared
        GPU the device-wide figure moves with other tenants; this one does not."""
        if gpu_vendor() != "nvidia":
            return 0
        try:
            import psutil

            me = psutil.Process()
            pids = {me.pid} | {c.pid for c in me.children(recursive = True)} | set(self.extra_pids() or [])
            out = smi(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], timeout = 10).stdout
            return sum(int(used) for pid, used in (ln.split(",") for ln in out.strip().splitlines() if "," in ln)
                       if int(pid) in pids)
        except Exception:  # noqa: BLE001
            return 0

    def run(self) -> None:
        while not self._stop_ev.is_set():
            q = gpu_query(self.device)
            if q:
                self.peak_mib = max(self.peak_mib, q["used_mib"])
            self.own_peak_mib = max(self.own_peak_mib, self._own_mib())
            self._stop_ev.wait(self.interval)

    def reset(self) -> None:
        # Seeded with a reading, so a cell shorter than one interval does not report 0.
        q = gpu_query(self.device)
        self.peak_mib = q.get("used_mib", 0) if q else 0
        self.own_peak_mib = self._own_mib()

    def stop(self) -> None:
        self._stop_ev.set()


def gpu_gate(max_util: int = 10, min_free_mib: int = 0, timeout_s: float = 1800, poll_s: float = 20,
             device: Optional[str] = None, log=print, max_foreign_mib: Optional[int] = 1024,
             own_pids: Optional[Iterable[int]] = None) -> dict:
    """Wait until the device is quiet before a timed cell. Returns the last reading plus ``gate``:
    ok, timeout (ran anyway; the record is flagged), or no_smi. Foreign compute processes (outside ``own_pids``,
    default this process's tree) are counted: more than ``max_foreign_mib`` held by them keeps the gate closed even
    when utilisation is low, since an idle tenant's allocations still move the free-memory planner."""
    import telemetry as T

    device = device or visible_gpu()
    deadline = time.time() + timeout_s
    own = set(own_pids) if own_pids is not None else T.tree_pids([os.getpid()])
    while True:
        q = gpu_query(device)
        if not q:
            return {"gate": "no_smi"}
        f = T.foreign(device, own)
        q = {**q, "foreign_procs": f.get("procs"), "foreign_mib": f.get("mib")}
        free = q["total_mib"] - q["used_mib"]
        foreign_ok = max_foreign_mib is None or f.get("mib") is None or f["mib"] <= max_foreign_mib
        if q["util"] <= max_util and free >= min_free_mib and foreign_ok:
            return {**q, "gate": "ok"}
        if time.time() > deadline:
            log(f"gpu gate TIMEOUT on {device}: util {q['util']}% free {free} MiB, {f.get('procs')} foreign "
                f"process(es) holding {f.get('mib')} MiB; running anyway, flagged")
            return {**q, "gate": "timeout"}
        time.sleep(poll_s)


def capability_gate(need: Any, gpu: Optional[dict] = None, device: Optional[str] = None) -> tuple:
    """(ok, reason) for a cell's ``min_compute_cap`` ("8.9"). An NVIDIA compute capability is meaningless on other
    vendors: gfx1151 reports (11, 5) through torch, which an NVIDIA ladder ranks as Blackwell, so an AMD / Apple /
    CPU device is unsupported for such a cell rather than "unknown, run it". An NVIDIA card whose capability cannot
    be read runs (the cell judges for itself)."""
    if not need:
        return True, None
    want = tuple(int(x) for x in str(need).split("."))
    vendor = gpu_vendor()
    cc = (gpu or {}).get("cc")
    have = tuple(int(x) for x in str(cc).split(".")) if cc else gpu_compute_cap(device)
    if vendor != "nvidia" and not cc:
        return False, f"unsupported: needs NVIDIA compute capability {need}, device vendor is {vendor}"
    if have is None:
        return True, None
    if tuple(have) < want:
        return False, f"unsupported: needs compute capability {need}, device is {have[0]}.{have[1]}"
    return True, None


def auto_dtype(cc: Optional[str]) -> str:
    """bf16 on compute capability 8.0+ (and on non-NVIDIA devices that report none), fp16 below (T4 is 7.5: bf16 is
    emulated there, several times slower, and some kernels refuse it)."""
    if not cc:
        return "bf16"
    major = int(str(cc).split(".")[0])
    return "bf16" if major >= 8 else "fp16"


# ---------------------------------------------------------------------------------------------- host memory
def host_memory() -> dict:
    """RSS of THIS process split into anonymous and file-backed pages (Linux), plus its lifetime peak. For an
    out-of-process backend (BlineUI, sd.cpp, Studio over HTTP) this is the polling client, not the model: the cell's
    host-RAM figure is telemetry.Sampler's process-tree peak (memory.peak_rss_tree_gib), which this only annotates."""
    out: dict = {}
    try:
        import psutil

        out["rss_gib"] = round(psutil.Process().memory_info().rss / 2**30, 3)
        vm = psutil.virtual_memory()
        out["host_available_gib"] = round(vm.available / 2**30, 2)
    except Exception:  # noqa: BLE001
        pass
    status = Path("/proc/self/status")
    if status.exists():
        fields = {ln.split(":")[0]: ln.split()[1] for ln in status.read_text().splitlines() if ln.startswith(("Rss", "VmHWM"))}
        for key, name in (("RssAnon", "rss_anon_gib"), ("RssFile", "rss_file_gib"), ("RssShmem", "rss_shmem_gib"),
                          ("VmHWM", "peak_rss_gib")):
            if key in fields:
                out[name] = round(int(fields[key]) / 2**20, 3)
    elif sys.platform != "win32":
        import resource

        scale = 2**30 if sys.platform == "darwin" else 2**20
        out["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale, 3)
    return out


def kill_tree(proc: Any, timeout: float = 30) -> None:
    """Terminate a child and everything under it, on every OS. psutil walks the tree when present (TERM, wait,
    KILL); otherwise POSIX kills the process group (children started with start_new_session) and Windows runs
    ``taskkill /T /F``. ``proc`` is a Popen or a pid. Never raises."""
    pid = getattr(proc, "pid", proc)
    if not pid:
        return
    try:
        import psutil

        try:
            root = psutil.Process(pid)
            procs = [root] + root.children(recursive = True)
        except psutil.NoSuchProcess:
            procs = []
        for p in procs:
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass
        _, alive = psutil.wait_procs(procs, timeout = timeout)
        for p in alive:
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass
        psutil.wait_procs(alive, timeout = 10)
    except ImportError:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output = True)
        elif hasattr(os, "killpg"):
            import signal

            for sig, wait in ((signal.SIGTERM, timeout), (signal.SIGKILL, 10)):
                try:
                    os.killpg(pid, sig)
                except (ProcessLookupError, PermissionError):
                    break
                if hasattr(proc, "wait"):
                    try:
                        proc.wait(timeout = wait)
                        break
                    except Exception:  # noqa: BLE001
                        continue
    if hasattr(proc, "wait"):
        try:
            proc.wait(timeout = 10)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------------------------- environment
def _version(mod: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(mod)
    except Exception:  # noqa: BLE001
        return None


def vcs_commit(dist: str) -> Optional[str]:
    """The git commit an installed distribution was built from (PEP 610 direct_url.json ``vcs_info.commit_id``, what
    ``pip install git+URL@ref`` records), else None. A version string alone does not identify a git install."""
    try:
        from importlib.metadata import distribution

        raw = distribution(dist).read_text("direct_url.json")
        return ((json.loads(raw) or {}).get("vcs_info") or {}).get("commit_id") if raw else None
    except Exception:  # noqa: BLE001
        return None


def git_rev(path: str | Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short=10", "HEAD"], capture_output = True,
                             text = True, timeout = 10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def scripts_sha() -> dict:
    """{sha, dirty, in_git} of the diffusion_bench tree that produced a record. A copy outside git (a per-round fork,
    a notebook bundle) is named, since its code exists nowhere else."""
    sha = git_rev(PKG)
    # the launcher's statement of the bundle's git sha (targets/vm_driver.py, the tester bundle):
    # DBENCH_SCRIPTS_SHA, DBENCH_BUNDLE_SHA its older name
    stated = os.environ.get("DBENCH_SCRIPTS_SHA") or os.environ.get("DBENCH_BUNDLE_SHA") or None
    if not sha:
        if not sys.stderr.closed:
            print(f"[diffusion_bench] warning: {PKG} is not in a git checkout; scripts_sha = "
                  f"{stated or 'unknown'} (bundle)", file = sys.stderr)
        return {"sha": stated, "in_git": False, "source": "launcher" if stated else None}
    dirty = None
    try:
        out = subprocess.run(["git", "-C", str(PKG), "status", "--porcelain", "--untracked-files=no", "--", "."],
                             capture_output = True, text = True, timeout = 10).stdout.strip()
        dirty = bool(out)
    except Exception:  # noqa: BLE001
        pass
    out = {"sha": sha, "in_git": True, "dirty": dirty}
    if stated and not sha.startswith(stated) and not stated.startswith(sha):
        out["launcher_sha"] = stated  # the launcher meant another revision than the checkout that ran
    return out


def env_fingerprint(trees: Optional[dict] = None, with_torch: bool = True) -> dict:
    """What produced a record: interpreter, platform, packages, device and driver, git revisions."""
    fp: dict = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "packages": {m: _version(m) for m in ("torch", "diffusers", "transformers", "accelerate", "torchao",
                                               "flashinfer-python", "sageattention", "unsloth", "unsloth_zoo",
                                               _ext("bline-kitchen"), "lpips")},
        "gpu": gpu_query(),
        "visible_devices": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
                            if os.environ.get(k) is not None},
    }
    fp["packages"] = {k: v for k, v in fp["packages"].items() if v}
    commits = {m: vcs_commit(m) for m in ("unsloth", "unsloth_zoo")}
    if any(commits.values()):
        fp["package_commits"] = {k: v for k, v in commits.items() if v}
    if with_torch and "torch" in sys.modules:
        torch = sys.modules["torch"]
        fp["torch_cuda"] = getattr(torch.version, "cuda", None)
        fp["torch_hip"] = getattr(torch.version, "hip", None)
        try:
            if torch.cuda.is_available():
                fp["device_name"] = torch.cuda.get_device_name(0)
                fp["device_capability"] = list(torch.cuda.get_device_capability(0))
                fp["gcn_arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
        except Exception:  # noqa: BLE001
            pass
    if trees:
        fp["trees"] = {name: git_rev(path) for name, path in trees.items() if path}
    return fp


def _ext(name: str) -> str:
    import extnames  # noqa: PLC0415

    return extnames.ext(name)


def unsafe_relpath(name) -> "str | None":
    """Why ``name`` (an archive member, a tester-supplied tag) must not become a path under an output dir, else None:
    empty, absolute (POSIX or Windows), a drive letter / UNC prefix, or a ``..`` component."""
    import re as _re

    s = str(name or "")
    if not s.strip():
        return "empty name"
    if s.startswith(("/", "\\")) or _re.match(r"^[A-Za-z]:", s):
        return f"absolute path {s!r}"
    if ".." in _re.split(r"[\\/]+", s):
        return f"parent component in {s!r}"
    return None


def contained(root, name) -> "Path | None":
    """``root / name`` resolved (existing symlinks followed) when it stays inside ``root``, else None."""
    from pathlib import Path as _P

    if unsafe_relpath(name):
        return None
    base = _P(root).resolve()
    dest = (base / str(name)).resolve()
    return dest if dest == base or base in dest.parents else None


def scrub_tokens() -> None:
    """A render never needs a token (models are local or public), and one in the environment ends up in logs
    and child processes. Callers that must download pass it explicitly to the downloader instead."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------------------------- media
def to_pil(obj: Any):
    """PIL image from a PIL image, a (data-)URL/base64 string, a numpy HWC array, or a file path."""
    from PIL import Image

    if hasattr(obj, "save") and hasattr(obj, "convert"):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(obj))
    if isinstance(obj, str):
        if Path(obj).exists():
            return Image.open(obj)
        return Image.open(io.BytesIO(base64.b64decode(obj.split(",", 1)[-1])))
    import numpy as np

    arr = np.asarray(obj)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return Image.fromarray(arr)


def save_image(obj: Any, path: Path) -> Path:
    to_pil(obj).convert("RGB").save(path)
    return path


def frames_array(frames: Any):
    """uint8 [T, H, W, 3] from a list of PIL frames / arrays or one array (floats in [0, 1] are scaled)."""
    import numpy as np

    if isinstance(frames, (list, tuple)):
        arr = np.stack([np.asarray(to_pil(f).convert("RGB")) if not isinstance(f, np.ndarray) else f for f in frames])
    else:
        arr = np.asarray(frames)
    while arr.ndim > 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
    return arr


def save_video(frames: Any, stem: Path, fps: int = 16, every: int = 1) -> dict:
    """frames.npz (every k-th frame, what the scorer reads) and an .mp4 when imageio can write one."""
    import numpy as np

    arr = frames_array(frames)
    np.savez_compressed(f"{stem}.npz", frames = arr[::max(1, every)])
    out = {"file": f"{stem.name}.npz", "frames": int(arr.shape[0]), "shape": list(arr.shape[1:])}
    try:
        import imageio.v3 as iio

        iio.imwrite(f"{stem}.mp4", arr, fps = fps or 16, codec = "libx264")
        out["mp4"] = f"{stem.name}.mp4"
    except Exception:  # noqa: BLE001 - scoring only needs the npz
        pass
    return out


# ---------------------------------------------------------------------------------------------- records
def stat_median(values: Iterable) -> Optional[float]:
    """THE median every record, scorer and report uses: statistics.median over the finite numbers (None, NaN and
    Infinity dropped), unrounded. Raw samples always stay in the record next to it."""
    import math

    vals = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
    return statistics.median(vals) if vals else None


def median(values: list) -> Optional[float]:
    """stat_median rounded to 4 decimals, for display fields."""
    m = stat_median(values)
    return round(m, 4) if m is not None else None


# Levers whose step skipping makes a short run unrepresentative of a long one (the cache warms over the run).
STEP_CACHE_OPTIONS = ("transformer_cache", "easycache", "fbcache", "first_block_cache", "teacache", "magcache",
                      "static_skip", "step_cache")


def step_cache_on(cell: dict) -> Optional[str]:
    opts = cell.get("options") or {}
    # Studio's "max" tier engages a step cache / static step skip on its own (diffusion_cache.AUTO_STEP_CACHE_TIER)
    if str(opts.get("speed_mode") or (opts.get("load") or {}).get("speed_mode") or "").lower() == "max":
        return "speed_mode=max"
    for key in STEP_CACHE_OPTIONS:
        v = opts.get(key)
        if v not in (None, False, "none", "off", 0, ""):
            return key
        if key in (opts.get("load") or {}) and (opts["load"][key] not in (None, False, "none", "off", 0, "")):
            return key
    extra = opts.get("graph_extra") or {}
    for key in STEP_CACHE_OPTIONS:
        if extra.get(key) not in (None, False, "none", "off", 0, ""):
            return f"graph_extra.{key}"
    return None


def polled_step_s(step_times: list, from_step: int = 3) -> Optional[float]:
    """Median per-step duration of one request from its polled step timestamps [(step, t), ...], from step
    ``from_step`` on (step 1 carries first-call overheads, step 2 often a recompile or autotune)."""
    seen: dict = {}
    for step, t in step_times or []:
        if step is not None and step not in seen:
            seen[int(step)] = float(t)
    ks = sorted(seen)
    deltas = [(seen[b] - seen[a]) / (b - a) for a, b in zip(ks, ks[1:]) if b > a and b >= from_step]
    return stat_median(deltas)


def finalize_record(rec: dict) -> dict:
    """Summary numbers every scorer reads, derived identically for every backend: the v1 fields (kept for older
    readers) and the v2 timing block from ``requests``."""
    cell = rec.get("cell", {})
    walls = [r["wall_s"] for r in rec.get("renders", [])]
    rec["wall_s_median"] = median(walls)
    rec["step_s_reported_median"] = median([r.get("step_s") for r in rec.get("renders", [])])
    long_, short = rec.get("long", []), rec.get("short", [])
    derived = None
    if long_ and short and cell.get("steps") and cell.get("short_steps") and cell["steps"] > cell["short_steps"]:
        derived = (stat_median(long_) - stat_median(short)) / (cell["steps"] - cell["short_steps"])
        rec["step_s_derived"] = round(derived, 4)
    rec["steady_s_median"] = median(long_)
    peaks = [r.get("peak_alloc_gib") for r in rec.get("renders", []) if r.get("peak_alloc_gib") is not None]
    if peaks:
        rec["peak_alloc_gib"] = max(peaks)
    if rec.get("schema_version") != 2:
        return rec
    reqs = rec.get("requests") or []
    t = rec.setdefault("timing", {})
    new = [r for r in reqs if r.get("kind") == "new"]
    same_long = [r for r in reqs if r.get("kind") == "same" and r.get("steps") == cell.get("steps")]
    t["new_s_median"] = median([r.get("total_s") for r in new])
    t["steady_s_median"] = median([r.get("total_s") for r in same_long]) if same_long else t["new_s_median"]
    polled = [r.get("step_s_polled") for r in new + same_long]
    t["step_s_polled_median"] = median(polled)
    t["step_s_derived"] = round(derived, 4) if derived is not None else None
    flags = []
    cache = step_cache_on(cell)
    if derived is not None and cache:
        flags.append(f"step_cache_on:{cache} (short run never reaches the cached regime; use step_s_polled_median)")
    if derived is not None and derived <= 0:
        flags.append("non_positive (noise exceeds the long-short gap)")
    t["step_s_derived_flag"] = "; ".join(flags) or None
    firsts = [r.get("first_step_s") for r in new]
    t["first_step_new_s_median"] = median(firsts)
    for key in ("te_s", "denoise_s", "vae_s"):
        t[f"{key}_median"] = median([(r.get("phases") or {}).get(key) for r in new])
    return rec


def write_record(out: Path, rec: dict) -> Path:
    """Atomic, strictly finite JSON. A non-finite value is replaced by None and named in ``nonfinite``; an "ok"
    record carrying one becomes VOID (a NaN timing is a harness bug, never a number to report)."""
    import schema

    out.mkdir(parents = True, exist_ok = True)
    path = out / RECORD
    rec, bad = schema.sanitize(rec)
    if bad:
        rec["nonfinite"] = bad[:50]
        if rec.get("verdict") == "ok":
            rec["verdict"] = "VOID"
            rec["reason"] = f"non-finite values in the record: {bad[:5]}"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(schema.dumps(rec))
    os.replace(tmp, path)
    return path


def read_record(cell_dir: Path) -> Optional[dict]:
    """The record as written (v1 or v2); None when absent, unreadable or not strictly finite JSON."""
    import schema

    path = Path(cell_dir) / RECORD
    try:
        return schema.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def record_ok(rec: Optional[dict]) -> bool:
    return bool(rec) and rec.get("verdict") == "ok"


def log(msg: str) -> None:
    print(f"[diffusion_bench {time.strftime('%H:%M:%S')}] {msg}", flush = True)


# ---------------------------------------------------------------------------------------------- Studio's resolved tier
# What a Studio load RAN, read from its status (not the request): a "default" speed tier is whatever Studio resolves
# on that card and model, so records, medians and parity are keyed by this. TIER_COMPUTE_KEYS change the numbers a
# render computes (a quantized GEMM, INT8 attention, a step cache); the rest change only how fast it runs.
TIER_KEYS = ("speed_mode", "compiled", "cuda_graph", "attention_backend", "int8_attention", "transformer_quant",
             "text_encoder_quant", "transformer_cache", "offload_policy")
TIER_COMPUTE_KEYS = ("attention_backend", "int8_attention", "transformer_quant", "text_encoder_quant",
                     "transformer_cache")


def resolved_tier(status) -> Optional[dict]:
    """{speed_mode, compiled, cuda_graph, attention_backend, int8_attention, transformer_quant, text_encoder_quant,
    transformer_cache, offload_policy} from a Studio status / status_after dict; None when there is no status.
    ``compiled`` / ``cuda_graph`` come from status.speed_optims (None when Studio reported no optimisation list:
    unknown, not off)."""
    if not isinstance(status, dict) or not status:
        return None
    optims = status.get("speed_optims")
    optims = [str(x) for x in optims] if isinstance(optims, (list, tuple)) else None
    att = status.get("attention_backend")
    cache = status.get("transformer_cache")
    # Studio's own per-step audit (status.resolved.speed_mode.steps) is its decision record; when its cuda_graph step
    # disagrees with speed_optims (Wan2.2-TI2V-5B on the Nov 19 L4: audit "engaged", the list without it, same
    # timings) the audit wins and the conflict is kept in the record
    steps = (((status.get("resolved") or {}).get("speed_mode") or {}).get("steps") or [])
    audit = {str(x.get("step")): x.get("status") for x in steps if isinstance(x, dict)}
    graph_list = None if optims is None else "cuda_graph" in optims
    graph_audit = None if "cuda_graph" not in audit else audit["cuda_graph"] == "engaged"
    if isinstance(cache, dict):
        cache = cache.get("mode") or cache.get("kind") or cache.get("name") or "on"
    return {"speed_mode": status.get("speed_mode"),
            "compiled": None if optims is None else "compiled" in optims,
            "cuda_graph": graph_audit if graph_audit is not None else graph_list,
            "cuda_graph_conflict": (graph_audit is not None and graph_list is not None and graph_audit != graph_list)
            or None,
            "attention_backend": att,
            "int8_attention": None if att is None and optims is None else ("int8" in str(att or "").lower()),
            "transformer_quant": status.get("transformer_quant"),
            "text_encoder_quant": status.get("text_encoder_quant"),
            "transformer_cache": cache or None,
            "offload_policy": status.get("offload_policy")}


def tier_label(tier: Optional[dict]) -> str:
    """One line for a resolved tier: 'default: compiled + CUDA graph, attention unsloth_int8 (INT8), DiT int8,
    TE int8'."""
    if not tier:
        return "UNAVAILABLE (no Studio status)"
    run = []
    if tier.get("compiled") is not None:
        run.append("compiled" if tier["compiled"] else "eager")
    if tier.get("cuda_graph"):
        run.append("CUDA graph")
    parts = [" + ".join(run) or "compile unknown"]
    if tier.get("attention_backend") is not None or tier.get("int8_attention") is not None:
        parts.append(f"attention {tier.get('attention_backend') or 'default'}"
                     + (" (INT8)" if tier.get("int8_attention") else ""))
    parts.append(f"DiT {tier.get('transformer_quant') or 'none'}")
    if tier.get("text_encoder_quant"):
        parts.append(f"TE {tier['text_encoder_quant']}")
    if tier.get("transformer_cache"):
        parts.append(f"step cache {tier['transformer_cache']}")
    if tier.get("offload_policy") not in (None, "none"):
        parts.append(f"offload {tier['offload_policy']}")
    return f"{tier.get('speed_mode') or '?'}: " + ", ".join(parts)


def tier_compute_diff(a: Optional[dict], b: Optional[dict]) -> list:
    """['key: a vs b'] for the compute-relevant tier fields two Studio runs disagree on ([] = same computation)."""
    if not a or not b:
        return []
    return [f"{k}: {a.get(k)!r} vs {b.get(k)!r}" for k in TIER_COMPUTE_KEYS if a.get(k) != b.get(k)]
