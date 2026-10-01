"""Shared contract for scripts/jobs/*.py. Each job flushes one metrics JSON per step (a crash
keeps its evidence) and prints one `JOB_RESULT {...}` line. Exit 0 = all checks pass, 1 = failed
check or crash, 3 (SKIP_EXIT_CODE) = JobRecorder.skip, cannot run here.

    {"job", "model", "backend", "device", "requested_backend", "config", "versions", "revisions",
     "steps": [{"step", "loss", "grad_norm", "time_ms", "tokens", "tokens_per_s", "peak_mem_gb", ...}],
     "summary": {...}, "checks": {name: {"ok": bool, "detail": str}}, "passed": bool, "error": str|None,
     "finished": bool}  (finished=false on disk = the run never reached __exit__: interrupted)

`steps` matches StatisticsCallback.save_logs (compare_training_runs reads it). Missing metric = None.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

PACKAGES = ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "vllm", "unsloth",
            "unsloth_zoo", "mlx", "mlx-lm", "mlx-vlm", "bitsandbytes", "triton")


def base_parser(job, default_model, tiny_model, default_steps=5):
    p = argparse.ArgumentParser(description=f"Unsloth sample job: {job}")
    p.add_argument("--model", default=None, help=f"default {default_model}; --tiny uses {tiny_model}")
    p.add_argument("--tiny", action="store_true", help="tiny model + data (CPU / CI smoke)")
    p.add_argument("--max-steps", type=int, default=default_steps)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--backend", default="auto", choices=("auto", "unsloth", "hf"),
                   help="unsloth: require Unsloth; hf: plain transformers/TRL reference")
    p.add_argument("--out", default=None, help=f"metrics JSON (default outputs/jobs/{job}.json)")
    p.add_argument("--compile-cache", default="fresh", choices=("fresh", "keep"),
                   help="fresh: new inductor / triton / Unsloth compile dirs per run, deleted on exit "
                        "(compile + warmup cost is measured every run, no cross-run reuse)")
    p.set_defaults(_job=job, _default_model=default_model, _tiny_model=tiny_model)
    return p


def resolve_args(p, argv=None):
    a = p.parse_args(argv)
    a.model = a.model or (a._tiny_model if a.tiny else a._default_model)
    a.out = a.out or f"outputs/jobs/{a._job}.json"
    if getattr(a, "compile_cache", "keep") == "fresh":
        fresh_compile_cache(a._job)
    # Unsloth needs an accelerator: `auto` on CPU (staging portability legs) runs plain HF.
    if a.backend == "auto" and a._job != "mlx_sft" and detect_device() == "cpu":
        a.backend = "hf"
    return a


_FRESH_CACHE = None


def fresh_compile_cache(job):
    """Point inductor, triton and the Unsloth compiler at a new empty dir (before they are first
    used) and drop in-process dynamo / inductor caches, so every run pays compile cost afresh."""
    global _FRESH_CACHE
    import tempfile
    root = Path(os.environ.get("WORKSPACE") or ".") / "temp" / "jobs_compile"
    root.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix=f"{job}_", dir=root))
    for var, sub_ in (("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton"),
                      ("UNSLOTH_COMPILE_LOCATION", "unsloth_compiled_cache")):
        os.environ[var] = str(d / sub_)
    if "torch" in sys.modules:
        try:
            import torch
            torch._dynamo.reset()
            from torch._inductor.utils import clear_inductor_caches
            clear_inductor_caches()
        except Exception:
            pass
    _FRESH_CACHE = d
    return d


def detect_device():
    """cuda | rocm | xpu | mps | mlx | cpu; imports torch only if installed."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx.core  # noqa: F401
            return "mlx"
        except Exception:
            pass
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "rocm" if getattr(torch.version, "hip", None) else "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


SHARED_GPU_MIN_GB = 1.0


def gpu_share():
    """Other compute processes on this job's GPU (NVML; None when unavailable). A GPU shared with
    another busy process makes step time / tok/s noise, not signal: compare.py does not gate perf
    on it. Loss and memory (per-process allocator peak) stay comparable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        dev = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip()
        h = (pynvml.nvmlDeviceGetHandleByUUID(dev) if dev.startswith(("GPU-", "MIG-"))
             else pynvml.nvmlDeviceGetHandleByIndex(int(dev)))
        me = os.getpid()
        others = [p for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h) if p.pid != me]
        gb = sum((p.usedGpuMemory or 0) for p in others) / 1024**3
        busy = [p for p in others if (p.usedGpuMemory or 0) / 1024**3 >= SHARED_GPU_MIN_GB]
        return {"util_pct": pynvml.nvmlDeviceGetUtilizationRates(h).gpu, "other_procs": len(busy),
                "other_mem_gb": round(gb, 2), "shared": bool(busy)}
    except Exception:
        return None


def versions():
    out = {}
    for name in PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    out["python"] = platform.python_version()
    out["platform"] = f"{platform.system()}-{platform.machine()}"
    return out


def _git(path, *args):
    try:
        r = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True,
                           timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _git_sha(path, pkg_dir=None):
    """HEAD of `path`'s repo (+`-dirty` if `pkg_dir` edited). With `pkg_dir`, the repo top must be
    pkg_dir's parent, so a wheel in a venv inside another checkout reports no revision."""
    top = _git(path, "rev-parse", "--show-toplevel")
    if not top:
        return None
    if pkg_dir is not None and Path(top).resolve() != Path(pkg_dir).resolve().parent:
        return None
    sha = _git(path, "rev-parse", "HEAD")
    if sha and pkg_dir is not None and _git(top, "status", "--porcelain", "--", str(pkg_dir)):
        sha += "-dirty"
    return sha


def revisions():
    """Commit of each checkout-installed Unsloth package, plus jobs/ itself."""
    out = {"jobs": _git_sha(Path(__file__).resolve().parent)}
    for mod in ("unsloth", "unsloth_zoo"):
        try:
            spec = importlib.util.find_spec(mod)
            pkg_dir = Path(spec.origin).resolve().parent if spec and spec.origin else None
            out[mod] = _git_sha(pkg_dir, pkg_dir) if pkg_dir else None
        except Exception:
            out[mod] = None
    return out


class SkipJob(Exception):
    """JobRecorder.skip: wrong OS / no accelerator."""


SKIP_EXIT_CODE = 3


def _finite(x):
    # bool is an int subclass: a stray True/False is not a metric.
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


WARMUP_STEPS = 2


def step_time_summary(steps):
    """first_step_ms (compile + warmup) and steady_step_ms / steady_tokens_per_s: medians over
    steps after the first WARMUP_STEPS (all steps when there are too few)."""
    ts = [s.get("time_ms") for s in steps]
    tps = [s.get("tokens_per_s") for s in steps]
    k = WARMUP_STEPS if len(steps) > WARMUP_STEPS + 1 else 0
    med = lambda xs: (sorted(xs)[len(xs) // 2] if len(xs) % 2 else sum(sorted(xs)[len(xs) // 2 - 1:len(xs) // 2 + 1]) / 2) if xs else None
    return {"first_step_ms": ts[0] if ts and _finite(ts[0]) else None,
            "steady_step_ms": med([x for x in ts[k:] if _finite(x)]),
            "steady_tokens_per_s": med([x for x in tps[k:] if _finite(x)]),
            "warmup_steps_excluded": k}


class JobRecorder:
    """Collects steps/checks, flushes JSON, prints JOB_RESULT, sets the exit code."""

    def __init__(self, args, backend_hint=None):
        self.path = Path(args.out)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_steps = args.max_steps
        self.requested_backend = args.backend
        self.data = {
            "job": args._job, "model": args.model, "tiny": bool(args.tiny),
            "backend": backend_hint, "requested_backend": args.backend, "device": detect_device(),
            "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
            "versions": versions(), "revisions": revisions(),
            "compile_cache": str(_FRESH_CACHE) if _FRESH_CACHE else "keep",
            "gpu_share": {"start": gpu_share()},
            "steps": [], "summary": {}, "checks": {}, "passed": False, "error": None,
            "finished": False,  # True only once __exit__ ran: False on disk = interrupted snapshot
        }
        self._t0 = time.perf_counter()

    # -- recording --
    def set_backend(self, backend):
        self.data["backend"] = backend

    def summary(self, **kv):
        self.data["summary"].update(kv)
        self.flush()

    def step(self, **metrics):
        self.data["steps"].append(metrics)
        self.flush()

    def skip(self, reason):
        """Record the reason, exit SKIP_EXIT_CODE; never a pass."""
        self.data["skipped"] = str(reason)
        raise SkipJob(reason)

    def check(self, name, ok, detail=""):
        self.data["checks"][name] = {"ok": bool(ok), "detail": str(detail)}
        self.flush()
        return bool(ok)

    def flush(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str))
        tmp.replace(self.path)

    # -- standard checks --
    def standard_training_checks(self, grad_max=1e3):
        steps = [s for s in self.data["steps"] if s.get("loss") is not None or s.get("grad_norm") is not None]
        self.check("completed_steps", len(steps) == self.max_steps,
                   f"{len(steps)} logged, {self.max_steps} requested")
        losses = [s.get("loss") for s in steps]
        self.check("finite_loss", bool(losses) and all(_finite(l) for l in losses), losses)
        gns = [s.get("grad_norm") for s in steps if s.get("grad_norm") is not None]
        self.check("grad_norm_bounded", bool(gns) and all(_finite(g) and 0 < g < grad_max for g in gns),
                   gns if gns else "no grad_norm logged")

    def backend_check(self):
        want, got = self.requested_backend, self.data["backend"]
        ok = want == "auto" or (got or "").startswith(want)
        self.check("backend_matches_request", ok, f"requested {want}, ran {got}")

    # -- context --
    def __enter__(self):
        self.flush()
        return self

    def __exit__(self, et, ev, tb):
        skipped = et is SkipJob
        if et is not None and et is not SystemExit and not skipped:
            self.data["error"] = "".join(traceback.format_exception(et, ev, tb))[-4000:]
        elif et is SystemExit and ev.code not in (None, 0):
            # SystemExit("reason"): keep the reason (sys.exit below replaces it)
            self.data["error"] = f"SystemExit: {ev.code}"
            print(self.data["error"], file=sys.stderr)
        self.data["summary"]["wall_s"] = round(time.perf_counter() - self._t0, 2)
        self.data["summary"].update(step_time_summary(self.data["steps"]))
        gs = self.data.setdefault("gpu_share", {})
        gs["end"] = gpu_share()
        gs["shared"] = any(bool(v and v.get("shared")) for v in (gs.get("start"), gs["end"]))
        self.data["finished"] = True
        checks = self.data["checks"]
        self.data["passed"] = not skipped and self.data["error"] is None and bool(checks) and all(
            c["ok"] for c in checks.values())
        self.flush()
        brief = {k: self.data[k] for k in ("job", "model", "backend", "device", "passed")}
        if skipped:
            brief["skipped"] = self.data["skipped"]
        brief["failed_checks"] = [k for k, c in checks.items() if not c["ok"]]
        brief["out"] = str(self.path)
        print("JOB_RESULT " + json.dumps(brief), flush=True)
        if et is not None and et is not SystemExit and not skipped:
            traceback.print_exception(et, ev, tb)
        if _FRESH_CACHE and self.data["passed"]:  # failed runs keep generated code for debugging
            import shutil
            shutil.rmtree(_FRESH_CACHE, ignore_errors=True)
        sys.exit(SKIP_EXIT_CODE if skipped else 0 if self.data["passed"] else 1)


# -- torch / HF helpers (lazy imports: MLX job never needs torch) --

def adapter_fingerprint(model):
    """sum|w| and numel over trainable (LoRA) params, for before/after diffs."""
    import torch
    total, n = 0.0, 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                total += p.detach().float().abs().sum().item()
                n += p.numel()
    return {"abs_sum": total, "numel": n}


def adapter_changed(before, after, rel=1e-7):
    if not before["numel"]:
        return False, "no trainable parameters"
    delta = abs(after["abs_sum"] - before["abs_sum"])
    return delta > rel * max(before["abs_sum"], 1.0), f"abs_sum {before['abs_sum']:.6g} -> {after['abs_sum']:.6g}"


def peak_memory_gb(reset=False):
    try:
        import torch
    except Exception:
        return None
    if torch.cuda.is_available():
        v = torch.cuda.max_memory_allocated() / 1024**3
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return round(v, 4)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return round(torch.mps.driver_allocated_memory() / 1024**3, 4)
    return None


def metrics_callback(rec):
    """TrainerCallback -> JobRecorder.step per optimizer step: every numeric log value plus step
    time, tokens (needs include_num_input_tokens_seen=True) and peak memory."""
    from transformers import TrainerCallback

    class _CB(TrainerCallback):
        def __init__(self):
            self.t, self.tokens_prev, self.overall_peak = None, 0, 0.0

        def on_train_begin(self, args, state, control, **kw):
            peak_memory_gb(reset=True)

        def on_step_begin(self, args, state, control, **kw):
            self.t = time.perf_counter()

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs or not any(k in logs for k in ("loss", "grad_norm")):
                return
            dt = (time.perf_counter() - self.t) if self.t else None
            seen = getattr(state, "num_input_tokens_seen", 0) or 0
            tokens = seen - self.tokens_prev if seen else None
            self.tokens_prev = seen or self.tokens_prev
            peak = peak_memory_gb(reset=True)
            if peak:
                self.overall_peak = max(self.overall_peak, peak)
            entry = {"step": state.global_step}
            entry.update({k: v for k, v in logs.items() if isinstance(v, (int, float))})
            entry.update({"time_ms": round(dt * 1000, 2) if dt else None, "tokens": tokens,
                          "tokens_per_s": round(tokens / dt, 2) if tokens and dt else None,
                          "peak_mem_gb": peak})
            rec.step(**entry)
            rec.data["summary"]["peak_mem_gb"] = self.overall_peak or None

    return _CB()


def seed_everything(seed):
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
