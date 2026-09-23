"""Shared contract for scripts/jobs/*.py sample runs.

Every job writes one metrics JSON (flushed after each step, so a crash keeps its evidence) and
prints a single `JOB_RESULT {...}` line. Exit code 0 iff every check passed, 1 on a failed check
or crash, 3 (SKIP_EXIT_CODE) when the job called JobRecorder.skip because it cannot run here.

    {"job", "model", "backend", "device", "requested_backend", "config", "versions", "revisions",
     "steps": [{"step", "loss", "grad_norm", "time_ms", "tokens", "tokens_per_s", "peak_mem_gb", ...}],
     "summary": {...}, "checks": {name: {"ok": bool, "detail": str}}, "passed": bool, "error": str|None}

`steps` keeps the list-of-dicts shape StatisticsCallback.save_logs writes, so
torch_debugging_utils.compare_training_runs still reads it. Unavailable metrics are None, never 0.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import importlib.util
import json
import math
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
    p.add_argument("--tiny", action="store_true", help="tiny model + tiny data, for CPU / CI smoke")
    p.add_argument("--max-steps", type=int, default=default_steps)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--backend", default="auto", choices=("auto", "unsloth", "hf"),
                   help="unsloth = must run through Unsloth; hf = plain transformers/TRL reference")
    p.add_argument("--out", default=None, help=f"metrics JSON (default outputs/jobs/{job}.json)")
    p.set_defaults(_job=job, _default_model=default_model, _tiny_model=tiny_model)
    return p


def resolve_args(p, argv=None):
    a = p.parse_args(argv)
    a.model = a.model or (a._tiny_model if a.tiny else a._default_model)
    a.out = a.out or f"outputs/jobs/{a._job}.json"
    # Unsloth refuses to import without an accelerator, so `auto` on a CPU-only host (the
    # staging Linux/Windows portability legs) runs the plain transformers/TRL path.
    if a.backend == "auto" and a._job != "mlx_sft" and detect_device() == "cpu":
        a.backend = "hf"
    return a


def detect_device():
    """cuda | rocm | xpu | mps | mlx | cpu, without importing torch unless it is installed."""
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
    """HEAD of the repo holding `path`, with `-dirty` when `pkg_dir` has local edits.

    With `pkg_dir`, only a repo whose top level is pkg_dir's parent counts: a wheel installed
    into a venv that happens to live inside some other checkout must not report that
    checkout's commit as the package revision.
    """
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
    """Commit of each Unsloth package when installed from a checkout, plus this helper's own."""
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
    """Raised by JobRecorder.skip: the job cannot run here (wrong OS / no accelerator)."""


SKIP_EXIT_CODE = 3


def _finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


class JobRecorder:
    """Context manager: collects steps and checks, flushes JSON, prints JOB_RESULT, sets exit code."""

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
            "steps": [], "summary": {}, "checks": {}, "passed": False, "error": None,
        }
        self._t0 = time.perf_counter()

    # -- recording --------------------------------------------------------------------------
    def set_backend(self, backend):
        self.data["backend"] = backend

    def summary(self, **kv):
        self.data["summary"].update(kv)
        self.flush()

    def step(self, **metrics):
        self.data["steps"].append(metrics)
        self.flush()

    def skip(self, reason):
        """Record why this job cannot run here and exit SKIP_EXIT_CODE. Never counts as a pass."""
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

    # -- standard checks ----------------------------------------------------------------------
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

    # -- context ------------------------------------------------------------------------------
    def __enter__(self):
        self.flush()
        return self

    def __exit__(self, et, ev, tb):
        skipped = et is SkipJob
        if et is not None and et is not SystemExit and not skipped:
            self.data["error"] = "".join(traceback.format_exception(et, ev, tb))[-4000:]
        elif et is SystemExit and ev.code not in (None, 0):
            # raise SystemExit("reason") inside a job: keep the reason, sys.exit below replaces it
            self.data["error"] = f"SystemExit: {ev.code}"
            print(self.data["error"], file=sys.stderr)
        self.data["summary"]["wall_s"] = round(time.perf_counter() - self._t0, 2)
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
        sys.exit(SKIP_EXIT_CODE if skipped else 0 if self.data["passed"] else 1)


# -- torch / HF helpers (imported lazily so the MLX job never needs torch) ------------------------

def adapter_fingerprint(model):
    """Sum of |w| and element count over trainable (LoRA) params; compare before vs after."""
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
    """transformers TrainerCallback that feeds JobRecorder.step once per optimizer step.

    Merges every numeric value the trainer logs (loss, grad_norm, rewards, kl, ...) with wall
    time per step, tokens seen (needs include_num_input_tokens_seen=True) and peak memory.
    """
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
