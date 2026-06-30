# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU benchmark of ONE diffusion engine on ONE model: latency + peak RAM.

Two engines, the SAME quantized transformer GGUF:
  diffusers  the live product path -- get_diffusion_backend().begin_load(...) + generate(...),
             forced onto CPU (the orchestrator launches this with CUDA_VISIBLE_DEVICES=""), which
             dequantizes the GGUF to fp32 and runs torch ops on CPU.
  sdcpp      the native engine -- SdCppEngine(...).generate(...) shelling out to sd-cli, which runs
             the GGUF quantized weights natively on ggml CPU kernels.

Peak RAM is sampled the SAME way for both: a thread tracking max RSS of this process + all children
(~5 Hz), so the in-process diffusers and the sd-cli subprocess are measured identically. Run ONE
engine per process (fresh process => clean peak). Prints a single RESULT line the orchestrator parses.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "studio" / "backend"
PROMPT = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed"


class _RSSSampler:
    """Background sampler of peak RSS (this process + all descendants), in bytes."""

    def __init__(self, hz: float = 5.0) -> None:
        import psutil

        self._psutil = psutil
        self._proc = psutil.Process(os.getpid())
        self._interval = 1.0 / hz
        self._peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target = self._loop, daemon = True)

    def _sample(self) -> int:
        total = 0
        try:
            total += self._proc.memory_info().rss
            for c in self._proc.children(recursive = True):
                try:
                    total += c.memory_info().rss
                except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                    pass
        except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
            pass
        return total

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._peak = max(self._peak, self._sample())
            time.sleep(self._interval)

    def __enter__(self) -> "_RSSSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout = 2.0)
        self._peak = max(self._peak, self._sample())

    @property
    def peak_gb(self) -> float:
        return self._peak / 1e9


def _emit(engine: str, family: str, *, status: str, **kw) -> None:
    parts = [f"engine={engine}", f"family={family}", f"status={status}"]
    parts += [f"{k}={v}" for k, v in kw.items()]
    print("RESULT " + " ".join(parts), flush = True)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# ── diffusers (live backend, CPU) ────────────────────────────────────────────


def run_diffusers(args) -> int:
    sys.path.insert(0, str(BACKEND))
    import torch

    torch.set_num_threads(int(args.threads))
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    gguf_dir = str(Path(args.gguf).resolve().parent)
    gguf_name = Path(args.gguf).name

    def _wait_ready(timeout_s = 7200) -> None:
        deadline = time.time() + timeout_s
        last = None
        while time.time() < deadline:
            p = backend.load_progress()
            phase = p.get("phase")
            if phase != last:
                last = phase
                print(f"  load phase={phase} frac={p.get('fraction') or 0:.2f}", flush = True)
            if phase == "ready":
                return
            if phase == "error":
                raise RuntimeError(f"load error: {p.get('error')}")
            time.sleep(2)
        raise TimeoutError("load did not reach ready")

    def _gen():
        r = backend.generate(
            prompt = PROMPT, width = args.width, height = args.height,
            steps = args.steps, guidance = args.guidance, seed = args.seed, batch_size = 1,
        )
        return r["images"][0]

    with _RSSSampler() as rss:
        try:
            backend.begin_load(
                gguf_dir,
                gguf_filename = gguf_name,
                base_repo = args.base_repo,
                family_override = args.family,
                hf_token = os.environ.get("HF_TOKEN"),
                memory_mode = "fast",
                speed_mode = "off",
            )
            _wait_ready()
            for _ in range(max(0, args.warmup)):
                _gen()
            lat = []
            img = None
            for i in range(max(1, args.iters)):
                t0 = time.time()
                img = _gen()
                lat.append(time.time() - t0)
                print(f"  gen[{i}] {lat[-1]:.1f}s", flush = True)
            out = Path(args.out)
            out.parent.mkdir(parents = True, exist_ok = True)
            img.save(out)
        except Exception as exc:  # noqa: BLE001
            _emit("diffusers", args.family, status = "blocked", reason = f"{type(exc).__name__}:{str(exc)[:80]}")
            return 1
        finally:
            try:
                backend.unload()
            except Exception:  # noqa: BLE001
                pass
    _emit(
        "diffusers", args.family, status = "OK",
        latency_s = f"{_median(lat):.1f}", peak_rss_gb = f"{rss.peak_gb:.1f}",
        steps = args.steps, res = f"{args.width}x{args.height}", image = str(args.out),
    )
    return 0


# ── sd.cpp (native sd-cli) ───────────────────────────────────────────────────


def run_sdcpp(args) -> int:
    sys.path.insert(0, str(BACKEND))
    from core.inference.sd_cpp_engine import SdCppEngine
    from core.inference.sd_cpp_args import SdCppGenParams, SdCppModelFiles

    engine = SdCppEngine(binary = args.sd_cli or None)
    if not engine.is_available():
        _emit("sdcpp", args.family, status = "blocked", reason = "sd-cli not found")
        return 1

    files = SdCppModelFiles(
        diffusion_model = str(Path(args.gguf).resolve()),
        vae = args.vae,
        clip_l = args.clip_l,
        t5xxl = args.t5xxl,
        llm = args.llm,
    )
    extra = []
    if args.vae_format:
        extra += ["--vae-format", args.vae_format]
    params = SdCppGenParams(
        prompt = PROMPT, width = args.width, height = args.height, steps = args.steps,
        cfg_scale = args.guidance, seed = args.seed, sampling_method = "euler",
    )

    out = Path(args.out)
    lat = []
    with _RSSSampler() as rss:
        try:
            for _ in range(max(0, args.warmup)):
                engine.generate(files, params, output_path = str(out), threads = int(args.threads),
                                extra_args = extra, timeout = 7200)
            for i in range(max(1, args.iters)):
                t0 = time.time()
                engine.generate(files, params, output_path = str(out), threads = int(args.threads),
                                extra_args = extra, timeout = 7200)
                lat.append(time.time() - t0)
                print(f"  gen[{i}] {lat[-1]:.1f}s", flush = True)
        except Exception as exc:  # noqa: BLE001
            _emit("sdcpp", args.family, status = "blocked", reason = f"{type(exc).__name__}:{str(exc)[:120]}")
            return 1
    _emit(
        "sdcpp", args.family, status = "OK",
        latency_s = f"{_median(lat):.1f}", peak_rss_gb = f"{rss.peak_gb:.1f}",
        steps = args.steps, res = f"{args.width}x{args.height}", image = str(args.out),
    )
    return 0


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices = ["diffusers", "sdcpp"], required = True)
    p.add_argument("--family", required = True)
    p.add_argument("--gguf", required = True, help = "transformer GGUF path (same for both engines)")
    p.add_argument("--base-repo", default = None, help = "diffusers base repo (VAE+TE) for diffusers")
    p.add_argument("--vae", default = None)
    p.add_argument("--clip-l", dest = "clip_l", default = None)
    p.add_argument("--t5xxl", default = None)
    p.add_argument("--llm", default = None)
    p.add_argument("--vae-format", dest = "vae_format", default = None)
    p.add_argument("--sd-cli", dest = "sd_cli", default = None)
    p.add_argument("--width", type = int, default = 512)
    p.add_argument("--height", type = int, default = 512)
    p.add_argument("--steps", type = int, default = 8)
    p.add_argument("--guidance", type = float, default = 1.0)
    p.add_argument("--seed", type = int, default = 12345)
    p.add_argument("--threads", type = int, default = 64)
    p.add_argument("--iters", type = int, default = 1)
    p.add_argument("--warmup", type = int, default = 0)
    p.add_argument("--out", required = True)
    args = p.parse_args(argv)
    rc = run_diffusers(args) if args.engine == "diffusers" else run_sdcpp(args)
    print("SDCPP-CPU-BENCH-DONE", flush = True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
