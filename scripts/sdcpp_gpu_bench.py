# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GPU benchmark of the native sd.cpp (CUDA) engine across the supported models.

Runs the CUDA-built sd-cli on one GPU per model, measuring end-to-end latency (sd-cli's own
"generate_image completed in" + wall time) and PEAK VRAM (nvidia-smi memory.used sampled on the
target GPU, minus baseline). Same Q4_K_M transformer GGUFs + VAE/encoders as the CPU bench.
Output -> outputs/sdcpp_gpu/results.csv, so sd.cpp-GPU sits next to the diffusers+opts benchall.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path("/mnt/disks/unslothai/ubuntu/workspace_81")
SD = ROOT / "stable-diffusion.cpp" / "build" / "bin" / "sd-cli"
G = ROOT / "unsloth" / "outputs" / "sdcpp_cpu" / "gguf"
A = ROOT / "unsloth" / "outputs" / "sdcpp_cpu" / "assets"
OUT = ROOT / "unsloth" / "outputs" / "sdcpp_gpu"
PROMPT = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed"

# family -> config. steps/cfg match the model's natural settings (full steps on GPU).
MANIFEST = [
    dict(family = "z-image", gguf = "z-image-turbo-Q4_K_M.gguf", vae = "zimage_vae.safetensors",
         llm = "qwen_3_4b.safetensors", steps = 8, cfg = 1.0),
    dict(family = "flux.2-klein", gguf = "flux-2-klein-4b-Q4_K_M.gguf", vae = "flux2_ae.safetensors",
         llm = "qwen_3_4b.safetensors", vae_format = "flux2", steps = 4, cfg = 1.0),
    dict(family = "flux.1", gguf = "flux1-schnell-Q4_K_M.gguf", vae = "flux1_ae.safetensors",
         clip_l = "clip_l.safetensors", t5xxl = "t5xxl_fp16.safetensors", steps = 4, cfg = 1.0),
    dict(family = "qwen-image", gguf = "qwen-image-Q4_K_M.gguf", vae = "qwen_image_vae.safetensors",
         llm = "qwen2.5vl_Q4_K_M.gguf", steps = 20, cfg = 1.0),
]


class _VRAMSampler:
    """Peak VRAM (MiB) on a physical GPU via nvidia-smi, minus the baseline at start."""

    def __init__(self, gpu: int, hz: float = 10.0) -> None:
        self.gpu = gpu
        self._interval = 1.0 / hz
        self._peak = 0
        self._base = self._used()
        self._stop = threading.Event()
        self._t = threading.Thread(target = self._loop, daemon = True)

    def _used(self) -> int:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", f"--id={self.gpu}"],
                text = True, timeout = 5,
            )
            return int(out.strip().splitlines()[0])
        except Exception:  # noqa: BLE001
            return 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._peak = max(self._peak, self._used())
            time.sleep(self._interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout = 2.0)
        self._peak = max(self._peak, self._used())

    @property
    def peak_gb(self) -> float:
        return max(0, self._peak) / 1024.0

    @property
    def delta_gb(self) -> float:
        return max(0, self._peak - self._base) / 1024.0


def run_one(cfg: dict, gpu: int, width: int, height: int, logf) -> dict:
    out_img = OUT / "images" / f"{cfg['family'].replace('.', '_')}_gpu.png"
    out_img.parent.mkdir(parents = True, exist_ok = True)
    cmd = [str(SD), "--mode", "img_gen", "--diffusion-model", str(G / cfg["gguf"]),
           "--vae", str(A / cfg["vae"])]
    for flag, key in (("--clip_l", "clip_l"), ("--t5xxl", "t5xxl"), ("--llm", "llm")):
        if cfg.get(key):
            cmd += [flag, str(A / cfg[key])]
    if cfg.get("vae_format"):
        cmd += ["--vae-format", cfg["vae_format"]]
    cmd += ["-p", PROMPT, "--cfg-scale", str(cfg["cfg"]), "--steps", str(cfg["steps"]),
            "-W", str(width), "-H", str(height), "--diffusion-fa", "--output", str(out_img)]
    env = {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "LD_LIBRARY_PATH": f"{SD.parent}",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/usr/local/cuda/bin",
    }
    import os
    env = dict(os.environ, **env)
    logf.write(f"\n=== {cfg['family']} gpu{gpu} :: {' '.join(cmd)}\n")
    logf.flush()
    gen_s = None
    sampling_s = None
    with _VRAMSampler(gpu) as vram:
        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, env = env, text = True)
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            m = re.search(r"generate_image completed in ([\d.]+)s", line)
            if m:
                gen_s = float(m.group(1))
            m2 = re.search(r"sampling completed, taking ([\d.]+)s", line)
            if m2:
                sampling_s = float(m2.group(1))
        proc.wait()
        wall = time.time() - t0
    ok = proc.returncode == 0 and out_img.is_file()
    return {
        "family": cfg["family"], "steps": cfg["steps"], "res": f"{width}x{height}",
        "status": "OK" if ok else f"FAIL_rc{proc.returncode}",
        "gen_s": f"{gen_s:.2f}" if gen_s else "", "sampling_s": f"{sampling_s:.2f}" if sampling_s else "",
        "wall_s": f"{wall:.2f}", "peak_vram_gb": f"{vram.peak_gb:.1f}", "vram_delta_gb": f"{vram.delta_gb:.1f}",
    }


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type = int, default = 4)
    p.add_argument("--width", type = int, default = 1024)
    p.add_argument("--height", type = int, default = 1024)
    p.add_argument("--only", default = None)
    args = p.parse_args(argv)
    only = set(args.only.split(",")) if args.only else None
    OUT.mkdir(parents = True, exist_ok = True)
    log = open(OUT / "sdcpp_gpu.log", "a")
    rows = []
    for cfg in MANIFEST:
        if only and cfg["family"] not in only:
            continue
        print(f">>> {cfg['family']} (gpu {args.gpu}, {args.width}x{args.height}, {cfg['steps']} steps)", flush = True)
        # warmup (load + compile kernels), discarded
        run_one(cfg, args.gpu, args.width, args.height, log)
        r = run_one(cfg, args.gpu, args.width, args.height, log)
        rows.append(r)
        print(f"    -> {r['status']}  gen={r['gen_s']}s (sampling {r['sampling_s']}s)  peakVRAM={r['peak_vram_gb']}GB", flush = True)
    csv_path = OUT / "results.csv"
    with open(csv_path, "w", newline = "") as f:
        w = csv.DictWriter(f, fieldnames = ["family", "steps", "res", "status", "gen_s", "sampling_s", "wall_s", "peak_vram_gb", "vram_delta_gb"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}", flush = True)
    print("SDCPP-GPU-BENCH-DONE", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
