# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Benchmark external diffusion engines (sglang-diffusion, vllm-omni) on one B200.

Runs each engine for each model in a fresh subprocess (clean VRAM), at 1024x1024
resident bf16, parses the engine's own reported generate/denoise latency, samples
peak VRAM via nvidia-smi on the target GPU, and saves the image. Aggregates to
outputs/ext_engines/results.csv. Our diffusers+opts numbers come from the existing
outputs/threeway run (same GPU/resolution) and are added in the SUMMARY, not here.

FastGen (distillation training) and sglang-omni (audio) are out of scope; Dynamo is
an orchestration layer over sglang-diffusion, not its own engine.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import threading
import time
from pathlib import Path

WS = Path("/mnt/disks/unslothai/ubuntu/workspace_81")
ROOT = WS / "unsloth"
OUT = ROOT / "outputs" / "ext_engines"
GPU = "7"  # physical index (cleanest); from the assigned set 4,5,6,7
PROMPT = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed"

SGLANG = WS / ".venv_sglang" / "bin" / "sglang"
VLLM_PY = WS / ".venv_vllm_omni" / "bin" / "python"
VLLM_EX = WS / "vllm-omni" / "examples" / "offline_inference" / "text_to_image" / "text_to_image.py"

# model -> (repo, steps, sd_cfg(sglang/vllm cfg-scale), guidance)
MODELS = {
    "z-image": dict(repo = "Tongyi-MAI/Z-Image-Turbo", steps = 8, cfg = 1.0, guidance = 0.0),
    "flux.1": dict(repo = "black-forest-labs/FLUX.1-schnell", steps = 4, cfg = 1.0, guidance = 0.0),
    "qwen-image": dict(repo = "Qwen/Qwen-Image", steps = 20, cfg = 4.0, guidance = 1.0),
}

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _strip(s: str) -> str:
    return _ANSI.sub("", s)


def _gpu_used_mib(idx: str) -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", idx],
            capture_output = True, text = True, check = False, timeout = 10,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return 0.0


class _VramSampler(threading.Thread):
    def __init__(self, idx: str):
        super().__init__(daemon = True)
        self.idx = idx
        self.stop = threading.Event()
        self.base = _gpu_used_mib(idx)
        self.peak = 0.0

    def run(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, _gpu_used_mib(self.idx) - self.base)
            time.sleep(0.25)


def _build_cmd(engine: str, model: str, out_img: Path):
    cfg = MODELS[model]
    if engine == "sglang":
        return [
            str(SGLANG), "generate", "--model-path", cfg["repo"],
            "--performance-mode", "speed",  # GPU-resident (no CPU offload)
            "--num-inference-steps", str(cfg["steps"]), "--height", "1024", "--width", "1024",
            "--seed", "1234", "--prompt", PROMPT,
            "--output-file-path", str(out_img), "--save-output",
        ], None
    # vllm-omni
    return [
        str(VLLM_PY), str(VLLM_EX), "--model", cfg["repo"],
        "--num-inference-steps", str(cfg["steps"]), "--height", "1024", "--width", "1024",
        "--seed", "1234", "--prompt", PROMPT,
        "--cfg-scale", str(cfg["cfg"]), "--guidance-scale", str(cfg["guidance"]),
        "--output", str(out_img),
    ], str(VLLM_EX.parent)


def _parse(engine: str, text: str) -> dict:
    t = _strip(text)
    out = {"total_s": "", "denoise_s": "", "per_step_ms": "", "self_peak_mb": ""}
    if engine == "sglang":
        m = re.search(r"Pixel data generated successfully in\s+([\d.]+)", t)
        if m:
            out["total_s"] = float(m.group(1))
        m = re.search(r"\[DenoisingStage\] finished in\s+([\d.]+)", t)
        if m:
            out["denoise_s"] = float(m.group(1))
        m = re.search(r"average time per step:\s*([\d.]+)", t)
        if m:
            out["per_step_ms"] = round(float(m.group(1)) * 1000.0, 1)
    else:
        m = re.search(r"Total generation time:\s*([\d.]+)\s*seconds", t)
        if m:
            out["total_s"] = float(m.group(1))
        m = re.search(r"denoise_step_latency_ms'?:\s*([\d.]+)", t)
        if m:
            out["per_step_ms"] = round(float(m.group(1)), 1)
        m = re.search(r"peak_memory_mb[=':\s]+([\d.]+)", t)
        if m:
            out["self_peak_mb"] = float(m.group(1))
    return out


def run_cell(engine: str, model: str, timeout: float) -> dict:
    OUT.mkdir(parents = True, exist_ok = True)
    out_img = OUT / f"{engine}_{model.replace('.', '_')}.png"
    cmd, cwd = _build_cmd(engine, model, out_img)
    import os

    env = dict(os.environ, CUDA_VISIBLE_DEVICES = GPU)
    sampler = _VramSampler(GPU)
    sampler.start()
    t0 = time.time()
    status, reason = "OK", ""
    try:
        proc = subprocess.run(
            cmd, env = env, cwd = cwd, capture_output = True, text = True,
            errors = "replace", timeout = timeout, check = False,
        )
        wall = time.time() - t0
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        (OUT / f"{engine}_{model.replace('.', '_')}.log").write_text(text)
        if proc.returncode != 0 or not out_img.is_file():
            status = "blocked"
            tail = _strip(text).strip().splitlines()[-4:]
            reason = " | ".join(tail)[:300]
        parsed = _parse(engine, text)
    except subprocess.TimeoutExpired:
        wall = time.time() - t0
        status, reason, parsed = "timeout", f">{timeout}s", _parse(engine, "")
    finally:
        sampler.stop.set()
        sampler.join(timeout = 2)
    row = dict(
        engine = engine, model = model, steps = MODELS[model]["steps"],
        wall_s = round(wall, 2), status = status,
        nvsmi_peak_gb = round(sampler.peak / 1024.0, 2),
        **parsed, reason = reason,
    )
    print("RESULT", row, flush = True)
    return row


def main(argv = None):
    p = argparse.ArgumentParser()
    p.add_argument("--engines", default = "vllm,sglang")
    p.add_argument("--models", default = "z-image,flux.1,qwen-image")
    p.add_argument("--timeout", type = float, default = 2400.0)
    args = p.parse_args(argv)

    rows = []
    for model in args.models.split(","):
        for engine in args.engines.split(","):
            print(f"\n===== {engine} / {model} =====", flush = True)
            rows.append(run_cell(engine.strip(), model.strip(), args.timeout))

    OUT.mkdir(parents = True, exist_ok = True)
    csv_path = OUT / "results.csv"
    cols = ["engine", "model", "steps", "wall_s", "total_s", "denoise_s", "per_step_ms",
            "nvsmi_peak_gb", "self_peak_mb", "status", "reason"]
    with open(csv_path, "w", newline = "") as f:
        w = csv.DictWriter(f, fieldnames = cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\nwrote {csv_path}", flush = True)
    print("EXT-ENGINE-BENCH-DONE", flush = True)


if __name__ == "__main__":
    main()
