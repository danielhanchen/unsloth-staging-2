# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run the CPU diffusers-vs-sd.cpp benchmark for every supported model and aggregate.

Per family, runs sdcpp_cpu_bench.py for BOTH engines (fresh subprocess each, CPU forced via
CUDA_VISIBLE_DEVICES=""), on the SAME Q4_K_M transformer GGUF, with matched threads / resolution /
steps. CFG is kept off on both engines (1 forward/step) for tractability unless a family needs it.
Sequential (CPU-bound; parallel runs would contend and skew latency/RSS). -> outputs/sdcpp_cpu/results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/disks/unslothai/ubuntu/workspace_81/unsloth")
OUT = ROOT / "outputs" / "sdcpp_cpu"
GGUF = OUT / "gguf"
ASSETS = OUT / "assets"
IMAGES = OUT / "images"
BENCH = ROOT / "scripts" / "sdcpp_cpu_bench.py"
SD_CLI = "/home/ubuntu/.unsloth/stable-diffusion.cpp/sd-cli"

# family -> config. diff_g = diffusers guidance (0 = CFG off for distilled; for qwen true_cfg);
# sd_cfg = sd-cli --cfg-scale (1.0 = CFG off). Both kept CFG-off here (1 forward/step) for tractability.
MANIFEST = [
    dict(family = "z-image", gguf = "z-image-turbo-Q4_K_M.gguf", base = "Tongyi-MAI/Z-Image-Turbo",
         vae = "zimage_vae.safetensors", llm = "qwen_3_4b.safetensors",
         steps = 8, diff_g = 0.0, sd_cfg = 1.0, covers = "Z-Image-Turbo, Z-Image"),
    dict(family = "flux.2-klein", gguf = "flux-2-klein-4b-Q4_K_M.gguf", base = "black-forest-labs/FLUX.2-klein-4B",
         vae = "flux2_ae.safetensors", llm = "qwen_3_4b.safetensors", vae_format = "flux2",
         steps = 4, diff_g = 0.0, sd_cfg = 1.0, covers = "FLUX.2-klein-4B"),
    dict(family = "flux.1", gguf = "flux1-schnell-Q4_K_M.gguf", base = "black-forest-labs/FLUX.1-schnell",
         vae = "flux1_ae.safetensors", clip_l = "clip_l.safetensors", t5xxl = "t5xxl_fp16.safetensors",
         steps = 4, diff_g = 0.0, sd_cfg = 1.0, covers = "FLUX.1-schnell, FLUX.1-dev"),
    dict(family = "qwen-image", gguf = "qwen-image-Q4_K_M.gguf", base = "Qwen/Qwen-Image",
         vae = "qwen_image_vae.safetensors", llm = "qwen2.5vl_Q4_K_M.gguf",
         steps = 8, diff_g = 1.0, sd_cfg = 1.0, covers = "Qwen-Image, Qwen-Image-2512"),
]


def _run(engine: str, cfg: dict, *, width: int, height: int, threads: int, logf) -> dict:
    gguf = str(GGUF / cfg["gguf"])
    out_img = str(IMAGES / f"{cfg['family'].replace('.', '_')}_{engine}.png")
    guidance = cfg["diff_g"] if engine == "diffusers" else cfg["sd_cfg"]
    cmd = [
        sys.executable, "-u", str(BENCH),
        "--engine", engine, "--family", cfg["family"], "--gguf", gguf,
        "--width", str(width), "--height", str(height), "--steps", str(cfg["steps"]),
        "--guidance", str(guidance), "--threads", str(threads), "--out", out_img,
    ]
    if engine == "diffusers":
        cmd += ["--base-repo", cfg["base"]]
    else:
        cmd += ["--sd-cli", SD_CLI]
        for flag, key in (("--vae", "vae"), ("--clip-l", "clip_l"), ("--t5xxl", "t5xxl"), ("--llm", "llm")):
            if cfg.get(key):
                cmd += [flag, str(ASSETS / cfg[key])]
        if cfg.get("vae_format"):
            cmd += ["--vae-format", cfg["vae_format"]]

    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", OMP_NUM_THREADS = str(threads), MKL_NUM_THREADS = str(threads))
    logf.write(f"\n=== {engine} {cfg['family']} :: {' '.join(cmd)}\n")
    logf.flush()
    res = {}
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, env = env, text = True)
    for line in proc.stdout:
        logf.write(line)
        logf.flush()
        if line.startswith("RESULT "):
            for tok in line.split()[1:]:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    res[k] = v
    proc.wait()
    res["_rc"] = proc.returncode
    res["_wall_s"] = round(time.time() - t0, 1)
    return res


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default = None, help = "comma-separated family names")
    p.add_argument("--width", type = int, default = 512)
    p.add_argument("--height", type = int, default = 512)
    p.add_argument("--threads", type = int, default = 64)
    args = p.parse_args(argv)
    only = set(args.only.split(",")) if args.only else None

    IMAGES.mkdir(parents = True, exist_ok = True)
    logdir = OUT / "logs"
    logdir.mkdir(parents = True, exist_ok = True)
    rows = []
    for cfg in MANIFEST:
        if only and cfg["family"] not in only:
            continue
        log = open(logdir / f"{cfg['family'].replace('.', '_')}.log", "a")
        row = {"family": cfg["family"], "covers": cfg["covers"], "steps": cfg["steps"],
               "res": f"{args.width}x{args.height}", "threads": args.threads}
        for engine in ("diffusers", "sdcpp"):
            print(f"\n>>> {engine} {cfg['family']} ...", flush = True)
            r = _run(engine, cfg, width = args.width, height = args.height, threads = args.threads, logf = log)
            row[f"{engine}_status"] = r.get("status", f"rc{r.get('_rc')}")
            row[f"{engine}_latency_s"] = r.get("latency_s", "")
            row[f"{engine}_peak_rss_gb"] = r.get("peak_rss_gb", "")
            row[f"{engine}_reason"] = r.get("reason", "")
            print(f"    -> {engine}: {row[f'{engine}_status']} "
                  f"lat={row[f'{engine}_latency_s']}s rss={row[f'{engine}_peak_rss_gb']}GB", flush = True)
        # ratios (diffusers / sdcpp) when both numeric
        try:
            dl, sl = float(row["diffusers_latency_s"]), float(row["sdcpp_latency_s"])
            row["speedup_sdcpp_x"] = f"{dl / sl:.2f}" if sl > 0 else ""
        except ValueError:
            row["speedup_sdcpp_x"] = ""
        try:
            dr, sr = float(row["diffusers_peak_rss_gb"]), float(row["sdcpp_peak_rss_gb"])
            row["rss_ratio_x"] = f"{dr / sr:.2f}" if sr > 0 else ""
        except ValueError:
            row["rss_ratio_x"] = ""
        rows.append(row)
        log.close()

    csv_path = OUT / "results.csv"
    fields = ["family", "covers", "steps", "res", "threads",
              "diffusers_latency_s", "sdcpp_latency_s", "speedup_sdcpp_x",
              "diffusers_peak_rss_gb", "sdcpp_peak_rss_gb", "rss_ratio_x",
              "diffusers_status", "sdcpp_status", "diffusers_reason", "sdcpp_reason"]
    with open(csv_path, "w", newline = "") as f:
        w = csv.DictWriter(f, fieldnames = fields, extrasaction = "ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {csv_path} ({len(rows)} rows)", flush = True)
    for r in rows:
        print(f"  {r['family']:14s} diffusers {r.get('diffusers_latency_s','?'):>7}s/"
              f"{r.get('diffusers_peak_rss_gb','?'):>5}GB  vs  sdcpp {r.get('sdcpp_latency_s','?'):>7}s/"
              f"{r.get('sdcpp_peak_rss_gb','?'):>5}GB  (speed x{r.get('speedup_sdcpp_x','?')}, rss x{r.get('rss_ratio_x','?')})",
              flush = True)
    print("SDCPP-CPU-ORCH-DONE", flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
