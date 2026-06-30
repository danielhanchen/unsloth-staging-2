# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Orchestrate the comprehensive diffusion benchmark: all 8 supported model variants x the GGUF
and dense (safetensors-quantised) paths x the shipped optimization levers, on one GPU per worker.

Each config runs `scripts/diffusion_bench.py --write-baseline <dir>/metrics.json --out-dir <dir>`,
which records load + generation metrics AND saves the generated image as <dir>/reference.png. The
per-model `G0` config (GGUF, speed=off) is the reference; the aggregator computes speedup and PSNR
of every other config of that model against its G0. Runs are resumable (a config whose metrics.json
already exists is skipped).

Usage:
  python benchall_orchestrator.py --gpu 5 --variants z-image-turbo flux.1-schnell   # one worker
  python benchall_orchestrator.py --list                                            # print plan
  python benchall_orchestrator.py --aggregate                                        # build CSV+summary
  python benchall_orchestrator.py --dry-run --gpu 5 --variants z-image-turbo         # reduced smoke
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "scripts" / "diffusion_bench.py"
OUT = ROOT / "outputs" / "diffusion_benchall"
SEED = 12345

# variant -> (gguf_repo, gguf_file, base_repo, steps, guidance, many_step)
MANIFEST = {
    "z-image-turbo":   ("unsloth/Z-Image-Turbo-GGUF",   "z-image-turbo-Q4_K_M.gguf",  "Tongyi-MAI/Z-Image-Turbo",        8,  0.0, False),
    "z-image":         ("unsloth/Z-Image-GGUF",         "z-image-Q4_K_M.gguf",        "Tongyi-MAI/Z-Image",              20, 4.0, True),
    "qwen-image-2512": ("unsloth/Qwen-Image-2512-GGUF", "qwen-image-2512-Q4_K_M.gguf","Qwen/Qwen-Image-2512",            20, 4.0, True),
    "qwen-image":      ("unsloth/Qwen-Image-GGUF",      "qwen-image-Q4_K_M.gguf",     "Qwen/Qwen-Image",                 20, 4.0, True),
    "flux.1-schnell":  ("unsloth/FLUX.1-schnell-GGUF",  "flux1-schnell-Q4_K_M.gguf",  "black-forest-labs/FLUX.1-schnell", 4,  0.0, False),
    "flux.1-dev":      ("unsloth/FLUX.1-dev-GGUF",      "flux1-dev-Q4_K_M.gguf",      "black-forest-labs/FLUX.1-dev",    28, 3.5, True),
    "flux.2-klein-4b": ("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_K_M.gguf","black-forest-labs/FLUX.2-klein-4B", 4, 0.0, False),
    "flux.2-klein-9b": ("unsloth/FLUX.2-klein-9B-GGUF", "flux-2-klein-9b-Q4_K_M.gguf","black-forest-labs/FLUX.2-klein-9B", 4, 0.0, False),
}

# Targeted single-model ablations (kept off the full cross-product). variant -> list of (cfg, flags)
ABLATIONS = {
    "z-image-turbo": [("G1_max", ["--speed-mode", "max"])],
    "flux.1-dev": [
        ("G1_max", ["--speed-mode", "max"]),
        ("D_nvfp4", ["--speed-mode", "default", "--transformer-quant", "nvfp4"]),
        ("D_mxfp8", ["--speed-mode", "default", "--transformer-quant", "mxfp8"]),
        ("D_fp8_fastaccum_on", ["--speed-mode", "default", "--transformer-quant", "fp8", "--fp8-fast-accum", "on"]),
        ("D_fp8_fastaccum_off", ["--speed-mode", "default", "--transformer-quant", "fp8", "--fp8-fast-accum", "off"]),
        ("G1_teq_fp8", ["--speed-mode", "default", "--text-encoder-quant", "fp8"]),
    ],
    "qwen-image": [("G1_teq_fp8", ["--speed-mode", "default", "--text-encoder-quant", "fp8"])],
}


def configs_for(variant: str) -> list[tuple[str, list[str]]]:
    """The ordered config list for a variant. G0 first (it is the PSNR reference)."""
    many = MANIFEST[variant][5]
    cfgs: list[tuple[str, list[str]]] = [
        ("G0_eager",       ["--speed-mode", "off"]),                                  # reference
        ("G1_default",     ["--speed-mode", "default"]),                              # lossless speed stack
        ("D1_fp8",         ["--speed-mode", "default", "--transformer-quant", "fp8"]),
        ("D2_int8",        ["--speed-mode", "default", "--transformer-quant", "int8"]),
    ]
    if many:
        cfgs.append(("G2_fbcache", ["--speed-mode", "default", "--transformer-cache", "fbcache"]))
    cfgs.extend(ABLATIONS.get(variant, []))
    return cfgs


def run_config(variant: str, cfg: str, flags: list[str], gpu: int, *, dry: bool) -> dict:
    gguf_repo, gguf_file, base_repo, steps, guidance, _ = MANIFEST[variant]
    d = (OUT / "_dryrun" / variant / cfg) if dry else (OUT / variant / cfg)
    d.mkdir(parents=True, exist_ok=True)
    metrics = d / "metrics.json"
    if metrics.exists() and not dry:
        return {"variant": variant, "cfg": cfg, "status": "skipped-exists"}
    steps_eff = 4 if dry else steps
    iters = 1 if dry else 3
    cmd = [
        sys.executable, "-u", str(BENCH),
        "--model", gguf_repo, "--gguf", gguf_file, "--base-repo", base_repo,
        "--memory-mode", "fast", "--steps", str(steps_eff), "--guidance", str(guidance),
        "--seed", str(SEED), "--iters", str(iters), "--warmup", "1",
        "--write-baseline", str(metrics), "--out-dir", str(d),
        *flags,
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    log = d / "run.log"
    t0 = time.time()
    with open(log, "w") as lf:
        proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    ok = proc.returncode == 0 and metrics.exists()
    rec = {"variant": variant, "cfg": cfg, "gpu": gpu, "rc": proc.returncode,
           "status": "ok" if ok else "FAILED", "wall_s": round(time.time() - t0, 1)}
    print(f"[gpu{gpu}] {variant}/{cfg}: {rec['status']} ({rec['wall_s']}s)", flush=True)
    return rec


def worker(gpu: int, variants: list[str], dry: bool) -> None:
    for v in variants:
        for cfg, flags in configs_for(v):
            run_config(v, cfg, flags, gpu, dry=dry)


def _psnr(a_png: Path, b_png: Path) -> float:
    import numpy as np
    from PIL import Image
    a = np.array(Image.open(a_png).convert("RGB")).astype(np.float64)
    b = np.array(Image.open(b_png).convert("RGB")).astype(np.float64)
    if a.shape != b.shape:
        return 0.0
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * float(np.log10(255.0)) - 10.0 * float(np.log10(mse))


def aggregate() -> None:
    import csv
    rows = []
    for variant in MANIFEST:
        ref_dir = OUT / variant / "G0_eager"
        ref_json = ref_dir / "metrics.json"
        if not ref_json.exists():
            continue
        ref = json.loads(ref_json.read_text())
        ref_lat = ref["generate"]["median_latency_s"]
        ref_png = ref_dir / "reference.png"
        for cfg, _ in configs_for(variant):
            mj = OUT / variant / cfg / "metrics.json"
            if not mj.exists():
                rows.append({"variant": variant, "cfg": cfg, "status": "MISSING"})
                continue
            m = json.loads(mj.read_text())
            g, ld = m["generate"], m["load"]
            png = OUT / variant / cfg / "reference.png"
            psnr = _psnr(ref_png, png) if (png.exists() and ref_png.exists()) else None
            lat = g["median_latency_s"]
            rows.append({
                "variant": variant, "cfg": cfg, "status": "ok",
                "median_latency_s": round(lat, 3),
                "speedup_vs_G0": round(ref_lat / lat, 2) if lat else None,
                "p90_latency_s": round(g.get("p90_latency_s", 0), 3),
                "gen_peak_GB": round(g["peak_vram_bytes"] / 1e9, 2),
                "load_peak_GB": round(ld["peak_vram_bytes"] / 1e9, 2),
                "load_wall_s": round(ld["wall_seconds"], 1),
                "psnr_vs_G0_dB": ("inf" if psnr == float("inf") else (round(psnr, 2) if psnr is not None else None)),
            })
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["variant", "cfg", "status", "median_latency_s", "speedup_vs_G0", "p90_latency_s",
            "gen_peak_GB", "load_peak_GB", "load_wall_s", "psnr_vs_G0_dB"]
    csv_path = OUT / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {csv_path} ({len(rows)} rows)")
    # console table
    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['variant']:16s} {r['cfg']:20s} {r['status']}")
            continue
        print(f"  {r['variant']:16s} {r['cfg']:20s} {r['median_latency_s']:>7}s "
              f"{str(r['speedup_vs_G0'])+'x':>7} gen {r['gen_peak_GB']:>6}G load {r['load_peak_GB']:>6}G "
              f"PSNR {r['psnr_vs_G0_dB']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int)
    p.add_argument("--variants", nargs="*")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    args = p.parse_args(argv)
    if args.list:
        total = 0
        for v in MANIFEST:
            cs = [c for c, _ in configs_for(v)]
            total += len(cs)
            print(f"{v:16s} ({len(cs)}): {', '.join(cs)}")
        print(f"TOTAL configs: {total}")
        return 0
    if args.aggregate:
        aggregate()
        return 0
    if args.gpu is None or not args.variants:
        p.error("need --gpu and --variants (or --list/--aggregate)")
    worker(args.gpu, args.variants, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
