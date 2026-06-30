# SPDX-License-Identifier: AGPL-3.0-only
"""Validate the new eager-fast tier: latency + accuracy vs the bare `off` reference.

For each family runs three speed modes in their own subprocess on a pool of GPUs:
  off    -> bare reference (no opts, no patches)        == the accuracy + latency anchor
  eager  -> channels_last + cudnn + attention + patches == the new tier (NO compile)
  default-> the existing compiled tier                   == confirm no compile regression

Aggregates warm latency + PSNR(vs that family's off image) into
``outputs/eager/results.csv`` + ``SUMMARY.md``. Reuses the shipped MANIFEST + _psnr so it
never drifts from the real benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

UNSLOTH = Path("/mnt/disks/unslothai/ubuntu/workspace_81/unsloth")
sys.path.insert(0, str(UNSLOTH / "scripts"))
import benchall_orchestrator as bo  # noqa: E402

BENCH = bo.BENCH
OUT = UNSLOTH / "outputs" / "eager"
VARIANTS = ["z-image-turbo", "qwen-image", "flux.1-schnell", "flux.2-klein-4b"]
MODES = ["off", "eager", "default"]


def run_one(variant: str, mode: str, gpu: int) -> None:
    gguf_repo, gguf_file, base_repo, steps, guidance, _ = bo.MANIFEST[variant]
    d = OUT / variant / mode
    d.mkdir(parents=True, exist_ok=True)
    if (d / "metrics.json").exists():
        print(f"[gpu{gpu}] {variant}/{mode}: skip (exists)", flush=True)
        return
    cmd = [
        sys.executable, "-u", str(BENCH),
        "--model", gguf_repo, "--gguf", gguf_file, "--base-repo", base_repo,
        "--memory-mode", "fast", "--steps", str(steps), "--guidance", str(guidance),
        "--seed", str(bo.SEED), "--warmup", "1", "--iters", "3",
        "--speed-mode", mode, "--write-baseline", str(d / "metrics.json"),
        "--out-dir", str(d),
    ]
    import os
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    t0 = time.time()
    with open(d / "run.log", "w") as lf:
        rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    print(f"[gpu{gpu}] {variant}/{mode}: {'ok' if rc == 0 else 'FAIL'} ({time.time()-t0:.0f}s)", flush=True)


def worker(gpu: int, q: "queue.Queue") -> None:
    while True:
        try:
            v, m = q.get_nowait()
        except queue.Empty:
            return
        try:
            run_one(v, m, gpu)
        finally:
            q.task_done()


def run(gpus: list[int]) -> None:
    q: "queue.Queue" = queue.Queue()
    for v in VARIANTS:
        for m in MODES:
            q.put((v, m))
    ts = [threading.Thread(target=worker, args=(g, q), daemon=True) for g in gpus]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("=== EAGER_BENCH_DONE ===", flush=True)


def _warm(variant: str, mode: str):
    mj = OUT / variant / mode / "metrics.json"
    if not mj.exists():
        return None
    m = json.loads(mj.read_text())
    lat = m["generate"].get("latencies_s") or []
    if not lat:
        return None
    return {"warm": statistics.median(lat), "optims": m.get("status", {}).get("speed_optims")}


def aggregate() -> None:
    rows = []
    for v in VARIANTS:
        ref_png = OUT / v / "off" / "reference.png"
        base = _warm(v, "off")
        for m in MODES:
            s = _warm(v, m)
            if s is None:
                rows.append({"variant": v, "mode": m, "status": "MISSING"})
                continue
            png = OUT / v / m / "reference.png"
            psnr = bo._psnr(ref_png, png) if (png.exists() and ref_png.exists()) else None
            spd = (base["warm"] / s["warm"]) if base and s["warm"] else None
            rows.append({
                "variant": v, "mode": m, "status": "ok",
                "warm_s": round(s["warm"], 3),
                "speedup_vs_off": round(spd, 2) if spd else None,
                "psnr_vs_off_dB": ("inf" if psnr == float("inf") else (round(psnr, 1) if psnr else None)),
                "optims": ",".join(s["optims"] or []),
            })
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["variant", "mode", "status", "warm_s", "speedup_vs_off", "psnr_vs_off_dB", "optims"]
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    _summary(rows)
    for r in rows:
        print("  " + " ".join(f"{r.get(c,'')}" for c in cols), flush=True)


def _summary(rows) -> None:
    lines = [
        "# Eager-fast tier: latency + accuracy vs the bare `off` reference",
        "",
        "`eager` = channels_last + cudnn + attention backend + the shared eager monkey-patches",
        "(fused RMSNorm / AdaLayerNorm), NO torch.compile. `default` = the existing compiled tier",
        "(shown to confirm it is unchanged). PSNR is vs the same family's `off` image (higher = closer;",
        "inf = identical).",
        "",
        "| model | mode | warm s | speedup vs off | PSNR vs off | optims |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"| {r['variant']} | {r['mode']} | - | - | - | {r.get('status')} |")
            continue
        lines.append(
            f"| {r['variant']} | {r['mode']} | {r['warm_s']} | "
            f"{r.get('speedup_vs_off','-')} | {r.get('psnr_vs_off_dB','-')} | {r.get('optims','')} |"
        )
    (OUT / "SUMMARY.md").write_text("\n".join(lines))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--gpus", type=int, nargs="*", default=[4, 5, 6, 7])
    a = p.parse_args(argv)
    if a.aggregate:
        aggregate()
        return 0
    if a.run:
        run(a.gpus)
        return 0
    p.error("need --run or --aggregate")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
