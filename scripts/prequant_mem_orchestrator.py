# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Drive prequant_mem_bench.py across the supported diffusion transformers and {int8, fp8}.

For each (family, scheme) job: run BEFORE-BUILD then AFTER (AFTER needs the checkpoint the first
step writes) on a single dedicated GPU, in fresh subprocesses. Jobs are fanned across the physical
GPUs in --gpus by a worker-per-GPU pool. Parses the RESULT lines into outputs/prequant_mem/results.csv.

  python scripts/prequant_mem_orchestrator.py --gpus 4,5,6,7
  python scripts/prequant_mem_orchestrator.py --gpus 4 --only flux.2-klein-4B   # dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/mnt/disks/unslothai/ubuntu/workspace_81/unsloth")
OUT = ROOT / "outputs" / "prequant_mem"
DENSE = OUT / "dense"
CKPTS = OUT / "ckpts"
BENCH = ROOT / "scripts" / "prequant_mem_bench.py"

# (family, base repo, local dense dir name, covered variants). Heaviest first so the long pole
# (qwen 40 GB) starts immediately on the first free GPU.
MANIFEST = [
    ("qwen-image",     "Qwen/Qwen-Image",                  "qwen-image",     "Qwen-Image, Qwen-Image-2512"),
    ("flux.1",         "black-forest-labs/FLUX.1-schnell", "flux.1",         "FLUX.1-schnell, FLUX.1-dev"),
    ("flux.2-klein",   "black-forest-labs/FLUX.2-klein-4B","flux.2-klein-4b","FLUX.2-klein-4B"),
    ("z-image",        "Tongyi-MAI/Z-Image",               "z-image",        "Z-Image-Turbo, Z-Image"),
]
SCHEMES = ["fp8", "int8"]


def _parse_result(line: str) -> dict:
    out = {}
    for tok in line.split()[1:]:  # drop "RESULT"
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def _run(mode: str, family: str, base: str, scheme: str, ckpt: Path, gpu: int, logf) -> dict:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = str(gpu))
    cmd = [
        sys.executable, "-u", str(BENCH),
        "--mode", mode, "--base", base, "--family", family, "--scheme", scheme, "--ckpt", str(ckpt),
    ]
    logf.write(f"\n=== [gpu {gpu}] {mode} {family}/{scheme} :: {' '.join(cmd)}\n")
    logf.flush()
    res = {}
    proc = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, env = env, text = True)
    for line in proc.stdout:
        logf.write(line)
        logf.flush()
        if line.startswith("RESULT "):
            res = _parse_result(line.strip())
    proc.wait()
    res["_rc"] = proc.returncode
    return res


def worker(gpu: int, jobs: "queue.Queue", rows: list, lock: threading.Lock, logdir: Path):
    log = open(logdir / f"gpu{gpu}.log", "a")
    while True:
        try:
            family, base, dirname, covers, scheme = jobs.get_nowait()
        except queue.Empty:
            break
        try:
            local = DENSE / dirname
            base_arg = str(local) if (local / "transformer").is_dir() else base
            ckpt = CKPTS / f"{dirname}_{scheme}.pt"
            bb = _run("before-build", family, base_arg, scheme, ckpt, gpu, log)
            af = _run("after", family, base_arg, scheme, ckpt, gpu, log) if bb.get("status") == "OK" else {}
            row = {
                "family": family, "base": base, "covers": covers, "scheme": scheme, "gpu": gpu,
                "before_status": bb.get("status", "?"),
                "before_peak_gb": bb.get("before_peak_gb", ""),
                "ckpt_disk_gb": bb.get("ckpt_disk_gb", ""),
                "before_reason": bb.get("reason", ""),
                "after_status": af.get("status", "?"),
                "after_peak_gb": af.get("after_peak_gb", ""),
                "after_reason": af.get("reason", ""),
                "n_cuda": af.get("n_cuda", ""), "n_cpu": af.get("n_cpu", ""), "n_meta": af.get("n_meta", ""),
                "marker": af.get("marker", ""),
            }
            try:
                b, a = float(bb.get("before_peak_gb", "nan")), float(af.get("after_peak_gb", "nan"))
                row["reduction_pct"] = f"{100.0 * (b - a) / b:.1f}" if b == b and a == a and b > 0 else ""
            except ValueError:
                row["reduction_pct"] = ""
            with lock:
                rows.append(row)
                log.write(f"ROW {row}\n")
                log.flush()
        except Exception as exc:  # noqa: BLE001
            with lock:
                rows.append({"family": family, "scheme": scheme, "gpu": gpu, "before_status": f"EXC:{exc}"})
        finally:
            jobs.task_done()
    log.close()


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default = "4,5,6,7")
    p.add_argument("--only", default = None, help = "comma-separated family names to restrict to")
    p.add_argument("--schemes", default = ",".join(SCHEMES))
    args = p.parse_args(argv)

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    only = set(args.only.split(",")) if args.only else None
    OUT.mkdir(parents = True, exist_ok = True)
    CKPTS.mkdir(parents = True, exist_ok = True)
    logdir = OUT / "logs"
    logdir.mkdir(exist_ok = True)

    jobs: "queue.Queue" = queue.Queue()
    n = 0
    for family, base, dirname, covers in MANIFEST:
        if only and family not in only and dirname not in only:
            continue
        for scheme in schemes:
            jobs.put((family, base, dirname, covers, scheme))
            n += 1
    print(f"queued {n} (family,scheme) jobs across GPUs {gpus}", flush = True)

    rows: list = []
    lock = threading.Lock()
    t0 = time.time()
    threads = [threading.Thread(target = worker, args = (g, jobs, rows, lock, logdir)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows.sort(key = lambda r: (r.get("family", ""), r.get("scheme", "")))
    csv_path = OUT / "results.csv"
    fields = [
        "family", "covers", "scheme", "gpu", "before_status", "before_peak_gb", "after_status",
        "after_peak_gb", "reduction_pct", "ckpt_disk_gb", "n_cuda", "n_cpu", "n_meta", "marker",
        "before_reason", "after_reason", "base",
    ]
    with open(csv_path, "w", newline = "") as f:
        w = csv.DictWriter(f, fieldnames = fields, extrasaction = "ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {csv_path} ({len(rows)} rows) in {time.time() - t0:.0f}s", flush = True)
    for r in rows:
        print(
            f"  {r.get('family'):14s} {r.get('scheme'):4s} "
            f"before={r.get('before_peak_gb','?'):>6} -> after={r.get('after_peak_gb','?'):>6} GB "
            f"({r.get('reduction_pct','?')}%)  after_status={r.get('after_status','?')}",
            flush = True,
        )
    print("PREQUANT-MEM-ORCH-DONE", flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
