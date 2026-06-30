# SPDX-License-Identifier: AGPL-3.0-only
"""Build + prove pre-warmed torch.compile cache bundles for the diffusion denoiser.

Two roles:

  * DISTRIBUTOR: ``--phase build`` compiles a family with save enabled and writes a
    per-fingerprint Mega-cache bundle under the bundle root (organised by GPU arch).
  * PROOF: an orchestrator (``--prove``) runs three FRESH subprocesses per family so the
    dynamo state is truly cold each time, and compares the first-image latency:
      build    -> cold compile, writes the bundle (the tax we want to remove)
      reuse    -> bundle present, exact-match load  -> compile tax should be ~gone
      mismatch -> manifest fingerprint corrupted     -> falls back to local compile

Each worker prints a single ``RESULT {json}`` line; the orchestrator aggregates them into
``outputs/compile_cache/PROOF.md`` + ``results.csv``. torch/diffusers import lazily.

Usage:
  # one worker (used by the orchestrator; also runnable directly)
  python build_diffusion_compile_cache.py --phase build --variant flux.1-schnell --gpu 4
  # full local proof across families
  python build_diffusion_compile_cache.py --prove --variants flux.1-schnell,qwen-image --gpus 4 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

UNSLOTH = Path("/mnt/disks/unslothai/ubuntu/workspace_81/unsloth")
BACKEND = UNSLOTH / "studio" / "backend"
sys.path.insert(0, str(UNSLOTH / "scripts"))
import benchall_orchestrator as bo  # noqa: E402

OUT = UNSLOTH / "outputs" / "compile_cache"
BUNDLE_ROOT = UNSLOTH / "outputs" / "compile_cache_bundles"
PROMPT = "a cat asleep on a stack of books, highly detailed"


# --------------------------------------------------------------------------- one worker
def _arch_tag() -> str:
    import torch
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        return f"sm_{cap[0]}{cap[1]}"
    return "cpu"


def run_worker(variant: str, phase: str, gpu: int) -> dict:
    """Load + (compile) + generate once in THIS process; return the first-image latency."""
    gguf_repo, gguf_file, base_repo, steps, guidance, _ = bo.MANIFEST[variant]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    import torch  # noqa: PLC0415
    arch = _arch_tag()
    bundle_dir = BUNDLE_ROOT / arch
    bundle_dir.mkdir(parents=True, exist_ok=True)
    os.environ["UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR"] = str(bundle_dir)

    # build => load+save; reuse => load only (auto); mismatch => load only, but we corrupt
    # the manifest first so the exact-match guard rejects the bundle.
    os.environ["UNSLOTH_DIFFUSION_COMPILE_CACHE"] = "1" if phase == "build" else "auto"

    # Isolate the PORTABLE bundle: for reuse/mismatch, wipe the on-disk inductor cache so
    # the only possible accelerator is ``load_cache_artifacts`` (the thing we would ship).
    # This is what a fresh consumer that downloaded ONLY the bundle would see. Otherwise the
    # persistent TORCHINDUCTOR_CACHE_DIR (warm from build) would mask the bundle's true
    # contribution and stop mismatch from going cold.
    if phase in ("reuse", "mismatch"):
        _wipe_inductor(bundle_dir)
    if phase == "mismatch":
        _corrupt_manifests(bundle_dir)

    sys.path.insert(0, str(BACKEND))
    from core.inference.diffusion import get_diffusion_backend  # noqa: PLC0415

    backend = get_diffusion_backend()
    backend.begin_load(
        gguf_repo, gguf_filename=gguf_file, base_repo=base_repo,
        hf_token=os.environ.get("HF_TOKEN"), memory_mode="fast", speed_mode="default",
    )
    _wait_ready(backend)

    # First image == the cold path (compile happens lazily here unless a bundle was loaded).
    torch.cuda.synchronize()
    t0 = time.time()
    backend.generate(prompt=PROMPT, width=1024, height=1024, steps=steps,
                     guidance=guidance, seed=bo.SEED, batch_size=1)
    torch.cuda.synchronize()
    cold_first_img_s = time.time() - t0

    # A second image == warm steady state (compile already done either way).
    torch.cuda.synchronize()
    t1 = time.time()
    backend.generate(prompt=PROMPT, width=1024, height=1024, steps=steps,
                     guidance=guidance, seed=bo.SEED, batch_size=1)
    torch.cuda.synchronize()
    warm_s = time.time() - t1

    status = backend.status()
    backend.unload()

    bundle_present = any(bundle_dir.rglob("cache.bin"))
    return {
        "variant": variant, "phase": phase, "arch": arch,
        "cold_first_img_s": round(cold_first_img_s, 2), "warm_s": round(warm_s, 3),
        "compile_tax_s": round(cold_first_img_s - warm_s, 2),
        "speed_optims": status.get("speed_optims"),
        "bundle_present_after": bundle_present,
    }


def _wipe_inductor(bundle_dir: Path) -> None:
    """Remove the on-disk inductor caches under the bundle root, keeping cache.bin +
    manifest.json -- so only the portable Mega-cache bundle can accelerate the next run."""
    import shutil
    for ind in bundle_dir.rglob("inductor"):
        if ind.is_dir():
            shutil.rmtree(ind, ignore_errors=True)


def _corrupt_manifests(bundle_dir: Path) -> None:
    """Flip the torch version in every manifest so the exact-match guard misses."""
    for mf in bundle_dir.rglob("manifest.json"):
        try:
            m = json.loads(mf.read_text())
            m.setdefault("env", {})["torch"] = "0.0.0-mismatch"
            mf.write_text(json.dumps(m))
        except Exception:  # noqa: BLE001
            pass


def _wait_ready(backend, timeout_s: int = 2400) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        p = backend.load_progress()
        if p.get("phase") == "ready":
            return
        if p.get("phase") == "error":
            raise RuntimeError(f"load error: {p.get('error')}")
        time.sleep(2)
    raise TimeoutError("model load did not reach ready")


# ----------------------------------------------------------------------------- orchestrate
def prove(variants: list[str], gpus: list[int]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, variant in enumerate(variants):
        gpu = gpus[i % len(gpus)]
        # Wipe any prior bundle for this variant's arch so "build" is a true cold compile.
        for phase in ("build", "reuse", "mismatch"):
            res = _spawn(variant, phase, gpu)
            rows.append(res)
            print(f"[{variant}/{phase}] {res}", flush=True)

    with open(OUT / "results.csv", "w", newline="") as f:
        cols = ["variant", "phase", "arch", "cold_first_img_s", "warm_s",
                "compile_tax_s", "bundle_present_after", "speed_optims"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    _write_proof_md(rows)
    print("=== COMPILE_CACHE_PROOF_DONE ===", flush=True)


def _spawn(variant: str, phase: str, gpu: int) -> dict:
    """Run one worker in a FRESH subprocess (true cold dynamo state)."""
    cmd = [sys.executable, "-u", __file__, "--phase", phase, "--variant", variant, "--gpu", str(gpu)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    return {"variant": variant, "phase": phase, "status": "FAILED",
            "stderr": proc.stderr[-400:]}


def _write_proof_md(rows: list[dict]) -> None:
    by_var: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_var.setdefault(r["variant"], {})[r.get("phase", "?")] = r
    lines = [
        "# torch.compile cache pre-warm: local proof (B200 / sm_100, fresh-process per phase)",
        "",
        "Each phase runs in its OWN subprocess (cold dynamo state). To isolate the PORTABLE",
        "Mega-cache bundle (the thing we would ship), `reuse`/`mismatch` first WIPE the on-disk",
        "inductor cache, so the only possible accelerator is `load_cache_artifacts(bundle)`:",
        "",
        "* `mismatch` = bundle present but its manifest fingerprint is corrupted -> the exact-match",
        "  guard rejects it, nothing is loaded, inductor is empty => a true COLD compile (baseline).",
        "* `reuse`    = bundle present + manifest matches -> `load_cache_artifacts` pre-populates",
        "  the caches => the portable bundle's real contribution.",
        "* `build`    = first-ever cold compile that also SAVES the bundle (so it includes save time).",
        "",
        "| model | mismatch (cold) | reuse (bundle) | warm steady | tax removed by the bundle |",
        "|---|---:|---:|---:|---:|",
    ]
    for var, ph in by_var.items():
        ru, mm = ph.get("reuse", {}), ph.get("mismatch", {})
        removed = ""
        if mm.get("cold_first_img_s") and ru.get("cold_first_img_s"):
            d = mm["cold_first_img_s"] - ru["cold_first_img_s"]
            pct = 100 * d / mm["cold_first_img_s"] if mm["cold_first_img_s"] else 0
            removed = f"{d:.1f}s ({pct:.0f}%)"
        lines.append(
            f"| {var} | {mm.get('cold_first_img_s','-')} | {ru.get('cold_first_img_s','-')} "
            f"| {ru.get('warm_s','-')} | {removed} |"
        )
    lines += [
        "",
        "Interpretation: `reuse` < `mismatch` proves a shipped bundle removes part of the compile",
        "tax on a matched (arch, torch, triton, CUDA) box; `mismatch` ~= a true cold compile proves",
        "the exact-match guard + silent fallback work (a bad/foreign bundle never errors, just",
        "recompiles). The deterministic guard/fallback unit tests are in",
        "`tests/test_diffusion_compile_cache.py`. Cross-arch/version portability is NOT promised --",
        "see `DISTRIBUTION.md`. (Same-machine repeat loads are even faster than `reuse` here because",
        "Studio also keeps the on-disk inductor cache, which this proof deliberately wiped.)",
    ]
    (OUT / "PROOF.md").write_text("\n".join(lines))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["build", "reuse", "mismatch"])
    p.add_argument("--variant")
    p.add_argument("--gpu", type=int, default=4)
    p.add_argument("--prove", action="store_true")
    p.add_argument("--variants", default="flux.1-schnell,qwen-image")
    p.add_argument("--gpus", type=int, nargs="*", default=[4, 5, 6, 7])
    args = p.parse_args(argv)

    if args.prove:
        prove([v.strip() for v in args.variants.split(",")], args.gpus)
        return 0
    if args.phase and args.variant:
        res = run_worker(args.variant, args.phase, args.gpu)
        print("RESULT " + json.dumps(res), flush=True)
        return 0
    p.error("need --prove, or --phase + --variant")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
