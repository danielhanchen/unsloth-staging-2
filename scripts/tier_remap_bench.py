# SPDX-License-Identifier: AGPL-3.0-only
"""End-to-end validation of the remapped GGUF speed tiers, THROUGH the backend.

Confirms the production wiring (not just a probe monkeypatch):
  * off     -> bit-identical reference (no speed optims).
  * eager   -> global weight buffer + eager patches (no compile).
  * default -> compiled dequant (torch.compile dequantize_gguf_tensor, dynamic=True)
               + global weight buffer (NO regional block compile).
  * max     -> regional block compile (max-autotune) + TF32 + fused-QKV.

For each model + tier we record: the speed_optims the backend actually engaged (so we
verify default=={compiled_dequant,weight_buffer} and max=={compiled,...}), cold
first-image latency, warm median latency, peak VRAM, and PSNR vs the `off` reference
(a corruption sanity check -- cudnn.benchmark makes it nondeterministic, so precise
accuracy lives in the unit tests). Plus a weight-buffer isolation in the eager tier
(UNSLOTH_DIFFUSION_GGUF_WEIGHT_BUFFER 0 vs 1) to size the buffer's own contribution.

  python tier_remap_bench.py --all --gpus 4 5 6 7
  python tier_remap_bench.py --model qwen-image --gpu 4
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

UNSLOTH = Path("/mnt/disks/unslothai/ubuntu/workspace_81/unsloth")
BACKEND = UNSLOTH / "studio" / "backend"
sys.path.insert(0, str(UNSLOTH / "scripts"))
import benchall_orchestrator as bo  # noqa: E402

OUT = UNSLOTH / "outputs" / "tier_remap"
PROMPT = "a cat asleep on a stack of books, highly detailed"
MODELS = ["z-image-turbo", "qwen-image", "flux.1-schnell", "flux.2-klein-4b"]


def _wait_ready(backend, timeout_s=2400):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        p = backend.load_progress()
        if p.get("phase") == "ready":
            return
        if p.get("phase") == "error":
            raise RuntimeError(f"load error: {p.get('error')}")
        time.sleep(2)
    raise TimeoutError("load did not become ready")


def _gen(backend, steps, guidance, out_path=None):
    res = backend.generate(prompt=PROMPT, width=1024, height=1024, steps=steps,
                           guidance=guidance, seed=bo.SEED, batch_size=1)
    if out_path is not None:
        res["images"][0].save(str(out_path))
    return res


def _measure_tier(backend, gguf_repo, gguf_file, base_repo, steps, guidance, speed_mode,
                  ref_png=None, save_png=None, iters=4):
    """Load one tier, return engaged optims + cold/warm latency + VRAM (+PSNR vs ref)."""
    import torch  # noqa: PLC0415

    backend.begin_load(gguf_repo, gguf_filename=gguf_file, base_repo=base_repo,
                       hf_token=os.environ.get("HF_TOKEN"), memory_mode="fast",
                       speed_mode=speed_mode)
    _wait_ready(backend)
    status = backend.status()
    optims = sorted(status.get("speed_optims", []))

    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    _gen(backend, steps, guidance)            # cold: includes any one-time compile
    torch.cuda.synchronize()
    cold_s = time.time() - t0

    lat = []
    for _ in range(iters):
        torch.cuda.synchronize(); t = time.time()
        _gen(backend, steps, guidance)
        torch.cuda.synchronize(); lat.append(time.time() - t)
    warm_s = statistics.median(lat)
    vram = torch.cuda.max_memory_allocated() / 1e9

    psnr = None
    if save_png is not None:
        _gen(backend, steps, guidance, out_path=save_png)
        if ref_png is not None and Path(ref_png).exists() and Path(save_png).exists():
            psnr = bo._psnr(Path(ref_png), Path(save_png))

    backend.unload()
    return {
        "speed_mode": speed_mode, "speed_optims": optims,
        "cold_s": round(cold_s, 2), "warm_s": round(warm_s, 3),
        "compile_tax_s": round(cold_s - warm_s, 1), "peak_vram_GB": round(vram, 1),
        "psnr_vs_off_dB": ("inf" if psnr == float("inf") else (round(psnr, 1) if psnr else None)),
    }


def run_one(model: str, gpu: int) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gguf_repo, gguf_file, base_repo, steps, guidance, _ = bo.MANIFEST[model]
    sys.path.insert(0, str(BACKEND))
    from core.inference.diffusion import get_diffusion_backend  # noqa: PLC0415

    d = OUT / model
    d.mkdir(parents=True, exist_ok=True)
    backend = get_diffusion_backend()
    common = dict(gguf_repo=gguf_repo, gguf_file=gguf_file, base_repo=base_repo,
                  steps=steps, guidance=guidance)
    res = {"model": model, "gpu": gpu, "tiers": {}}

    # off = bit-identical reference (image + latency).
    res["tiers"]["off"] = _measure_tier(backend, **common, speed_mode="off",
                                        save_png=d / "off.png")
    ref = d / "off.png"

    # weight-buffer isolation in the eager tier: OFF vs ON (same tier, only the buffer differs).
    os.environ["UNSLOTH_DIFFUSION_GGUF_WEIGHT_BUFFER"] = "0"
    res["tiers"]["eager_buf_off"] = _measure_tier(backend, **common, speed_mode="eager",
                                                  ref_png=ref, save_png=d / "eager_buf_off.png")
    os.environ.pop("UNSLOTH_DIFFUSION_GGUF_WEIGHT_BUFFER", None)
    res["tiers"]["eager"] = _measure_tier(backend, **common, speed_mode="eager",
                                          ref_png=ref, save_png=d / "eager.png")
    res["tiers"]["default"] = _measure_tier(backend, **common, speed_mode="default",
                                            ref_png=ref, save_png=d / "default.png")
    res["tiers"]["max"] = _measure_tier(backend, **common, speed_mode="max",
                                        ref_png=ref, save_png=d / "max.png")

    OUT.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(res, indent=2))
    return res


def aggregate() -> None:
    rows = [json.loads((OUT / m / "result.json").read_text())
            for m in MODELS if (OUT / m / "result.json").exists()]
    lines = [
        "# Remapped GGUF speed tiers, validated end-to-end through the backend",
        "",
        "default = compile-dequant (`torch.compile(dequantize_gguf_tensor, dynamic=True)`) + global",
        "weight buffer; max = full regional block compile (max-autotune) + TF32 + fused-QKV. eager =",
        "weight buffer + eager patches, no compile. PSNR is vs the `off` reference (a corruption check;",
        "cudnn.benchmark makes it nondeterministic -- precise accuracy is in the unit tests).",
        "",
    ]
    for r in rows:
        t = r["tiers"]
        off = t["off"]["warm_s"]

        def spd(tier):
            return round(off / t[tier]["warm_s"], 2) if t.get(tier) and t[tier]["warm_s"] else None

        lines += [
            f"## {r['model']} (gpu{r['gpu']}, off warm = {off}s)",
            "",
            "| tier | engaged speed_optims | warm (s) | speedup vs off | cold (s) | compile tax (s) | VRAM (GB) | PSNR vs off |",
            "|---|---|--:|--:|--:|--:|--:|--:|",
        ]
        for tier in ("off", "eager_buf_off", "eager", "default", "max"):
            x = t.get(tier)
            if not x:
                continue
            lines.append(
                f"| {tier} | {', '.join(x['speed_optims']) or '-'} | {x['warm_s']} | "
                f"{spd(tier) or '-'} | {x['cold_s']} | {x['compile_tax_s']} | "
                f"{x['peak_vram_GB']} | {x['psnr_vs_off_dB']} |"
            )
        # weight-buffer contribution = eager(on) vs eager(off).
        eb_off, eb_on = t.get("eager_buf_off"), t.get("eager")
        if eb_off and eb_on and eb_off["warm_s"] and eb_on["warm_s"]:
            buf_pct = round((eb_off["warm_s"] / eb_on["warm_s"] - 1) * 100, 1)
            lines += ["", f"weight-buffer contribution (eager on vs off): {buf_pct:+}%"]
        lines.append("")
    (OUT / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


def _spawn(model, gpu):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{model}.log", "w") as lf:
        return subprocess.Popen([sys.executable, "-u", __file__, "--model", model, "--gpu", str(gpu)],
                                stdout=lf, stderr=subprocess.STDOUT)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model"); p.add_argument("--gpu", type=int, default=4)
    p.add_argument("--all", action="store_true"); p.add_argument("--aggregate", action="store_true")
    p.add_argument("--gpus", type=int, nargs="*", default=[4, 5, 6, 7])
    a = p.parse_args(argv)
    if a.aggregate:
        aggregate(); return 0
    if a.all:
        procs = [_spawn(m, a.gpus[i % len(a.gpus)]) for i, m in enumerate(MODELS)]
        for pr in procs:
            pr.wait()
        aggregate(); print("=== TIER_REMAP_DONE ===", flush=True); return 0
    if a.model:
        print(json.dumps(run_one(a.model, a.gpu), indent=2)); return 0
    p.error("need --model, --all, or --aggregate"); return 2


if __name__ == "__main__":
    raise SystemExit(main())
