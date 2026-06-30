# SPDX-License-Identifier: AGPL-3.0-only
"""Verify the remapped speed tiers on the NON-GGUF (torchao dense-quant) transformer path.

The compile-dequant `default` lever is GGUF-specific (it fuses diffusers' pure-PyTorch GGUF
dequant). The non-GGUF tiers are the torchao DYNAMIC-quant schemes (fp8 / int8 / nvfp4 /
mxfp8): the low-precision matmul runs on tensor cores via fused kernels (_scaled_mm /
torch._int_mm) -- there is no separate dequant op chain to compile, AND those schemes need
the REGIONAL block compile (dynamic quant is ~30x slower eager). So the remap must route a
non-GGUF transformer to the regional compile under `default` (NOT compile-dequant).

This loads the dense bf16 transformer + torchao-quantises it (transformer_quant=<scheme>),
then for each speed tier records the engaged speed_optims + latency + VRAM. The correctness
assertion: under `default`, a non-GGUF transformer engages `compiled` (regional) and NOT
`compiled_dequant` -- proving the is_gguf fix (gguf_filename is still set as the fallback).

  python nongguf_tier_check.py --quant fp8 --gpu 4
  python nongguf_tier_check.py --quant int8 --gpu 5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

UNSLOTH = Path("/mnt/disks/unslothai/ubuntu/workspace_81/unsloth")
BACKEND = UNSLOTH / "studio" / "backend"
sys.path.insert(0, str(UNSLOTH / "scripts"))
import benchall_orchestrator as bo  # noqa: E402

OUT = UNSLOTH / "outputs" / "nongguf_tier"
PROMPT = "a cat asleep on a stack of books, highly detailed"
MODEL = "z-image-turbo"  # smallest dense transformer -> cheapest dense load
TIERS = ["default", "max"]  # eager skipped: dynamic quant is ~30x slower without compile


def _wait_ready(backend, timeout_s=3600):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        p = backend.load_progress()
        if p.get("phase") == "ready":
            return
        if p.get("phase") == "error":
            raise RuntimeError(f"load error: {p.get('error')}")
        time.sleep(2)
    raise TimeoutError("load did not become ready")


def _gen(backend, steps, guidance):
    backend.generate(prompt=PROMPT, width=1024, height=1024, steps=steps,
                     guidance=guidance, seed=bo.SEED, batch_size=1)


def _measure(backend, gguf_repo, gguf_file, base_repo, steps, guidance, quant, speed, iters=4):
    import torch  # noqa: PLC0415

    backend.begin_load(gguf_repo, gguf_filename=gguf_file, base_repo=base_repo,
                       hf_token=os.environ.get("HF_TOKEN"), memory_mode="fast",
                       speed_mode=speed, transformer_quant=quant)
    _wait_ready(backend)
    status = backend.status()
    engaged_quant = status.get("transformer_quant")
    optims = sorted(status.get("speed_optims", []))

    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); _gen(backend, steps, guidance); torch.cuda.synchronize()
    cold_s = time.time() - t0

    lat = []
    for _ in range(iters):
        torch.cuda.synchronize(); t = time.time()
        _gen(backend, steps, guidance)
        torch.cuda.synchronize(); lat.append(time.time() - t)
    warm_s = statistics.median(lat)
    vram = torch.cuda.max_memory_allocated() / 1e9
    backend.unload()

    # The whole point of the fix: a non-GGUF transformer must NOT engage the GGUF dequant
    # levers, and must get the regional compile under default.
    routing_ok = ("compiled_dequant" not in optims and "weight_buffer" not in optims
                  and "compiled" in optims)
    return {
        "speed_mode": speed, "transformer_quant_engaged": engaged_quant,
        "speed_optims": optims, "routing_ok": routing_ok,
        "cold_s": round(cold_s, 2), "warm_s": round(warm_s, 3),
        "compile_tax_s": round(cold_s - warm_s, 1), "peak_vram_GB": round(vram, 1),
    }


def run_one(quant: str, gpu: int) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gguf_repo, gguf_file, base_repo, steps, guidance, _ = bo.MANIFEST[MODEL]
    sys.path.insert(0, str(BACKEND))
    from core.inference.diffusion import get_diffusion_backend  # noqa: PLC0415

    backend = get_diffusion_backend()
    res = {"model": MODEL, "quant_requested": quant, "gpu": gpu, "tiers": {}}
    for speed in TIERS:
        try:
            res["tiers"][speed] = _measure(backend, gguf_repo, gguf_file, base_repo,
                                           steps, guidance, quant, speed)
        except Exception as exc:  # noqa: BLE001
            res["tiers"][speed] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{quant}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


def aggregate() -> None:
    rows = [json.loads((OUT / f"{q}.json").read_text())
            for q in ("fp8", "int8", "nvfp4", "auto") if (OUT / f"{q}.json").exists()]
    lines = [
        "# Remapped speed tiers on the NON-GGUF (torchao dense-quant) path",
        "",
        "fp8 / int8 use torchao dynamic quant: the low-precision matmul is a fused tensor-core",
        "kernel (no separate dequant op chain), and the scheme NEEDS the regional block compile",
        "(dynamic quant ~30x slower eager). So `default`/`max` must route a non-GGUF transformer",
        "to `compiled` (regional), NOT to the GGUF `compiled_dequant`. routing_ok verifies that.",
        "",
        "| quant | tier | engaged | speed_optims | routing_ok | warm (s) | cold (s) | tax (s) | VRAM (GB) |",
        "|---|---|---|---|:--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        for tier, x in r["tiers"].items():
            if "error" in x:
                lines.append(f"| {r['quant_requested']} | {tier} | - | ERROR: {x['error']} | - | - | - | - | - |")
                continue
            lines.append(
                f"| {r['quant_requested']} | {tier} | {x['transformer_quant_engaged']} | "
                f"{', '.join(x['speed_optims'])} | {'YES' if x['routing_ok'] else '**NO**'} | "
                f"{x['warm_s']} | {x['cold_s']} | {x['compile_tax_s']} | {x['peak_vram_GB']} |"
            )
    (OUT / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quant", choices=["fp8", "int8", "nvfp4", "auto"])
    p.add_argument("--gpu", type=int, default=4)
    p.add_argument("--aggregate", action="store_true")
    a = p.parse_args(argv)
    if a.aggregate:
        aggregate(); return 0
    if a.quant:
        run_one(a.quant, a.gpu)
        print("=== NONGGUF_TIER_DONE ===", flush=True); return 0
    p.error("need --quant or --aggregate"); return 2


if __name__ == "__main__":
    raise SystemExit(main())
