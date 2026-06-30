# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Gate measurement for a consumer-aware attention default: is SageAttention (INT8 QK) a
near-lossless drop-in vs the native/cuDNN attention the auto path uses today?

INT8 tensor cores run at full rate on consumer GPUs (where FP16-with-FP16-accumulate attention
is half-rate and FP16-with-FP32-accumulate is quarter-rate), so on consumer hardware sage would
be a real speedup -- but only worth defaulting to if its quality cost is within the bar. This
measures PSNR / (optional) LPIPS of sage vs native at a fixed seed on one CUDA GPU. Speed on a
data-center GPU (full-rate everything) is not the consumer signal, so we report quality first.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "studio" / "backend"))

BASE = "unsloth/Z-Image-Turbo-GGUF"
GGUF = "z-image-turbo-Q4_K_M.gguf"
PROMPTS = [
    "a sloth astronaut floating in space, cinematic, highly detailed",
    "a cozy bookstore cafe in autumn, warm light, photorealistic",
    "a koi pond with lily pads at dawn, soft mist",
]
SEED = 12345


def _psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)


def _load(backend, attn):
    backend.begin_load(
        BASE, gguf_filename=GGUF, memory_mode="fast", speed_mode="default", attention_backend=attn
    )
    while True:
        prog = backend.load_progress()
        if prog.get("phase") == "ready":
            return
        if prog.get("error"):
            raise RuntimeError(prog["error"])
        time.sleep(2)


def _gen(backend, prompt):
    out = backend.generate(prompt=prompt, width=1024, height=1024, steps=8, guidance=0.0, seed=SEED)
    return np.array(out["images"][0])


def main() -> int:
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()

    # Reference: explicit native SDPA (the bit-identical-ish baseline).
    print("== loading reference (attention=native) ==", flush=True)
    _load(backend, "native")
    ref = []
    t0 = time.time()
    for p in PROMPTS:
        ref.append(_gen(backend, p))
    ref_dt = (time.time() - t0) / len(PROMPTS)
    backend.unload()
    print(f"  native median-ish per-gen {ref_dt:.3f}s", flush=True)

    # Candidate: sage (INT8 QK) attention.
    print("== loading candidate (attention=sage) ==", flush=True)
    try:
        _load(backend, "sage")
    except Exception as exc:  # noqa: BLE001
        print(f"  sage load FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 1
    status = backend.status()
    print(f"  status attention_backend={status.get('attention_backend')}", flush=True)
    t0 = time.time()
    psnrs = []
    for i, p in enumerate(PROMPTS):
        img = _gen(backend, p)
        ps = _psnr(ref[i], img)
        psnrs.append(ps)
        print(f"  prompt[{i}] PSNR(sage vs native) = {ps:.2f} dB", flush=True)
    cand_dt = (time.time() - t0) / len(PROMPTS)
    backend.unload()

    print("\n==== SUMMARY (Z-Image-Turbo, sage vs native attention) ====", flush=True)
    print(f"  per-gen latency: native {ref_dt:.3f}s  sage {cand_dt:.3f}s", flush=True)
    finite = [p for p in psnrs if p != float("inf")]
    mean_psnr = sum(finite) / len(finite) if finite else float("inf")
    print(f"  mean PSNR(sage vs native) = {mean_psnr:.2f} dB  (per-prompt: "
          f"{', '.join(f'{p:.1f}' for p in psnrs)})", flush=True)
    # Rule of thumb: >35 dB is visually near-identical; 30-35 minor; <30 visible.
    verdict = "NEAR-LOSSLESS" if mean_psnr >= 35 else ("MINOR" if mean_psnr >= 30 else "VISIBLE LOSS")
    print(f"  verdict: {verdict}", flush=True)
    print("SAGE-QUALITY-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
