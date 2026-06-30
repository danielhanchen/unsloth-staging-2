# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Measure transformer GPU load-peak VRAM BEFORE vs AFTER pre-quantized loading.

BEFORE (on-the-fly): from_pretrained the dense bf16 transformer to the GPU, then run the real
product ``quantize_transformer`` (int8/fp8). Peak ~ full bf16 footprint.
AFTER (prequant): the real product ``load_prequantized_transformer`` (meta-init +
load_state_dict(assign=True)) on a checkpoint built here. Peak ~ quantized footprint.

Modes (run each in its OWN subprocess -- CUDA peak is process-global):
  before-build  load dense, quantize_transformer (=BEFORE peak), then torch.save the quantized
                state dict in PREQUANT_FORMAT so --mode after can load it. One dense load.
  after         load_prequantized_transformer from that checkpoint (=AFTER peak), then ASSERT
                every param/buffer is resident on cuda (the planner's #1 false-positive guard:
                a quantized tensor left on CPU would fake a low peak).

Transformer-only: the peak is read right after the transformer is on the GPU, before any pipeline
assembly, so no VAE / text-encoder download is needed. One model on one CUDA GPU.

  python scripts/prequant_mem_bench.py --mode before-build \
      --base <local_dir_or_repo> --family z-image --scheme int8 --ckpt out/transformer_int8.pt
  python scripts/prequant_mem_bench.py --mode after \
      --base <local_dir_or_repo> --family z-image --scheme int8 --ckpt out/transformer_int8.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import types
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "studio" / "backend"

logging.basicConfig(level = logging.INFO, format = "%(message)s")
LOGGER = logging.getLogger("prequant_mem_bench")


def _gb(nbytes: int) -> float:
    return nbytes / 1e9


def _emit(mode: str, family: str, scheme: str, **kw) -> None:
    """One machine-parseable line the orchestrator greps for."""
    parts = [f"mode={mode}", f"family={family}", f"scheme={scheme}"]
    parts += [f"{k}={v}" for k, v in kw.items()]
    print("RESULT " + " ".join(parts), flush = True)


def _transformer_cls(base: str, family: str):
    import diffusers
    from core.inference.diffusion_families import detect_family

    fam = detect_family(base, override = family)
    if fam is None:
        raise SystemExit(f"unknown family '{family}' for base '{base}'")
    return getattr(diffusers, fam.transformer_class), fam


def _device_summary(module):
    """(n_cuda, n_cpu, n_meta) over params+buffers -- the residency proof for AFTER."""
    import torch  # noqa: F401

    n_cuda = n_cpu = n_meta = 0
    for t in list(module.parameters()) + list(module.buffers()):
        if getattr(t, "is_meta", False):
            n_meta += 1
        elif getattr(getattr(t, "device", None), "type", None) == "cuda":
            n_cuda += 1
        else:
            n_cpu += 1
    return n_cuda, n_cpu, n_meta


def run_before_build(base: str, family: str, scheme: str, ckpt: str, hf_token, min_features: int) -> int:
    import torch
    import torchao
    import diffusers
    from core.inference.diffusion_prequant import PREQUANT_FORMAT
    from core.inference.diffusion_transformer_quant import quantize_transformer

    cls, fam = _transformer_cls(base, family)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    # No device_map / offload: load dense bf16 then move the whole module to the GPU.
    transformer = cls.from_pretrained(
        base, subfolder = "transformer", torch_dtype = torch.bfloat16, token = hf_token
    ).to("cuda")
    pipe = types.SimpleNamespace(transformer = transformer)
    target = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
    engaged = quantize_transformer(pipe, target, mode = scheme, min_features = min_features, logger = LOGGER)
    torch.cuda.synchronize()
    before_peak = _gb(torch.cuda.max_memory_allocated())
    reserved = _gb(torch.cuda.max_memory_reserved())
    if engaged != scheme:
        _emit("before-build", fam.name, scheme, status = "FAIL", reason = f"engaged={engaged}")
        return 1

    # Save the quantized state dict in the exact format diffusion_prequant expects, mirroring
    # scripts/build_prequant_checkpoint.py (quantize_transformer already applied the same filter,
    # incl. the int8 M=1 exclusion, so this checkpoint == the product builder's).
    state_dict = {
        k: (v.detach().to("cpu") if hasattr(v, "detach") else v)
        for k, v in pipe.transformer.state_dict().items()
    }
    payload = {
        "format": PREQUANT_FORMAT,
        "metadata": {
            "base_model_id": base,
            "family": fam.name,
            "scheme": scheme,
            "min_features": min_features,
            "torch_dtype": "bfloat16",
            "quant_backend": "torchao",
            "transformer_class": fam.transformer_class,
            "torch_version": torch.__version__,
            "torchao_version": getattr(torchao, "__version__", "?"),
            "diffusers_version": diffusers.__version__,
        },
        "state_dict": state_dict,
    }
    out = Path(ckpt)
    out.parent.mkdir(parents = True, exist_ok = True)
    torch.save(payload, out)
    disk_gb = _gb(out.stat().st_size)
    _emit(
        "before-build", fam.name, scheme,
        status = "OK", before_peak_gb = f"{before_peak:.2f}", reserved_gb = f"{reserved:.2f}",
        ckpt_disk_gb = f"{disk_gb:.2f}", secs = f"{time.time() - t0:.0f}",
    )
    return 0


def run_after(base: str, family: str, scheme: str, ckpt: str, hf_token) -> int:
    import torch
    from core.inference.diffusion_prequant import PrequantSource, load_prequantized_transformer

    cls, fam = _transformer_cls(base, family)
    if not Path(ckpt).is_file():
        _emit("after", fam.name, scheme, status = "FAIL", reason = "missing_ckpt")
        return 1

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    source = PrequantSource(kind = "path", location = ckpt, filename = None)
    transformer = load_prequantized_transformer(
        cls, base, source,
        device = "cuda", dtype = torch.bfloat16, hf_token = hf_token, scheme = scheme, logger = LOGGER,
    )
    if transformer is None:
        _emit("after", fam.name, scheme, status = "FAIL", reason = "load_returned_none")
        return 1
    torch.cuda.synchronize()
    after_peak = _gb(torch.cuda.max_memory_allocated())
    reserved = _gb(torch.cuda.max_memory_reserved())

    # Residency proof: everything must be on cuda, nothing on cpu/meta, else the peak is a lie.
    n_cuda, n_cpu, n_meta = _device_summary(transformer)
    marker = getattr(transformer, "_unsloth_runtime_quant", None)
    resident = (n_cpu == 0 and n_meta == 0 and n_cuda > 0)
    _emit(
        "after", fam.name, scheme,
        status = "OK" if resident else "INVALID",
        after_peak_gb = f"{after_peak:.2f}", reserved_gb = f"{reserved:.2f}",
        n_cuda = n_cuda, n_cpu = n_cpu, n_meta = n_meta, marker = marker,
        secs = f"{time.time() - t0:.0f}",
    )
    return 0 if resident else 2


def main(argv = None) -> int:
    sys.path.insert(0, str(BACKEND))
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices = ["before-build", "after"], required = True)
    p.add_argument("--base", required = True, help = "local dir or HF repo with a transformer/ subfolder")
    p.add_argument("--family", required = True)
    p.add_argument("--scheme", required = True, choices = ["int8", "fp8"])
    p.add_argument("--ckpt", required = True)
    p.add_argument("--min-features", type = int, default = 512)
    p.add_argument("--hf-token", default = None)
    args = p.parse_args(argv)
    import os

    tok = args.hf_token or os.environ.get("HF_TOKEN")
    if args.mode == "before-build":
        rc = run_before_build(args.base, args.family, args.scheme, args.ckpt, tok, args.min_features)
    else:
        rc = run_after(args.base, args.family, args.scheme, args.ckpt, tok)
    print("PREQUANT-MEM-BENCH-DONE", flush = True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
