# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Benchmark ComfyUI (GGUF path) per model: latency + peak VRAM, via the /prompt HTTP API.

Assumes a ComfyUI server is already running (start_server.sh / the orchestrator launches it). For each
family it submits an API-format workflow graph using the SAME Q4_K_M transformer GGUF + VAE/encoders as
our other engines, runs a warmup (loads + compiles + caches the model) then a timed run, and samples
peak VRAM on the target GPU via nvidia-smi. -> outputs/comfyui/results.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path("/mnt/disks/unslothai/ubuntu/workspace_81")
OUT = ROOT / "unsloth" / "outputs" / "comfyui"
PROMPT = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed"


def _workflow(family: str, *, w: int, h: int, seed: int, compile: bool = False) -> dict:
    """API-format ComfyUI graph for one family (GGUF transformer).

    ``compile`` inserts a TorchCompileModel node after the GGUF loader (ComfyUI's best-perf knob),
    so the KSampler runs the compiled model -- the fair 'ComfyUI best' comparison vs our compiled path.
    """
    if family == "flux.1":
        g = {
            "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "flux1-schnell-Q4_K_M.gguf"}},
            "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp16.safetensors", "type": "flux"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux1_ae.safetensors"}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["2", 0]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
            "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "comfy_flux1"}},
        }
    elif family == "qwen-image":
        g = {
            "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-Q4_K_M.gguf"}},
            "2": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": "qwen2.5vl_Q4_K_M.gguf", "type": "qwen_image"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["2", 0]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
            "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": seed, "steps": 20, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "comfy_qwen"}},
        }
    elif family == "flux.2-klein":
        g = {
            "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "flux-2-klein-4b-Q4_K_M.gguf"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2_ae.safetensors"}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["2", 0]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
            "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "comfy_klein"}},
        }
    elif family == "z-image":
        # No z_image CLIP type in 0.26.0; Z-Image is Lumina-style with a Qwen3 encoder -> try lumina2.
        g = {
            "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "z-image-turbo-Q4_K_M.gguf"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "zimage_vae.safetensors"}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["2", 0]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
            "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "comfy_zimage"}},
        }
    elif family == "flux.1-fp8":
        g = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-schnell-fp8.safetensors", "weight_dtype": "default"}},
            "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp16.safetensors", "type": "flux"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux1_ae.safetensors"}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["2", 0]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
            "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "comfy_flux1_fp8"}},
        }
    elif family in ("qwen-image-fp8", "qwen-image-nvfp4"):
        unet = "qwen_image_fp8_e4m3fn.safetensors" if family.endswith("fp8") else "qwen_image_nvfp4.safetensors"
        g = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": "qwen2.5vl_Q4_K_M.gguf", "type": "qwen_image"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["2", 0]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
            "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": seed, "steps": 20, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "comfy_qwen_q"}},
        }
    elif family == "flux.2-klein-fp8":
        g = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b.safetensors", "weight_dtype": "fp8_e4m3fn_fast"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2_ae.safetensors"}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["2", 0]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
            "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "comfy_klein_fp8"}},
        }
    elif family == "z-image-fp8":
        g = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "fp8_e4m3fn_fast"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "zimage_vae.safetensors"}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["2", 0]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
            "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "comfy_zimage_fp8"}},
        }
    else:
        raise ValueError(f"no workflow template for {family}")
    if compile:
        g["10"] = {"class_type": "TorchCompileModel", "inputs": {"model": ["1", 0], "backend": "inductor"}}
        g["7"]["inputs"]["model"] = ["10", 0]
    return g


class _VRAM:
    def __init__(self, gpu: int, hz: float = 10.0):
        self.gpu, self._i, self._peak = gpu, 1.0 / hz, 0
        self._stop = threading.Event(); self._t = threading.Thread(target=self._loop, daemon=True)
    def _used(self):
        try:
            o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", f"--id={self.gpu}"], text=True, timeout=5)
            return int(o.strip().splitlines()[0])
        except Exception:
            return 0
    def _loop(self):
        while not self._stop.is_set():
            self._peak = max(self._peak, self._used()); time.sleep(self._i)
    def __enter__(self): self._t.start(); return self
    def __exit__(self, *a): self._stop.set(); self._t.join(timeout=2); self._peak = max(self._peak, self._used())
    @property
    def peak_gb(self): return self._peak / 1024.0


def _post(port: int, graph: dict) -> str:
    data = json.dumps({"prompt": graph}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/prompt", data=data, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]


def _wait(port: int, pid: str, timeout=1200) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            h = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{pid}", timeout=30).read())
        except Exception:
            h = {}
        if pid in h:
            return h[pid]
        time.sleep(0.5)
    raise TimeoutError("comfy prompt did not finish")


def run_family(family: str, port: int, gpu: int, w: int, h: int, seed: int, compile: bool = False) -> dict:
    graph = _workflow(family, w=w, h=h, seed=seed, compile=compile)
    # warmup (load + compile + cache); compile can be slow on the first run -> absorbed here
    _wait(port, _post(port, graph), timeout=2400)
    # timed (model cached -> pure inference); re-seed to force re-run
    graph2 = _workflow(family, w=w, h=h, seed=seed + 1, compile=compile)
    with _VRAM(gpu) as vram:
        t0 = time.time()
        rec = _wait(port, _post(port, graph2))
        lat = time.time() - t0
    ok = rec.get("status", {}).get("status_str") == "success" or "outputs" in rec
    return {"family": family, "res": f"{w}x{h}", "compile": compile, "status": "OK" if ok else "FAIL",
            "latency_s": f"{lat:.2f}", "peak_vram_gb": f"{vram.peak_gb:.1f}"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8231)
    p.add_argument("--gpu", type=int, default=5)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--families", default="flux.1")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--compile", action="store_true", help="insert TorchCompileModel (ComfyUI best)")
    p.add_argument("--tag", default="", help="suffix for the results csv")
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for fam in args.families.split(","):
        fam = fam.strip()
        print(f">>> comfy {fam} (compile={args.compile}) ...", flush=True)
        try:
            r = run_family(fam, args.port, args.gpu, args.width, args.height, args.seed, compile=args.compile)
        except Exception as exc:  # noqa: BLE001
            r = {"family": fam, "compile": args.compile, "status": f"ERR:{type(exc).__name__}:{str(exc)[:90]}", "latency_s": "", "peak_vram_gb": ""}
        rows.append(r)
        print(f"    -> {r['status']} lat={r.get('latency_s')}s vram={r.get('peak_vram_gb')}GB", flush=True)
    import csv
    name = f"results{('_' + args.tag) if args.tag else ''}.csv"
    with open(OUT / name, "w", newline="") as f:
        wt = csv.DictWriter(f, fieldnames=["family", "res", "compile", "status", "latency_s", "peak_vram_gb"]); wt.writeheader(); wt.writerows(rows)
    print(f"wrote {OUT/'results.csv'}", flush=True)
    print("COMFY-BENCH-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
