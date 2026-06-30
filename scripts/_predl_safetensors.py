"""Predownload the curated safetensors diffusion models into the shared HF cache."""

import time

from huggingface_hub import hf_hub_download, snapshot_download

t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:.0f}s] {msg}", flush=True)


for repo in [
    "unsloth/Z-Image-Turbo-unsloth-bnb-4bit",
    "unsloth/Qwen-Image-2512-unsloth-bnb-4bit",
]:
    log(f"snapshot {repo} ...")
    snapshot_download(
        repo, ignore_patterns=["*.md", "*.png", "*.jpg", "assets/*"], max_workers=8
    )
    log(f"done {repo}")

log("fp8 single file ...")
f = hf_hub_download("unsloth/Qwen-Image-2512-FP8", "qwen-image-2512-fp8.safetensors")
log(f"-> {f}")

log("base Qwen/Qwen-Image-2512 (skip its transformer weights) ...")
snapshot_download(
    "Qwen/Qwen-Image-2512",
    ignore_patterns=["*.md", "*.png", "*.jpg", "transformer/*.safetensors"],
    max_workers=8,
)
log("ALL DONE")
