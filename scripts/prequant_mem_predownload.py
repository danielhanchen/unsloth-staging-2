# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-download ONLY the transformer subfolder of each supported base repo into the workspace.

Transformer-only keeps the download to ~85 GB (no VAE / text encoders): the memory benchmark
measures the transformer load peak, which needs just the dense bf16 transformer weights + config.
Gated repos (FLUX.2-klein-9B) are attempted with HF_TOKEN and recorded as gated on 403.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

OUT = Path("/mnt/disks/unslothai/ubuntu/workspace_81/unsloth/outputs/prequant_mem/dense")

# (local dir name, repo). Includes klein-9B (attempted; license-gated download).
REPOS = [
    ("z-image",         "Tongyi-MAI/Z-Image"),
    ("flux.2-klein-4b", "black-forest-labs/FLUX.2-klein-4B"),
    ("flux.1",          "black-forest-labs/FLUX.1-schnell"),
    ("qwen-image",      "Qwen/Qwen-Image"),
    ("flux.2-klein-9b", "black-forest-labs/FLUX.2-klein-9B"),
]


def main() -> int:
    from huggingface_hub import snapshot_download

    tok = os.environ.get("HF_TOKEN")
    OUT.mkdir(parents = True, exist_ok = True)
    for name, repo in REPOS:
        dest = OUT / name
        t0 = time.time()
        print(f"\n=== {name} <- {repo} (transformer/*) ===", flush = True)
        try:
            snapshot_download(
                repo_id = repo,
                allow_patterns = ["transformer/*"],
                local_dir = str(dest),
                token = tok,
            )
            files = list((dest / "transformer").glob("*")) if (dest / "transformer").is_dir() else []
            nbytes = sum(f.stat().st_size for f in files if f.is_file())
            print(f"  OK {name}: {len(files)} files, {nbytes / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush = True)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {name}: {type(exc).__name__}: {str(exc)[:160]}", flush = True)
    print("PREDOWNLOAD-DONE", flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
