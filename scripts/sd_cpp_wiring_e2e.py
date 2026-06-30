# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Real end-to-end check of the wired sd.cpp no-GPU engine (no HTTP, no GPU).

Forces CPU (CUDA hidden), drives the SdCppDiffusionBackend exactly as the route would:
begin_load (fetches the per-family single-file VAE + text encoder from the registry,
reuses the local transformer GGUF), polls load_progress until ready, then generates a
small image and asserts a PIL image comes back. Proves the registry repos + flag
mapping + real sd-cli all line up.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force the no-GPU path

BACKEND = Path("/mnt/disks/unslothai/ubuntu/workspace_81/unsloth/studio/backend")
GGUF_DIR = "/mnt/disks/unslothai/ubuntu/workspace_81/unsloth/outputs/sdcpp_cpu/gguf"
OUT = Path("/mnt/disks/unslothai/ubuntu/workspace_81/unsloth/outputs/sd_cpp_wiring")


def main() -> int:
    sys.path.insert(0, str(BACKEND))
    from core.inference.diffusion_device import resolve_diffusion_device_target
    from core.inference.sd_cpp_backend import SdCppDiffusionBackend, ensure_sd_cpp_binary

    OUT.mkdir(parents = True, exist_ok = True)
    target = resolve_diffusion_device_target()
    print(f"device backend={target.backend} device={target.device}", flush = True)
    assert target.backend == "cpu", f"expected cpu, got {target.backend}"

    binary = ensure_sd_cpp_binary()
    print(f"sd-cli binary: {binary}", flush = True)
    assert binary, "no sd-cli binary"

    be = SdCppDiffusionBackend()
    # Reuse the locally-cached transformer GGUF (local repo_id), so only the VAE +
    # Qwen3 encoder download from the registry repos.
    be.begin_load(
        GGUF_DIR,
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        family_override = "z-image",
    )
    print("begin_load returned; polling load_progress ...", flush = True)
    t0 = time.time()
    while True:
        p = be.load_progress()
        if p["phase"] == "error":
            print(f"LOAD ERROR: {p['error']}", flush = True)
            return 1
        if p["phase"] == "ready":
            break
        if time.time() - t0 > 1800:
            print("LOAD TIMEOUT", flush = True)
            return 1
        print(f"  phase={p['phase']} frac={p['fraction']:.2f} "
              f"{p['bytes_downloaded']/1e9:.2f}/{p['bytes_total']/1e9:.2f} GB", flush = True)
        time.sleep(5)

    st = be.status()
    print(f"loaded status: engine={st['engine']} family={st['family']} device={st['device']}", flush = True)
    assert st["engine"] == "sd_cpp" and st["loaded"] is True

    print("generating 256x256 (8 steps) ...", flush = True)
    g0 = time.time()
    res = be.generate(prompt = "a red fox in a snowy forest", width = 256, height = 256,
                      steps = 8, seed = 1234, batch_size = 1)
    dt = time.time() - g0
    img = res["images"][0]
    out_png = OUT / "zimage_sd_cpp_wired.png"
    img.save(out_png)
    print(f"generated 1 image {img.size} seed={res['seed']} in {dt:.1f}s -> {out_png}", flush = True)
    assert img.size == (256, 256)
    print("SD_CPP_WIRING_E2E_OK", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
