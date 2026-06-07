#!/usr/bin/env python3
"""Cross-OS validation helper (throwaway pr5963-cross-os branch).

Loads studio/install_llama_prebuilt.py and prints what detect_host() reports on
this GitHub-hosted runner, so each per-OS job shows the real platform/arch the
installer infers (Linux x64/arm64, Windows x64, macOS arm64/intel). GitHub
runners have no GPU, so has_usable_nvidia / has_rocm are expected False here;
the GPU selection branches are exercised by the synthetic-host unit tests.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "studio" / "install_llama_prebuilt.py"
spec = importlib.util.spec_from_file_location("ilp_hostprobe", MODULE_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

host = mod.detect_host()
fields = [
    "system", "machine",
    "is_windows", "is_linux", "is_macos",
    "is_x86_64", "is_arm64",
    "has_physical_nvidia", "has_usable_nvidia", "has_rocm",
    "rocm_gfx_target", "compute_caps", "driver_cuda_version", "macos_version",
]
print("=== detect_host() on this runner ===")
for f in fields:
    print(f"{f:22s} = {getattr(host, f, '<missing>')!r}")

# Sanity: the detected platform booleans must be internally consistent.
assert host.is_windows + host.is_linux + host.is_macos == 1, "exactly one OS flag must be set"
assert host.is_x86_64 + host.is_arm64 <= 1, "arch flags must not both be set"
print("host-detection consistency: OK")
