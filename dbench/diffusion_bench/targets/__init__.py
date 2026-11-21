"""Target adapters: where a plan's cells run. One module per target kind, each with

  prepare(plan, out_dir, **opts) -> dict   launch artifacts (notebooks, workflows, bundle, launch script), offline
  collect(run_dir) -> list[dict]           the records a finished run left (record.json per cell), tagged with the
                                           plan's target / arm when the record lacks them

Target names: "colab:T4|L4|A100-40|A100-80|G4", "kaggle:T4x2", "local:B200", "amd:gfx1151-linux|gfx1151-windows",
"mac:macos-15", "cpu" (Colab CPU High-RAM via cloud_pool --cpu), "tester:<label>" (export_tester_bundle.py).
"""

from __future__ import annotations

import importlib

# Per-target facts the scheduler needs. vram_gb / dtypes / cu_per_h follow cloud_pool.TIERS (2026-11); timing says
# whether a timed arm may run there at all (local GPUs are shared: correctness only); max_vm_min caps one VM's
# planned work below the launcher's wall timeout (colab 22000 s, kaggle 42500 s, GitHub jobs 330 / 600 min).
# disk_gb: free disk after setup (Colab VMs: a 236 GB root, ~16 GB used by the two stacks' venvs); unknown = None.
TARGETS: dict = {
    "colab:T4": {"kind": "colab", "gpu": "T4", "vram_gb": 15.0, "ram_gb": 12.7, "dtypes": ["fp16", "fp32"],
                 "cc": "7.5", "cu_per_h": 1.07, "lanes": 1, "timing": True, "max_vm_min": 330, "setup_min": 25, "disk_gb": 220.0},
    "colab:L4": {"kind": "colab", "gpu": "L4", "vram_gb": 22.5, "ram_gb": 53.0,
                 "dtypes": ["fp16", "bf16", "fp8", "fp32"], "cc": "8.9", "cu_per_h": 1.54, "lanes": 1,
                 "timing": True, "max_vm_min": 330, "setup_min": 20, "disk_gb": 220.0},
    "colab:A100-40": {"kind": "colab", "gpu": "A100", "vram_gb": 40.0, "ram_gb": 83.5,
                      "dtypes": ["fp16", "bf16", "fp32"], "cc": "8.0", "cu_per_h": 5.3, "lanes": 1, "timing": True,
                      "max_vm_min": 330, "setup_min": 15, "disk_gb": 220.0},
    "colab:A100-80": {"kind": "colab", "gpu": "A100-HM", "vram_gb": 80.0, "ram_gb": 167.0,
                      "dtypes": ["fp16", "bf16", "fp32"], "cc": "8.0", "cu_per_h": 6.77, "lanes": 1, "timing": True,
                      "max_vm_min": 330, "setup_min": 15, "disk_gb": 220.0},
    "colab:G4": {"kind": "colab", "gpu": "G4", "vram_gb": 95.6, "ram_gb": 176.9,
                 "dtypes": ["fp16", "bf16", "fp8", "nvfp4", "fp32"], "cc": "12.0", "cu_per_h": 8.9, "lanes": 1,
                 "timing": True, "max_vm_min": 330, "setup_min": 12, "disk_gb": 220.0},
    "kaggle:T4x2": {"kind": "kaggle", "gpu": "T4x2", "vram_gb": 15.0, "ram_gb": 29.0, "dtypes": ["fp16", "fp32"],
                    "cc": "7.5", "cu_per_h": 0.0, "lanes": 2, "timing": True, "max_vm_min": 690, "setup_min": 30},
    "local:B200": {"kind": "local", "gpu": "B200", "vram_gb": 180.0, "ram_gb": 1000.0,
                   "dtypes": ["fp16", "bf16", "fp8", "nvfp4", "fp32"], "cc": "10.0", "cu_per_h": 0.0, "lanes": 1,
                   "timing": False, "max_vm_min": 100000, "setup_min": 0},
    "amd:gfx1151-linux": {"kind": "amd", "gpu": "gfx1151", "vram_gb": 62.5, "ram_gb": 62.5,
                          "dtypes": ["fp16", "bf16", "fp32"], "cc": None, "cu_per_h": 0.0, "lanes": 1,
                          "timing": True, "max_vm_min": 540, "setup_min": 60, "os": "linux"},
    "amd:gfx1151-windows": {"kind": "amd", "gpu": "gfx1151", "vram_gb": 62.5, "ram_gb": 62.5,
                            "dtypes": ["fp16", "bf16", "fp32"], "cc": None, "cu_per_h": 0.0, "lanes": 1,
                            "timing": True, "max_vm_min": 540, "setup_min": 60, "os": "windows"},
    "mac:macos-15": {"kind": "mac", "gpu": "MPS", "vram_gb": 7.0, "ram_gb": 7.0, "dtypes": ["fp16", "bf16", "fp32"],
                     "cc": None, "cu_per_h": 0.0, "lanes": 1, "timing": True, "max_vm_min": 300, "setup_min": 20},
    "cpu": {"kind": "cpu", "gpu": "CPU", "vram_gb": 0.0, "ram_gb": 51.0, "dtypes": ["fp32", "bf16", "fp16"],
            "cc": None, "cu_per_h": 0.26, "lanes": 1, "timing": True, "max_vm_min": 600, "setup_min": 20},
}
KINDS = ("colab", "kaggle", "local", "amd", "mac", "cpu", "tester")


def target_info(name: str) -> dict:
    if name in TARGETS:
        return {"name": name, **TARGETS[name]}
    if name.startswith("tester:"):
        return {"name": name, "kind": "tester", "gpu": name.split(":", 1)[1], "vram_gb": None, "ram_gb": None,
                "dtypes": ["fp16", "bf16", "fp8", "fp32"], "cc": None, "cu_per_h": 0.0, "lanes": 1, "timing": True,
                "max_vm_min": 100000, "setup_min": 0}
    raise KeyError(f"unknown target {name!r}; known: {sorted(TARGETS)} or tester:<label>")


def slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in name.lower()).strip("-")


def adapter(kind_or_name: str):
    """The adapter module for a target kind ("colab") or name ("colab:L4")."""
    kind = kind_or_name.split(":", 1)[0]
    if kind not in KINDS or kind == "tester":
        raise KeyError(f"no adapter for {kind_or_name!r} (tester: use export_tester_bundle.py)")
    return importlib.import_module(f"{__name__}.{kind}")
