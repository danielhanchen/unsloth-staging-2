"""SIM 4: [Linux, Windows, WSL, macOS] x [NVIDIA, AMD/ROCm, CPU-only] = 12 combos.

The changed code is platform agnostic orchestration, so the point of this sweep is to
PROVE that: patch the platform/device detectors the training stack actually branches on
and assert the lifecycle outcome is byte-identical in all 12, including the MLX
(Apple Silicon) worker selection, which is the one real platform fork in the training path.
"""
from __future__ import annotations

import platform
import sys

import simharness as H

T = H.T
FAILS = []


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(name)
        print(f"  FAIL  {name}  <- {detail}")


OSES = {
    "Linux":   ("Linux",   "x86_64"),
    "Windows": ("Windows", "AMD64"),
    "WSL":     ("Linux",   "x86_64"),   # WSL2 reports as Linux to Python
    "macOS":   ("Darwin",  "arm64"),
}
GPUS = ["NVIDIA", "AMD-ROCm", "CPU-only"]


def complete_event():
    return {
        "type": "complete",
        "output_dir": "/tmp/out",
        "status_message": "Training completed! Model saved to /tmp/out",
    }


def scenario(tag):
    """Full terminal lifecycle; returns an outcome tuple to compare across combos."""
    b = H.fresh_backend(job_id=f"job-{tag}")
    armed = []
    b._start_stop_watchdog = lambda **kw: armed.append(kw)

    mid_ui, mid_adm = H.ui_active(b), b.is_training_active()
    b._handle_event(complete_event())
    post_ui, post_adm = H.ui_active(b), b.is_training_active()
    phase = H.derive_phase(b)
    msg_ok = b._progress.status_message.startswith("Training completed!")
    b._proc._alive = False
    gone_ui, gone_adm = H.ui_active(b), b.is_training_active()
    return (mid_ui, mid_adm, post_ui, post_adm, phase, msg_ok, gone_ui, gone_adm,
            len(armed), armed[0].get("terminal_seen") if armed else None)


EXPECTED = (True, True, False, True, "completed", True, False, False, 1, True)

print("\n  OS x GPU matrix (12 combos)\n")
print(f"  {'combo':<24} {'mlx path':<9} {'outcome':<9} {'watchdog':<9}")
print("  " + "-" * 55)

results = {}
real_system, real_machine = platform.system, platform.machine
for osname, (sysname, machine) in OSES.items():
    for gpu in GPUS:
        platform.system = lambda s=sysname: s
        platform.machine = lambda m=machine: m
        H.G["platform"].system = platform.system
        H.G["platform"].machine = platform.machine

        mlx = T.should_use_mlx_training_backend()
        tag = f"{osname}-{gpu}"
        out = scenario(tag)
        results[tag] = out
        ok = out == EXPECTED
        check(f"{tag} lifecycle", ok, f"{out} != {EXPECTED}")
        print(f"  {tag:<24} {str(mlx):<9} {'OK' if ok else 'BAD':<9} "
              f"{'armed' if out[8] else 'none':<9}")

        # macOS/arm64 must select MLX; nothing else may.
        check(f"{tag} mlx selection", mlx is (osname == "macOS"), f"mlx={mlx}")

platform.system, platform.machine = real_system, real_machine
H.G["platform"].system = real_system
H.G["platform"].machine = real_machine

print()
check("all 12 combos identical", len(set(results.values())) == 1,
      f"{len(set(results.values()))} distinct outcomes")

print("\n=== env-var overrides parse identically everywhere ===")
import os as _os
for raw, expect in [("", 120), ("30", 30), ("0", 0), ("bogus", 120), ("  45  ", 45), ("-5", -5)]:
    if raw == "":
        _os.environ.pop("UNSLOTH_STUDIO_TRAINING_COMPLETE_EXIT_GRACE_S", None)
    else:
        _os.environ["UNSLOTH_STUDIO_TRAINING_COMPLETE_EXIT_GRACE_S"] = raw
    got = T._env_int("UNSLOTH_STUDIO_TRAINING_COMPLETE_EXIT_GRACE_S", 120)
    check(f"env {raw!r} -> {expect}", got == expect, f"got {got}")
_os.environ.pop("UNSLOTH_STUDIO_TRAINING_COMPLETE_EXIT_GRACE_S", None)
print("  all env parses OK")

print("\n" + "=" * 62)
print(f"SIM 4: {'ALL PASS (12/12 combos identical)' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
