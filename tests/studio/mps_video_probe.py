#!/usr/bin/env python3
"""Probe the Apple-Silicon video path on a REAL macOS runner.

Report-only: prints a labelled section per check and never fails the job, so one
run tells us the whole picture (a hard failure would hide every later section).
Run from the repo root with PYTHONPATH=studio/backend.
"""
import json
import os
import platform
import sys
import traceback

sys.path.insert(0, os.path.join(os.getcwd(), "studio", "backend"))

FINDINGS = {}


def section(name):
    print(f"\n===== {name} =====", flush=True)


def record(key, value):
    FINDINGS[key] = value
    print(f"  {key} = {value!r}", flush=True)


def guarded(key, fn):
    try:
        record(key, fn())
    except Exception as exc:
        record(key, f"RAISED {type(exc).__name__}: {exc}")
        traceback.print_exc()


section("platform")
record("system", platform.system())
record("machine", platform.machine())
record("mac_ver", platform.mac_ver()[0] if platform.system() == "Darwin" else None)
record("python", sys.version.split()[0])

section("torch")
try:
    import torch
    record("torch_version", torch.__version__)
    record("mps_is_built", torch.backends.mps.is_built())
    record("mps_is_available", torch.backends.mps.is_available())
    record("has_recommended_max_memory", hasattr(torch.mps, "recommended_max_memory"))
    record("has_driver_allocated_memory", hasattr(torch.mps, "driver_allocated_memory"))
except Exception as exc:
    record("torch_import", f"RAISED {type(exc).__name__}: {exc}")
    torch = None

MPS_USABLE = False
if torch is not None and torch.backends.mps.is_available():
    section("real MPS allocation (is_available() can be True on a GPU-less VM)")

    def alloc():
        global MPS_USABLE
        x = torch.ones(1024, 1024, device="mps")
        y = (x + x).sum().item()
        MPS_USABLE = True
        return y
    guarded("mps_allocation", alloc)
    guarded("mps_bfloat16", lambda: torch.ones(2, dtype=torch.bfloat16, device="mps").float().sum().item())
    guarded("recommended_max_memory_GiB",
            lambda: round(torch.mps.recommended_max_memory() / 1024**3, 2))
    guarded("driver_allocated_memory_MiB",
            lambda: round(torch.mps.driver_allocated_memory() / 1024**2, 2))
    section("float64 on MPS (the LTX-2 rope failure)")
    guarded("linspace_float64_on_mps",
            lambda: torch.linspace(0.0, 1.0, 8, dtype=torch.float64, device="mps").sum().item())

section("video_capability()")
guarded("video_capability", lambda: __import__(
    "utils.hardware.hardware", fromlist=["x"]).video_capability())

section("resolve_diffusion_device_target()")


def target():
    from core.inference.diffusion_device import resolve_diffusion_device_target
    t = resolve_diffusion_device_target()
    return t.as_public_dict()


guarded("device_target", target)

section("snapshot_device_memory()")


def mem():
    from core.inference.diffusion_device import resolve_diffusion_device_target
    from core.inference.diffusion_memory import snapshot_device_memory
    return snapshot_device_memory(resolve_diffusion_device_target()).as_public_dict()


guarded("device_memory", mem)

section("memory plan per video family (bf16 resident, default preset)")


def plans():
    from core.inference.diffusion_device import resolve_diffusion_device_target
    from core.inference.diffusion_memory import (
        snapshot_device_memory, plan_diffusion_memory, estimate_video_runtime_mib)
    from core.inference.video_families import _FAMILIES
    t = resolve_diffusion_device_target()
    dm = snapshot_device_memory(t)
    out = {}
    for fam in _FAMILIES:
        dit, te, vae = fam.bf16_components_gb
        total_mib = int((dit + te + vae) * 1024)
        w, h = fam.resolution_presets[0]
        head = estimate_video_runtime_mib(width=w, height=h, num_frames=fam.default_num_frames)
        plan = plan_diffusion_memory(
            target=t, device_memory=dm, model_dense_mib=total_mib,
            runtime_headroom_mib=head, companion_dense_mib=int((te + vae) * 1024))
        out[fam.name] = {
            "resident_required_mib": plan.estimates["resident_required_mib"],
            "safe_device_budget_mib": plan.estimates["safe_device_budget_mib"],
            "offload_policy": plan.offload_policy,
            "fits": (plan.estimates["safe_device_budget_mib"] is not None
                     and plan.estimates["resident_required_mib"] is not None
                     and plan.estimates["resident_required_mib"]
                     <= plan.estimates["safe_device_budget_mib"]),
            "reasons": list(plan.reasons),
        }
    return out


guarded("memory_plans", plans)

section("install_decoder_sync() against a real nn.Module decoder")


def decoder_sync():
    import torch.nn as nn
    from core.inference.diffusion_device import (
        resolve_diffusion_device_target, install_decoder_sync)

    class Vae:
        def __init__(self):
            self.decoder = nn.Identity()

    class Pipe:
        def __init__(self):
            self.vae = Vae()
    p = Pipe()
    installed = install_decoder_sync(p, resolve_diffusion_device_target())
    # fire the hook once to prove it does not replace the module output
    out = p.vae.decoder(torch.zeros(2))
    return {"installed": installed, "output_shape": list(out.shape)}


guarded("install_decoder_sync", decoder_sync)

section("force_float32_rope() against the real LTX-2 rope modules")


def rope():
    import torch.nn as nn
    from diffusers.pipelines.ltx2.connectors import LTX2RotaryPosEmbed1d
    from core.inference.diffusion_device import (
        resolve_diffusion_device_target, force_float32_rope)
    t = resolve_diffusion_device_target()

    class Conn(nn.Module):
        def __init__(self):
            super().__init__()
            self.r = LTX2RotaryPosEmbed1d(dim=64, rope_type="split")

    class Pipe:
        def __init__(self, c):
            self.connectors = c

        @property
        def components(self):
            return {"connectors": self.connectors}

    before = Conn().to(t.device) if t.device != "cpu" else Conn()
    res = {"device": t.device}
    try:
        before.r(batch_size=1, pos=16, device=t.device)
        res["before_fix"] = "ok"
    except Exception as exc:
        res["before_fix"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    after = Conn()
    changed = force_float32_rope(Pipe(after), t)
    res["modules_demoted"] = changed
    try:
        cos, _ = after.r(batch_size=1, pos=16, device=t.device)
        res["after_fix"] = f"ok dtype={cos.dtype}"
    except Exception as exc:
        res["after_fix"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return res


guarded("force_float32_rope", rope)

print("\n===== SUMMARY JSON =====")
print(json.dumps(FINDINGS, indent=2, default=str))
