"""Conventional video planner probe (one side): the tree's real VideoBackend.load_pipeline on the tree's own test fake
runtime (tests/test_video_backend.py fake_runtime), with spoofed device memory, capability and quant hooks, as
scripts/cloud_pred_video_sim.py. No weights load. Every hook is set with raising=False, so a tree that lacks one
still runs; a load that raises is a refused cell carrying the message.

    cd <src>/studio/backend && PLANNER_GRID=... PLANNER_OUT=out.json python -m pytest -q -p no:cacheprovider THIS
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.getcwd())

from tests.test_video_backend import (  # noqa: E402,F401  (fixtures, incl. the autouse one)
    _assume_the_restricted_load_is_available,
    fake_runtime,
)

GRID = json.loads(
    os.environ.get("PLANNER_GRID")
    or '{"vram_gib": [24], "memory": ["discrete"], "precisions": ["auto"]}'
)
CAP = (8, 9)
SUPPORTED = {"int8", "fp8"}
CASES = [
    ("Wan2.2-TI2V-5B", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
    ("Wan2.2-T2V-A14B", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"),
    ("HunyuanVideo-1.5-480p", "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"),
    ("LTX-2", "Lightricks/LTX-2"),
]
ROWS = []


def placement(plan):
    if plan is None:
        return "?"
    p = plan.offload_policy
    if p == "none":
        return "resident"
    if p == "group":
        return (
            "group:dit-streamed"
            if getattr(plan, "stream_transformer", True)
            else "group:dit-resident"
        )
    return str(p)


def _cell(monkeypatch, label, repo, gib, kind, req):
    import core.inference.video as V
    from core.inference import diffusion_transformer_quant as tq
    from core.inference.diffusion_device import DiffusionDeviceTarget
    from core.inference.diffusion_memory import DeviceMemory
    from core.inference.video import VideoBackend

    torch = sys.modules["torch"]  # the fake
    target = DiffusionDeviceTarget(
        device = "cuda",
        dtype = torch.bfloat16,
        backend = "cuda",
        vendor = "nvidia",
        supports_model_cpu_offload = True,
        supports_default_torch_compile = True,
        supports_pinned_transfer = True,
    )
    total = int(gib * 1024)
    mem = DeviceMemory(
        backend = "cuda",
        device = "cuda",
        memory_kind = "unified_memory" if kind == "unified" else "discrete_vram",
        free_mib = total - 500,
        total_mib = total,
    )
    for mod, name, val in (
        (V, "settled_snapshot_device_memory", lambda *a, **k: mem),
        (V, "snapshot_device_memory", lambda *a, **k: mem),
        (VideoBackend, "_device_target", lambda self, ordinal = None: target),
        (V, "resolve_diffusion_device_target", lambda *a, **k: target),
        (V, "dense_transformer_supported", lambda t: True),
        (tq, "dense_transformer_supported", lambda t: True),
        (tq, "_capability", lambda ordinal = None: CAP),
        (tq, "_scheme_supported", lambda s, d = None, unproven_ok = False: s in SUPPORTED),
        (tq, "_is_consumer_gpu", lambda d = None: gib <= 32),
        (tq, "native_quant_host", lambda t: False),
        (tq, "native_offload_host", lambda t: True),
        (V, "stored_denoiser_precision", lambda *a, **k: None),
        (V, "native_quant_reason", lambda m, s: f"native {s}"),
    ):
        monkeypatch.setattr(mod, name, val, raising = False)
    try:
        import core.inference.video_denoiser_prequant as vdp
        monkeypatch.setattr(
            vdp,
            "denoiser_prequant_pipe_kwargs",
            lambda fam, base_, *, scheme, **kw: {"transformer": object()},
            raising = False,
        )
    except ImportError:
        pass

    def _quantize(
        view,
        tgt,
        *,
        mode,
        family = None,
        logger = None,
        **kw,
    ):
        return (
            "int8"
            if kw.get("offload")
            else tq.select_transformer_quant_scheme(tgt, mode, family = family)
        )

    monkeypatch.setattr(V, "quantize_transformer", _quantize, raising = False)
    monkeypatch.setattr(
        V,
        "apply_speed_optims",
        lambda *a, **k: {"compiled": False, "compiled_dequant": False, "cuda_graph": False},
        raising = False,
    )
    plans = []

    def _apply(pipe, plan, *a, **k):
        plans.append(plan)
        return plan.offload_policy, plan.vae_tiling

    monkeypatch.setattr(V, "apply_memory_plan", _apply, raising = False)
    cell = {"engine": "video", "family": label, "vram_gib": gib, "memory": kind, "request": req}
    try:
        # unset is auto; the dense bf16 denoiser is what Speed "off" loads (an explicit scheme is honoured either way)
        status = VideoBackend().load_pipeline(
            repo,
            model_kind = "pipeline",
            transformer_quant = None if req in ("auto", "bf16") else req,
            **({"speed_mode": "off"} if req == "bf16" else {}),
        )
        plan = plans[-1] if plans else None
        cell.update(
            precision = (status or {}).get("transformer_quant") or "bf16",
            placement = placement(plan),
            admitted = True,
            reason = "; ".join(list(getattr(plan, "reasons", ()) or ())[:3])[:300],
        )
    except Exception as e:  # noqa: BLE001 - a refusal is the cell's result
        cell.update(
            precision = None,
            placement = "refused",
            admitted = False,
            reason = f"{type(e).__name__}: {str(e)[:200]}",
        )
    ROWS.append(cell)


@pytest.mark.parametrize("req", GRID["precisions"])
@pytest.mark.parametrize("kind", GRID["memory"])
@pytest.mark.parametrize("gib", GRID["vram_gib"])
@pytest.mark.parametrize("case", CASES, ids = [c[0] for c in CASES])
def test_cell(fake_runtime, monkeypatch, case, gib, kind, req):
    _cell(monkeypatch, case[0], case[1], gib, kind, req)


def teardown_module(module):
    out = os.environ.get("PLANNER_OUT")
    if out:
        with open(out, "w") as fh:
            json.dump(ROWS, fh, indent = 1, default = str)
