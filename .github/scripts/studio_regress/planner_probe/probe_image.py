"""Image planner probe (one side): what Studio's image loader would pick per family x VRAM x memory kind x precision.

Run by planner_matrix.py with the side's Studio python, cwd = <src>/studio/backend. CPU only, no weights:
torch.cuda, the device-memory snapshots, the scheme support probe and the cache scans are spoofed, the rest is
the tree's own planner (DiffusionBackend._plan_memory / _bf16_table_plan / _seeded_pipeline_plan,
diffusion_auto_policy.estimate_dense_quant, diffusion_transformer_quant.select_transformer_quant_scheme).
The auto path mirrors load_pipeline's in-place quant gate the way the cloud prediction sim
(scripts/cloud_pred_img_sim.py) does. A cell whose planner call raises is recorded as ERR with the exception.

    python probe_image.py OUT.json        grid from $PLANNER_GRID (JSON: {"vram_gib": [...], "memory": [...],
                                          "precisions": [...]})
"""

from __future__ import annotations

import json
import os
import sys
import types

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402

from core.inference import diffusion as D  # noqa: E402
from core.inference import diffusion_auto_policy as AP  # noqa: E402
from core.inference import diffusion_memory as M  # noqa: E402
from core.inference import diffusion_transformer_quant as TQ  # noqa: E402
from core.inference.diffusion_families import _FAMILIES  # noqa: E402

CAP = (
    8,
    9,
)  # an Ada consumer card: int8 and fp8 both supported, so an explicit pick is never refused for support
SUPPORTED = {"int8", "fp8"}
torch.cuda.get_device_capability = lambda *a, **k: CAP
torch.cuda.is_available = lambda: True
TQ._scheme_supported = lambda scheme, device = None, unproven_ok = False: scheme in SUPPORTED
TQ._is_consumer_gpu = lambda *a, **k: True
if hasattr(AP, "_hf_cache_free_mib"):
    AP._hf_cache_free_mib = lambda: 10**7
MEM = [None]
for mod in (D, M):
    for fn in (
        "snapshot_device_memory",
        "settled_snapshot_device_memory",
        "reclaimable_snapshot_device_memory",
    ):
        if hasattr(mod, fn):
            setattr(mod, fn, lambda target, *a, **k: MEM[0])
target = types.SimpleNamespace(
    device = "cuda",
    dtype = torch.bfloat16,
    backend = "cuda",
    supports_model_cpu_offload = True,
    supports_default_torch_compile = True,
    ordinal = None,
    capability = CAP,
    vendor = "nvidia",
    supports_pinned_transfer = True,
)
B = D.DiffusionBackend.__new__(D.DiffusionBackend)
B._resolve_device_target = lambda fam, *a, **k: target
B._target_for_ordinal = lambda fam, ordinal: target
B._cache_bytes = lambda *a, **k: 0
B._released_transformer_cached = lambda *a, **k: False
FAM = {f.name: f for f in _FAMILIES}

# (label, family, base repo): the catalog's pipeline rows whose bf16 component table is known.
ROWS = [
    ("Z-Image-Turbo", "z-image", "Tongyi-MAI/Z-Image-Turbo"),
    ("Qwen-Image-2.1", "qwen-image-2.1", "Qwen/Qwen-Image-2.1"),
    ("Qwen-Image-2512", "qwen-image", "Qwen/Qwen-Image-2512"),
    ("FLUX.1-schnell", "flux.1", "black-forest-labs/FLUX.1-schnell"),
    ("Krea-2-Turbo", "krea-2", "krea/Krea-2-Turbo"),
    ("Lumina-Image-2.0", "lumina-2", "Alpha-VLLM/Lumina-Image-2.0"),
    ("HunyuanImage-2.1", "hunyuanimage-2.1", "hunyuanvideo-community/HunyuanImage-2.1-Diffusers"),
    ("HiDream-I1-Full", "hidream-i1", "HiDream-ai/HiDream-I1-Full"),
]


def placement(plan):
    if plan is None:
        return "?"
    p = plan.offload_policy
    if p == M.OFFLOAD_NONE:
        return "resident"
    if p == M.OFFLOAD_GROUP:
        return (
            "group:dit-streamed"
            if getattr(plan, "stream_transformer", True)
            else "group:dit-resident"
        )
    return str(p)


def _overrides(est, fam, base):
    fn = getattr(B, "_candidate_companion_overrides", None)
    return fn(est, fam, base, target, None) if fn else {}


def _replan(fam, base, est):
    return B._plan_memory(
        target,
        None,
        base,
        fam,
        None,
        False,
        kind = "pipeline",
        repo_id = base,
        fetch_base = base,
        transformer_resident_override_mib = est.steady_transformer_mib,
        **_overrides(est, fam, base),
    )


def auto(fam, base):
    """(precision, plan) the auto load picks (cloud_pred_img_sim.pipeline_path, reduced)."""
    seed = B._pipeline_planned_denoiser_scheme(
        fam,
        base = base,
        kind = "pipeline",
        transformer_quant = None,
        speed_mode = None,
        repo_id = base,
        fetch_base = base,
    )
    declined = getattr(D, "PIPELINE_SEED_DECLINED", object())
    if seed not in (None, declined):
        sp = B._seeded_pipeline_plan(
            seed, target, base, fam, None, False, repo_id = base, base_local_dir = None, fetch_base = base
        )
        if sp is not None and M.plan_keeps_transformer_resident(sp):
            return f"{seed} (hosted)", sp
    plan = B._bf16_table_plan(
        target, fam, base, None, False, kind = "pipeline", repo_id = base, fetch_base = base
    )
    if plan is None:
        return "bf16", None
    keep = getattr(D, "_auto_keeps_bf16_reason", None)
    eager = getattr(D, "_auto_quant_eager_reason", None)
    if keep and eager and keep(fam, "pipeline") is not None and eager(fam, plan, None, "pipeline"):
        return "bf16", plan
    unc = getattr(D, "_pipeline_quant_uncompilable_reason", None)
    if unc and unc(target, fam, None, model_kind = "pipeline") is not None:
        return "bf16", plan
    scheme = TQ.select_transformer_quant_scheme(target, "auto", family = fam.name, base_repo = base)
    est = AP.estimate_dense_quant(fam, scheme, base_repo = base) if scheme else None
    if est is None:
        return "bf16", plan
    if plan.offload_policy != M.OFFLOAD_NONE:
        rp = _replan(fam, base, est)
        if M.plan_keeps_transformer_resident(rp):
            return scheme, rp
        if plan.offload_policy != M.OFFLOAD_MODEL:
            return "bf16", plan  # torchao declines a streamed DiT: the load keeps bf16
    return scheme, plan


def explicit(fam, base, scheme):
    if scheme == "bf16":
        return "bf16", B._bf16_table_plan(
            target, fam, base, None, False, kind = "pipeline", repo_id = base, fetch_base = base
        )
    if scheme not in SUPPORTED:
        raise RuntimeError(f"refused: {scheme} unsupported on sm_{CAP[0]}{CAP[1]}")
    est = AP.estimate_dense_quant(fam, scheme, base_repo = base)
    if est is None:
        raise RuntimeError(f"refused: no {scheme} estimate for {fam.name}")
    return scheme, _replan(fam, base, est)


def main(out_path):
    grid = json.loads(os.environ["PLANNER_GRID"])
    rows = []
    for label, famname, base in ROWS:
        fam = FAM.get(famname)
        for gib in grid["vram_gib"]:
            total = int(gib * 1024)
            for kind in grid["memory"]:
                MEM[0] = M.DeviceMemory(
                    "cuda",
                    "cuda",
                    "unified_memory" if kind == "unified" else "discrete_vram",
                    total - 500,
                    total,
                )
                for req in grid["precisions"]:
                    cell = {
                        "engine": "image",
                        "family": label,
                        "vram_gib": gib,
                        "memory": kind,
                        "request": req,
                    }
                    try:
                        if fam is None:
                            raise LookupError(f"family {famname} not in this tree")
                        prec, plan = auto(fam, base) if req == "auto" else explicit(fam, base, req)
                        cell.update(
                            precision = prec,
                            placement = placement(plan),
                            admitted = plan is not None,
                            reason = "; ".join(list(plan.reasons)[:3])[:300]
                            if plan is not None
                            else "no size table",
                        )
                    except Exception as e:  # noqa: BLE001 - a planner failure is a cell, not a crash
                        refused = str(e).startswith("refused")
                        cell.update(
                            precision = None,
                            placement = "refused" if refused else "ERR",
                            admitted = False,
                            reason = f"{type(e).__name__}: {str(e)[:200]}",
                        )
                    rows.append(cell)
    with open(out_path, "w") as fh:
        json.dump(rows, fh, indent = 1, default = str)
    print(f"image: {len(rows)} cells -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
