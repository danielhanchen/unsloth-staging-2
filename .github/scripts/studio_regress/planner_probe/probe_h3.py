"""MiniMax-H3 planner probe (one side): auto denoiser precision and the VRAM-floor admission per VRAM x memory kind x
precision, from the tree's own video.py / video_minimax_h3*.py (as scripts/cloud_pred/cloud_pred_h3_sim.py).
CPU only, no weights. Placement: "resident" when the pinned-denoiser floor fits the free memory, "streamed" when
only the streamed-denoiser floor does, else refused. Unified memory counts the same pool (no offload relief), so
only the resident floor applies.

    python probe_h3.py OUT.json         grid from $PLANNER_GRID, cwd = <src>/studio/backend
"""

from __future__ import annotations

import json
import os
import sys
import types

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402

# The tree's H3 API is resolved inside try: a renamed constant or module on one side must give ERR
# cells for that side (a PLAN_DIFF), not a crashed probe that VOIDs the whole target.
try:
    from core.inference import video as V
    from core.inference.video_families import detect_video_family
    from core.inference.video_minimax_h3 import (
        H3_TEXT_ENCODER_BF16_GB,
        estimate_h3_diffusers_vram_gb,
        h3_transformer_resident_gb,
    )
    from core.inference.video_minimax_h3_te import H3_TE_QUANT_DEFAULT, h3_te_resident_gb

    IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    IMPORT_ERROR = f"{type(e).__name__}: {str(e)[:200]}"

if IMPORT_ERROR is None:
    try:
        import core.inference.diffusion_prequant as P
        P.restricted_prequant_load_supported = lambda *a, **k: True
    except ImportError:
        pass
    V.video_family_prequant_available = lambda *a, **k: True
CAP = (8, 9)
SIZE = (960, 544, 124)  # the family default preset: width, height, frames


def floor(tq, streamed):
    kw = {
        "text_encoder_gb": h3_te_resident_gb(H3_TE_QUANT_DEFAULT, bf16_gb = H3_TEXT_ENCODER_BF16_GB),
        "transformer_gb": h3_transformer_resident_gb(tq),
        "transformer_pinned": tq is not None and not streamed,
    }
    if streamed:
        kw["transformer_streamed"] = True
    return estimate_h3_diffusers_vram_gb(*SIZE, **kw)


def _err_rows(grid, reason):
    return [
        {
            "engine": "h3",
            "family": "MiniMax-H3",
            "vram_gib": gib,
            "memory": kind,
            "request": req,
            "precision": None,
            "placement": "ERR",
            "admitted": False,
            "reason": reason,
        }
        for gib in grid["vram_gib"]
        for kind in grid["memory"]
        for req in grid["precisions"]
    ]


def main(out_path):
    grid = json.loads(os.environ["PLANNER_GRID"])
    if IMPORT_ERROR is not None:
        rows = _err_rows(grid, f"import: {IMPORT_ERROR}")
        with open(out_path, "w") as fh:
            json.dump(rows, fh, indent = 1, default = str)
        print(f"h3: {len(rows)} ERR cells (import failed) -> {out_path}")
        return
    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    rows = []
    for gib in grid["vram_gib"]:
        total = gib * 1024**3
        free = total - 500 * 1024**2
        target = types.SimpleNamespace(
            device = "cuda", dtype = torch.bfloat16, backend = "cuda", capability = CAP
        )
        V._h3_auto_precision_target = lambda t = None, _t = target: _t
        for kind in grid["memory"]:
            for req in grid["precisions"]:
                cell = {
                    "engine": "h3",
                    "family": "MiniMax-H3",
                    "vram_gib": gib,
                    "memory": kind,
                    "request": req,
                }
                try:
                    if req == "auto":
                        tq = V._h3_auto_denoiser_scheme(
                            fam,
                            target = target,
                            dtype = torch.bfloat16,
                            device = "cuda",
                            te_scheme = H3_TE_QUANT_DEFAULT,
                            task = "fl2va",
                            base_repo = "MiniMaxAI/MiniMax-H3",
                            speed_mode = None,
                            free_reader = lambda d, _v = free: _v,
                        )
                    else:
                        tq = None if req == "bf16" else req
                    free_gb = free / 1e9 + 0.25
                    if floor(tq, False) <= free_gb:
                        place = "resident"
                    elif kind != "unified" and floor(tq, True) <= free_gb:
                        place = "streamed"
                    else:
                        place = "refused"
                    cell.update(
                        precision = tq or "bf16",
                        placement = place,
                        admitted = place != "refused",
                        reason = f"floor {floor(tq, False):.1f} GB vs {free_gb:.1f} GB free",
                    )
                except Exception as e:  # noqa: BLE001
                    cell.update(
                        precision = None,
                        placement = "ERR",
                        admitted = False,
                        reason = f"{type(e).__name__}: {str(e)[:200]}",
                    )
                rows.append(cell)
    with open(out_path, "w") as fh:
        json.dump(rows, fh, indent = 1, default = str)
    print(f"h3: {len(rows)} cells -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
