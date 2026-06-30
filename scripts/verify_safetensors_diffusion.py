"""B200 verification for the non-GGUF (safetensors) diffusion load path.

Drives the REAL Studio diffusion backend (core.inference.diffusion) end-to-end for
each curated safetensors model: resolve kind -> validate -> begin_load -> poll ->
generate -> save a PNG. Also asserts the Layered variant is rejected at load. Run
from anywhere; it puts the backend root on sys.path like the app does.

    HF_HOME=.../workspace_81/hf_home CUDA_VISIBLE_DEVICES=7 python scripts/verify_safetensors_diffusion.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "studio" / "backend"
sys.path.insert(0, str(BACKEND))

OUT = REPO / "outputs" / "safetensors_verify"
OUT.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")

# (repo_id, model_kind, single_file_name, steps, guidance)
MODELS = [
    ("unsloth/Z-Image-Turbo-unsloth-bnb-4bit", "pipeline", None, 9, 0.0),
    ("unsloth/Qwen-Image-2512-unsloth-bnb-4bit", "pipeline", None, 15, 4.0),
    ("unsloth/Qwen-Image-2512-FP8", "single_file", "qwen-image-2512-fp8.safetensors", 15, 4.0),
]

PROMPT = "a highly detailed photograph of a red panda wearing a tiny astronaut helmet, studio lighting"


def _vram_peak_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 3
    except Exception:
        pass
    return 0.0


def _reset_vram_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def verify_one(backend, resolve_model_kind, select_and_activate_engine, spec) -> dict:
    repo_id, model_kind, single_file, steps, guidance = spec
    result = {"repo": repo_id, "kind": model_kind, "ok": False}
    _reset_vram_peak()
    t0 = time.time()
    kind = resolve_model_kind(single_file, model_kind)
    fam = backend.validate_load_request(
        repo_id, gguf_filename=single_file, model_kind=kind
    )
    result["family"] = fam.name
    engine = select_and_activate_engine(fam, hf_token=HF_TOKEN, model_kind=kind)
    engine.begin_load(
        repo_id, gguf_filename=single_file, model_kind=kind, hf_token=HF_TOKEN
    )
    # Poll the background load.
    while True:
        prog = engine.load_progress()
        phase = prog.get("phase")
        if phase == "error":
            raise RuntimeError(f"load failed: {prog.get('error')}")
        if phase == "ready":
            break
        time.sleep(2.0)
    result["load_s"] = round(time.time() - t0, 1)
    result["load_vram_gb"] = round(_vram_peak_gb(), 2)

    g0 = time.time()
    out = engine.generate(
        prompt=PROMPT, width=1024, height=1024, steps=steps, guidance=guidance, seed=42
    )
    result["gen_s"] = round(time.time() - g0, 1)
    result["peak_vram_gb"] = round(_vram_peak_gb(), 2)
    img = out["images"][0]
    png = OUT / f"{repo_id.split('/')[-1]}.png"
    img.save(png)
    result["png"] = str(png)
    result["png_bytes"] = png.stat().st_size
    result["ok"] = png.stat().st_size > 1000
    engine.unload()
    return result


def main() -> int:
    from core.inference.diffusion import get_diffusion_backend, resolve_model_kind
    from core.inference.diffusion_engine_router import select_and_activate_engine

    backend = get_diffusion_backend()

    print("=== Layered reject check ===", flush=True)
    layered_rejected = False
    try:
        backend.validate_load_request(
            "unsloth/Qwen-Image-Layered-GGUF", gguf_filename="x.gguf"
        )
    except (ValueError, FileNotFoundError) as exc:
        layered_rejected = True
        print(f"  Layered correctly rejected at validate: {exc}", flush=True)
    if not layered_rejected:
        print("  FAIL: Layered was NOT rejected", flush=True)

    print("\n=== Non-unsloth reject check ===", flush=True)
    nonunsloth_rejected = False
    try:
        backend.validate_load_request("randomorg/Z-Image-bnb-4bit")
    except ValueError as exc:
        nonunsloth_rejected = True
        print(f"  Non-unsloth pipeline correctly rejected: {exc}", flush=True)
    if not nonunsloth_rejected:
        print("  FAIL: non-unsloth repo was NOT rejected", flush=True)

    results = []
    for spec in MODELS:
        print(f"\n=== {spec[0]} ({spec[1]}) ===", flush=True)
        try:
            r = verify_one(backend, resolve_model_kind, select_and_activate_engine, spec)
            print(f"  {r}", flush=True)
        except Exception as exc:  # noqa: BLE001
            r = {"repo": spec[0], "kind": spec[1], "ok": False, "error": str(exc)}
            print(f"  ERROR: {exc}", flush=True)
            traceback.print_exc()
            try:
                backend.unload()
            except Exception:
                pass
        results.append(r)

    print("\n=== SUMMARY ===", flush=True)
    print(f"  layered_rejected={layered_rejected} nonunsloth_rejected={nonunsloth_rejected}", flush=True)
    for r in results:
        status = "OK" if r.get("ok") else "FAIL"
        extra = (
            f"load={r.get('load_s')}s gen={r.get('gen_s')}s vram={r.get('peak_vram_gb')}GB png={r.get('png_bytes')}B"
            if r.get("ok")
            else r.get("error", "?")
        )
        print(f"  [{status}] {r['repo']} ({r['kind']}): {extra}", flush=True)

    all_ok = (
        layered_rejected
        and nonunsloth_rejected
        and all(r.get("ok") for r in results)
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
