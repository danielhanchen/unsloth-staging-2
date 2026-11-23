"""Studio's real load path on a local Qwen-Image-2.1 pipeline dir: DiffusionBackend.begin_load (default speed tier,
eager patches at their default), then generate(); records the q21_mps engagement counters and the image.
python studio_load.py --backend-dir B --model DIR --out OUT --tag T [--size 256 --steps 4]"""
import argparse
import base64
import io
import json
import logging
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--backend-dir", required = True)
ap.add_argument("--model", required = True)
ap.add_argument("--out", required = True)
ap.add_argument("--tag", required = True)
ap.add_argument("--size", type = int, default = 256)
ap.add_argument("--steps", type = int, default = 4)
ap.add_argument("--repeats", type = int, default = 3)
a = ap.parse_args()
os.makedirs(a.out, exist_ok = True)
sys.path.insert(0, os.path.abspath(a.backend_dir))
logging.basicConfig(level = logging.INFO, stream = sys.stdout, format = "%(asctime)s %(name)s %(message)s")
rec = {"tag": a.tag, "model": a.model, "env": {k: v for k, v in os.environ.items() if "Q21_MPS" in k or "EAGER" in k}}


def dump():
    with open(os.path.join(a.out, f"{a.tag}.json"), "w") as f:
        json.dump(rec, f, indent = 1, default = str)


import torch  # noqa: E402
from core.inference.diffusion import get_diffusion_backend  # noqa: E402

try:
    from core.inference import diffusion_qwenimage21_mps as fused
except ImportError:
    fused = None
rec["torch"] = torch.__version__
rec["mps"] = torch.backends.mps.is_available()
backend = get_diffusion_backend()
t = time.perf_counter()
backend.begin_load(a.model, family_override = "qwen-image-2.1", local_files_only = True)
while True:
    p = backend.load_progress()
    if p.get("phase") in ("ready", "error"):
        break
    time.sleep(0.5)
rec["load_s"] = round(time.perf_counter() - t, 1)
rec["progress"] = p
st = backend.status()
rec["status"] = {k: st.get(k) for k in ("device", "dtype", "family", "speed_mode", "effective_speed", "optimizations", "patches", "loaded") if k in st}
dump()
if p.get("phase") == "error":
    raise SystemExit(f"load failed: {p}")
pipe = backend._state.pipe if getattr(backend, "_state", None) is not None and hasattr(backend._state, "pipe") else None
tr = getattr(pipe, "transformer", None)
if tr is not None:
    blocks = getattr(tr, "transformer_blocks", [])
    rec["transformer_device"] = str(next(tr.parameters()).device)
    rec["blocks_patched"] = sum(1 for b in blocks if "_unsloth_q21_mps_block" in b.__dict__)
    rec["attn_prepare_set"] = sum(1 for b in blocks if "_unsloth_q21_mps_prepare" in b.attn.__dict__)
    rec["processor"] = type(blocks[0].attn.processor).__name__ if blocks else None
if fused is not None:
    rec["probes"] = dict(fused._PROBED)
gens = []
for i in range(a.repeats):
    before = dict(fused.STATS) if fused is not None else {}
    t = time.perf_counter()
    out = backend.generate(prompt = "a lighthouse on a cliff at sunset", width = a.size, height = a.size,
                           steps = a.steps, guidance = 1.0, seed = 7)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    dt = time.perf_counter() - t
    after = dict(fused.STATS) if fused is not None else {}
    gens.append({"s": round(dt, 3), "stats_delta": {k: after[k] - before.get(k, 0) for k in after}})
    imgs = out.get("images") or out.get("image") or []
    if isinstance(imgs, str):
        imgs = [imgs]
    if imgs and i == 0:
        img = imgs[0]
        if isinstance(img, str):
            from PIL import Image
            raw = img.split(",", 1)[1] if img.startswith("data:") else img
            img = Image.open(io.BytesIO(base64.b64decode(raw)))
        img.save(os.path.join(a.out, f"{a.tag}.png"))
    rec["gen_keys"] = sorted(out.keys())
    rec["generations"] = gens
    dump()
print(json.dumps(rec, indent = 1, default = str))
