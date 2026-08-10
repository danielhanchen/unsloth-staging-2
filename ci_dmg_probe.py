"""Probe: how far does the rendered DMG background drift from the committed asset?

Run inside an isolated venv with a specific Pillow/numpy build. Prints one JSON line so
the matrix driver can collect results. Also re-checks the stale-asset case, so a green
result means "matches the current asset AND still rejects the previous one".
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageSequence


REPO = Path(sys.argv[1]).resolve()
ASSET = REPO / "studio/src-tauri/dmg/background.tiff"
STALE = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def drift_against(asset: Path, expected) -> dict:
    pages = [p.convert("RGB") for p in ImageSequence.Iterator(Image.open(asset))]
    if [p.size for p in pages] != [p.size for p in expected]:
        return {"sizes_match": False}
    out = {"sizes_match": True, "max": [], "mean": [], "over2": []}
    for page, ref in zip(pages, expected):
        d = np.abs(np.asarray(page, dtype = np.int16) - np.asarray(ref, dtype = np.int16))
        out["max"].append(int(d.max()))
        out["mean"].append(round(float(d.mean()), 5))
        out["over2"].append(int((d > 2).sum()))
    return out


renderer = load_module(REPO / "scripts/make_dmg_background.py")
image = renderer.build()
expected = [
    image.resize((renderer.WIN_W, renderer.WIN_H), Image.LANCZOS).convert("RGB"),
    image.convert("RGB"),
]

# where the drift lives matters: the chevron is ImageDraw + a LANCZOS downscale, the
# rest is pure numpy, so a version-sensitive rasteriser shows up as chevron-only drift
hidpi = np.abs(
    np.asarray([p.convert("RGB") for p in ImageSequence.Iterator(Image.open(ASSET))][1], dtype = np.int16)
    - np.asarray(expected[1], dtype = np.int16)
)
chev = hidpi[280:400, 600:720]

result = {
    "python": ".".join(str(v) for v in sys.version_info[:3]),
    "pillow": PIL.__version__,
    "numpy": np.__version__,
    "current": drift_against(ASSET, expected),
    "chevron_region_max": int(chev.max()),
    "outside_chevron_max": int(np.delete(hidpi.reshape(-1), []).max()),
}
if STALE is not None:
    result["stale"] = drift_against(STALE, expected)
print("PROBE " + json.dumps(result, indent = 2))

if STALE is None:
    ok = result["current"]["sizes_match"] and max(result["current"]["max"]) <= 2
    print("current asset drift within tolerance:", ok)
else:
    ok = result["stale"]["sizes_match"] and max(result["stale"]["max"]) > 2
    print("stale asset rejected:", ok)
sys.exit(0 if ok else 1)
