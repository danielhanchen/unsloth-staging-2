"""`diffusion_imports`: every image and video family's diffusers classes import under the side's installed pins.

    <install>/unsloth_studio/bin/python studio_regress/diffusion_imports.py --src <tree> --out <dir>

For each family in the tree's image (diffusion_families._FAMILIES) and video (video_families._FAMILIES) registries,
resolves every class name it declares (pipeline, transformer, transformer2, img2img / inpaint pipelines,
ControlNet pipeline + model) with getattr(diffusers, name), which is what imports the submodule, the way the
loaders do, after running the tree's import shims (`core/inference/*_import_compat.py` `ensure_*`) as Studio
does. A class that is missing, raises on import, or is diffusers' dummy placeholder (a backend it needs
is not installed) FAILs. One child imports them all; if it crashes or hangs, one child per family
isolates the culprit, so a hard crash costs one family's rows, not the run.

Writes <out>/results.json in the diffusion_bench shape ({"results": [{"surface", "check", "status", "error"}]})
so external.py's per-check base comparison applies. Exit 1 on any FAIL, 2 when the registries do not load.
A class the tree marks as loaded outside diffusers (a family whose module provides its own loader) is
labelled `known` when diffusers lacks it on this side, so it does not force a base run by itself.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CLASS_FIELDS = (
    "pipeline_class",
    "transformer_class",
    "transformer2_class",
    "img2img_pipeline_class",
    "inpaint_pipeline_class",
    "controlnet_pipeline_class",
    "controlnet_model_class",
)
CHILD_TIMEOUT_S = 300

LIST = r"""
import json, sys
sys.path.insert(0, ".")
from core.inference import diffusion_families as DF, video_families as VF
out = []
for engine, mod in (("image", DF), ("video", VF)):
    for f in mod._FAMILIES:
        names = {k: getattr(f, k, None) for k in %r}
        out.append({"engine": engine, "family": f.name, "classes": {k: v for k, v in names.items() if v}})
print(json.dumps(out))
""" % (CLASS_FIELDS,)

PROBE = r"""
import glob, importlib, inspect, json, os, sys, time
names = json.loads(sys.argv[1])
res = {}
sys.path.insert(0, ".")
# Studio runs its import shims (core/inference/*_import_compat.py, e.g. LTX-2's) before a loader resolves a
# class; the probe must too, or it reports failures no user can hit.
for path in sorted(glob.glob("core/inference/*_import_compat.py")):
    try:
        mod = importlib.import_module("core.inference." + os.path.basename(path)[:-3])
        for fname, fn in vars(mod).items():
            if fname.startswith("ensure_") and inspect.isfunction(fn) and fn.__module__ == mod.__name__:
                fn()
    except BaseException as e:
        print(f"shim {path} raised {type(e).__name__}: {e}", file=sys.stderr)
import diffusers
for n in names:
    t0 = time.time()
    try:
        cls = getattr(diffusers, n, None)
        if cls is None:
            res[n] = ("FAIL", f"diffusers {diffusers.__version__} has no {n}")
        elif type(cls).__name__ == "DummyObject" or getattr(cls, "_backends", None):
            res[n] = ("FAIL", f"{n} is a placeholder: needs {getattr(cls, '_backends', '?')}")
        else:
            res[n] = ("PASS", f"{cls.__module__} ({time.time() - t0:.1f}s)")
    except BaseException as e:
        res[n] = ("FAIL", f"{type(e).__name__}: {str(e)[:300]}")
print("RESULT " + json.dumps(res))
"""


def families(python, backend):
    p = subprocess.run(
        [python, "-c", LIST],
        cwd = str(backend),
        capture_output = True,
        text = True,
        timeout = CHILD_TIMEOUT_S,
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "")},
    )
    if p.returncode:
        raise RuntimeError(f"family registries did not load: {p.stderr.strip()[-800:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def probe(python, backend, names, timeout):
    """{name: (status, detail)} from one child importing `names`, or None when it crashed / hung."""
    try:
        p = subprocess.run(
            [python, "-c", PROBE, json.dumps(names)],
            cwd = str(backend),
            capture_output = True,
            text = True,
            timeout = timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT ")), None)
    return {n: tuple(v) for n, v in json.loads(line[7:]).items()} if line else None


def custom_loader_classes(backend):
    """Class names the tree assembles itself (a `load_<x>_pipeline` helper exists for the family), so diffusers
    lacking them is expected on an older pin."""
    known = set()
    for f in (backend / "core" / "inference").glob("*.py"):
        text = f.read_text(errors = "replace")
        if "def load_" in text and "_pipeline(" in text:
            for line in text.splitlines():
                if line.startswith("class ") and ("Pipeline" in line or "Transformer" in line):
                    known.add(line.split()[1].split("(")[0].rstrip(":"))
    return known


def main(argv = None):
    ap = argparse.ArgumentParser(description = __doc__.split("\n")[0])
    ap.add_argument("--src", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--python", default = sys.executable)
    a = ap.parse_args(argv)
    backend = Path(a.src) / "studio" / "backend"
    out = Path(a.out)
    out.mkdir(parents = True, exist_ok = True)
    t0 = time.time()
    try:
        fams = families(a.python, backend)
    except Exception as e:  # noqa: BLE001
        (out / "results.json").write_text(json.dumps({"error": str(e), "results": []}, indent = 1))
        print(e, file = sys.stderr)
        return 2
    custom = custom_loader_classes(backend)
    everything = sorted({n for fam in fams for n in fam["classes"].values()})
    res = probe(a.python, backend, everything, CHILD_TIMEOUT_S * 3)
    if res is None:  # the one-shot child crashed or hung: one child per family isolates the culprit
        res = {}
        for fam in fams:
            names = sorted(set(fam["classes"].values()))
            res.update(
                probe(a.python, backend, names, CHILD_TIMEOUT_S)
                or {n: ("FAIL", "import crashed or hung the interpreter") for n in names}
            )
    rows = []
    for fam in fams:
        for field, name in fam["classes"].items():
            status, detail = res.get(name, ("FAIL", "not probed"))
            row = {
                "surface": fam["engine"],
                "check": f"{fam['family']}:{field}={name}",
                "status": status,
                "tier": "fast",
                "wall_s": 0,
                "error": None if status == "PASS" else detail,
                "failures": [],
                "infos": {"detail": detail} if status == "PASS" else {},
                "evidence": {},
            }
            if status == "FAIL" and name in custom:
                row["known"] = "the tree provides this class itself (custom loader)"
            rows.append(row)
    fails = [r for r in rows if r["status"] == "FAIL"]
    try:  # from the backend (its shims / sys.path), bounded: a hanging import must not stall the target
        import_ok = subprocess.run(
            [
                a.python,
                "-c",
                "import diffusers, transformers; print(diffusers.__version__, "
                "transformers.__version__)",
            ],
            cwd = str(backend),
            capture_output = True,
            text = True,
            timeout = CHILD_TIMEOUT_S,
        ).stdout.strip()
    except subprocess.TimeoutExpired:
        import_ok = f"pins unknown: import timed out after {CHILD_TIMEOUT_S}s"
    (out / "results.json").write_text(
        json.dumps(
            {
                "meta": {
                    "pins": import_ok,
                    "wall_s": round(time.time() - t0, 1),
                    "families": len(fams),
                },
                "results": rows,
            },
            indent = 1,
        )
    )
    lines = [
        f"# diffusion_imports: {len(rows) - len(fails)} of {len(rows)} classes import ({import_ok})",
        "",
    ]
    lines += [
        f"- FAIL{' (known)' if r.get('known') else ''} {r['surface']} {r['check']}: {r['error']}"
        for r in fails
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
