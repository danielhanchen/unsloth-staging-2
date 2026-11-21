#!/usr/bin/env python3
"""Snapshot the hf-internal-testing tiny random pipelines the tiny tier reads into $WORKSPACE/hf_tiny/<name>.

The tiny tier (edge/run_edge.py --tier tiny, edge/edge_bline_sdcpp.py --tier tiny, specs/tiny_plumbing.json,
specs/studio_tiny.json) expects one directory per repo name under $WORKSPACE/hf_tiny. This fills it: public repos
only, no token (the ambient one is dropped), local_dir so nothing lands in ~/.cache/huggingface, and a repo that is
already complete is skipped. All 14 together are about 340 MB. DERIVED variants (below) are built locally from a
fetched pipe, never downloaded.

  python diffusion_bench/fetch_tiny.py                    # every pipe any tiny spec or check names
  python diffusion_bench/fetch_tiny.py --set edge         # only what run_edge.py --tier tiny needs
  python diffusion_bench/fetch_tiny.py --only tiny-wan-pipe --root /elsewhere --force
  python diffusion_bench/fetch_tiny.py --list

Exit 0 when every requested pipe is present, 1 when any download failed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

ORG = "hf-internal-testing"
# Edge suite (run_edge.py --tier tiny): the img_a / img_b / vid roles plus tiny.family_matrix and
# tiny.rope_overflow_isolated. Missing matrix rows are reported as "missing", not failed, so a partial set still runs.
EDGE_CORE = ["tiny-stable-diffusion-xl-pipe", "tiny-flux-pipe", "tiny-wan-pipe"]
EDGE_MATRIX = ["tiny-zimage-pipe", "tiny-qwenimage-pipe", "tiny-qwenimage21-pipe", "tiny-lumina2-pipe",
               "tiny-flux-kontext-pipe", "tiny-qwenimage-edit-pipe", "tiny-krea2-turbo-modular-pipe",
               "tiny-sd3-pipe", "tiny-sana-pipe", "tiny-random-hunyuanvideo", "tiny-cogvideox-pipe"]
SETS = {
    "core": EDGE_CORE,
    "edge": EDGE_CORE + EDGE_MATRIX,
    # specs/tiny_plumbing.json, specs/studio_tiny.json, specs/amd_strix_small.json tier 1
    "matrix": ["tiny-stable-diffusion-xl-pipe", "tiny-flux-pipe", "tiny-wan-pipe", "tiny-zimage-pipe",
               "tiny-qwenimage-pipe", "tiny-qwenimage21-pipe", "tiny-lumina2-pipe", "tiny-sd3-pipe"],
}
SETS["all"] = list(dict.fromkeys(SETS["edge"] + SETS["matrix"]))
DONE = ".fetch_complete"

# Local variants derived from a fetched pipe (never downloaded). tiny-wan-pipe ships rope_max_seq_len 32, so its
# rope table covers at most a 32 x 32 patch grid; Studio's HTTP /video route only accepts the family presets
# (1280x704 / 704x1280 for Wan2.2-TI2V-5B), a 44 x 80 grid, and diffusers raises "shape '[1, 44, 1, -1]' is invalid".
# The rope buffers are non-persistent (rebuilt from the config at load), so raising the limit changes no weight and
# leaves every position under 32 bit-identical: the same random pipe, now able to render a preset over HTTP.
DERIVED = {
    "tiny-wan-pipe-rope128": ("tiny-wan-pipe", {"transformer/config.json": {"rope_max_seq_len": 128}}),
    # Studio's offload int8 path (torchao-free W8A8) only takes linears at least DEFAULT_MIN_LINEAR_FEATURES (512)
    # wide; the hub tiny DiTs are 32 / 24 wide, so int8 under group or whole-module offload engages on 0 linears and
    # the load is refused. These rebuild the DiT 512 wide (random weights, seed 0, some MB); text encoders, VAE and
    # scheduler stay the fetched ones, so the context / pooled dims still match.
    "tiny-flux-pipe-w512": ("tiny-flux-pipe", {"transformer/config.json": {
        "attention_head_dim": 128, "num_attention_heads": 4, "axes_dims_rope": [32, 32, 64]}}, ("transformer",)),
    "tiny-wan-pipe-w512": ("tiny-wan-pipe", {"transformer/config.json": {
        "attention_head_dim": 128, "num_attention_heads": 4, "ffn_dim": 2048}}, ("transformer",)),
    # FluxKontextPipeline renders at max_area (1024^2) whatever the size asked, as real Kontext does. The hub tiny
    # VAE has one block (vae_scale_factor 1), so 1024^2 is ~524k joint tokens and a T4 (math SDPA) asks for 4 TiB.
    # Four blocks make it 8x like the real FLUX VAE (8k tokens); latent_channels stays 1, so the DiT is unchanged.
    "tiny-flux-kontext-pipe-vae8": ("tiny-flux-kontext-pipe", {"vae/config.json": {
        "block_out_channels": [4, 4, 4, 4], "down_block_types": ["DownEncoderBlock2D"] * 4,
        "up_block_types": ["UpDecoderBlock2D"] * 4}}, ("vae",)),
}
SETS["matrix"] += ["tiny-flux-pipe-w512", "tiny-wan-pipe-w512"]
for _set in ("core", "edge", "all"):
    SETS[_set] = SETS[_set] + [n for n, spec in DERIVED.items() if spec[0] in SETS[_set] and n not in SETS[_set]]


def fetch(name: str, root: Path, force: bool = False, retries: int = 4) -> dict:
    from huggingface_hub import snapshot_download

    target = root / name
    marker = target / DONE
    if marker.exists() and not force:
        return {"name": name, "status": "present", "path": str(target)}
    t0 = time.time()
    last = None
    for attempt in range(retries):
        try:
            rev = snapshot_download(repo_id = f"{ORG}/{name}", local_dir = str(target), token = False)
            last = None
            break
        except Exception as exc:  # noqa: BLE001 - hub 5xx happen; retry, then report
            last = f"{type(exc).__name__}: {str(exc)[:300]}"
            C.log(f"{name}: attempt {attempt + 1} failed: {last}")
            time.sleep(5 * (attempt + 1))
    if last:
        return {"name": name, "status": "failed", "error": last}
    marker.write_text(f"{ORG}/{name} {Path(rev).name if rev else ''} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts)
    return {"name": name, "status": "downloaded", "path": str(target), "mb": round(size / 1e6, 1),
            "s": round(time.time() - t0, 1)}


REBUILD = r"""
import json, sys
from pathlib import Path
import diffusers, torch
d = Path(sys.argv[1])
cfg = json.loads((d / "config.json").read_text())
torch.manual_seed(0)
model = getattr(diffusers, cfg["_class_name"]).from_config({k: v for k, v in cfg.items() if not k.startswith("_")})
for f in list(d.glob("*.safetensors")) + list(d.glob("*.bin")):
    f.unlink()
model.to(torch.bfloat16).save_pretrained(str(d), safe_serialization=True)
"""


def build_pythons() -> list:
    """Interpreters that may import diffusers + torch, in order: $DIFFUSION_BENCH_BUILD_PYTHON, this one, then the
    newest Studio install (a switchboard prefetch runs in a venv without diffusers): the workspace's, the run's
    shared store ($STUDIO_REGRESS_SHARED_DIR: $SB_CACHE/studio on a Colab / Kaggle VM), the host's."""
    out = [os.environ.get("DIFFUSION_BENCH_BUILD_PYTHON"), sys.executable]
    shared = os.environ.get("STUDIO_REGRESS_SHARED_DIR") or ""
    roots = [C.WS / "temp" / "studio_regress" / "installs"]
    if shared and shared.lower() not in ("off", "0", "none"):
        roots.append(Path(shared) / "installs")
    for root in roots + [Path("/mnt/disks/unslothai/shared/studio-regress-cache/installs")]:
        try:
            pys = [p for p in root.glob("*/unsloth_studio/bin/python") if os.access(p, os.X_OK)]
            pys.sort(key = lambda p: p.stat().st_mtime, reverse = True)
        except OSError:   # another user's install
            pys = []
        out += [str(p) for p in pys[:1]]
    return [p for p in dict.fromkeys(out) if p]


def rebuild_component(comp_dir: Path):
    """Random-init a component from its (patched) config and save it; None on success, else the last error."""
    import subprocess

    err = "no interpreter tried"
    for py in build_pythons():
        try:
            r = subprocess.run([py, "-c", REBUILD, str(comp_dir)], capture_output = True, text = True, timeout = 900,
                               env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        except (OSError, subprocess.TimeoutExpired) as e:
            err = f"{py}: {type(e).__name__}: {e}"
            continue
        if r.returncode == 0:
            return None
        err = f"{py}: {(r.stderr.strip().splitlines() or ['exit ' + str(r.returncode)])[-1][:300]}"
    return err


def derive(name: str, root: Path, force: bool = False) -> dict:
    """Build a DERIVED variant from its fetched source: a copy with a few config keys overridden."""
    import json
    import shutil

    src_name, patches, rebuild = (DERIVED[name] + ((),))[:3]
    src, target = root / src_name, root / name
    if not (src / DONE).exists():
        return {"name": name, "status": "failed", "error": f"{src_name} is not fetched"}
    if (target / DONE).exists() and not force:
        return {"name": name, "status": "present", "path": str(target)}
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(src, target, ignore = shutil.ignore_patterns(".cache", DONE))
    for rel, keys in patches.items():
        cfg = json.loads((target / rel).read_text())
        cfg.update(keys)
        (target / rel).write_text(json.dumps(cfg, indent = 2) + "\n")
    for comp in rebuild:   # new shapes need new weights: random init from the patched config
        err = rebuild_component(target / comp)
        if err:
            shutil.rmtree(target, ignore_errors = True)
            return {"name": name, "status": "failed", "error": f"rebuilding {comp}: {err}"}
    (target / DONE).write_text(f"derived from {src_name}: {json.dumps(patches)}"
                               + (f", rebuilt {list(rebuild)} (random, seed 0)" if rebuild else "") + "\n")
    return {"name": name, "status": "derived", "path": str(target), "from": src_name}


def ensure_aliases(models: dict, python: str | None = None, log=None) -> list:
    """Build every DERIVED pipe a spec's "models" table names whose local dir is missing (fetching its source
    first), with `python` as the build interpreter when given (matrix.py passes the cell venv, which has
    diffusers). A spec loads its aliases at once, so a missing derived pipe would fail every cell on both sides.
    -> [error strings] (empty: all present)."""
    errors = []
    for alias, entry in (models or {}).items():
        local = C.expand_env((entry or {}).get("local") or "")
        if not (entry or {}).get("derived") or not local or os.environ.get(C.model_env_key(alias)):
            continue
        target = Path(local)
        if (target / DONE).exists() or target.name not in DERIVED:
            continue
        root = target.parent
        os.environ.setdefault("HF_HOME", str(C.WS / "temp" / "hf_home"))   # as main(): never ~/.cache
        old = os.environ.get("DIFFUSION_BENCH_BUILD_PYTHON")
        if python:
            os.environ["DIFFUSION_BENCH_BUILD_PYTHON"] = python
        try:
            src = DERIVED[target.name][0]
            if not (root / src / DONE).exists():
                r = fetch(src, root)
                if r["status"] == "failed":
                    errors.append(f"{alias}: fetching {src}: {r.get('error')}")
                    continue
            r = derive(target.name, root)
            if r["status"] == "failed":
                errors.append(f"{alias}: {r.get('error')}")
            elif log:
                log(f"{alias}: {r['status']} {r.get('path', '')}")
        finally:
            if python:
                if old is None:
                    os.environ.pop("DIFFUSION_BENCH_BUILD_PYTHON", None)
                else:
                    os.environ["DIFFUSION_BENCH_BUILD_PYTHON"] = old
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default = str(C.WS / "hf_tiny"), help = "target root (default $WORKSPACE/hf_tiny)")
    ap.add_argument("--set", choices = sorted(SETS), default = "all")
    ap.add_argument("--only", action = "append", default = [], help = "repo name(s) under hf-internal-testing")
    ap.add_argument("--force", action = "store_true", help = "re-download even when complete")
    ap.add_argument("--list", action = "store_true")
    args = ap.parse_args()

    names = args.only or SETS[args.set]
    root = Path(args.root).resolve()
    if args.list:
        for n in names:
            src = f"derived from {DERIVED[n][0]}" if n in DERIVED else f"{ORG}/{n}"
            print(f"{n:<34} {'present' if (root / n / DONE).exists() else 'missing':<8} {src}")
        return 0
    C.scrub_tokens()
    os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)
    os.environ.pop("HF_HUB_TOKEN", None)
    # Keep the hub's own lock / metadata cache inside the workspace too, never ~/.cache.
    os.environ.setdefault("HF_HOME", str(C.WS / "temp" / "hf_home"))
    root.mkdir(parents = True, exist_ok = True)
    bad = 0
    for n in names:
        r = derive(n, root, force = args.force) if n in DERIVED else fetch(n, root, force = args.force)
        bad += r["status"] == "failed"
        C.log(" ".join(f"{k}={v}" for k, v in r.items()))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
