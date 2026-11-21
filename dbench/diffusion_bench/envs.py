#!/usr/bin/env python3
"""Create or reuse the environments the benchmark cells run in. One venv per (profile, python, torch pin,
index URL, extra pins) under $WORKSPACE/temp/diffusion_bench/venvs/, reused on every later call: a fresh venv
pulls multi-GB wheels into page cache, so never build one per run.

Profiles:
  bench    torch + lpips / scikit-image / imageio: the scorer and the fake/diffusers backends
  studio   an Unsloth Studio backend you can import in process (setup_studio.py): a checkout at a path, a git
           ref of unslothai/unsloth, or the released PyPI package; also used to launch a Studio server
  blineui  a BlineUI clone at a pinned commit with its own venv (setup_blineui.py)
  sdcpp    stable-diffusion.cpp (unslothai fork) built for CUDA / HIP / Vulkan / CPU (setup_sdcpp.py)

Usage:
  python diffusion_bench/envs.py ensure bench
  python diffusion_bench/envs.py ensure studio --studio-src $WORKSPACE/unsloth
  python diffusion_bench/envs.py list
Prints a JSON line {"python": ..., ...} on success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

ROOT = C.WS / "temp" / "diffusion_bench"
VENVS = ROOT / "venvs"
SRC = ROOT / "src"

BENCH_PACKAGES = ["numpy", "pillow", "psutil", "nvidia-ml-py", "lpips", "scikit-image", "imageio", "imageio-ffmpeg", "safetensors",
                  "diffusers", "transformers", "accelerate", "huggingface_hub", "sentencepiece", "protobuf"]


def detect_torch_index() -> str:
    """The PyTorch wheel index matching this host's GPU stack. Override with DIFFUSION_BENCH_TORCH_INDEX."""
    env = os.environ.get("DIFFUSION_BENCH_TORCH_INDEX")
    if env:
        return env
    vendor = C.gpu_vendor()
    if vendor == "nvidia":
        try:
            out = C.smi([], timeout = 20).stdout
            cuda = float(out.split("CUDA Version:")[1].split()[0])
        except Exception:  # noqa: BLE001
            cuda = 12.8
        tag = "cu130" if cuda >= 13.0 else "cu128" if cuda >= 12.8 else "cu126"
        return f"https://download.pytorch.org/whl/{tag}"
    if vendor == "amd":
        return os.environ.get("DIFFUSION_BENCH_ROCM_INDEX", "https://download.pytorch.org/whl/rocm6.4")
    return "https://download.pytorch.org/whl/cpu"


def venv_key(profile: str, python: str, torch_spec: str, index: str, extra: list) -> str:
    raw = json.dumps([profile, python, torch_spec, index, sorted(extra)])
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def venv_python(path: Path) -> Path:
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def uv() -> str:
    exe = shutil.which("uv")
    if not exe:
        raise RuntimeError("uv is required (pip install uv, or https://docs.astral.sh/uv/)")
    return exe


def sh(cmd: list, **kw) -> None:
    C.log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check = True, **kw)


def ensure_venv(profile: str, python: str = "3.12", torch_spec: str = "torch", index: Optional[str] = None,
                packages: Optional[list] = None, extra: Optional[list] = None, name: Optional[str] = None,
                audio: bool = False) -> dict:
    """A venv with torch from ``index`` plus ``packages``; reused when the same key exists and imports torch.
    ``extra`` are more requirement strings that belong in the key (pins that change behaviour). ``audio`` adds
    torchaudio from the SAME index in the same resolve as torch, so it matches the torch build (a PyPI torchaudio
    can drag in a different torch); BlineUI's audio nodes (MiniMax-H3, LTX-2.x) need it."""
    index = index or detect_torch_index()
    packages, extra = list(packages or []), list(extra or [])
    key = venv_key(profile, python, torch_spec, index, packages + extra + (["+torchaudio"] if audio else []))
    path = VENVS / (name or f"{profile}-py{python}-{key}")
    py = venv_python(path)
    marker = path / ".diffusion_bench.json"
    if py.exists() and marker.exists():
        probe = "import torch, torchaudio" if audio else "import torch"
        ok = subprocess.run([str(py), "-c", probe], capture_output = True).returncode == 0
        if ok:
            return json.loads(marker.read_text())
    VENVS.mkdir(parents = True, exist_ok = True)
    if not py.exists():
        sh([uv(), "venv", "--seed", "--python", python, path])
    env = {**os.environ, "VIRTUAL_ENV": str(path)}
    env.pop("UV_EXCLUDE_NEWER", None)
    sh([uv(), "pip", "install", "--python", py, torch_spec, "torchvision", *(["torchaudio"] if audio else []), "--index-url", index,
        "--extra-index-url", "https://pypi.org/simple", "--index-strategy", "unsafe-best-match"], env = env)
    if packages + extra:
        sh([uv(), "pip", "install", "--python", py, *packages, *extra], env = env)
    info = {"profile": profile, "python": str(py), "venv": str(path), "torch_spec": torch_spec, "index": index,
            "packages": packages + extra, "torchaudio": audio}
    marker.write_text(json.dumps(info, indent = 1))
    return info


def ensure(profile: str, **kw) -> dict:
    if profile == "bench":
        return ensure_venv("bench", python = kw.get("python") or "3.12", torch_spec = kw.get("torch") or "torch",
                           index = kw.get("index"), packages = BENCH_PACKAGES)
    if profile == "studio":
        import setup_studio

        return setup_studio.ensure(**kw)
    if profile == "blineui":
        import setup_blineui

        return setup_blineui.ensure(**kw)
    if profile == "sdcpp":
        import setup_sdcpp

        return setup_sdcpp.ensure(**kw)
    raise KeyError(f"unknown profile {profile!r}")


def resolve_python(venv: Optional[str | dict], **kw) -> str:
    """What a cell's "venv" field names: a profile name, {"profile": ..., <kwargs>}, a venv dir, or a python
    executable. None means this interpreter."""
    if not venv:
        return sys.executable
    if isinstance(venv, dict):
        venv = dict(venv)
        return ensure(venv.pop("profile"), **{**kw, **venv})["python"]
    if venv in ("bench", "studio", "blineui", "sdcpp"):
        # A profile name first: run from an unsloth checkout, "studio" is also a relative directory.
        return ensure(venv, **kw)["python"]
    path = Path(C.expand_env(venv))
    if path.is_file():
        return str(path)
    if path.is_dir():
        return str(venv_python(path))
    return ensure(venv, **kw)["python"]


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest = "cmd", required = True)
    e = sub.add_parser("ensure")
    e.add_argument("profile", choices = ["bench", "studio", "blineui", "sdcpp"])
    e.add_argument("--python", default = None)
    e.add_argument("--torch", default = None, help = "torch requirement, e.g. 'torch==2.10.*'")
    e.add_argument("--index", default = None, help = "torch wheel index (default: detected from the GPU)")
    e.add_argument("--studio-src", default = None, help = "studio: checkout path, git ref (e.g. main, pr/11766), or 'pypi'")
    e.add_argument("--bline-commit", default = None)
    e.add_argument("--sdcpp-ref", default = None)
    e.add_argument("--sdcpp-backend", default = None, help = "cuda | hip | vulkan | cpu (default: detected)")
    sub.add_parser("list")
    args = ap.parse_args()
    if args.cmd == "list":
        for marker in sorted(VENVS.glob("*/.diffusion_bench.json")):
            print(marker.read_text().replace("\n", " "))
        return 0
    kw = {k: v for k, v in vars(args).items() if k not in ("cmd", "profile") and v is not None}
    kw = {k.replace("-", "_"): v for k, v in kw.items()}
    print(json.dumps(ensure(args.profile, **kw)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
