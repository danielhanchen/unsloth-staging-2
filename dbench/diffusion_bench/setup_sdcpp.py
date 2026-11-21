#!/usr/bin/env python3
"""stable-diffusion.cpp (the unslothai/stable-diffusion.cpp fork Studio ships) for the ``sdcpp`` backend.

Three ways to get a binary, in the order ``ensure`` tries them:

  1. DIFFUSION_BENCH_SDCPP_BIN=<path to sd-cli, or a directory holding it>: reuse an existing build as is.
  2. method "source" (default): clone the fork at ``ref`` and build sd-cli + sd-server with CMake for one
     GPU backend into $WORKSPACE/temp/diffusion_bench/src/sdcpp-<sha10>-<backend>/build/bin. The flags follow
     the fork's own prebuilt workflow (.github/workflows/unsloth-sd-prebuilt.yml): Release, examples on,
     server frontend / webp / webm off, GGML_NATIVE off, plus the backend switch:
        cuda    -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<detected from nvidia-smi compute_cap, e.g. 100>
        hip     -DSD_HIPBLAS=ON -DGPU_TARGETS=<gfx from rocminfo> with ROCm's clang (docs/build.md)
        vulkan  -DSD_VULKAN=ON (needs libvulkan-dev + glslc)
        metal   -DSD_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON (macOS)
        cpu     no switch
     A finished build writes .diffusion_bench_sdcpp.json and is reused on every later call.
  3. method "prebuilt": download the fork's release zip for this host (Linux x86_64 cpu / cuda12, macOS, Windows
     cpu) for a release tag, the same assets Studio's install_sd_cpp_prebuilt.py installs.

The fork's master is the prebuilt's tree: the prebuilt workflow merges the PR pins in scripts/unsloth/pr-set.json
on top, and that list is empty at the pinned commit, so a source build of DEFAULT_REF is the same code as release
master-813-bfbef5b-u7bbffa3 (the last tag carrying Qwen-Image-2.1 support; u13b9d92 and older cannot load it).

Usage:
  python diffusion_bench/setup_sdcpp.py --backend cuda              # build (or reuse) and print JSON
  python diffusion_bench/setup_sdcpp.py --backend hip --plan        # print the commands without running them
  python diffusion_bench/setup_sdcpp.py --method prebuilt --ref master-813-bfbef5b-u7bbffa3
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

REPO = "unslothai/stable-diffusion.cpp"
REPO_URL = f"https://github.com/{REPO}"
# master on 2026-09-22 ("Pin every third-party action to a commit SHA"), == release master-813-bfbef5b-u7bbffa3.
DEFAULT_REF = "7bbffa35fc3571bcb9116212c60762f461e67e85"
DEFAULT_TAG = "master-813-bfbef5b-u7bbffa3"
SRC = C.WS / "temp" / "diffusion_bench" / "src"
MIRROR = SRC / "sdcpp-mirror.git"
MARKER = ".diffusion_bench_sdcpp.json"
BACKENDS = ("cuda", "hip", "vulkan", "metal", "cpu")


def log(msg: str) -> None:
    C.log(f"[setup_sdcpp] {msg}")


# ---------------------------------------------------------------------------------------------- detection
def detect_backend() -> str:
    if sys.platform == "darwin":
        return "metal"
    vendor = C.gpu_vendor()
    if vendor == "nvidia" and (shutil.which("nvcc") or Path("/usr/local/cuda/bin/nvcc").exists()):
        return "cuda"
    if vendor == "amd":
        return "hip"
    if shutil.which("glslc"):
        return "vulkan"
    return "cpu"


def cuda_archs() -> str:
    """CMAKE_CUDA_ARCHITECTURES for the GPUs on this host (10.0 -> 100), overridable with SDCPP_CUDA_ARCHS."""
    env = os.environ.get("SDCPP_CUDA_ARCHS")
    if env:
        return env
    try:
        out = C.smi(["--query-gpu=compute_cap", "--format=csv,noheader"], timeout = 20).stdout.split()
        caps = sorted({c.strip().replace(".", "") for c in out if c.strip()})
        if caps:
            return ";".join(caps)
    except Exception:  # noqa: BLE001
        pass
    return "native"


def gfx_target() -> Optional[str]:
    env = os.environ.get("SDCPP_GFX") or os.environ.get("GFX_NAME")
    if env:
        return env
    rocminfo = shutil.which("rocminfo") or "/opt/rocm/bin/rocminfo"
    try:
        out = subprocess.run([rocminfo], capture_output = True, text = True, timeout = 30).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "Name:" and parts[1].startswith("gfx") and parts[1][3:4] in "123456789":
                return parts[1]
    except Exception:  # noqa: BLE001
        pass
    return None


def cmake_flags(backend: str) -> tuple[list, dict]:
    """(cmake -D flags, extra env) for one backend, mirroring the fork's CI and docs/build.md."""
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DSD_BUILD_EXAMPLES=ON", "-DSD_SERVER_BUILD_FRONTEND=OFF",
             "-DSD_WEBP=OFF", "-DSD_WEBM=OFF", "-DGGML_NATIVE=OFF"]
    env: dict = {}
    if backend == "cuda":
        flags += ["-DSD_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs()}"]
        nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        if Path(nvcc).exists():
            flags.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")
    elif backend == "hip":
        gfx = gfx_target()
        if not gfx:
            raise RuntimeError("hip build: no gfx target (rocminfo found none); set SDCPP_GFX=gfx1151 or similar")
        rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
        cc, cxx = rocm / "llvm/bin/clang", rocm / "llvm/bin/clang++"
        flags += ["-DSD_HIPBLAS=ON", f"-DGPU_TARGETS={gfx}", f"-DAMDGPU_TARGETS={gfx}",
                  "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON", "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
                  f"-DCMAKE_C_COMPILER={cc if cc.exists() else 'clang'}",
                  f"-DCMAKE_CXX_COMPILER={cxx if cxx.exists() else 'clang++'}"]
    elif backend == "vulkan":
        if not shutil.which("glslc"):
            raise RuntimeError("vulkan build needs glslc and libvulkan-dev (apt install libvulkan-dev glslc spirv-headers)")
        flags += ["-DSD_VULKAN=ON"]
    elif backend == "metal":
        flags += ["-DSD_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON"]
    elif backend != "cpu":
        raise ValueError(f"unknown sd.cpp backend {backend!r}; one of {BACKENDS}")
    if shutil.which("ccache"):
        flags += ["-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"]
        if backend == "cuda":
            flags.append("-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    return flags, env


# ---------------------------------------------------------------------------------------------- binaries
def exe(name: str) -> str:
    return name + (".exe" if os.name == "nt" else "")


def locate(root: Path) -> dict:
    """sd-cli / sd-server under ``root`` (a build dir, an extracted prebuilt, or the binary's own directory)."""
    root = Path(root)
    if root.is_file():
        root = root.parent
    found: dict = {}
    for name, key in (("sd-cli", "bin"), ("sd-server", "server")):
        for cand in [root / exe(name), root / "build/bin" / exe(name), root / "bin" / exe(name)] + \
                sorted(root.glob(f"*/{exe(name)}")):
            if cand.is_file():
                found[key] = str(cand)
                break
    return found


def binary_backends(binary: str) -> list:
    """The ggml devices the binary sees (``--list-devices``), e.g. ['CUDA0', 'CPU']; [] when the flag is absent."""
    try:
        out = subprocess.run([binary, "--list-devices"], capture_output = True, text = True, timeout = 60,
                             env = runtime_env(binary))
        return [ln.split("\t")[0].strip() for ln in out.stdout.splitlines() if "\t" in ln]
    except Exception:  # noqa: BLE001
        return []


def runtime_env(binary: str) -> dict:
    """Prebuilt zips ship libcudart / libstable-diffusion next to the binary: put that directory on the loader path."""
    env = dict(os.environ)
    d = str(Path(binary).resolve().parent)
    key = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    env[key] = d + (os.pathsep + env[key] if env.get(key) else "")
    return env


def from_env() -> Optional[dict]:
    raw = os.environ.get("DIFFUSION_BENCH_SDCPP_BIN")
    if not raw:
        return None
    path = Path(C.expand_env(raw))
    found = locate(path) if path.exists() else {}
    if path.is_file() and "bin" not in found:
        found["bin"] = str(path)
    if "bin" not in found:
        raise FileNotFoundError(f"DIFFUSION_BENCH_SDCPP_BIN={raw}: no sd-cli there")
    return {"python": sys.executable, "method": "env", **found, "src": str(path if path.is_dir() else path.parent),
            "ref": None, "commit": None, "backend": None}


# ---------------------------------------------------------------------------------------------- source build
def git(*args, cwd: Optional[Path] = None, capture: bool = False) -> str:
    cmd = ["git", *[str(a) for a in args]]
    if not capture:
        log("$ " + " ".join(cmd))
    out = subprocess.run(cmd, cwd = cwd, check = True, capture_output = capture, text = True)
    return (out.stdout or "").strip() if capture else ""


def resolve_ref(ref: Optional[str]) -> str:
    """A full commit sha for ``ref`` (sha, branch, tag, or 'latest' = origin master), via a bare mirror clone."""
    ref = ref or DEFAULT_REF
    SRC.mkdir(parents = True, exist_ok = True)
    if not MIRROR.exists():
        git("clone", "--bare", "-q", REPO_URL, MIRROR)
    want = "master" if ref in ("latest", "master") else ref
    if ref in ("latest", "master"):
        git("-C", MIRROR, "fetch", "-q", "origin", "+refs/heads/*:refs/heads/*", "--tags")
    try:
        return git("-C", MIRROR, "rev-parse", "--verify", f"{want}^{{commit}}", capture = True)
    except subprocess.CalledProcessError:
        git("-C", MIRROR, "fetch", "-q", "origin", "+refs/heads/*:refs/heads/*", "--tags")
        try:
            return git("-C", MIRROR, "rev-parse", "--verify", f"{want}^{{commit}}", capture = True)
        except subprocess.CalledProcessError:
            git("-C", MIRROR, "fetch", "-q", "origin", want)
            return git("-C", MIRROR, "rev-parse", "--verify", "FETCH_HEAD^{commit}", capture = True)


def build_dir_for(sha: str, backend: str) -> Path:
    return SRC / f"sdcpp-{sha[:10]}-{backend}"


def plan_source(ref: Optional[str], backend: str, jobs: Optional[int] = None, sha: Optional[str] = None) -> dict:
    """The exact commands a source build runs (no side effects beyond resolving the ref when ``sha`` is None)."""
    sha = sha or resolve_ref(ref)
    root = build_dir_for(sha, backend)
    flags, env = cmake_flags(backend)
    gen = ["-G", "Ninja"] if shutil.which("ninja") else []
    jobs = jobs or max(1, min(64, (os.cpu_count() or 4) - 2))
    return {
        "ref": ref or DEFAULT_REF, "commit": sha, "backend": backend, "src": str(root),
        "commands": [
            ["git", "clone", "-q", "--no-checkout", str(MIRROR), str(root)],
            ["git", "-C", str(root), "remote", "set-url", "origin", REPO_URL],
            ["git", "-C", str(root), "checkout", "-q", sha],
            ["git", "-C", str(root), "submodule", "update", "--init", "--depth", "1", "ggml"],
            ["cmake", "-S", str(root), "-B", str(root / "build"), *gen, *flags],
            ["cmake", "--build", str(root / "build"), "--config", "Release", "-j", str(jobs), "--target", "sd-cli",
             "sd-server"],
        ],
        "env": env,
    }


def build_source(ref: Optional[str], backend: str, jobs: Optional[int] = None, force: bool = False) -> dict:
    sha = resolve_ref(ref)
    root = build_dir_for(sha, backend)
    marker = root / MARKER
    if marker.exists() and not force:
        info = json.loads(marker.read_text())
        if Path(info.get("bin", "")).exists():
            return info
    plan = plan_source(ref, backend, jobs, sha = sha)
    t0 = time.time()
    if not (root / ".git").exists():
        subprocess.run(plan["commands"][0], check = True)
        subprocess.run(plan["commands"][1], check = True)
    for cmd in plan["commands"][2:]:
        log("$ " + " ".join(cmd))
        subprocess.run(cmd, check = True, env = {**os.environ, **plan["env"]})
    found = locate(root / "build" / "bin")
    if "bin" not in found:
        raise RuntimeError(f"sd.cpp build finished but no sd-cli under {root / 'build/bin'}")
    info = {"python": sys.executable, "method": "source", **found, "src": str(root), "ref": plan["ref"],
            "commit": sha, "backend": backend, "cmake_flags": plan["commands"][4][5:],
            "build_s": round(time.time() - t0, 1), "devices": binary_backends(found["bin"])}
    marker.write_text(json.dumps(info, indent = 1))
    log(f"built sd.cpp {sha[:10]} ({backend}) in {info['build_s']}s -> {found['bin']}")
    return info


# ---------------------------------------------------------------------------------------------- prebuilt
def prebuilt_asset_label(backend: str) -> str:
    system, machine = platform.system(), platform.machine().lower()
    if system == "Linux" and machine in ("x86_64", "amd64"):
        return "Linux-Ubuntu-22.04-x86_64" + ("-cuda12" if backend == "cuda" else "")
    if system == "Linux":
        return "Linux-Ubuntu-24.04-aarch64"
    if system == "Darwin":
        return "Darwin-macOS-" + ("arm64" if machine == "arm64" else "x86_64")
    if system == "Windows":
        return "win-cpu-x64"
    raise RuntimeError(f"no fork prebuilt for {system}/{machine}")


def fetch_prebuilt(tag: Optional[str], backend: str) -> dict:
    tag = tag if tag and tag.startswith("master-") else DEFAULT_TAG
    if backend not in ("cuda", "cpu", "metal"):
        raise RuntimeError(f"the fork publishes no {backend} prebuilt; use method 'source'")
    label = prebuilt_asset_label(backend)
    root = SRC / f"sdcpp-{tag}-prebuilt-{label}"
    marker = root / MARKER
    if marker.exists():
        return json.loads(marker.read_text())
    name = f"sd-{tag}-bin-{label}.zip"
    url = f"{REPO_URL}/releases/download/{tag}/{name}"
    root.mkdir(parents = True, exist_ok = True)
    dest = root / name
    log(f"downloading {url}")
    with urllib.request.urlopen(url, timeout = 120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(dest) as zf:
        for member in zf.namelist():
            target = (root / member).resolve()
            if not str(target).startswith(str(root.resolve())):
                raise RuntimeError(f"refusing zip member outside the target: {member}")
        for info in zf.infolist():
            # zipfile writes a symlink member as a regular file holding the target path, which breaks the
            # lib*.so -> lib*.so.N links some bundles ship (Studio hit this as #9268): recreate them as links.
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                link = root / info.filename
                link.parent.mkdir(parents = True, exist_ok = True)
                target = zf.read(info).decode()
                if Path(target).is_absolute() or ".." in Path(target).parts:
                    raise RuntimeError(f"refusing symlink {info.filename} -> {target}")
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(target)
            else:
                zf.extract(info, root)
    dest.unlink()
    found = locate(root)
    if "bin" not in found:
        raise RuntimeError(f"{name} has no sd-cli")
    for key in ("bin", "server"):
        if key in found:
            os.chmod(found[key], 0o755)
    info = {"python": sys.executable, "method": "prebuilt", **found, "src": str(root), "ref": tag, "commit": None,
            "backend": backend, "asset": name, "devices": binary_backends(found["bin"])}
    marker.write_text(json.dumps(info, indent = 1))
    return info


# ---------------------------------------------------------------------------------------------- entry
def ensure(sdcpp_ref: Optional[str] = None, sdcpp_backend: Optional[str] = None, sdcpp_method: Optional[str] = None,
           jobs: Optional[int] = None, force: bool = False, **_) -> dict:
    """{"python", "bin", "server", "backend", "commit", ...}. ``python`` is this interpreter: the sdcpp backend
    only drives a subprocess, so any bench venv can run its cells."""
    env = from_env()
    if env:
        return env
    backend = sdcpp_backend or os.environ.get("DIFFUSION_BENCH_SDCPP_BACKEND") or detect_backend()
    method = sdcpp_method or os.environ.get("DIFFUSION_BENCH_SDCPP_METHOD") or "source"
    if method == "prebuilt":
        info = fetch_prebuilt(sdcpp_ref, backend)
    else:
        info = build_source(sdcpp_ref, backend, jobs = jobs, force = force)
    return {**info, "python": sys.executable}


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default = None, help = f"fork ref: sha, branch, tag or 'latest' (default {DEFAULT_REF[:10]})")
    ap.add_argument("--backend", default = None, choices = BACKENDS)
    ap.add_argument("--method", default = None, choices = ["source", "prebuilt"])
    ap.add_argument("--jobs", type = int, default = None)
    ap.add_argument("--force", action = "store_true", help = "rebuild even when a finished build exists")
    ap.add_argument("--plan", action = "store_true", help = "print the build commands, run nothing")
    args = ap.parse_args()
    backend = args.backend or detect_backend()
    if args.plan:
        print(json.dumps(plan_source(args.ref, backend, args.jobs), indent = 1))
        return 0
    print(json.dumps(ensure(args.ref, backend, args.method, args.jobs, args.force)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
