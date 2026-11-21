#!/usr/bin/env python3
"""BlineUI for the ``blineui`` backend: a clone at a pinned commit, its own venv, and per-cell model wiring.

  ensure(bline_commit=None, python=None, torch=None, index=None, kitchen=True, sage=False)
      -> {"python": <venv python>, "bline_dir": <clone>, "commit": <sha>, ...}

Clone: blineanonymous/BlineUI into $WORKSPACE/temp/diffusion_bench/src/BlineUI@<sha10>. ``bline_commit`` is a sha, a
tag, a branch, or "latest" (origin/master at call time); None is the release pin (default_ref): in order
$DIFFUSION_BENCH_BLINE_REF, models/BLINE_PIN.json ({"tag", "sha"}, the model-card owner's pin; sha wins), the newest
GitHub vX.Y.Z tag (resolved once, cached under temp/; not "latest release", which lags), then FALLBACK_TAG. The tag and sha land in every record.

Venv: envs.ensure_venv("blineui", ..., audio=True) with the clone's requirements.txt minus the torch family:
torch / torchvision / torchaudio all come from the detected wheel index in one resolve (a PyPI torchaudio can pull a
torch that does not match the index build; the MiniMax-H3 and LTX-2.x audio nodes need torchaudio). bline-kitchen
is in BlineUI's own requirements (the int8 / fp8 / nvfp4 kernels); ``sage=True`` adds sageattention.

Reuse instead of building: DIFFUSION_BENCH_BLINE_PYTHON=<python> and/or DIFFUSION_BENCH_BLINE_DIR=<clone>. Either
alone is honoured (the other is still resolved normally), so an existing clone can run in a fresh venv and
vice versa.

Models: ``wire_models(files, dest)`` symlinks the cell's files into <dest>/<folder>/ (diffusion_models,
text_encoders, vae, checkpoints, diffusers, loras) and writes <dest>/extra_model_paths.yaml with is_default: true,
passed to the server with --extra-model-paths-config, so the clone's own models/ tree is never written to and a
same-named file there cannot shadow the cell's.

Usage:
  python diffusion_bench/setup_blineui.py                    # clone + venv at the default commit, print JSON
  python diffusion_bench/setup_blineui.py --commit latest --plan
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import extnames as X  # noqa: E402


def repo_url() -> str:
    """The baseline UI's git URL ($BASELINE_REPO, else the private defaults; a public copy has no literal)."""
    return X.baseline_repo(True)


def repo_slug() -> str:
    """owner/name of the baseline repository, for the GitHub tags API."""
    return "/".join(repo_url().rstrip("/").removesuffix(".git").split("/")[-2:])
# Used only when the env, models/BLINE_PIN.json and the GitHub API all fail: the newest version tag on 2026-11-18
# (a277b5bf213e8a44fa34817b7770b54f86ef4cea; v0.45.6 = 45eb6e6f before Nov 18). GitHub's "latest release" lags the tags (v0.45.0 that day).
FALLBACK_TAG = "v0.46.0"
# The Sep 22 master tree (Qwen-Image-2.1 allopt round); kept so older specs can name it explicitly.
SEP22_COMMIT = "b5cc8830279eae909a59de030af1e50761c36751"
SRC = C.WS / "temp" / "diffusion_bench" / "src"
MIRROR = SRC / "BlineUI-mirror.git"
MODEL_FOLDERS = ("diffusion_models", "text_encoders", "vae", "checkpoints", "diffusers", "loras", "clip_vision",
                 "model_patches", "upscale_models", "latent_upscale_models")
_TORCH_FAMILY = re.compile(r"^(torch|torchvision|torchaudio)\s*([<>=!~;].*)?$", re.I)


def log(msg: str) -> None:
    C.log(f"[setup_blineui] {msg}")


def git(*args, capture: bool = False) -> str:
    cmd = ["git", *[str(a) for a in args]]
    if not capture:
        log("$ " + " ".join(cmd))
    out = subprocess.run(cmd, check = True, capture_output = capture, text = True)
    return (out.stdout or "").strip() if capture else ""


def _newest_version_tag() -> Optional[str]:
    """The highest vX.Y.Z tag of BlineUI (not GitHub's "latest release", which lags the tags), cached 6 h."""
    cache = SRC / "newest_tag.json"
    try:
        data = json.loads(cache.read_text())
        if time.time() - data["at"] < 6 * 3600:
            return data["tag"]
    except Exception:  # noqa: BLE001
        pass
    names: list = []
    if shutil.which("gh"):
        try:
            names = subprocess.run(["gh", "api", f"repos/{repo_slug()}/tags?per_page=100", "--jq", ".[].name"],
                                   capture_output = True, text = True, timeout = 30).stdout.split()
        except Exception:  # noqa: BLE001
            names = []
    if not names:
        try:
            import urllib.request

            with urllib.request.urlopen(f"https://api.github.com/repos/{repo_slug()}/tags?per_page=100",
                                        timeout = 20) as r:
                names = [t.get("name") for t in json.loads(r.read())]
        except Exception:  # noqa: BLE001
            names = []
    vers = [(tuple(int(x) for x in m.groups()), n) for n in names if n
            for m in [re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", n)] if m]
    tag = max(vers)[1] if vers else None
    if tag:
        SRC.mkdir(parents = True, exist_ok = True)
        cache.write_text(json.dumps({"tag": tag, "at": time.time()}))
    return tag


def default_ref() -> tuple:
    """(ref, source) of the BlineUI pin when a cell names none."""
    env = os.environ.get("DIFFUSION_BENCH_BLINE_REF")
    if env:
        return env, "env DIFFUSION_BENCH_BLINE_REF"
    pin = C.PKG / "models" / "BLINE_PIN.json"
    if pin.exists():
        try:
            data = json.loads(pin.read_text())
            ref = data.get("sha") or data.get("commit") or data.get("tag")
            if ref:
                return ref, f"models/BLINE_PIN.json ({data.get('tag') or ref})"
        except Exception:  # noqa: BLE001
            pass
    tag = _newest_version_tag()
    if tag:
        return tag, "newest GitHub version tag"
    return FALLBACK_TAG, "fallback constant (no env, no BLINE_PIN.json, GitHub unreachable)"


def tag_of(sha: str, repo: Optional[Path] = None) -> Optional[str]:
    """The release tag pointing at ``sha`` in the mirror (or ``repo``), if any."""
    try:
        return git("-C", repo or MIRROR, "describe", "--tags", "--exact-match", sha, capture = True) or None
    except Exception:  # noqa: BLE001
        return None


def resolve_commit(ref: Optional[str]) -> str:
    ref = ref or default_ref()[0]
    SRC.mkdir(parents = True, exist_ok = True)
    if not MIRROR.exists():
        git("clone", "--bare", "-q", repo_url(), MIRROR)
    want = "master" if ref in ("latest", "master") else ref
    if ref in ("latest", "master"):
        git("-C", MIRROR, "fetch", "-q", "origin", "+refs/heads/*:refs/heads/*", "--tags")
    for attempt in range(2):
        try:
            return git("-C", MIRROR, "rev-parse", "--verify", f"{want}^{{commit}}", capture = True)
        except subprocess.CalledProcessError:
            if attempt == 0:
                git("-C", MIRROR, "fetch", "-q", "origin", "+refs/heads/*:refs/heads/*", "--tags")
    git("-C", MIRROR, "fetch", "-q", "origin", want)
    return git("-C", MIRROR, "rev-parse", "--verify", "FETCH_HEAD^{commit}", capture = True)


def clone_dir(sha: str) -> Path:
    return SRC / f"BlineUI@{sha[:10]}"


def ensure_clone(ref: Optional[str]) -> tuple[Path, str]:
    sha = resolve_commit(ref)
    root = clone_dir(sha)
    if not (root / "main.py").exists():
        git("clone", "-q", "--no-checkout", MIRROR, root)
        git("-C", root, "remote", "set-url", "origin", repo_url())
        git("-C", root, "checkout", "-q", sha)
    return root, sha


def requirements(bline_dir: Path, sha: Optional[str] = None) -> list:
    """requirements.txt minus comments and the torch family (torch/torchvision come from the index). Read from the
    bare mirror at ``sha`` when the clone does not exist yet (plan mode)."""
    out = []
    path = bline_dir / "requirements.txt"
    text = path.read_text() if path.exists() else git("-C", MIRROR, "show", f"{sha}:requirements.txt", capture = True)
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not _TORCH_FAMILY.match(line):
            out.append(line)
    return out


def ensure(bline_commit: Optional[str] = None, python: Optional[str] = None, torch: Optional[str] = None,
           index: Optional[str] = None, kitchen: bool = True, sage: bool = False, plan: bool = False, **_) -> dict:
    # DIFFUSION_BENCH_BASELINE_*: the same, under the neutral names a public workflow uses (amd/gen_workflow.py)
    env_py = os.environ.get("DIFFUSION_BENCH_BLINE_PYTHON") or os.environ.get("DIFFUSION_BENCH_BASELINE_PYTHON")
    env_dir = os.environ.get("DIFFUSION_BENCH_BLINE_DIR") or os.environ.get("DIFFUSION_BENCH_BASELINE_DIR")
    if env_dir:
        bline_dir = Path(C.expand_env(env_dir)).resolve()
        if not (bline_dir / "main.py").exists():
            raise FileNotFoundError(f"DIFFUSION_BENCH_BLINE_DIR={env_dir}: no BlineUI main.py there")
        sha = C.git_rev(bline_dir) or "unknown"
        try:
            sha = git("-C", bline_dir, "rev-parse", "HEAD", capture = True)
        except Exception:  # noqa: BLE001
            pass
    elif plan:
        sha = resolve_commit(bline_commit)
        bline_dir = clone_dir(sha)
    else:
        bline_dir, sha = ensure_clone(bline_commit)
    pin_source = "cell bline.commit" if bline_commit else default_ref()[1]
    info = {"bline_dir": str(bline_dir), "commit": sha, "source": "env" if env_dir else "clone",
            "tag": tag_of(sha, bline_dir if env_dir else None) if (env_dir or MIRROR.exists()) else None, "pin_source": "env DIFFUSION_BENCH_BLINE_DIR" if env_dir
            else pin_source}
    if env_py:
        py = Path(C.expand_env(env_py))
        if not py.exists():
            raise FileNotFoundError(f"DIFFUSION_BENCH_BLINE_PYTHON={env_py} does not exist")
        return {**info, "python": str(py), "venv_source": "env"}
    import envs

    reqs = requirements(bline_dir, sha)
    if not kitchen:
        reqs = [r for r in reqs if not X.norm(r).lower().startswith("bline-kitchen")]
    extra = ["sageattention"] if sage else []
    if plan:
        return {**info, "python": None, "plan": {"profile": "blineui", "python": python or "3.12",
                                                "torch": torch or "torch", "torchaudio": True,
                                                "index": index or envs.detect_torch_index(),
                                                "packages": reqs, "extra": extra}}
    venv = envs.ensure_venv("blineui", python = python or "3.12", torch_spec = torch or "torch", index = index,
                            packages = reqs, extra = extra, audio = True)
    return {**info, "python": venv["python"], "venv": venv.get("venv"), "venv_source": "envs"}


# ---------------------------------------------------------------------------------------------- model wiring
def wire_models(files: dict, dest: Path) -> Path:
    """``files`` maps a BlineUI model folder to paths (a str or a list). Each path is symlinked into
    <dest>/<folder>/<basename> (a directory for ``diffusers``); returns the extra_model_paths.yaml to pass with
    --extra-model-paths-config. Missing sources raise FileNotFoundError naming the folder, before any server starts."""
    dest = Path(dest)
    for folder, paths in files.items():
        if not paths:
            continue
        folder = {"unet": "diffusion_models", "clip": "text_encoders"}.get(folder, folder)
        sub = dest / folder
        sub.mkdir(parents = True, exist_ok = True)
        for p in [paths] if isinstance(paths, (str, Path)) else paths:
            src = Path(C.expand_env(str(p)))
            if not src.exists():
                raise FileNotFoundError(f"BlineUI {folder} file not found: {src}")
            link = sub / X.ext(src.name)  # the name the graph asks for once submitted (extnames.ext)
            if link.is_symlink() or link.exists():
                if link.is_symlink() and Path(os.readlink(link)) == src.resolve():
                    continue
                link.unlink()
            link.symlink_to(src.resolve(), target_is_directory = src.is_dir())
    lines = ["diffusion_bench:", f"    base_path: {dest.resolve()}", "    is_default: true"]
    lines += [f"    {f}: {f}" for f in MODEL_FOLDERS]
    yaml = dest / "extra_model_paths.yaml"
    yaml.write_text("\n".join(lines) + "\n")
    return yaml


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", default = None, help = "sha / tag / branch / latest (default: the release pin, "
                    "see default_ref)")
    ap.add_argument("--python", default = None)
    ap.add_argument("--torch", default = None)
    ap.add_argument("--index", default = None)
    ap.add_argument("--no-kitchen", action = "store_true")
    ap.add_argument("--sage", action = "store_true", help = "add sageattention to the venv")
    ap.add_argument("--plan", action = "store_true", help = "resolve the commit and print the venv plan; build nothing")
    args = ap.parse_args()
    print(json.dumps(ensure(args.commit, args.python, args.torch, args.index, kitchen = not args.no_kitchen,
                            sage = args.sage, plan = args.plan), indent = 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
