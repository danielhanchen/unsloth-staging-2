#!/usr/bin/env python3
"""Resolve an Unsloth Studio source tree, build (or reuse) the venv its diffusion / video backends import in,
and launch or attach to a real Studio server for the HTTP backend and the edge suite.

Sources (``studio_src``):
  <path>             a local checkout of unslothai/unsloth (or its studio/backend directory)
  main | <branch>    a branch of https://github.com/unslothai/unsloth, fetched into
                     $WORKSPACE/temp/diffusion_bench/src/unsloth with one worktree per resolved commit
  pr/<N>             the head of pull request N
  <sha>              a commit (full or abbreviated, when the remote can resolve it)
  pypi               the released ``unsloth`` wheel, which bundles studio/backend under site-packages
  None               $DIFFUSION_BENCH_STUDIO_SRC, else ``main``

Venv: ``DIFFUSION_BENCH_STUDIO_PYTHON`` reuses an existing interpreter as is (nothing installed). Otherwise one
venv per (python, torch pin, index, Studio requirement-file contents) via envs.ensure_venv, then Studio's own
requirement files from the resolved tree (the same files install_python_stack.py installs), Diffusers main or the
release pin (``DIFFUSION_BENCH_STUDIO_DIFFUSERS=pin``), and the torchao build Studio's installer pairs with the
installed torch.

Server: ``launch()`` starts a Studio API server from the tree with its own UNSLOTH_STUDIO_HOME under
$WORKSPACE/temp, through studio_test_kit.lifecycle.launch_studio (a two-line ``bin/unsloth`` shim maps the kit's
``unsloth studio -p PORT`` onto ``python studio/backend/run.py``); a home that already holds a real install (from
lifecycle.install_studio or install.sh) is launched with its own CLI. ``attach()`` talks to a running Studio or
Unsloth Desktop from DIFFUSION_BENCH_STUDIO_URL plus _USER / _PASSWORD or _TOKEN.

CLI:
  python diffusion_bench/setup_studio.py tree --studio-src pr/11766
  python diffusion_bench/setup_studio.py ensure --studio-src main [--dry-run]
  python diffusion_bench/setup_studio.py launch --studio-src $WORKSPACE/unsloth --gpu 5     # prints JSON, leaves it up
  python diffusion_bench/setup_studio.py stop --home <home printed by launch>
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

ROOT = C.WS / "temp" / "diffusion_bench"
SRC = ROOT / "src"
HOMES = ROOT / "studio_homes"
UPSTREAM = "https://github.com/unslothai/unsloth"
KIT = Path(__file__).resolve().parent.parent / "studio_test_kit"
DEFAULT_USER = "unsloth"

# Beyond Studio's own requirement files: what the bench records and the edge suite need in the same venv.
BENCH_EXTRAS = ["numpy", "pillow", "psutil", "nvidia-ml-py", "imageio", "imageio-ffmpeg", "av"]


# ---------------------------------------------------------------------------------------------- source tree
def _git(*args, cwd: Optional[Path] = None, check: bool = True, timeout: int = 900) -> str:
    out = subprocess.run(["git", *[str(a) for a in args]], cwd = str(cwd) if cwd else None, capture_output = True,
                         text = True, timeout = timeout)
    if check and out.returncode != 0:
        raise RuntimeError(f"git {' '.join(map(str, args))} failed: {out.stderr.strip()[:800]}")
    return out.stdout.strip()


def _backend_of(path: Path) -> Optional[Path]:
    """studio/backend under a checkout, or the path itself when it already is one."""
    for cand in (path / "studio" / "backend", path):
        if (cand / "core" / "inference" / "diffusion.py").exists():
            return cand
    return None


def _pypi_backend(python: str) -> Path:
    code = ("import importlib.util, os; s = importlib.util.find_spec('studio'); "
            "print(os.path.dirname(s.origin) if s and s.origin else (list(s.submodule_search_locations)[0] if s else ''))")
    out = subprocess.run([python, "-c", code], capture_output = True, text = True, timeout = 120,
                         env = _clean_env()).stdout.strip()
    backend = Path(out) / "backend" if out else None
    if not backend or not (backend / "core" / "inference" / "diffusion.py").exists():
        raise FileNotFoundError(f"{python} has no bundled studio/backend (is the released unsloth wheel installed?)")
    return backend


def _clone() -> Path:
    repo = SRC / "unsloth"
    if not (repo / ".git").exists():
        SRC.mkdir(parents = True, exist_ok = True)
        C.log(f"cloning {UPSTREAM} -> {repo} (blobless)")
        # filter-source: UPSTREAM is the github https constant.
        _git("clone", "--filter=blob:none", "--no-checkout", UPSTREAM, repo, timeout = 3600)
    return repo


def _fetch_ref(repo: Path, ref: str) -> str:
    """The commit ``ref`` names on the upstream, fetched into the shared clone."""
    m = re.fullmatch(r"(?:pr/|pull/|#)?(\d+)(?:/head)?", ref) if ref.lower().startswith(("pr/", "pull/", "#")) else None
    if m:
        local = f"refs/remotes/origin/pr/{m.group(1)}"
        _git("fetch", "--force", "origin", f"pull/{m.group(1)}/head:{local}", cwd = repo)
        return _git("rev-parse", local, cwd = repo)
    if re.fullmatch(r"[0-9a-f]{7,40}", ref):
        try:
            return _git("rev-parse", "--verify", f"{ref}^{{commit}}", cwd = repo)
        except RuntimeError:
            _git("fetch", "origin", ref, cwd = repo, check = False)
            try:
                return _git("rev-parse", "--verify", f"{ref}^{{commit}}", cwd = repo)
            except RuntimeError:
                _git("fetch", "origin", cwd = repo)
                return _git("rev-parse", "--verify", f"{ref}^{{commit}}", cwd = repo)
    _git("fetch", "--force", "origin", f"{ref}:refs/remotes/origin/{ref}", cwd = repo)
    return _git("rev-parse", f"refs/remotes/origin/{ref}", cwd = repo)


def resolve_tree(studio_src: Optional[str] = None, python: Optional[str] = None) -> dict:
    """{"tree", "studio_backend", "rev", "mode"} for ``studio_src`` without touching any venv (``pypi`` needs the
    ``python`` whose site-packages holds the wheel)."""
    src = studio_src or os.environ.get("DIFFUSION_BENCH_STUDIO_SRC") or "main"
    src = C.expand_env(str(src))
    if src == "pypi":
        py = python or os.environ.get("DIFFUSION_BENCH_STUDIO_PYTHON") or sys.executable
        backend = _pypi_backend(py)
        ver = subprocess.run([py, "-c", "from importlib.metadata import version; print(version('unsloth'))"],
                             capture_output = True, text = True, env = _clean_env()).stdout.strip()
        tree = backend.parent.parent
        if "site-packages" not in str(backend):  # an editable install of a checkout, not the released wheel
            return {"tree": str(tree), "studio_backend": str(backend), "rev": C.git_rev(tree) or f"editable-{ver}",
                    "mode": "installed-editable", "version": ver}
        return {"tree": str(tree), "studio_backend": str(backend), "rev": f"pypi-{ver or '?'}", "mode": "pypi",
                "version": ver}
    path = Path(src).expanduser()
    if not path.exists() and (src.startswith(("/", "./", "../", "~", "$")) or "\\" in src or path.suffix == ".py"):
        # A path-shaped source that is not on disk: never hand it to `git fetch` as a ref name
        # (that built `git fetch origin /abs/x:refs/remotes/origin//abs/x`).
        raise FileNotFoundError(f"studio source {src!r} looks like a local path but does not exist "
                                f"(cwd {os.getcwd()}); pass an existing checkout, or main / <branch> / pr/<N> / <sha>")
    if path.exists():
        backend = _backend_of(path.resolve())
        if backend is None:
            raise FileNotFoundError(f"{path} is neither an unsloth checkout nor a studio/backend directory")
        tree = backend.parent.parent if backend.name == "backend" and backend.parent.name == "studio" else backend
        return {"tree": str(tree), "studio_backend": str(backend), "rev": C.git_rev(tree) or "unknown", "mode": "path"}
    if subprocess.run(["git", "check-ref-format", "--allow-onelevel", src], capture_output = True).returncode != 0 \
            and not re.fullmatch(r"(?:pr/|pull/|#)\d+(?:/head)?", src):
        raise ValueError(f"studio source {src!r} is not an existing path nor a valid git ref "
                         f"(main / <branch> / pr/<N> / <sha>)")
    repo = _clone()
    sha = _fetch_ref(repo, src)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", src)[:40]
    wt = SRC / "worktrees" / f"{safe}-{sha[:10]}"
    if not (wt / ".git").exists():
        C.log(f"worktree {src} @ {sha[:10]} -> {wt}")
        _git("worktree", "add", "--detach", "--force", wt, sha, cwd = repo, timeout = 3600)
    backend = _backend_of(wt)
    if backend is None:
        raise FileNotFoundError(f"{src} @ {sha[:10]} has no studio/backend")
    return {"tree": str(wt), "studio_backend": str(backend), "rev": sha[:10], "mode": "git", "ref": src}


# ---------------------------------------------------------------------------------------------- venv
def _clean_env(extra: Optional[dict] = None) -> dict:
    """The parent's environment minus what would leak into a Studio process: tokens, a PYTHONPATH that can shadow
    Studio's own top-level packages (``models``, ``core``, ``utils``), and the parent's venv."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV", "HF_TOKEN",
                                                              "HUGGING_FACE_HUB_TOKEN", "PYTHONHOME")}
    env.update(extra or {})
    return env


def _python_works(py: str) -> bool:
    try:
        return subprocess.run([py, "-c", "import torch, diffusers, fastapi"], capture_output = True, timeout = 300,
                              env = _clean_env()).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _torchao_spec(tree: Path, torch_version: str) -> str:
    """Studio's own torch -> torchao mapping, read out of studio/install_python_stack.py without importing it
    (the module runs an installer at import). Falls back to the unpinned package."""
    src = tree / "studio" / "install_python_stack.py"
    try:
        mod = ast.parse(src.read_text())
        keep = []
        for node in mod.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith("_TORCHAO")
                                                    for t in node.targets):
                keep.append(node)
            if isinstance(node, ast.FunctionDef) and node.name in ("_select_torchao_spec", "_cuda_major_from_torch_version"):
                keep.append(node)
        ns: dict = {"re": re}
        exec(compile(ast.Module(body = keep, type_ignores = []), str(src), "exec"), ns)  # noqa: S102
        return ns["_select_torchao_spec"](torch_version)
    except Exception:  # noqa: BLE001
        return "torchao"


def requirement_plan(tree: dict) -> dict:
    """The requirement files (from the resolved tree) that make an importable Studio diffusion / video backend,
    in install_python_stack.py's order, and a hash of their contents that keys the venv."""
    req = Path(tree["studio_backend"]) / "requirements"
    diffusers = os.environ.get("DIFFUSION_BENCH_STUDIO_DIFFUSERS", "main")
    steps = [
        ("no-torch-runtime.txt", []),
        ("extras-no-deps.txt", ["--no-deps"]),
        ("studio.txt", []),
        ("diffusers-pin.txt", []),
    ]
    if diffusers != "pin" and (req / "diffusers-main.txt").exists():
        steps.append(("diffusers-main.txt", []))
    present = [(name, flags) for name, flags in steps if (req / name).exists()]
    digest = hashlib.sha1()
    for name, flags in present:
        digest.update(name.encode() + b"\0" + " ".join(flags).encode() + b"\0" + (req / name).read_bytes())
    return {"dir": str(req), "steps": present, "hash": digest.hexdigest()[:10], "diffusers": diffusers}


def ensure(studio_src: Optional[str] = None, python: Optional[str] = None, torch: Optional[str] = None,
           index: Optional[str] = None, dry_run: bool = False, **_) -> dict:
    """{"python", "studio_backend", "tree", "rev", "mode"} (+ "venv", "reused") ready to import Studio's backends.

    ``DIFFUSION_BENCH_STUDIO_PYTHON`` short-circuits the build: that interpreter is used as is."""
    reuse = os.environ.get("DIFFUSION_BENCH_STUDIO_PYTHON")
    if reuse:
        if not dry_run and not _python_works(reuse):
            raise RuntimeError(f"DIFFUSION_BENCH_STUDIO_PYTHON={reuse} cannot import torch + diffusers + fastapi")
        tree = resolve_tree(studio_src, python = reuse)
        return {**tree, "python": reuse, "reused": True}

    import envs as V

    py_ver = python or "3.12"
    torch_spec = torch or "torch"
    index = index or V.detect_torch_index()
    if (studio_src or os.environ.get("DIFFUSION_BENCH_STUDIO_SRC")) == "pypi":
        # The wheel carries studio/backend, so the requirements come from the wheel after it is installed.
        info = {"profile": "studio", "packages": ["unsloth", "unsloth-zoo"], "python_version": py_ver}
        if dry_run:
            return {"mode": "pypi", "dry_run": True, "python": None, "plan": {**info, "torch": torch_spec,
                                                                              "index": index}}
        venv = V.ensure_venv("studio", python = py_ver, torch_spec = torch_spec, index = index,
                             packages = ["unsloth", "unsloth-zoo"], name = f"studio-pypi-py{py_ver}")
        tree = resolve_tree("pypi", python = venv["python"])
        _install_requirements(venv, tree, requirement_plan(tree))
        return {**tree, "python": venv["python"], "venv": venv["venv"], "reused": False}

    tree = resolve_tree(studio_src)
    plan = requirement_plan(tree)
    key = V.venv_key("studio", py_ver, torch_spec, index, [plan["hash"]])
    name = f"studio-py{py_ver}-{key}"
    if dry_run:
        cmds = [f"uv venv {V.VENVS / name} --python {py_ver}",
                f"uv pip install {torch_spec} torchvision --index-url {index}",
                "uv pip install unsloth unsloth-zoo"]
        cmds += [f"uv pip install {' '.join(flags)} -r {plan['dir']}/{f}".replace("  ", " ") for f, flags in plan["steps"]]
        cmds += ["uv pip install <torchao matched to torch by install_python_stack._select_torchao_spec>",
                 f"uv pip install {' '.join(BENCH_EXTRAS)}"]
        exists = (V.VENVS / name / ".diffusion_bench.json").exists()
        return {**tree, "python": str(V.venv_python(V.VENVS / name)), "venv": str(V.VENVS / name), "dry_run": True,
                "exists": exists, "commands": cmds}
    venv = V.ensure_venv("studio", python = py_ver, torch_spec = torch_spec, index = index,
                         packages = ["unsloth", "unsloth-zoo"], name = name)
    _install_requirements(venv, tree, plan)
    return {**tree, "python": venv["python"], "venv": venv["venv"], "reused": False}


def _install_requirements(venv: dict, tree: dict, plan: dict) -> None:
    import envs as V

    marker = Path(venv["venv"]) / ".studio_requirements.json"
    if marker.exists() and json.loads(marker.read_text()).get("hash") == plan["hash"] and _python_works(venv["python"]):
        return
    py = venv["python"]
    env = _clean_env({"VIRTUAL_ENV": venv["venv"]})
    env.pop("UV_EXCLUDE_NEWER", None)
    for name, flags in plan["steps"]:
        V.sh([V.uv(), "pip", "install", "--python", py, *flags, "-r", str(Path(plan["dir"]) / name)], env = env)
    tv = subprocess.run([py, "-c", "import torch; print(torch.__version__)"], capture_output = True, text = True,
                        env = env).stdout.strip()
    spec = _torchao_spec(Path(tree["tree"]), tv)
    V.sh([V.uv(), "pip", "install", "--python", py, "--no-deps", spec, "--index-url", venv.get("index") or
          V.detect_torch_index(), "--extra-index-url", "https://pypi.org/simple", "--index-strategy",
          "unsafe-best-match"], env = env)
    V.sh([V.uv(), "pip", "install", "--python", py, *BENCH_EXTRAS], env = env)
    marker.write_text(json.dumps({"hash": plan["hash"], "tree": tree, "torch": tv, "torchao": spec}, indent = 1))


# ---------------------------------------------------------------------------------------------- server
def _kit(name: str):
    """A studio_test_kit module loaded from its file. The package __init__ imports Playwright, which a render
    venv does not carry; lifecycle.py itself is stdlib only."""
    path = KIT / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"studio_test_kit not found at {KIT} (it ships next to diffusion_bench)")
    spec = importlib.util.spec_from_file_location(f"_dbench_kit_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


def free_port(start: int = 18800, end: int = 19800) -> int:
    for port in range(start + secrets.randbelow(200), end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port")


@dataclass
class StudioServer:
    base_url: str
    username: str = DEFAULT_USER
    password: Optional[str] = None
    token: Optional[str] = None
    home: Optional[str] = None
    port: Optional[int] = None
    leader_pid: Optional[int] = None  # the setsid session the kit started (bash | tee around the server)
    server_pid: Optional[int] = None  # the python process serving requests (VRAM / RSS are read from it)
    log: Optional[str] = None
    tree: Optional[str] = None
    rev: Optional[str] = None
    python: Optional[str] = None
    attached: bool = False
    extra: dict = field(default_factory = dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d["password"] = "***" if self.password else None
        d["token"] = "***" if self.token else None
        return d


PROCFS = Path("/proc/self").exists()


def _psutil():
    try:
        import psutil  # noqa: PLC0415

        return psutil
    except Exception:  # noqa: BLE001
        return None


def _alive(pid: int) -> bool:
    if PROCFS:
        return Path(f"/proc/{pid}").exists()
    ps = _psutil()
    return bool(ps and ps.pid_exists(pid))


def _pids() -> list:
    if PROCFS:
        return [int(d.name) for d in Path("/proc").iterdir() if d.name.isdigit()]
    ps = _psutil()
    return list(ps.pids()) if ps else []


def _proc_env(pid: int) -> dict:
    if not PROCFS:  # Windows / macOS: psutil, else nothing (then nothing is ever matched or killed)
        ps = _psutil()
        try:
            return dict(ps.Process(pid).environ()) if ps else {}
        except Exception:  # noqa: BLE001
            return {}
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        return dict(x.decode(errors = "replace").split("=", 1) for x in raw if b"=" in x)
    except OSError:
        return {}


def _proc_cmd(pid: int) -> str:
    if not PROCFS:
        ps = _psutil()
        try:
            return " ".join(ps.Process(pid).cmdline()) if ps else ""
        except Exception:  # noqa: BLE001
            return ""
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors = "replace")
    except OSError:
        return ""


def _is_leader(pid: int) -> bool:
    getpgid = getattr(os, "getpgid", None)
    if getpgid is None:
        return False
    try:
        return getpgid(pid) == pid
    except OSError:
        return False


def _find_server_pid(home: Path, port: int) -> tuple[Optional[int], Optional[int]]:
    """(session leader, serving python) for the Studio started with this home and port. Only a process whose
    environment names OUR home is returned, so nothing else on the host can be picked up and later killed."""
    leader, server = None, None
    for pid in _pids():
        if _proc_env(pid).get("UNSLOTH_STUDIO_HOME") != str(home):
            continue
        cmd = _proc_cmd(pid)
        if "python" in cmd.split(" ")[0] and f"--port {port}" in cmd and "run.py" in cmd:
            server = pid  # the shim's exec'd backend
        elif "python" in cmd.split(" ")[0] and f"studio -p {port}" in cmd:
            server = pid  # a real install's CLI
        if f"studio -p {port}" in cmd and "tee" in cmd and _is_leader(pid):
            leader = pid  # the kit's `setsid bash -c '... | tee'` session leader
    return leader, server


def _write_shim(home: Path, python: str, backend: Path, tree: Path, pythonpath_tree: bool) -> Path:
    """``<home>/bin/unsloth`` that turns the kit's ``unsloth studio -p PORT`` into ``python run.py --port PORT``."""
    shim = home / "bin" / "unsloth"
    shim.parent.mkdir(parents = True, exist_ok = True)
    pp = f"{backend}:{tree}" if pythonpath_tree else str(backend)
    # Last on the path: a sitecustomize registering faulthandler on SIGUSR1 (edge/surfaces.dump_stacks on a Hang)
    pp += f":{Path(__file__).resolve().parent / 'edge' / 'stackdump'}"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        "# diffusion_bench: studio_test_kit launches `unsloth studio -p PORT`; run this tree's backend instead.\n"
        'port=8888; while [ $# -gt 0 ]; do case "$1" in -p|--port) port="$2"; shift 2;; *) shift;; esac; done\n'
        # DBENCH_PYTHONPATH_PREFIX (budget.child_env): the bench shim (simulated VRAM total, parity hooks) goes first
        f'export PYTHONPATH="${{DBENCH_PYTHONPATH_PREFIX:+$DBENCH_PYTHONPATH_PREFIX:}}"{shlex.quote(pp)}\n'
        f"cd {shlex.quote(str(home))}\n"
        f'exec {shlex.quote(python)} {shlex.quote(str(backend / "run.py"))} --api-only --host 127.0.0.1 --port "$port"\n')
    shim.chmod(0o755)
    return shim


def launch(studio_src: Optional[str] = None, python: Optional[str] = None, port: Optional[int] = None,
           home: Optional[str] = None, gpu: Optional[str] = None, password: Optional[str] = None,
           env: Optional[dict] = None, timeout_s: float = 300, fresh_home: bool = True,
           pythonpath_tree: bool = False) -> StudioServer:
    """Start a Studio API server and log in. ``home`` holding a real install (bin/unsloth from install.sh) is
    launched with its own CLI; otherwise the tree from ``studio_src`` runs under ``python`` (default: ensure())."""
    lifecycle = _kit("lifecycle")
    port = port or free_port()
    home_p = Path(home).resolve() if home else HOMES / f"studio_{port}_{time.strftime('%Y%m%d_%H%M%S')}"
    real_install = (home_p / "bin" / "unsloth").exists() and not (home_p / "bin" / ".dbench_shim").exists()
    tree_info: dict = {}
    if not real_install:
        if fresh_home and home_p.exists() and not home:
            raise FileExistsError(home_p)
        home_p.mkdir(parents = True, exist_ok = True)
        if python:
            tree_info = {**resolve_tree(studio_src, python = python), "python": python}
        else:
            tree_info = ensure(studio_src)
        missing = subprocess.run([tree_info["python"], "-c", "import uvicorn, fastapi, datasets, structlog, jwt"],
                                 capture_output = True, text = True, env = _clean_env()).stderr.strip().splitlines()
        if missing:
            raise RuntimeError(f"{tree_info['python']} cannot run a Studio server ({missing[-1]}). Point "
                               "DIFFUSION_BENCH_STUDIO_PYTHON at a Studio install's unsloth_studio/bin/python, or let "
                               "ensure() build the studio venv (it installs studio.txt).")
        _write_shim(home_p, tree_info["python"], Path(tree_info["studio_backend"]), Path(tree_info["tree"]),
                    pythonpath_tree)
        (home_p / "bin" / ".dbench_shim").write_text("1")
    password = password or os.environ.get("DIFFUSION_BENCH_STUDIO_NEW_PASSWORD") or f"dbench-{secrets.token_urlsafe(12)}"
    extra_env = {"UNSLOTH_STUDIO_PASSWORD": password, "PYTHONUNBUFFERED": "1", **(env or {})}
    if gpu is not None:
        extra_env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "HIP_VISIBLE_DEVICES": str(gpu)})
    log = C.WS / "logs" / f"dbench_studio_{port}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    install = lifecycle.StudioInstall(home = home_p, repo = Path(tree_info.get("tree") or home_p), branch = "")
    saved = {k: os.environ.get(k) for k in ("PYTHONPATH", "VIRTUAL_ENV", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    for k in saved:  # launch_studio passes os.environ through; the shim sets its own PYTHONPATH
        os.environ.pop(k, None)
    try:
        # wait_for_healthz=False: the kit polls /healthz, which current Studio does not serve (404 on an
        # --api-only server); /api/health is the liveness route, polled below.
        lifecycle.launch_studio(install, port = port, log_path = log, extra_env = extra_env,
                                wait_for_healthz = False, password_timeout_s = 2)
        wait_healthy(f"http://127.0.0.1:{port}", timeout_s, log = log, home = home_p, port = port)
    except Exception:
        leader, server = _find_server_pid(home_p, port)
        _kill_group(leader or server)
        raise
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    leader, server = _find_server_pid(home_p, port)
    srv = StudioServer(base_url = f"http://127.0.0.1:{port}", password = password, home = str(home_p), port = port,
                       leader_pid = leader, server_pid = server, log = str(log), tree = tree_info.get("tree"),
                       rev = tree_info.get("rev"), python = tree_info.get("python"))
    (home_p / "dbench_server.json").write_text(json.dumps({**srv.to_json(), "password": password}, indent = 1))
    # A kit-parsed bootstrap password means the supplied one did not apply (an existing home): log in with it.
    if install.bootstrap_password and install.bootstrap_password != password:
        srv.extra["bootstrap_password"] = install.bootstrap_password
    return srv


def wait_healthy(base_url: str, timeout_s: float, log: Optional[Path] = None, home: Optional[Path] = None,
                 port: Optional[int] = None) -> None:
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/health", timeout = 3) as r:
                # 200 alone is not ready: the SPA catch-all answers unknown paths 200 with index.html
                if r.status == 200 and isinstance(json.loads(r.read(65536) or b"null"), dict):
                    return
        except Exception:  # noqa: BLE001
            pass
        if home is not None and port is not None and time.time() > deadline - timeout_s + 20:
            leader, server = _find_server_pid(home, port)
            if leader is None and server is None:
                tail = Path(log).read_text(errors = "replace")[-3000:] if log and Path(log).exists() else ""
                raise RuntimeError(f"Studio on :{port} exited during startup. Log tail:\n{tail}")
        time.sleep(1)
    raise TimeoutError(f"{base_url}/api/health not 200 JSON within {timeout_s:.0f}s (log: {log})")


def attach(base_url: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None,
           token: Optional[str] = None, server_pid: Optional[int] = None) -> StudioServer:
    """A Studio / Unsloth Desktop that is already running. Credentials from arguments or DIFFUSION_BENCH_STUDIO_*."""
    url = base_url or os.environ.get("DIFFUSION_BENCH_STUDIO_URL")
    if not url:
        raise ValueError("attach needs base_url or DIFFUSION_BENCH_STUDIO_URL")
    pid = server_pid or os.environ.get("DIFFUSION_BENCH_STUDIO_PID")
    return StudioServer(base_url = url.rstrip("/"), username = username or os.environ.get("DIFFUSION_BENCH_STUDIO_USER")
                        or DEFAULT_USER, password = password or os.environ.get("DIFFUSION_BENCH_STUDIO_PASSWORD"),
                        token = token or os.environ.get("DIFFUSION_BENCH_STUDIO_TOKEN"), attached = True,
                        server_pid = int(pid) if pid else None)


def server_from_env_or_launch(opts: dict) -> StudioServer:
    """What the HTTP backend uses: attach when DIFFUSION_BENCH_STUDIO_URL (or options.base_url) is set, else launch."""
    if opts.get("base_url") or os.environ.get("DIFFUSION_BENCH_STUDIO_URL"):
        return attach(opts.get("base_url"), opts.get("username"), opts.get("password"), opts.get("token"))
    # options.studio_home (run_cell._cache_env: per cell for "fresh", the shared per-stack home for "returning"), so
    # the record's cache.studio_home is the home the server really used
    return launch(studio_src = opts.get("studio_src"), python = opts.get("studio_python")
                  or os.environ.get("DIFFUSION_BENCH_STUDIO_PYTHON") or None, gpu = opts.get("gpu") or C.visible_gpu(),
                  env = opts.get("server_env"), home = opts.get("studio_home"))


def process_memory(pid: Optional[int]) -> dict:
    """{vram_mib, rss_gib, rss_anon_gib, rss_file_gib} for one process: VRAM from the driver's per-process
    accounting (other tenants of the device excluded), RSS from /proc."""
    out: dict = {}
    if not pid:
        return out
    try:
        fields = {ln.split(":")[0]: int(ln.split()[1]) for ln in Path(f"/proc/{pid}/status").read_text().splitlines()
                  if ln.startswith(("VmRSS", "RssAnon", "RssFile"))}
        out.update({"rss_gib": round(fields.get("VmRSS", 0) / 2**20, 3),
                    "rss_anon_gib": round(fields.get("RssAnon", 0) / 2**20, 3),
                    "rss_file_gib": round(fields.get("RssFile", 0) / 2**20, 3)})
    except OSError:
        out["gone"] = True
    if C.gpu_vendor() == "nvidia":
        try:
            rows = C.smi(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         timeout = 15).stdout.splitlines()
            out["vram_mib"] = sum(int(r.split(",")[1]) for r in rows if r.split(",")[0].strip() == str(pid))
        except Exception:  # noqa: BLE001
            pass
    return out


def _kill_group(pid: Optional[int], timeout: float = 30) -> bool:
    if not pid:
        return False
    if os.name == "nt" or not hasattr(os, "killpg"):  # no process groups / SIGKILL: psutil tree, else taskkill /T
        C.kill_tree(pid, timeout = timeout)
        return not _alive(pid)
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return True
    if pgid in (os.getpgid(0),):  # never our own group
        return False
    for sig, wait in ((signal.SIGTERM, timeout), (signal.SIGKILL, 10)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return True
        deadline = time.time() + wait
        while time.time() < deadline:
            if not _alive(pid):
                return True
            time.sleep(0.5)
    return not _alive(pid)


def stop(server: StudioServer) -> bool:
    """Stop a server this module launched (never an attached one). Re-finds the pids from /proc by home + port,
    so only processes whose environment names that home are signalled."""
    if server.attached or not server.home:
        return False
    leader, srv = _find_server_pid(Path(server.home), int(server.port))
    ok = True
    for pid in [p for p in (leader, srv, server.leader_pid, server.server_pid) if p]:
        env_home = _proc_env(pid).get("UNSLOTH_STUDIO_HOME")
        if env_home not in (None, server.home):
            continue  # pid reused by something else: not ours
        if _alive(pid):
            ok = _kill_group(pid) and ok
    return ok


# ---------------------------------------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest = "cmd", required = True)
    for name in ("tree", "ensure", "launch"):
        p = sub.add_parser(name)
        p.add_argument("--studio-src", default = None)
        p.add_argument("--python", default = None, help = "ensure: python version; launch: interpreter to run")
        p.add_argument("--torch", default = None)
        p.add_argument("--index", default = None)
        p.add_argument("--dry-run", action = "store_true")
        p.add_argument("--gpu", default = None)
        p.add_argument("--port", type = int, default = None)
    s = sub.add_parser("stop")
    s.add_argument("--home", required = True)
    args = ap.parse_args()
    if args.cmd == "tree":
        print(json.dumps(resolve_tree(args.studio_src)))
    elif args.cmd == "ensure":
        print(json.dumps(ensure(args.studio_src, python = args.python, torch = args.torch, index = args.index,
                                dry_run = args.dry_run), indent = 1))
    elif args.cmd == "launch":
        srv = launch(args.studio_src, python = args.python or os.environ.get("DIFFUSION_BENCH_STUDIO_PYTHON"),
                     port = args.port, gpu = args.gpu)
        print(json.dumps(srv.to_json(), indent = 1))
    elif args.cmd == "stop":
        info = json.loads((Path(args.home) / "dbench_server.json").read_text())
        srv = StudioServer(**{k: v for k, v in info.items() if k in StudioServer.__dataclass_fields__})
        srv.attached = False
        print(json.dumps({"stopped": stop(srv)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
