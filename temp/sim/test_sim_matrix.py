"""Cross-platform / cross-GPU simulation matrix for the three Studio fixes.

Run on the studio/all-fixes-integration branch:

    PYTHONPATH=studio/backend ../temp/venv_studio/bin/python -m pytest temp/sim -q

Covers the cartesian product the maintainer asked for:
  [Windows, Linux, WSL, Mac] x [NVIDIA, AMD, CPU] for the GPU-validation fix,
  plus an exhaustive (installed, latest) grid for the downgrade fix and every
  wording state for the Cloudflare line.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "studio" / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import utils.llama_cpp_freshness as freshness  # noqa: E402
import utils.llama_cpp_update as upd  # noqa: E402

_ILP_PATH = REPO / "studio" / "install_llama_prebuilt.py"
_spec = importlib.util.spec_from_file_location("ilp_sim", _ILP_PATH)
ilp = importlib.util.module_from_spec(_spec)
sys.modules["ilp_sim"] = ilp
_spec.loader.exec_module(ilp)

RUN_PY = (BACKEND / "run.py").read_text()


# ───────────────────────── helpers ─────────────────────────


def _reset():
    freshness.reset_caches()
    upd._reset_job_for_tests()
    upd._resolve_memo.clear()


def _marker_install(tmp_path: Path, tag: str) -> str:
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    binary = bin_dir / "llama-server"
    binary.write_text("#!/bin/sh\necho stub\n")
    (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(
        '{"tag": "%s", "release_tag": "%s", "published_repo": "unslothai/llama.cpp",'
        ' "installed_at_utc": "2020-01-01T00:00:00Z"}' % (tag, tag)
    )
    return str(binary)


def _host(*, system, machine, nvidia = False, rocm = False, arm64 = False):
    return ilp.HostInfo(
        system = system,
        machine = machine,
        is_windows = system == "Windows",
        is_linux = system == "Linux",
        is_macos = system == "Darwin",
        is_x86_64 = machine in ("x86_64", "AMD64"),
        is_arm64 = arm64,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = ["10.0"] if nvidia else [],
        visible_cuda_devices = None,
        has_physical_nvidia = nvidia,
        has_usable_nvidia = nvidia,
        has_rocm = rocm,
        rocm_gfx_target = "gfx1100" if rocm else None,
        macos_version = (14, 0) if system == "Darwin" else None,
    )


def _capture_validate_command(monkeypatch, tmp_path, *, host, install_kind, integrity_verified):
    captured: dict = {}

    class _P:
        def __init__(self, command, **_):
            captured["command"] = list(command)

        def poll(self):
            return None

        def wait(self, timeout = None):
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

    class _Resp:
        status = 200

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(ilp.subprocess, "Popen", lambda command, **kw: _P(command, **kw))
    monkeypatch.setattr(ilp.urllib.request, "urlopen", lambda *a, **k: _Resp())
    monkeypatch.setattr(ilp, "binary_env", lambda *a, **k: {})
    ilp.validate_server(
        tmp_path / "llama-server",
        tmp_path / "probe.gguf",
        host,
        tmp_path / "install",
        install_kind = install_kind,
        integrity_verified = integrity_verified,
    )
    return captured["command"]


# ──────────────── Issue 1: downgrade grid (platform-independent) ────────────────

# (installed_build, latest_build, expect_update_available)
_PAIRS = [
    (9585, 9518, False),   # the reported downgrade (older partial release)
    (9596, 9594, False),   # the Reddit case
    (9585, 9585, False),   # equal -> no-op
    (9518, 9585, True),    # genuine upgrade
    (9000, 9585, True),    # large upgrade
    (9585, 9000, False),   # large downgrade
    (1, 9585, True),       # unknown-ish installed, real latest
]


@pytest.mark.parametrize("installed,latest,expect", _PAIRS)
def test_marker_path_never_downgrades(monkeypatch, tmp_path, installed, latest, expect):
    _reset()
    binary = _marker_install(tmp_path, f"b{installed}")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: f"b{latest}")
    st = upd.get_update_status(force_refresh = True)
    assert st["update_available"] is expect, (installed, latest, st)
    _reset()


@pytest.mark.parametrize("installed,latest,expect", _PAIRS)
def test_source_build_path_never_downgrades(monkeypatch, tmp_path, installed, latest, expect):
    _reset()
    binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    payload = {
        "prebuilt_available": True,
        "repo": "unslothai/llama.cpp",
        "release_tag": f"b{latest}",
        "llama_tag": f"b{latest}",
        "asset": "x.tar.gz",
        "install_kind": "macos-arm64",
    }
    monkeypatch.setattr(upd, "_resolve_prebuilt_for_host", lambda *, force_refresh = False: payload)
    monkeypatch.setattr(upd, "_installed_build_number", lambda b: installed)
    st = upd.get_update_status()
    assert st["update_available"] is expect, (installed, latest, st)
    _reset()


def test_source_build_unknown_version_still_offered(monkeypatch, tmp_path):
    # The genuine involuntary source-build case (version unknown) still gets the
    # prebuilt -- but only when latest parses to a real build number.
    _reset()
    binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    monkeypatch.setattr(upd, "_installed_build_number", lambda b: None)
    for latest, expect in [("b9585", True), ("master", False), ("", False)]:
        payload = {
            "prebuilt_available": bool(latest),
            "repo": "unslothai/llama.cpp",
            "release_tag": latest,
            "llama_tag": latest,
            "asset": "x.tar.gz",
            "install_kind": "macos-arm64",
        }
        monkeypatch.setattr(upd, "_resolve_prebuilt_for_host", lambda *, force_refresh = False, p = payload: p)
        st = upd.get_update_status()
        assert st["update_available"] is expect, (latest, st)
    _reset()


# ──────────────── Issue 2: OS x GPU x integrity x env GPU-gating matrix ────────────────

# install_kind -> is it a GPU bundle kind (gets --n-gpu-layers when unapproved)
_INSTALL_KINDS = {
    # Linux / WSL
    "linux-cuda": True,         # Linux+NVIDIA, WSL+NVIDIA
    "linux-arm64-cuda": True,   # Linux arm64 + NVIDIA (Jetson/GH200)
    "linux-rocm": True,         # Linux+AMD, WSL+AMD
    "linux-cpu": False,         # Linux+CPU, WSL+CPU
    "linux-x64": False,         # Linux CPU bundle
    # Windows
    "windows-cuda": True,       # Windows+NVIDIA
    "windows-hip": True,        # Windows+AMD (HIP)
    "windows-rocm": True,       # Windows+AMD (ROCm)
    "windows-cpu": False,       # Windows+CPU
    # macOS
    "macos-arm64": True,        # Mac Apple Silicon (Metal)
    "macos-x64": False,         # Mac Intel (CPU)
}

_GENERIC_HOST = _host(system = "Linux", machine = "x86_64", nvidia = True)


@pytest.mark.parametrize("install_kind,is_gpu_kind", list(_INSTALL_KINDS.items()))
def test_gpu_gating_unapproved(monkeypatch, tmp_path, install_kind, is_gpu_kind):
    # Unapproved/lemonade (integrity_verified=False): full GPU smoke test only on
    # GPU bundle kinds; CPU bundles never get --n-gpu-layers.
    monkeypatch.delenv("UNSLOTH_LLAMA_SKIP_GPU_VALIDATION", raising = False)
    cmd = _capture_validate_command(
        monkeypatch, tmp_path, host = _GENERIC_HOST,
        install_kind = install_kind, integrity_verified = False,
    )
    assert ("--n-gpu-layers" in cmd) is is_gpu_kind, install_kind


@pytest.mark.parametrize("install_kind", list(_INSTALL_KINDS))
def test_gpu_gating_approved_always_cpu(monkeypatch, tmp_path, install_kind):
    # Approved (sha256-verified) build: never pay the GPU JIT, on every OS/GPU.
    monkeypatch.delenv("UNSLOTH_LLAMA_SKIP_GPU_VALIDATION", raising = False)
    cmd = _capture_validate_command(
        monkeypatch, tmp_path, host = _GENERIC_HOST,
        install_kind = install_kind, integrity_verified = True,
    )
    assert "--n-gpu-layers" not in cmd, install_kind


@pytest.mark.parametrize("install_kind,is_gpu_kind", list(_INSTALL_KINDS.items()))
def test_gpu_gating_env_flag_forces_cpu(monkeypatch, tmp_path, install_kind, is_gpu_kind):
    # UNSLOTH_LLAMA_SKIP_GPU_VALIDATION forces CPU even for unapproved GPU kinds.
    monkeypatch.setenv("UNSLOTH_LLAMA_SKIP_GPU_VALIDATION", "1")
    cmd = _capture_validate_command(
        monkeypatch, tmp_path, host = _GENERIC_HOST,
        install_kind = install_kind, integrity_verified = False,
    )
    assert "--n-gpu-layers" not in cmd, install_kind


# install_kind=None fallback (older call sites) -> host detection across OS x GPU
_HOST_CASES = [
    ("Linux", "x86_64", dict(nvidia = True), True),    # Linux + NVIDIA
    ("Linux", "x86_64", dict(rocm = True), True),      # Linux + AMD
    ("Linux", "x86_64", dict(), False),                # Linux + CPU
    ("Linux", "aarch64", dict(nvidia = True, arm64 = True), True),   # WSL/ARM + NVIDIA
    ("Windows", "AMD64", dict(nvidia = True), True),   # Windows + NVIDIA
    ("Windows", "AMD64", dict(rocm = True), True),     # Windows + AMD
    ("Windows", "AMD64", dict(), False),               # Windows + CPU
    ("Darwin", "arm64", dict(arm64 = True), True),     # Mac Apple Silicon (Metal)
    ("Darwin", "x86_64", dict(), False),               # Mac Intel (CPU)
]


@pytest.mark.parametrize("system,machine,flags,is_gpu", _HOST_CASES)
def test_gpu_gating_host_detection_fallback(monkeypatch, tmp_path, system, machine, flags, is_gpu):
    monkeypatch.delenv("UNSLOTH_LLAMA_SKIP_GPU_VALIDATION", raising = False)
    host = _host(system = system, machine = machine, **flags)
    # install_kind=None exercises the host-detection branch.
    cmd = _capture_validate_command(
        monkeypatch, tmp_path, host = host, install_kind = None, integrity_verified = False,
    )
    assert ("--n-gpu-layers" in cmd) is is_gpu, (system, machine, flags)
    # Approved build is always CPU regardless of host.
    cmd2 = _capture_validate_command(
        monkeypatch, tmp_path, host = host, install_kind = None, integrity_verified = True,
    )
    assert "--n-gpu-layers" not in cmd2, (system, machine, flags)


# ──────────────── Issue 4: Cloudflare wording matrix ────────────────


def _print_cloudflare_line(monkeypatch, *, url, reachable):
    func_src = next(
        ast.get_source_segment(RUN_PY, n)
        for n in ast.walk(ast.parse(RUN_PY))
        if isinstance(n, ast.FunctionDef) and n.name == "_print_cloudflare_line"
    )
    stub = types.ModuleType("startup_banner")
    stub.stdout_supports_color = lambda: False
    monkeypatch.setitem(sys.modules, "startup_banner", stub)
    out: list = []
    ns = {"_cloudflare_url": url, "_public_reachable": reachable,
          "print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    exec(compile(func_src, "<x>", "exec"), ns)
    ns["_print_cloudflare_line"]()
    return "\n".join(out)


_URL = "https://demo.trycloudflare.com"


@pytest.mark.parametrize("reachable,url,expect", [
    (False, _URL, "Also, the secure link access via Cloudflare works: " + _URL),
    (True, _URL, "Secure link access via Cloudflare: " + _URL),
    (None, _URL, "Secure link access via Cloudflare: " + _URL),
    (False, None, ""),
    (True, None, ""),
    (None, None, ""),
])
def test_cloudflare_wording_matrix(monkeypatch, reachable, url, expect):
    out = _print_cloudflare_line(monkeypatch, url = url, reachable = reachable)
    if expect == "":
        assert out == ""
    else:
        assert expect in out
        if reachable is False:
            assert "Also, the secure link" in out
        else:
            assert "Also, the secure link" not in out
