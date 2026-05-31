"""Cross-OS reproduction of the Windows cuda13 prebuilt miss after ggml-org
bumped their published Windows CUDA-13 build from cuda-13.1 to cuda-13.3.

No GPU and no network: the Windows asset selector in install_llama_prebuilt.py
is a pure function of HostInfo.driver_cuda_version plus the upstream release
asset list. We spoof a Windows NVIDIA host whose driver advertises CUDA 13.x and
feed a frozen snapshot of the real ggml-org b9437 Windows CUDA asset names.

This file runs identically on ubuntu-latest, macos-14 and windows-latest, which
demonstrates the selection bug is OS-independent: even on a real Windows runner,
current main hands a CUDA-13 host the cuda-12.4 build.

- test_main_*  : current studio/install_llama_prebuilt.py -> CUDA-13 host gets cuda-12.4 (bug)
- test_fixed_* : studio/install_llama_prebuilt_fixed.py    -> CUDA-13 host gets cuda-13.3 (fixed)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PATH = REPO_ROOT / "studio" / "install_llama_prebuilt.py"
FIXED_PATH = REPO_ROOT / "studio" / "install_llama_prebuilt_fixed.py"

# Frozen snapshot of ggml-org/llama.cpp@b9437 Windows CUDA assets (2026-05-31).
# Upstream ships 12.4 and 13.3 only; there is no 13.1 asset anymore.
LLAMA_TAG = "b9437"
UPSTREAM_ASSETS = {
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "https://example/cudart-12.4.zip",
    "cudart-llama-bin-win-cuda-13.3-x64.zip": "https://example/cudart-13.3.zip",
    "llama-b9437-bin-win-cuda-12.4-x64.zip": "https://example/llama-12.4.zip",
    "llama-b9437-bin-win-cuda-13.3-x64.zip": "https://example/llama-13.3.zip",
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # so @dataclass can resolve cls.__module__
    spec.loader.exec_module(mod)
    return mod


def _spoof_windows_host(mod, driver_cuda: tuple[int, int], compute_cap: str = "89"):
    return mod.HostInfo(
        system="Windows",
        machine="AMD64",
        is_windows=True,
        is_linux=False,
        is_macos=False,
        is_x86_64=True,
        is_arm64=False,
        nvidia_smi="nvidia-smi",
        driver_cuda_version=driver_cuda,
        compute_caps=[compute_cap],
        visible_cuda_devices=None,
        has_physical_nvidia=True,
        has_usable_nvidia=True,
    )


def _selected_prebuilt(mod, driver_cuda: tuple[int, int], compute_cap: str = "89"):
    """Drive the real resolver and return the top-choice asset name (or None).

    detected_windows_runtime_lines is forced empty so the result depends only on
    the driver version and the upstream asset list, identically on every OS
    (a real Windows CI runner has no CUDA runtime DLLs on PATH anyway).
    """
    host = _spoof_windows_host(mod, driver_cuda, compute_cap)
    orig = mod.detected_windows_runtime_lines
    mod.detected_windows_runtime_lines = lambda: ([], {})
    try:
        attempts = mod.windows_cuda_attempts(host, LLAMA_TAG, UPSTREAM_ASSETS, None)
    finally:
        mod.detected_windows_runtime_lines = orig
    return attempts[0].name if attempts else None


@pytest.fixture(scope="module")
def main_mod():
    return _load(MAIN_PATH, "ilp_main")


@pytest.fixture(scope="module")
def fixed_mod():
    return _load(FIXED_PATH, "ilp_fixed")


CUDA13_DRIVERS = [(13, 3), (13, 2), (13, 1)]


@pytest.mark.parametrize("driver", CUDA13_DRIVERS)
def test_main_cuda13_host_misses_133_and_falls_back_to_124(main_mod, driver):
    """Current main: a CUDA-13 host is handed the cuda-12.4 build; the cuda-13.3
    bundle ggml-org actually publishes is never selected."""
    selected = _selected_prebuilt(main_mod, driver)
    assert selected == "llama-b9437-bin-win-cuda-12.4-x64.zip", (
        f"driver {driver} selected {selected!r}; expected the buggy 12.4 fallback"
    )
    # Confirm the root cause: it asks for the non-existent 13.1 asset.
    assert main_mod.pick_windows_cuda_runtime(
        _spoof_windows_host(main_mod, driver)
    ) == "13.1"


def test_main_no_cuda13_host_can_reach_133(main_mod):
    """Headline regression: across every CUDA-13 driver, the 13.3 prebuilt is
    unreachable on current main."""
    reached = {
        driver: _selected_prebuilt(main_mod, driver) for driver in CUDA13_DRIVERS
    }
    assert all("cuda-13" not in (name or "") for name in reached.values()), reached


@pytest.mark.parametrize("driver", CUDA13_DRIVERS)
def test_fixed_cuda13_host_selects_133(fixed_mod, driver):
    """With dynamic minor discovery, a CUDA-13 host resolves to the published
    cuda-13.3 build, with cuda-12.4 retained as the fallback."""
    selected = _selected_prebuilt(fixed_mod, driver)
    assert selected == "llama-b9437-bin-win-cuda-13.3-x64.zip", (
        f"driver {driver} selected {selected!r}; expected cuda-13.3"
    )


def test_fixed_cuda12_host_unaffected(fixed_mod):
    """The fix must not disturb CUDA-12 hosts: a 12.8 driver still gets 12.4."""
    assert _selected_prebuilt(fixed_mod, (12, 8)) == "llama-b9437-bin-win-cuda-12.4-x64.zip"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
