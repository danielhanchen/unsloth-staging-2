# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AMD-GPU spoofing matrix for the prebuilt llama.cpp selector (PRs 5301 + 5303).

No AMD hardware is required: we construct fake ``HostInfo`` objects with
``has_rocm = True`` and a spoofed ``rocm_gfx_target``, stub the GitHub release
JSON, and assert that ``install_llama_prebuilt`` selects the correct
``lemonade-sdk/llamacpp-rocm`` GPU prebuilt for every supported AMD gfx target
across Linux / WSL / Windows, and that the CPU / NVIDIA / macOS paths are
unchanged. Runs identically on a real Windows / macOS / Linux CI runner because
the selection logic keys off the fake ``HostInfo`` flags, not the live platform.

Goal these PRs deliver: use a PREBUILT llama.cpp binary everywhere -- a GPU
build when a GPU is present, a CPU build otherwise -- instead of compiling
llama.cpp from source for AMD machines.

Run as pytest, or directly (``python test_amd_spoof_prebuilt_matrix.py``) to
print the spoof matrix table.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# studio/backend/tests/ -> studio/
_studio = Path(__file__).resolve().parent.parent.parent
if str(_studio) not in sys.path:
    sys.path.insert(0, str(_studio))

_mod = importlib.import_module("install_llama_prebuilt")

HostInfo = _mod.HostInfo
AssetChoice = _mod.AssetChoice
PrebuiltFallback = _mod.PrebuiltFallback
resolve_lemonade_rocm_choice = _mod.resolve_lemonade_rocm_choice
direct_linux_release_plan = _mod.direct_linux_release_plan
direct_upstream_release_plan = _mod.direct_upstream_release_plan
_lemonade_gfx_family = _mod._lemonade_gfx_family

# Every AMD gfx target covered by lemonade-sdk/llamacpp-rocm, plus the family
# each one must map to (RDNA2 -> gfx103X, RDNA3 -> gfx110X, Strix -> gfx1150 /
# gfx1151, RDNA4 -> gfx120X).
GFX_TO_FAMILY = {
    "gfx1030": "gfx103X",  # RX 6800/6900 (RDNA2)
    "gfx1100": "gfx110X",  # RX 7900 XTX (RDNA3)
    "gfx1101": "gfx110X",  # RX 7800 XT
    "gfx1102": "gfx110X",  # RX 7600 XT
    "gfx1103": "gfx110X",  # Radeon 780M iGPU
    "gfx1150": "gfx1150",  # Strix Point APU
    "gfx1151": "gfx1151",  # Strix Halo APU (Radeon 8060S)
    "gfx1200": "gfx120X",  # RX 9070 (RDNA4)
    "gfx1201": "gfx120X",  # Radeon AI PRO R9700 (RDNA4)
}
ALL_GFX = list(GFX_TO_FAMILY)

# gfx targets lemonade does NOT publish -> must fall through, never silent CPU.
UNSUPPORTED_GFX = ["gfx900", "gfx906", "gfx908", "gfx999"]

# Full AMD catalog: new + old + data center. Each row is
# (gfx, product, era, expected_lemonade_family_or_None). lemonade-sdk only
# publishes RDNA *consumer* families (gfx103X / gfx110X / gfx1150 / gfx1151 /
# gfx120X), so everything else -- in particular every CDNA / Instinct data
# center accelerator -- must map to None and fall through to the upstream HIP
# prebuilt (Windows) or the HIP source build (Linux), never a silent CPU
# binary. gfx -> product mapping per AMD ROCm GPU-arch docs + LLVM targets.
AMD_CATALOG: list[tuple[str, str, str, str | None]] = [
    # --- Data center / Instinct (Vega20 + CDNA 1/2/3/4) -> NOT in lemonade ---
    ("gfx906", "Instinct MI50 / MI60 (Vega20)",          "datacenter", None),
    ("gfx908", "Instinct MI100 (CDNA1)",                 "datacenter", None),
    ("gfx90a", "Instinct MI210/MI250/MI250X (CDNA2)",    "datacenter", None),
    ("gfx940", "Instinct MI300A pre-release (CDNA3)",    "datacenter", None),
    ("gfx941", "Instinct MI300 (CDNA3)",                 "datacenter", None),
    ("gfx942", "Instinct MI300X / MI325X (CDNA3)",       "datacenter", None),
    ("gfx950", "Instinct MI350X / MI355X (CDNA4)",       "datacenter", None),
    # --- Old consumer (Vega10 + RDNA1) -> NOT in lemonade ---
    ("gfx900", "Radeon RX Vega 56/64 (Vega10)",          "old",        None),
    ("gfx1010", "Radeon RX 5700 XT (RDNA1)",             "old",        None),
    ("gfx1011", "Radeon Pro V520 (RDNA1)",               "old",        None),
    ("gfx1012", "Radeon RX 5500 XT (RDNA1)",             "old",        None),
    # --- RDNA2 (gfx103X) -> covered ---
    ("gfx1030", "Radeon RX 6800 / 6900 XT (RDNA2)",      "rdna2",      "gfx103X"),
    ("gfx1031", "Radeon RX 6700 XT (RDNA2)",             "rdna2",      "gfx103X"),
    ("gfx1032", "Radeon RX 6600 (RDNA2)",                "rdna2",      "gfx103X"),
    ("gfx1034", "Radeon RX 6500 XT (RDNA2)",             "rdna2",      "gfx103X"),
    ("gfx1035", "Radeon 680M iGPU (RDNA2)",              "rdna2",      "gfx103X"),
    ("gfx1036", "Raphael iGPU (RDNA2)",                  "rdna2",      "gfx103X"),
    # --- RDNA3 (gfx110X) -> covered ---
    ("gfx1100", "Radeon RX 7900 XTX (RDNA3)",            "rdna3",      "gfx110X"),
    ("gfx1101", "Radeon RX 7800 / 7700 XT (RDNA3)",      "rdna3",      "gfx110X"),
    ("gfx1102", "Radeon RX 7600 (RDNA3)",                "rdna3",      "gfx110X"),
    ("gfx1103", "Radeon 780M iGPU (RDNA3)",              "rdna3",      "gfx110X"),
    # --- RDNA3.5 APU (Strix) -> covered (exact targets only) ---
    ("gfx1150", "Strix Point 880M iGPU (RDNA3.5)",       "rdna3.5",    "gfx1150"),
    ("gfx1151", "Strix Halo 8060S iGPU (RDNA3.5)",       "rdna3.5",    "gfx1151"),
    # --- RDNA4 (gfx120X) -> covered ---
    ("gfx1200", "Radeon RX 9060 (RDNA4)",                "rdna4",      "gfx120X"),
    ("gfx1201", "Radeon RX 9070 / AI PRO R9700 (RDNA4)", "rdna4",      "gfx120X"),
]
CATALOG_COVERED = [g for g, _, _, fam in AMD_CATALOG if fam is not None]
CATALOG_DATACENTER = [g for g, _, era, _ in AMD_CATALOG if era == "datacenter"]
CATALOG_OLD = [g for g, _, era, _ in AMD_CATALOG if era == "old"]

_STUB_TAG = "b9334"
_LEMONADE_OS_PREFIXES = ("ubuntu", "windows")


# --------------------------------------------------------------------------- #
# Fake hosts + stub release payloads
# --------------------------------------------------------------------------- #
def make_host(
    *,
    system: str,
    machine: str,
    is_windows: bool = False,
    is_linux: bool = False,
    is_macos: bool = False,
    is_x86_64: bool = True,
    is_arm64: bool = False,
    has_physical_nvidia: bool = False,
    has_usable_nvidia: bool = False,
    has_rocm: bool = False,
    rocm_gfx_target: str | None = None,
) -> HostInfo:
    return HostInfo(
        system = system,
        machine = machine,
        is_windows = is_windows,
        is_linux = is_linux,
        is_macos = is_macos,
        is_x86_64 = is_x86_64,
        is_arm64 = is_arm64,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = has_physical_nvidia,
        has_usable_nvidia = has_usable_nvidia,
        has_rocm = has_rocm,
        rocm_gfx_target = rocm_gfx_target,
    )


def linux_amd(gfx: str) -> HostInfo:
    return make_host(system = "Linux", machine = "x86_64", is_linux = True,
                     has_rocm = True, rocm_gfx_target = gfx)


def wsl_amd(gfx: str) -> HostInfo:
    # WSL reports platform.system() == "Linux"; there is no separate WSL branch
    # in the prebuilt selector, so a WSL AMD host is constructed exactly like a
    # native Linux AMD host. Modelled explicitly to prove WSL == Linux.
    return make_host(system = "Linux", machine = "x86_64", is_linux = True,
                     has_rocm = True, rocm_gfx_target = gfx)


def windows_amd(gfx: str) -> HostInfo:
    return make_host(system = "Windows", machine = "AMD64", is_windows = True,
                     has_rocm = True, rocm_gfx_target = gfx)


def linux_nvidia() -> HostInfo:
    return make_host(system = "Linux", machine = "x86_64", is_linux = True,
                     has_physical_nvidia = True, has_usable_nvidia = True)


def windows_nvidia() -> HostInfo:
    return make_host(system = "Windows", machine = "AMD64", is_windows = True,
                     has_physical_nvidia = True, has_usable_nvidia = True)


def linux_cpu() -> HostInfo:
    return make_host(system = "Linux", machine = "x86_64", is_linux = True)


def windows_cpu() -> HostInfo:
    return make_host(system = "Windows", machine = "AMD64", is_windows = True)


def macos_arm() -> HostInfo:
    return make_host(system = "Darwin", machine = "arm64", is_macos = True,
                     is_x86_64 = False, is_arm64 = True)


def macos_x64() -> HostInfo:
    return make_host(system = "Darwin", machine = "x86_64", is_macos = True)


def stub_lemonade_release(tag: str = _STUB_TAG) -> dict:
    """A lemonade release exposing a per-GPU ROCm zip for every family/OS."""
    families = sorted(set(GFX_TO_FAMILY.values()))
    assets = [
        {
            "name": f"llama-{tag}-{prefix}-rocm-{family}-x64.zip",
            "browser_download_url": (
                "https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/"
                f"{tag}/llama-{tag}-{prefix}-rocm-{family}-x64.zip"
            ),
        }
        for prefix in _LEMONADE_OS_PREFIXES
        for family in families
    ]
    return {"tag_name": tag, "assets": assets}


def stub_upstream_release(tag: str = _STUB_TAG) -> dict:
    """A ggml-org/llama.cpp release exposing the CPU/HIP/macOS prebuilt assets."""
    names = [
        f"llama-{tag}-bin-win-cpu-x64.zip",
        f"llama-{tag}-bin-win-hip-radeon-x64.zip",
        f"llama-{tag}-bin-win-cpu-arm64.zip",
        f"llama-{tag}-bin-macos-arm64.tar.gz",
        f"llama-{tag}-bin-macos-x64.tar.gz",
        f"llama-{tag}-bin-ubuntu-x64.tar.gz",
        f"llama-{tag}-bin-ubuntu-arm64.tar.gz",
    ]
    return {
        "tag_name": tag,
        "assets": [
            {
                "name": n,
                "browser_download_url": (
                    f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{n}"
                ),
            }
            for n in names
        ],
    }


def stub_unsloth_linux_release(tag: str = _STUB_TAG) -> dict:
    """Minimal unslothai/llama.cpp direct-Linux bundle (CPU baseline asset).

    direct_linux_release_plan requires one app-<label>-linux-x64.tar.gz so the
    bundle parses; the lemonade ROCm choice is then layered on top for AMD GPUs.
    """
    name = f"app-{tag}-linux-x64.tar.gz"
    return {
        "tag_name": tag,
        "name": tag,
        "assets": [{"name": name, "browser_download_url": f"https://example.invalid/{name}"}],
    }


@pytest.fixture(autouse = True)
def _clear_lemonade_cache(monkeypatch):
    """resolve_lemonade_rocm_choice memoises the release fetch; clear it around
    each test so stubs do not leak, and ensure the opt-out env is unset."""
    monkeypatch.delenv("UNSLOTH_DISABLE_LEMONADE_ROCM", raising = False)
    cache = getattr(_mod, "_fetch_lemonade_release_cached", None)
    if cache is not None and hasattr(cache, "cache_clear"):
        cache.cache_clear()
    yield
    if cache is not None and hasattr(cache, "cache_clear"):
        cache.cache_clear()


def _patch_lemonade(monkeypatch, release: dict | None = None):
    monkeypatch.setattr(_mod, "fetch_json",
                        lambda *a, **k: release or stub_lemonade_release())


# --------------------------------------------------------------------------- #
# 1. AMD GPU -> lemonade GPU prebuilt (resolver level), every gfx x OS
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gfx", ALL_GFX)
@pytest.mark.parametrize(
    "os_prefix,install_kind,host_factory",
    [
        ("ubuntu", "linux-rocm", linux_amd),
        ("ubuntu", "linux-rocm", wsl_amd),   # WSL == Linux
        ("windows", "windows-hip", windows_amd),
    ],
)
def test_amd_gpu_resolves_lemonade_prebuilt(monkeypatch, gfx, os_prefix, install_kind, host_factory):
    _patch_lemonade(monkeypatch)
    host = host_factory(gfx)
    choice = resolve_lemonade_rocm_choice(host, os_prefix, install_kind, llama_tag = "latest")
    assert choice is not None, f"no lemonade prebuilt selected for {gfx} ({os_prefix})"
    assert choice.repo == "lemonade-sdk/llamacpp-rocm"
    assert choice.source_label == "lemonade"
    assert choice.install_kind == install_kind
    family = GFX_TO_FAMILY[gfx]
    assert _lemonade_gfx_family(gfx) == family
    assert family in choice.name
    assert choice.name == f"llama-{_STUB_TAG}-{os_prefix}-rocm-{family}-x64.zip"
    assert _mod._is_trusted_github_release_url(choice.url, "lemonade-sdk/llamacpp-rocm")


# --------------------------------------------------------------------------- #
# 2. Planner wiring: Linux GPU -> lemonade linux-rocm, NO cpu attempt
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gfx", ["gfx1030", "gfx1100", "gfx1151", "gfx1201"])
def test_linux_amd_planner_picks_lemonade_no_cpu(monkeypatch, gfx):
    _patch_lemonade(monkeypatch)
    host = linux_amd(gfx)
    plan = direct_linux_release_plan(
        stub_unsloth_linux_release(), host, "unslothai/llama.cpp", "latest"
    )
    assert plan is not None
    kinds = [a.install_kind for a in plan.attempts]
    assert "linux-rocm" in kinds, f"no lemonade linux-rocm attempt for {gfx}; got {kinds}"
    rocm = next(a for a in plan.attempts if a.install_kind == "linux-rocm")
    assert rocm.source_label == "lemonade"
    assert GFX_TO_FAMILY[gfx] in rocm.name
    # ROCm-only host must NOT silently fall back to a CPU binary.
    assert "linux-cpu" not in kinds, f"CPU attempt leaked onto ROCm host: {kinds}"


# --------------------------------------------------------------------------- #
# 3. Planner wiring: Windows GPU -> lemonade windows-hip first, upstream HIP fallback
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gfx", ["gfx1030", "gfx1102", "gfx1151", "gfx1200"])
def test_windows_amd_planner_prefers_lemonade_then_upstream_hip(monkeypatch, gfx):
    _patch_lemonade(monkeypatch)
    host = windows_amd(gfx)
    plan = direct_upstream_release_plan(
        stub_upstream_release(), host, "ggml-org/llama.cpp", "latest"
    )
    assert plan is not None
    kinds = [a.install_kind for a in plan.attempts]
    assert kinds.count("windows-hip") >= 1
    # lemonade GPU prebuilt must come before the CPU asset.
    assert kinds.index("windows-hip") < kinds.index("windows-cpu")
    first_hip = next(a for a in plan.attempts if a.install_kind == "windows-hip")
    assert first_hip.source_label == "lemonade", "lemonade prebuilt must be preferred over upstream"
    assert first_hip.repo == "lemonade-sdk/llamacpp-rocm"
    # The upstream HIP asset is still present as a GPU fallback.
    assert any(a.install_kind == "windows-hip" and a.source_label == "upstream"
               for a in plan.attempts)


# --------------------------------------------------------------------------- #
# 4. No GPU -> CPU prebuilt (Linux / Windows / macOS), lemonade not consulted
# --------------------------------------------------------------------------- #
def test_linux_cpu_picks_cpu_prebuilt(monkeypatch):
    _patch_lemonade(monkeypatch)
    host = linux_cpu()
    assert resolve_lemonade_rocm_choice(host, "ubuntu", "linux-rocm") is None
    plan = direct_upstream_release_plan(
        stub_upstream_release(), host, "ggml-org/llama.cpp", "latest"
    )
    kinds = [a.install_kind for a in plan.attempts]
    assert kinds == ["linux-cpu"]
    assert all(a.source_label != "lemonade" for a in plan.attempts)


def test_windows_cpu_picks_cpu_prebuilt(monkeypatch):
    _patch_lemonade(monkeypatch)
    host = windows_cpu()
    assert resolve_lemonade_rocm_choice(host, "windows", "windows-hip") is None
    plan = direct_upstream_release_plan(
        stub_upstream_release(), host, "ggml-org/llama.cpp", "latest"
    )
    kinds = [a.install_kind for a in plan.attempts]
    assert "windows-cpu" in kinds
    assert "windows-hip" not in kinds
    assert all(a.source_label != "lemonade" for a in plan.attempts)


@pytest.mark.parametrize(
    "host_factory,expected_kind",
    [(macos_arm, "macos-arm64"), (macos_x64, "macos-x64")],
)
def test_macos_picks_prebuilt_tarball(monkeypatch, host_factory, expected_kind):
    _patch_lemonade(monkeypatch)
    host = host_factory()
    assert resolve_lemonade_rocm_choice(host, "ubuntu", "linux-rocm") is None
    plan = direct_upstream_release_plan(
        stub_upstream_release(), host, "ggml-org/llama.cpp", "latest"
    )
    kinds = [a.install_kind for a in plan.attempts]
    assert kinds == [expected_kind]
    assert plan.attempts[0].name.endswith(".tar.gz")
    assert all(a.source_label != "lemonade" for a in plan.attempts)


# --------------------------------------------------------------------------- #
# 5. NVIDIA -> lemonade never consulted
# --------------------------------------------------------------------------- #
def test_nvidia_never_uses_lemonade(monkeypatch):
    _patch_lemonade(monkeypatch)
    for host in (linux_nvidia(), windows_nvidia()):
        assert resolve_lemonade_rocm_choice(host, "ubuntu", "linux-rocm") is None
    # Linux NVIDIA host with a CPU-only bundle: CUDA selection yields nothing,
    # falls to CPU -- but never a lemonade attempt.
    plan = direct_linux_release_plan(
        stub_unsloth_linux_release(), linux_nvidia(), "unslothai/llama.cpp", "latest"
    )
    assert plan is not None
    assert all(a.source_label != "lemonade" for a in plan.attempts)


# --------------------------------------------------------------------------- #
# 6. Unknown / non-lemonade AMD gfx -> fall through, NEVER silent CPU
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gfx", UNSUPPORTED_GFX)
def test_unknown_amd_gfx_resolver_returns_none(monkeypatch, gfx):
    _patch_lemonade(monkeypatch)
    assert resolve_lemonade_rocm_choice(linux_amd(gfx), "ubuntu", "linux-rocm") is None
    assert resolve_lemonade_rocm_choice(windows_amd(gfx), "windows", "windows-hip") is None


@pytest.mark.parametrize("gfx", UNSUPPORTED_GFX)
def test_unknown_amd_gfx_linux_raises_fallback_not_cpu(monkeypatch, gfx):
    # Linux ROCm host + no lemonade asset -> PrebuiltFallback (=> HIP source
    # build), NOT a CPU binary.
    _patch_lemonade(monkeypatch)
    with pytest.raises(PrebuiltFallback):
        direct_linux_release_plan(
            stub_unsloth_linux_release(), linux_amd(gfx), "unslothai/llama.cpp", "latest"
        )


@pytest.mark.parametrize("gfx", UNSUPPORTED_GFX)
def test_unknown_amd_gfx_windows_falls_to_upstream_hip(monkeypatch, gfx):
    # Windows ROCm host + no lemonade asset -> upstream HIP prebuilt (GPU),
    # which still precedes the CPU asset.
    _patch_lemonade(monkeypatch)
    plan = direct_upstream_release_plan(
        stub_upstream_release(), windows_amd(gfx), "ggml-org/llama.cpp", "latest"
    )
    kinds = [a.install_kind for a in plan.attempts]
    assert "windows-hip" in kinds
    assert kinds.index("windows-hip") < kinds.index("windows-cpu")
    hip = next(a for a in plan.attempts if a.install_kind == "windows-hip")
    assert hip.source_label == "upstream"
    assert all(a.source_label != "lemonade" for a in plan.attempts)


# --------------------------------------------------------------------------- #
# 7. Opt-out + URL trust
# --------------------------------------------------------------------------- #
def test_opt_out_env_disables_lemonade(monkeypatch):
    _patch_lemonade(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DISABLE_LEMONADE_ROCM", "1")
    assert resolve_lemonade_rocm_choice(linux_amd("gfx1151"), "ubuntu", "linux-rocm") is None


def test_offhost_download_url_rejected(monkeypatch):
    bad = {
        "tag_name": _STUB_TAG,
        "assets": [{
            "name": f"llama-{_STUB_TAG}-ubuntu-rocm-gfx1151-x64.zip",
            "browser_download_url": "https://attacker.invalid/llama.zip",
        }],
    }
    monkeypatch.setattr(_mod, "fetch_json", lambda *a, **k: bad)
    assert resolve_lemonade_rocm_choice(linux_amd("gfx1151"), "ubuntu", "linux-rocm") is None


# --------------------------------------------------------------------------- #
# 8. Full AMD catalog (new + old + data center): family mapping + detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gfx,product,era,expected_family", AMD_CATALOG)
def test_full_catalog_family_and_detection(gfx, product, era, expected_family):
    # The gfx must be detected from a rocminfo/hipinfo-style probe ...
    probe = f"***\nAgent 1\n***\n  Name:                    {gfx}\n  ISA: amdgcn-amd-amdhsa--{gfx}"
    assert _mod._pick_rocm_gfx_target(probe) == gfx, f"{gfx} ({product}) not detected"
    # ... and map to exactly the documented lemonade family (or None).
    assert _lemonade_gfx_family(gfx) == expected_family, (
        f"{gfx} ({product}, {era}) family mismatch"
    )


# --------------------------------------------------------------------------- #
# 9. Data center / Instinct (CDNA) GPUs: NO lemonade prebuilt -> fall through
#    (Linux -> PrebuiltFallback => HIP source build; never a silent CPU binary)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gfx", CATALOG_DATACENTER)
def test_datacenter_gpu_has_no_lemonade_prebuilt(monkeypatch, gfx):
    _patch_lemonade(monkeypatch)
    # No lemonade GPU prebuilt for any data center accelerator, on any OS.
    assert resolve_lemonade_rocm_choice(linux_amd(gfx), "ubuntu", "linux-rocm") is None
    assert resolve_lemonade_rocm_choice(windows_amd(gfx), "windows", "windows-hip") is None
    # Linux Instinct host -> source-build fallback, NOT a CPU downgrade.
    with pytest.raises(PrebuiltFallback):
        direct_linux_release_plan(
            stub_unsloth_linux_release(), linux_amd(gfx), "unslothai/llama.cpp", "latest"
        )


# --------------------------------------------------------------------------- #
# 10. Old GPUs (Vega10 / RDNA1): also not in lemonade -> fall through
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gfx", CATALOG_OLD)
def test_old_gpu_falls_through(monkeypatch, gfx):
    _patch_lemonade(monkeypatch)
    assert resolve_lemonade_rocm_choice(linux_amd(gfx), "ubuntu", "linux-rocm") is None
    with pytest.raises(PrebuiltFallback):
        direct_linux_release_plan(
            stub_unsloth_linux_release(), linux_amd(gfx), "unslothai/llama.cpp", "latest"
        )


# --------------------------------------------------------------------------- #
# 11. Every covered consumer gfx (incl. RDNA2 6700/6600, RDNA4) -> lemonade GPU
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gfx", CATALOG_COVERED)
def test_every_covered_gfx_gets_lemonade(monkeypatch, gfx):
    _patch_lemonade(monkeypatch)
    family = _lemonade_gfx_family(gfx)
    for host, prefix, kind in (
        (linux_amd(gfx), "ubuntu", "linux-rocm"),
        (windows_amd(gfx), "windows", "windows-hip"),
    ):
        c = resolve_lemonade_rocm_choice(host, prefix, kind, llama_tag = "latest")
        assert c is not None and c.repo == "lemonade-sdk/llamacpp-rocm"
        assert c.name == f"llama-{_STUB_TAG}-{prefix}-rocm-{family}-x64.zip"


# --------------------------------------------------------------------------- #
# Standalone matrix printer (python test_amd_spoof_prebuilt_matrix.py)
# --------------------------------------------------------------------------- #
def _build_matrix_rows() -> list[tuple[str, str, str, str]]:
    """Return (OS, GPU, selected binary, source) rows for the spoof matrix."""
    import unittest.mock as _um
    rows: list[tuple[str, str, str, str]] = []
    fam = GFX_TO_FAMILY

    def _resolve(host, prefix, kind):
        with _um.patch.object(_mod, "fetch_json", lambda *a, **k: stub_lemonade_release()):
            cache = _mod._fetch_lemonade_release_cached
            cache.cache_clear()
            try:
                return resolve_lemonade_rocm_choice(host, prefix, kind, llama_tag = "latest")
            finally:
                cache.cache_clear()

    for gfx in ALL_GFX:
        c = _resolve(linux_amd(gfx), "ubuntu", "linux-rocm")
        rows.append(("Linux/WSL", f"AMD {gfx} ({fam[gfx]})", c.name if c else "-",
                     c.repo if c else "-"))
    for gfx in ALL_GFX:
        c = _resolve(windows_amd(gfx), "windows", "windows-hip")
        rows.append(("Windows", f"AMD {gfx} ({fam[gfx]})", c.name if c else "-",
                     c.repo if c else "-"))
    for gfx in UNSUPPORTED_GFX:
        c = _resolve(linux_amd(gfx), "ubuntu", "linux-rocm")
        rows.append(("Linux/WSL", f"AMD {gfx} (unsupported)",
                     "lemonade->None => upstream/source HIP build", "-"))
    rows.append(("Linux/WSL", "NVIDIA", "linux-cuda prebuilt", "ggml-org/llama.cpp"))
    rows.append(("Windows", "NVIDIA", "windows-cuda prebuilt", "ggml-org/llama.cpp"))
    rows.append(("Linux/WSL", "none (CPU)", f"llama-{_STUB_TAG}-bin-ubuntu-x64.tar.gz",
                 "ggml-org/llama.cpp"))
    rows.append(("Windows", "none (CPU)", f"llama-{_STUB_TAG}-bin-win-cpu-x64.zip",
                 "ggml-org/llama.cpp"))
    rows.append(("macOS arm64", "Metal", f"llama-{_STUB_TAG}-bin-macos-arm64.tar.gz",
                 "ggml-org/llama.cpp"))
    rows.append(("macOS x64", "none (CPU)", f"llama-{_STUB_TAG}-bin-macos-x64.tar.gz",
                 "ggml-org/llama.cpp"))
    return rows


def _build_catalog_rows() -> list[tuple[str, str, str, str]]:
    """Return (era, gfx/product, lemonade prebuilt?, what gets installed) rows
    for the full new+old+data center AMD catalog."""
    import unittest.mock as _um
    rows: list[tuple[str, str, str, str]] = []
    with _um.patch.object(_mod, "fetch_json", lambda *a, **k: stub_lemonade_release()):
        for gfx, product, era, expected in AMD_CATALOG:
            _mod._fetch_lemonade_release_cached.cache_clear()
            fam = _lemonade_gfx_family(gfx)
            assert fam == expected
            if fam is not None:
                prebuilt = f"YES ({fam})"
                installed = f"lemonade {fam} GPU prebuilt"
            else:
                prebuilt = "no"
                installed = "Linux: HIP source build / Win: upstream HIP prebuilt"
            rows.append((era, f"{gfx}  {product}", prebuilt, installed))
    _mod._fetch_lemonade_release_cached.cache_clear()
    return rows


def main() -> None:
    rows = _build_matrix_rows()
    w0 = max(len(r[0]) for r in rows) + 2
    w1 = max(len(r[1]) for r in rows) + 2
    w2 = max(len(r[2]) for r in rows) + 2
    print(f"\nAMD-spoof prebuilt llama.cpp selection matrix (tag {_STUB_TAG})")
    print(f"running on: {sys.platform}\n")
    hdr = f"{'OS':<{w0}}{'GPU':<{w1}}{'selected binary':<{w2}}{'binary source'}"
    print(hdr)
    print("-" * len(hdr))
    for os_, gpu, binary, src in rows:
        print(f"{os_:<{w0}}{gpu:<{w1}}{binary:<{w2}}{src}")
    print()

    crows = _build_catalog_rows()
    c0 = max(len(r[0]) for r in crows) + 2
    c1 = max(len(r[1]) for r in crows) + 2
    c2 = max(len(r[2]) for r in crows) + 2
    print("Full AMD catalog (new + old + data center) lemonade prebuilt coverage\n")
    chdr = f"{'era':<{c0}}{'gfx / product':<{c1}}{'lemonade prebuilt?':<{c2}}{'what gets installed'}"
    print(chdr)
    print("-" * len(chdr))
    for era, prod, prebuilt, installed in crows:
        print(f"{era:<{c0}}{prod:<{c1}}{prebuilt:<{c2}}{installed}")
    print()


if __name__ == "__main__":
    main()
