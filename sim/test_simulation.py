"""Comprehensive offline simulation of the NVIDIA prebuilt selector change.

Loads the pre-PR (old) and post-PR (new) modules side by side and proves:
  1. the bug existed before and is fixed after,
  2. behaviour is IDENTICAL for every already-supported driver (no regression),
  3. the new code never crashes across an exhaustive driver x OS x upstream grid,
  4. forward cases (13.4, 14.x, multi-digit minors) resolve correctly,
  5. macOS / non-NVIDIA / edge inputs degrade gracefully.

Pure stdlib; runs in an isolated uv venv with only pytest installed.
"""
from __future__ import annotations

import contextlib
import importlib.util
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load_mod(filename, name):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


OLD = load_mod("old_ilp.py", "old_ilp")
NEW = load_mod("new_ilp.py", "new_ilp")

TAG = "b9999"


def mk_host(mod, system, driver, sm="120"):
    return mod.HostInfo(
        system=system,
        machine=("AMD64" if system == "Windows" else ("arm64" if system == "Darwin" else "x86_64")),
        is_windows=system == "Windows",
        is_linux=system == "Linux",
        is_macos=system == "Darwin",
        is_x86_64=system != "Darwin",
        is_arm64=system == "Darwin",
        nvidia_smi=None if system == "Darwin" else "nvidia-smi",
        driver_cuda_version=driver,
        compute_caps=[sm] if sm else [],
        visible_cuda_devices=None,
        has_physical_nvidia=system != "Darwin",
        has_usable_nvidia=system != "Darwin",
    )


@contextlib.contextmanager
def win_detection(mod, lines):
    orig = mod.detected_windows_runtime_lines
    mod.detected_windows_runtime_lines = lambda: (list(lines), {l: ["C:\\x"] for l in lines})
    try:
        yield
    finally:
        mod.detected_windows_runtime_lines = orig


@contextlib.contextmanager
def lin_detection(mod, lines):
    orig = mod.detected_linux_runtime_lines
    mod.detected_linux_runtime_lines = lambda: (list(lines), {l: ["/x"] for l in lines})
    try:
        yield
    finally:
        mod.detected_linux_runtime_lines = orig


def win_assets(*minors, cudart=True):
    d = {}
    for m in minors:
        d[f"llama-{TAG}-bin-win-cuda-{m}-x64.zip"] = "u"
        if cudart:
            d[f"cudart-llama-bin-win-cuda-{m}-x64.zip"] = "u"
    return d


def win_select(mod, system, driver, assets, detected=()):
    host = mk_host(mod, system, driver)
    with win_detection(mod, detected):
        attempts = mod.windows_cuda_attempts(host, TAG, assets, None)
    return attempts[0].name if attempts else None


def lin_release(mod, *targets):
    names = [f"app-{TAG}-linux-x64-{t}.tar.gz" for t in targets]
    rel = {"tag_name": TAG,
           "assets": [{"name": n, "browser_download_url": "https://x/" + n} for n in names]}
    return mod.parse_direct_linux_release_bundle("unslothai/llama.cpp", rel)


def lin_select(mod, driver, bundle, detected, sm="120"):
    if bundle is None:
        return None
    host = mk_host(mod, "Linux", driver, sm=sm)
    with lin_detection(mod, detected):
        sel = mod.linux_cuda_choice_from_release(host, bundle)
    return sel.attempts[0].name if sel and sel.attempts else None


# Drivers spanning unsupported, every supported minor, and the future.
ALL_DRIVERS = [
    None, (11, 8), (12, 0), (12, 3), (12, 4), (12, 6), (12, 8),
    (13, 0), (13, 1), (13, 2), (13, 3), (13, 4), (13, 9), (13, 10),
    (14, 0), (14, 7), (15, 2), (99, 9),
]
SUPPORTED_TODAY = [d for d in ALL_DRIVERS if d and (d[0] == 13 or (d[0] == 12 and d[1] >= 4))]


# ---------------------------------------------------------------------------
# 1. Backward compatibility: identical line derivation for supported drivers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("driver", ALL_DRIVERS)
def test_windows_runtime_lines_unchanged_for_existing_majors(driver):
    host_old = mk_host(OLD, "Windows", driver)
    host_new = mk_host(NEW, "Windows", driver)
    old = OLD.compatible_windows_runtime_lines(host_old)
    new = NEW.compatible_windows_runtime_lines(host_new)
    if driver and driver[0] >= 14:
        # new adds the future major first, then the same older tail
        assert new[-len(old):] == old or old == []
    else:
        assert old == new, f"driver {driver}: old={old} new={new}"


@pytest.mark.parametrize("driver", ALL_DRIVERS)
def test_linux_runtime_lines_unchanged_for_existing_majors(driver):
    old = OLD.compatible_linux_runtime_lines(mk_host(OLD, "Linux", driver))
    new = NEW.compatible_linux_runtime_lines(mk_host(NEW, "Linux", driver))
    if driver and driver[0] >= 14:
        assert new[-len(old):] == old or old == []
    else:
        assert old == new, f"driver {driver}: old={old} new={new}"


# ---------------------------------------------------------------------------
# 2. The bug, before and after (real ggml-org b9437 asset shape: 12.4 + 13.3)
# ---------------------------------------------------------------------------
def test_13_3_driver_old_misses_new_fixes():
    # The #5861 case: driver advertises 13.3, ggml-org ships 12.4 + 13.3.
    assets = win_assets("12.4", "13.3")
    assert win_select(OLD, "Windows", (13, 3), assets) == f"llama-{TAG}-bin-win-cuda-12.4-x64.zip"
    assert win_select(NEW, "Windows", (13, 3), assets) == f"llama-{TAG}-bin-win-cuda-13.3-x64.zip"


@pytest.mark.parametrize("driver", [(13, 0), (13, 1), (13, 2)])
def test_sub_13_3_driver_gated_to_12_4(driver):
    # A driver below 13.3 cannot safely run the 13.3 build, so the new code
    # gates it to the 12.4 build (guaranteed by backward compatibility).
    assets = win_assets("12.4", "13.3")
    assert win_select(NEW, "Windows", driver, assets) == f"llama-{TAG}-bin-win-cuda-12.4-x64.zip"


def test_cuda12_host_identical_old_and_new():
    # A plain CUDA 12 host must behave exactly as before.
    assets = win_assets("12.4", "13.3")
    assert win_select(OLD, "Windows", (12, 8), assets) == win_select(NEW, "Windows", (12, 8), assets)


# ---------------------------------------------------------------------------
# 3. Exhaustive no-crash grid on the new code
# ---------------------------------------------------------------------------
UPSTREAMS = {
    "real_b9437": win_assets("12.4", "13.3"),
    "legacy_13_1": win_assets("12.4", "13.1"),
    "bump_13_4": win_assets("12.4", "13.4"),
    "new_major_14": win_assets("12.4", "13.3", "14.0"),
    "empty": {},
    "cudart_only": win_assets("12.4", "13.3", cudart=True),
    "multi_digit_minor": win_assets("12.4", "13.9", "13.10"),
    "weird": {"not-a-llama-asset.zip": "u", "llama-x-bin-win-cuda-x.y-x64.zip": "u"},
}


@pytest.mark.parametrize("system", ["Windows", "Linux", "Darwin"])
@pytest.mark.parametrize("driver", ALL_DRIVERS)
@pytest.mark.parametrize("up_name", list(UPSTREAMS))
def test_windows_attempts_never_crashes(system, driver, up_name):
    # windows_cuda_attempts is only called for Windows hosts in production, but
    # it must not raise for any host/asset combination.
    result = win_select(NEW, system, driver, UPSTREAMS[up_name], detected=())
    assert result is None or result.endswith("-x64.zip")


@pytest.mark.parametrize("driver", ALL_DRIVERS)
@pytest.mark.parametrize("detected", [(), ("cuda13", "cuda12"), ("cuda14", "cuda13", "cuda12")])
def test_windows_with_detected_dlls_never_crashes(driver, detected):
    result = win_select(NEW, "Windows", driver, win_assets("12.4", "13.3", "14.0"), detected=detected)
    assert result is None or result.endswith("-x64.zip")


# ---------------------------------------------------------------------------
# 4. Forward correctness: minor bump, multi-digit minor, new major
# ---------------------------------------------------------------------------
def test_multi_digit_minor_uses_int_not_string_order():
    # 13.10 must beat 13.9 (int compare, not lexicographic).
    assert win_select(NEW, "Windows", (13, 10), win_assets("12.4", "13.9", "13.10")) \
        == f"llama-{TAG}-bin-win-cuda-13.10-x64.zip"


def test_new_major_selected_only_when_driver_supports_it():
    assets = win_assets("12.4", "13.3", "14.0")
    # 14 driver with cuda14 libs present -> cuda14
    assert win_select(NEW, "Windows", (14, 0), assets, detected=("cuda14", "cuda13", "cuda12")) \
        == f"llama-{TAG}-bin-win-cuda-14.0-x64.zip"
    # 13 driver must NEVER get the 14.0 build (it cannot run it)
    assert win_select(NEW, "Windows", (13, 3), assets, detected=("cuda13", "cuda12")) \
        == f"llama-{TAG}-bin-win-cuda-13.3-x64.zip"


def test_new_major_degrades_when_no_cuda14_build():
    assert win_select(NEW, "Windows", (14, 0), win_assets("12.4", "13.3"), detected=("cuda13", "cuda12")) \
        == f"llama-{TAG}-bin-win-cuda-13.3-x64.zip"


# ---------------------------------------------------------------------------
# 5. Linux exhaustive + future-major + differential
# ---------------------------------------------------------------------------
LINUX_REAL = ("cuda12-older", "cuda12-newer", "cuda12-portable",
              "cuda13-older", "cuda13-newer", "cuda13-portable")


@pytest.mark.parametrize("driver", ALL_DRIVERS)
def test_linux_real_release_never_crashes(driver):
    bundle = lin_release(NEW, *LINUX_REAL)
    detected = ["cuda14", "cuda13", "cuda12"]
    result = lin_select(NEW, driver, bundle, detected)
    assert result is None or result.endswith(".tar.gz")


def test_linux_existing_selection_matches_old():
    # For supported drivers, old and new pick the same linux bundle.
    bundle_old = lin_release(OLD, *LINUX_REAL)
    bundle_new = lin_release(NEW, *LINUX_REAL)
    for driver in SUPPORTED_TODAY:
        old = lin_select(OLD, driver, bundle_old, ["cuda13", "cuda12"])
        new = lin_select(NEW, driver, bundle_new, ["cuda13", "cuda12"])
        assert old == new, f"driver {driver}: old={old} new={new}"


def test_linux_future_major_parsed_and_selected():
    bundle = lin_release(NEW, "cuda12-portable", "cuda13-portable", "cuda14-portable")
    assert lin_select(NEW, (14, 0), bundle, ["cuda14", "cuda13", "cuda12"]) \
        == f"app-{TAG}-linux-x64-cuda14-portable.tar.gz"
    # old code cannot even parse a cuda14 bundle
    assert lin_release(OLD, "cuda14-portable") is None


def test_linux_cpu_bundle_still_parses():
    bundle = lin_release(NEW, "cuda13-newer")
    rel = {"tag_name": TAG, "assets": [
        {"name": f"app-{TAG}-linux-x64.tar.gz", "browser_download_url": "u"},
        {"name": f"app-{TAG}-linux-x64-cpu.tar.gz", "browser_download_url": "u"},
        {"name": f"app-{TAG}-linux-x64-cuda13-newer.tar.gz", "browser_download_url": "u"},
    ]}
    b = NEW.parse_direct_linux_release_bundle("unslothai/llama.cpp", rel)
    kinds = sorted({a.install_kind for a in b.artifacts})
    assert "linux-cpu" in kinds and "linux-cuda" in kinds


# ---------------------------------------------------------------------------
# 6. Linux asset-name regex battery (no over/under matching)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,should_parse", [
    (f"app-{TAG}-linux-x64-cuda12-newer.tar.gz", True),
    (f"app-{TAG}-linux-x64-cuda13-portable.tar.gz", True),
    (f"app-{TAG}-linux-x64-cuda14-older.tar.gz", True),
    (f"app-{TAG}-linux-x64-cuda130-newer.tar.gz", True),
    (f"app-{TAG}-linux-x64.tar.gz", True),       # cpu
    (f"app-{TAG}-linux-x64-cpu.tar.gz", True),   # cpu
    (f"app-{TAG}-linux-x64-cuda13beta-newer.tar.gz", False),
    (f"app-{TAG}-linux-x64-cuda13-fancy.tar.gz", False),
    (f"app-{TAG}-linux-arm64-cuda13-newer.tar.gz", False),
    (f"llama-{TAG}-bin-win-cuda-13.3-x64.zip", False),
])
def test_linux_regex_battery(name, should_parse):
    rel = {"tag_name": TAG, "assets": [{"name": name, "browser_download_url": "u"}]}
    bundle = NEW.parse_direct_linux_release_bundle("unslothai/llama.cpp", rel)
    parsed = bundle is not None and len(bundle.artifacts) > 0
    assert parsed == should_parse, f"{name}: parsed={parsed} expected={should_parse}"


# ---------------------------------------------------------------------------
# 7. macOS / non-NVIDIA / edge inputs degrade gracefully
# ---------------------------------------------------------------------------
def test_macos_host_yields_no_cuda_lines():
    host = mk_host(NEW, "Darwin", None)
    assert NEW.compatible_windows_runtime_lines(host) == []
    assert NEW.compatible_linux_runtime_lines(host) == []


@pytest.mark.parametrize("driver", [None, (0, 0), (11, 8), (12, 3)])
def test_unsupported_drivers_select_nothing(driver):
    assert win_select(NEW, "Windows", driver, win_assets("12.4", "13.3")) is None


def test_empty_and_garbage_upstream():
    assert win_select(NEW, "Windows", (13, 3), {}) is None
    assert win_select(NEW, "Windows", (13, 3), {"garbage.zip": "u"}) is None


# ---------------------------------------------------------------------------
# 8. Performance: expanded per-major probing stays cheap
# ---------------------------------------------------------------------------
def test_runtime_line_info_generation_is_fast():
    t = time.perf_counter()
    for _ in range(10000):
        NEW.windows_runtime_line_info()
    assert time.perf_counter() - t < 2.0


def test_detection_generation_count():
    # 8 majors (12..19) x correct lib naming, newest first.
    info = NEW.windows_runtime_line_info()
    assert list(info)[0] == "cuda19" and list(info)[-1] == "cuda12"
    assert info["cuda14"] == ("cudart64_14*.dll", "cublas64_14*.dll", "cublasLt64_14*.dll")
