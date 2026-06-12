"""Import + call safety on non-Apple platforms.

apple.py must import cleanly and its public readers must return None (never
raise) on Linux / Windows / Intel-Mac, since the macOS frameworks are absent.
This is what keeps the PR backwards compatible for every non-MLX host.
"""

import importlib

import pytest


def test_module_imports_with_no_macos_frameworks(apple):
    # The fixture already imported it; assert the public surface exists.
    assert hasattr(apple, "read_gpu_temperature_c")
    assert hasattr(apple, "read_gpu_power_w")


def test_no_cdll_loaded_at_import_time(apple):
    # The IOKit/CoreFoundation/IOReport dylibs are only opened lazily inside the
    # read functions, never at module import -> importing on Linux is safe.
    src = importlib.util.find_spec("utils.hardware.apple").origin
    with open(src) as f:
        text = f.read()
    # ctypes.CDLL must appear only inside the _load_* helpers, not at col 0.
    module_level_cdll = [
        ln for ln in text.splitlines()
        if ln.startswith("ctypes.CDLL") or ln.startswith("    ctypes.CDLL(")
        and "def " not in ln
    ]
    assert module_level_cdll == []


def test_reads_return_none_on_linux(apple):
    # Real Linux host: no IOKit -> graceful None, no exception.
    assert apple.read_gpu_temperature_c() is None
    assert apple.read_gpu_power_w() is None


def test_reads_latch_after_first_failure(apple):
    apple.read_gpu_temperature_c()
    apple.read_gpu_power_w()
    assert apple._smc_failed is True
    assert apple._energy_failed is True


@pytest.mark.parametrize("system,machine", [
    ("Windows", "AMD64"),
    ("Linux", "x86_64"),
    ("Darwin", "x86_64"),   # Intel Mac
])
def test_reads_none_on_simulated_platforms(apple, monkeypatch, system, machine):
    import platform as _platform
    monkeypatch.setattr(_platform, "system", lambda: system)
    monkeypatch.setattr(_platform, "machine", lambda: machine)
    # Frameworks still absent on this Linux box -> None everywhere, no raise.
    assert apple.read_gpu_temperature_c() is None
    assert apple.read_gpu_power_w() is None
