"""End-to-end simulation of the IOReport GPU-power path on Linux.

Exercises real subscription/baseline/diff windowing, GPU-channel selection,
energy->watts conversion and failure latching against a scripted fake
IOReport. Also reproduces the Codex P2 finding (subscribed-channels descriptor).
"""

import pytest

from apple_sim_mocks import (
    FakeIOR,
    SUBSCRIBED_SENTINEL,
    gpu_channel,
    install_fake_ioreport,
)


@pytest.fixture
def power(apple, monkeypatch, clock_factory):
    """Helper returning a function that runs N polls against a scripted IOR."""

    def run(windows, clock_values, **ior_kwargs):
        ior = FakeIOR(windows, **ior_kwargs)
        install_fake_ioreport(apple, monkeypatch, ior)
        clock_factory(clock_values)
        results = [apple.read_gpu_power_w() for _ in clock_values]
        return ior, results

    return run


def test_power_first_poll_is_baseline_none(power):
    ior, results = power(
        windows=[[], [gpu_channel("GPU Energy", "mJ", 2000)]],
        clock_values=[0.0, 2.0],
    )
    assert results[0] is None            # baseline
    assert results[1] == pytest.approx(1.0)   # 2 J over 2 s = 1 W


def test_power_sums_gpu_die_variants(power):
    ior, results = power(
        windows=[
            [],
            [
                gpu_channel("GPU Energy", "mJ", 2000),        # 1.0 W
                gpu_channel("DIE_1_GPU Energy", "mJ", 1000),  # 0.5 W
            ],
        ],
        clock_values=[0.0, 2.0],
    )
    assert results[1] == pytest.approx(1.5)


def test_power_excludes_sram_cpu_ane(power):
    ior, results = power(
        windows=[
            [],
            [
                gpu_channel("GPU Energy", "mJ", 2000),        # counted -> 1.0 W
                gpu_channel("GPU SRAM Energy", "mJ", 8000),   # excluded
                gpu_channel("CPU Energy", "mJ", 9000),        # excluded
                gpu_channel("ANE Energy", "mJ", 4000),        # excluded
            ],
        ],
        clock_values=[0.0, 2.0],
    )
    assert results[1] == pytest.approx(1.0)


def test_power_microjoule_units(power):
    ior, results = power(
        windows=[[], [gpu_channel("GPU Energy", "uJ", 5_000_000)]],
        clock_values=[0.0, 1.0],
    )
    assert results[1] == pytest.approx(5.0)


def test_power_zero_draw_is_zero_not_none(power):
    ior, results = power(
        windows=[[], [gpu_channel("GPU Energy", "mJ", 0)]],
        clock_values=[0.0, 1.0],
    )
    assert results[1] == pytest.approx(0.0)


def test_power_no_gpu_channels_returns_none(power):
    ior, results = power(
        windows=[[], [gpu_channel("CPU Energy", "mJ", 9000)]],
        clock_values=[0.0, 1.0],
    )
    assert results[1] is None


def test_power_unknown_unit_channel_skipped(power):
    ior, results = power(
        windows=[[], [gpu_channel("GPU Energy", "J", 5)]],
        clock_values=[0.0, 1.0],
    )
    assert results[1] is None


def test_power_zero_elapsed_returns_none(power):
    # Two back-to-back polls with an identical clock reading -> div-by-zero guard.
    ior, results = power(
        windows=[[], [gpu_channel("GPU Energy", "mJ", 2000)]],
        clock_values=[5.0, 5.0],
    )
    assert results[1] is None


def test_power_subscription_failure_sets_latch(apple, monkeypatch, clock_factory):
    ior = FakeIOR(windows=[[], []], subscription_ok=False)
    install_fake_ioreport(apple, monkeypatch, ior)
    clock_factory([0.0, 1.0])
    assert apple.read_gpu_power_w() is None
    assert apple._energy_failed is True


def test_power_sample_none_returns_none(power):
    ior, results = power(
        windows=[[], []],
        clock_values=[0.0, 1.0],
        samples_ok=False,
    )
    assert results == [None, None]


def test_power_releases_previous_sample(apple, monkeypatch, clock_factory):
    ior = FakeIOR(windows=[[], [gpu_channel("GPU Energy", "mJ", 2000)]])
    install_fake_ioreport(apple, monkeypatch, ior)
    clock_factory([0.0, 2.0])
    apple.read_gpu_power_w()  # baseline, sample id 1
    apple.read_gpu_power_w()  # diff, releases prev sample id 1
    released = apple._energy._cf.released
    assert 1 in released  # the previous sample handle was CFRelease'd


def test_power_never_raises_on_broken_loader(apple, monkeypatch):
    monkeypatch.setattr(
        apple, "_load_ioreport",
        lambda: (_ for _ in ()).throw(OSError("dylib missing")),
    )
    assert apple.read_gpu_power_w() is None


# --------- Codex P2: must sample with the subscribed-channels descriptor ---------

def test_power_uses_subscribed_channels_descriptor(power):
    """IOReportCreateSubscription writes a subscribed-channels descriptor to its
    out-param; sampling must use *that*, not the original channel group (this is
    what macmon does). Pre-fix this fails: the code samples with ORIG_CHANNELS."""
    ior, results = power(
        windows=[[], [gpu_channel("GPU Energy", "mJ", 2000)]],
        clock_values=[0.0, 2.0],
    )
    assert ior.sample_channel_args, "IOReportCreateSamples was never called"
    assert all(arg == SUBSCRIBED_SENTINEL for arg in ior.sample_channel_args), (
        f"sampled with {ior.sample_channel_args!r}; expected the subscribed "
        f"descriptor {SUBSCRIBED_SENTINEL!r}"
    )
