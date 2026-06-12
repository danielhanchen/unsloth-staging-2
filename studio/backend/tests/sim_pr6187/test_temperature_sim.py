"""End-to-end simulation of the AppleSMC GPU-temperature path on Linux.

Exercises real key discovery, float decode, Tg-prefix filtering, averaging
and the failure-latched public API against an in-memory fake SMC.
"""

import pytest

from apple_sim_mocks import install_fake_smc


def test_temperature_happy_path(apple, monkeypatch):
    # Two GPU die sensors + unrelated CPU/other keys that must be ignored.
    install_fake_smc(apple, monkeypatch, float_keys={
        "Tg0D": 45.0,
        "Tg1D": 55.0,
        "TC0P": 60.0,   # CPU proximity -> not Tg, ignored
        "Tp01": 70.0,   # not Tg, ignored
    })
    # avg(45, 55) = 50.0
    assert apple.read_gpu_temperature_c() == pytest.approx(50.0)


def test_temperature_discovers_only_tg_float_keys(apple, monkeypatch):
    install_fake_smc(apple, monkeypatch, float_keys={
        "Tg0D": 80.0, "Tg2D": 90.0, "TgaB": 100.0,
    })
    # All three are Tg* float keys -> avg(80,90,100)=90.0
    assert apple.read_gpu_temperature_c() == pytest.approx(90.0)


def test_temperature_filters_invalid_readings(apple, monkeypatch):
    install_fake_smc(apple, monkeypatch, float_keys={
        "Tg0D": 0.0,      # invalid (<= 0) -> filtered
        "Tg1D": 200.0,    # invalid (> 150) -> filtered
        "Tg2D": 48.0,     # valid
    })
    assert apple.read_gpu_temperature_c() == pytest.approx(48.0)


def test_temperature_all_invalid_returns_none(apple, monkeypatch):
    install_fake_smc(apple, monkeypatch, float_keys={"Tg0D": 0.0, "Tg1D": 999.0})
    assert apple.read_gpu_temperature_c() is None


def test_temperature_no_tg_keys_returns_none(apple, monkeypatch):
    # Mac/OS without Tg* keys (e.g. macOS < 14): discovery finds none.
    install_fake_smc(apple, monkeypatch, float_keys={"TC0P": 55.0, "TB0T": 30.0})
    assert apple.read_gpu_temperature_c() is None


def test_temperature_open_failure_latches_to_none(apple, monkeypatch):
    install_fake_smc(apple, monkeypatch, float_keys={"Tg0D": 50.0}, fail_open=True)
    assert apple.read_gpu_temperature_c() is None
    assert apple._smc_failed is True


def test_temperature_never_retries_after_failure(apple, monkeypatch):
    calls = {"open": 0}
    orig = apple._SMCConnection._open

    install_fake_smc(apple, monkeypatch, float_keys={"Tg0D": 50.0}, fail_open=True)

    def counting_open(self):
        calls["open"] += 1
        raise OSError("boom")

    monkeypatch.setattr(apple._SMCConnection, "_open", counting_open)
    for _ in range(5):
        assert apple.read_gpu_temperature_c() is None
    # Latched: only the first call attempts to open the connection.
    assert calls["open"] == 1


def test_temperature_connection_is_cached_across_polls(apple, monkeypatch):
    install_fake_smc(apple, monkeypatch, float_keys={"Tg0D": 42.0, "Tg1D": 44.0})
    first = apple.read_gpu_temperature_c()
    conn = apple._smc
    second = apple.read_gpu_temperature_c()
    assert first == second == pytest.approx(43.0)
    # Same connection object reused (not rebuilt every poll).
    assert apple._smc is conn


def test_temperature_bad_result_code_on_one_key_is_filtered(apple, monkeypatch):
    # If a single key read raises mid-stream, it is dropped, others still count.
    install_fake_smc(
        apple, monkeypatch,
        float_keys={"Tg0D": 50.0, "Tg1D": 60.0},
        bad_result_keys=("Tg1D",),
    )
    # Tg1D raises during discovery read_float -> excluded; only Tg0D remains.
    assert apple.read_gpu_temperature_c() == pytest.approx(50.0)


def test_temperature_never_raises(apple, monkeypatch):
    # Even with a totally broken backend, the public API must not raise.
    monkeypatch.setattr(apple, "_load_iokit", lambda: (_ for _ in ()).throw(OSError("x")))
    assert apple.read_gpu_temperature_c() is None
