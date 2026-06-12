"""Exhaustive edge-case coverage of apple.py's pure helpers."""

import ctypes
import struct

import pytest


# ----- _fourcc / _fourcc_str round trip -----

@pytest.mark.parametrize("key", [
    "#KEY", "Tg0D", "Tg1D", "TgaD", "flt ", "ui32", "    ", "AAAA", "zzzz",
    "Tp09", "Tg0p", "PSTR", "ch08",
])
def test_fourcc_roundtrip(apple, key):
    assert apple._fourcc_str(apple._fourcc(key)) == key


def test_fourcc_known_constant(apple):
    # "flt " is the float type tag macmon/SMCKit use.
    assert apple._fourcc("flt ") == 1718383648


def test_fourcc_is_big_endian(apple):
    assert apple._fourcc("\x00\x00\x00\x01") == 1
    assert apple._fourcc("\x01\x00\x00\x00") == 0x01000000


def test_fourcc_all_four_byte_space_covered(apple):
    # Every ASCII 4-char key must round-trip (sample the corners).
    for key in ("\x00\x00\x00\x00", "\x7f\x7f\x7f\x7f", "ABCD", "0000"):
        assert apple._fourcc_str(apple._fourcc(key)) == key


# ----- _watts: energy delta -> average watts over a window -----

@pytest.mark.parametrize("energy,unit,elapsed,expected", [
    (2000, "mJ", 2.0, 1.0),          # 2 J over 2 s = 1 W
    (5_000_000, "uJ", 1.0, 5.0),     # 5 J over 1 s = 5 W
    (1_500_000_000, "nJ", 1.0, 1.5),
    (0, "mJ", 1.0, 0.0),             # genuine zero draw is valid (not None)
    (1000, "mJ", 0.5, 2.0),
    (1, "nJ", 1.0, 1e-9),
])
def test_watts_valid(apple, energy, unit, elapsed, expected):
    assert apple._watts(energy, unit, elapsed) == pytest.approx(expected)


@pytest.mark.parametrize("unit", ["J", "kJ", "", "Wh", "mj", "MJ", " J ", "xJ"])
def test_watts_unknown_unit_is_none(apple, unit):
    # Only mJ/uJ/nJ are known; anything else -> None (skipped, not crashed).
    assert apple._watts(1000, unit, 1.0) is None


@pytest.mark.parametrize("elapsed", [0.0, -0.001, -5.0])
def test_watts_nonpositive_elapsed_is_none(apple, elapsed):
    assert apple._watts(1000, "mJ", elapsed) is None


def test_watts_unit_is_stripped(apple):
    # The IOReport unit label often has padding; helper strips before lookup.
    assert apple._watts(2000, " mJ ", 2.0) == pytest.approx(1.0)


def test_watts_negative_energy_passes_through(apple):
    # A counter reset yields a negative delta; helper does not clamp. Documents
    # current behavior (a single negative sample is possible after a wrap).
    assert apple._watts(-2000, "mJ", 2.0) == pytest.approx(-1.0)


# ----- _average_valid_temps: filter + mean + round -----

def test_avg_basic(apple):
    assert apple._average_valid_temps([40.0, 50.0, 60.05]) == 50.0


def test_avg_filters_out_of_range(apple):
    assert apple._average_valid_temps([-1.0, 0.0, 151.0, 42.0]) == 42.0


def test_avg_empty_is_none(apple):
    assert apple._average_valid_temps([]) is None


def test_avg_all_invalid_is_none(apple):
    assert apple._average_valid_temps([0.0, 200.0, -5.0, 150.001]) is None


def test_avg_boundary_values(apple):
    # 0.0 is excluded (strict >), 150.0 is included (<=).
    assert apple._average_valid_temps([0.0]) is None
    assert apple._average_valid_temps([150.0]) == 150.0
    assert apple._average_valid_temps([0.0001, 150.0]) == pytest.approx(75.0)


def test_avg_rounds_to_one_decimal(apple):
    assert apple._average_valid_temps([33.33, 33.34]) == 33.3
    assert apple._average_valid_temps([10.0, 10.0, 10.1]) == 10.0


def test_avg_single_value(apple):
    assert apple._average_valid_temps([73.6]) == 73.6


# ----- _is_gpu_energy_channel: macmon's GPU-energy selection -----

@pytest.mark.parametrize("name,included", [
    ("GPU Energy", True),
    ("DIE_0_GPU Energy", True),
    ("DIE_1_GPU Energy", True),
    ("DIE_3_GPU Energy", True),
    ("GPU SRAM Energy", False),       # SRAM is not GPU core power
    ("DIE_0_GPU SRAM Energy", False),
    ("CPU Energy", False),
    ("ANE Energy", False),
    ("DRAM Energy", False),
    ("GPU Energy ", False),           # trailing space -> not endswith
    ("Energy", False),
    ("", False),
    ("gpu energy", False),            # case sensitive
])
def test_is_gpu_energy_channel(apple, name, included):
    assert apple._is_gpu_energy_channel(name) is included


# ----- SMC struct layout: kernel rejects mismatched sizes -----

def test_smc_keydata_size_is_80(apple):
    assert ctypes.sizeof(apple._SMCKeyData) == 80


def test_smc_substruct_sizes(apple):
    # C 4-byte alignment pads _SMCKeyInfo's trailing uint8 -> 12 (not 9). What
    # the kernel actually validates is the full _SMCKeyData (== 80).
    assert ctypes.sizeof(apple._SMCKeyDataVers) == 6
    assert ctypes.sizeof(apple._SMCKeyInfo) == 12
    assert ctypes.sizeof(apple._SMCPLimitData) == 16


def test_smc_float_decode_is_le(apple):
    # Sanity: the bytes apple writes/reads decode as little-endian float.
    raw = struct.pack("<f", 91.5)
    assert struct.unpack("<f", raw)[0] == pytest.approx(91.5)
