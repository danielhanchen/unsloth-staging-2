"""Shared fixtures for the PR #6187 cross-OS simulation suite.

Path-relative so it runs from a fresh checkout on any OS: it adds the studio
backend root (and this dir, for apple_sim_mocks) to sys.path, imports the real
utils.hardware.apple under test, and resets its failure-latched singletons
between cases so they do not leak state.
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.abspath(os.path.join(HERE, "..", ".."))  # tests/sim_pr6187 -> backend
for _p in (BACKEND, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture
def apple():
    from utils.hardware import apple as mod

    mod._smc = None
    mod._smc_failed = False
    mod._energy = None
    mod._energy_failed = False
    yield mod
    mod._smc = None
    mod._smc_failed = False
    mod._energy = None
    mod._energy_failed = False


class Clock:
    def __init__(self, values):
        self._values = list(values)
        self._i = 0

    def monotonic(self):
        v = self._values[min(self._i, len(self._values) - 1)]
        self._i += 1
        return v

    def sleep(self, _):
        pass


@pytest.fixture
def clock_factory(apple, monkeypatch):
    def _make(values):
        clk = Clock(values)
        monkeypatch.setattr(apple, "time", clk)
        return clk

    return _make
