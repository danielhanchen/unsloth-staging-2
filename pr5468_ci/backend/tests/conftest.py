"""Auto-reset the sandbox module's global state between every test in
this directory so the simulation tests cannot bleed cached probe state
into the PR's own test_sandbox.py."""

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import sandbox as sb  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_sandbox_globals():
    sb._sandbox_available_cache = None
    sb._linux_bwrap_path = None
    yield
    sb._sandbox_available_cache = None
    sb._linux_bwrap_path = None
