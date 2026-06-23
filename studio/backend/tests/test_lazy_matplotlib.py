# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lazy-matplotlib regression tests for core.training.training (PR #6596).

Studio must still boot when matplotlib's native wheel is blocked by a host
policy (e.g. Windows Smart App Control), because importing matplotlib triggers
loading _c_internal_utils, which raises under SAC. PR #6596 moves the import
behind _load_pyplot(); these tests pin that behavior.

GitHub-hosted Windows runners do NOT have Smart App Control, so we SIMULATE the
blocked wheel with a sys.meta_path finder that raises on `matplotlib` and any
`matplotlib.*` submodule, mirroring the real SAC error string.

Each scenario runs in a FRESH subprocess because training._pyplot /
training._pyplot_failed are module globals (cached once) and matplotlib import +
backend choice are sticky in sys.modules. A new interpreter is the only fully
reliable reset for CI.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# .../studio/backend/tests/ -> .../studio/backend
_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _run_child(body: str) -> subprocess.CompletedProcess:
    """Run `body` in a fresh interpreter with PYTHONPATH=studio/backend."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_BACKEND_ROOT) + (os.pathsep + existing if existing else "")
    env.setdefault("MPLBACKEND", "Agg")  # headless + deterministic
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True, text=True, env=env, timeout=120, cwd=str(_BACKEND_ROOT),
    )


_BLOCKED_CHILD = r'''
import sys, importlib, importlib.abc


class _SACBlocker(importlib.abc.MetaPathFinder):
    """Reject matplotlib like Windows Smart App Control blocking the wheel."""
    _MSG = ("An Application Control policy has blocked this file: "
            "matplotlib\\_c_internal_utils.pyd")

    def find_spec(self, fullname, path, target=None):
        if fullname == "matplotlib" or fullname.startswith("matplotlib."):
            raise ImportError(self._MSG, name=fullname)
        return None


# Evict any pre-imported matplotlib so the blocker governs the next import.
for _name in [m for m in list(sys.modules)
              if m == "matplotlib" or m.startswith("matplotlib.")]:
    del sys.modules[_name]
sys.meta_path.insert(0, _SACBlocker())

import json  # sanity: non-matplotlib imports still resolve

mod = importlib.import_module("core.training.training")
assert mod._pyplot is None
assert mod._pyplot_failed is False
assert mod._load_pyplot() is None, "_load_pyplot must return None when blocked"
assert mod._pyplot_failed is True, "_pyplot_failed must be set after failure"
assert mod._load_pyplot() is None  # idempotent, no re-raise
# Guard returns None before touching self/progress -> fake self is safe HERE only.
assert mod.TrainingBackend._create_loss_plot(object(), None, "light") is None
print("BLOCKED_OK")
'''


_PRESENT_CHILD = r'''
import importlib

mod = importlib.import_module("core.training.training")
assert mod._pyplot is None
assert mod._pyplot_failed is False
plt = mod._load_pyplot()
assert plt is not None, "matplotlib installed; _load_pyplot must return module"
import matplotlib
assert matplotlib.get_backend().lower() == "agg", matplotlib.get_backend()
assert mod._load_pyplot() is plt, "second call must return cached object"
assert mod._pyplot is plt
assert mod._pyplot_failed is False
# Do NOT call _create_loss_plot here: with matplotlib present the plt-is-None
# guard does not fire and a fake self would AttributeError on self.loss_history.
print("PRESENT_OK")
'''


def test_module_imports_with_matplotlib_blocked():
    """Server-import path survives a SAC-style matplotlib block."""
    proc = _run_child(_BLOCKED_CHILD)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "BLOCKED_OK" in proc.stdout, proc.stdout


def test_load_pyplot_returns_agg_when_present():
    """With matplotlib==3.11.0 installed, _load_pyplot returns a cached Agg module."""
    import importlib.util
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib not installed; present-path test is N/A")
    proc = _run_child(_PRESENT_CHILD)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "PRESENT_OK" in proc.stdout, proc.stdout
