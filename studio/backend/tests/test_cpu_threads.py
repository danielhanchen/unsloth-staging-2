# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for Studio's early CPU thread-pool configuration."""

import ast
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.cpu_threads import (
    _DEFAULT_IDLE_CAP,
    _THREAD_POOL_ENV_VARS,
    configure_cpu_threads,
)


_BACKEND_DIR = Path(__file__).resolve().parent.parent
_RUN_PY = _BACKEND_DIR / "run.py"
_MAIN_PY = _BACKEND_DIR / "main.py"


# ---------------------------------------------------------------------------
# Explicit positive integers
# ---------------------------------------------------------------------------

def test_cpu_thread_cap_seeds_native_pool_limits():
    env = {"UNSLOTH_CPU_THREADS": " 6 "}

    configure_cpu_threads(env)

    assert {variable: env[variable] for variable in _THREAD_POOL_ENV_VARS} == {
        variable: "6" for variable in _THREAD_POOL_ENV_VARS
    }


def test_cpu_thread_cap_preserves_runtime_specific_override():
    env = {"UNSLOTH_CPU_THREADS": "4", "OMP_NUM_THREADS": "2"}

    configure_cpu_threads(env)

    assert env["OMP_NUM_THREADS"] == "2"
    assert env["MKL_NUM_THREADS"] == "4"


@pytest.mark.parametrize("raw", ["+4", "007", "  4  "])
def test_cpu_thread_cap_normalises_valid_inputs(raw):
    env = {"UNSLOTH_CPU_THREADS": raw}

    configure_cpu_threads(env)

    assert env["OMP_NUM_THREADS"] == str(int(raw.strip()))


# ---------------------------------------------------------------------------
# Auto-default: unset / empty / "auto" / "default" -> min(cpu_count, 8)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [None, "", "   ", "\t", "auto", "AUTO", "default", "Default"])
def test_cpu_thread_cap_applies_idle_default(raw):
    env = {} if raw is None else {"UNSLOTH_CPU_THREADS": raw}

    with patch("utils.cpu_threads._resolve_cpu_count", return_value=192):
        configure_cpu_threads(env)

    expected = str(_DEFAULT_IDLE_CAP)
    assert {variable: env[variable] for variable in _THREAD_POOL_ENV_VARS} == {
        variable: expected for variable in _THREAD_POOL_ENV_VARS
    }


def test_cpu_thread_cap_default_floor_is_small_cpu_count():
    """On tiny VMs the cap collapses to cpu_count and effectively no-ops."""
    env = {}

    with patch("utils.cpu_threads._resolve_cpu_count", return_value=2):
        configure_cpu_threads(env)

    assert env["OMP_NUM_THREADS"] == "2"


def test_cpu_thread_cap_default_never_exceeds_eight():
    env = {}

    with patch("utils.cpu_threads._resolve_cpu_count", return_value=192):
        configure_cpu_threads(env)

    assert env["OMP_NUM_THREADS"] == str(_DEFAULT_IDLE_CAP)


def test_cpu_thread_cap_default_respects_explicit_override():
    """Auto-default still defers to an explicit per-library env var."""
    env = {"OMP_NUM_THREADS": "1"}

    with patch("utils.cpu_threads._resolve_cpu_count", return_value=192):
        configure_cpu_threads(env)

    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == str(_DEFAULT_IDLE_CAP)


# ---------------------------------------------------------------------------
# Opt-out sentinels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["off", "OFF", "none", "None", "disabled", "unlimited"])
def test_cpu_thread_cap_opt_out_sentinels_skip_seeding(raw):
    env = {"UNSLOTH_CPU_THREADS": raw}

    configure_cpu_threads(env)

    assert all(variable not in env for variable in _THREAD_POOL_ENV_VARS)


# ---------------------------------------------------------------------------
# Invalid input -> ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["zero", "0", "-3", "1.5", "abc", "8a", "0x4", "1e3", "4 0"])
def test_cpu_thread_cap_requires_positive_integer(raw):
    with pytest.raises(ValueError, match="must be a positive integer"):
        configure_cpu_threads({"UNSLOTH_CPU_THREADS": raw})


# ---------------------------------------------------------------------------
# Real os.environ path (env=None) -- covers the production call from run.py
# ---------------------------------------------------------------------------

def test_cpu_thread_cap_uses_os_environ_when_env_is_none(monkeypatch):
    for variable in (*_THREAD_POOL_ENV_VARS, "UNSLOTH_CPU_THREADS"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", "3")

    configure_cpu_threads()

    for variable in _THREAD_POOL_ENV_VARS:
        assert os.environ[variable] == "3"


def test_cpu_thread_cap_default_via_os_environ(monkeypatch):
    for variable in (*_THREAD_POOL_ENV_VARS, "UNSLOTH_CPU_THREADS"):
        monkeypatch.delenv(variable, raising=False)

    with patch("utils.cpu_threads._resolve_cpu_count", return_value=192):
        configure_cpu_threads()

    assert os.environ["OMP_NUM_THREADS"] == str(_DEFAULT_IDLE_CAP)


def test_cpu_thread_cap_idempotent(monkeypatch):
    """Calling twice with the same env must not flip any value."""
    for variable in (*_THREAD_POOL_ENV_VARS, "UNSLOTH_CPU_THREADS"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", "5")

    configure_cpu_threads()
    snapshot = {v: os.environ.get(v) for v in _THREAD_POOL_ENV_VARS}
    configure_cpu_threads()

    assert {v: os.environ.get(v) for v in _THREAD_POOL_ENV_VARS} == snapshot


# ---------------------------------------------------------------------------
# Early-call ordering: configure_cpu_threads() must precede _platform_compat
# in BOTH run.py and main.py. AST-based so reformatting doesn't break it.
# ---------------------------------------------------------------------------

def _ast_line_of_configure_call(source: str) -> int:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "configure_cpu_threads"
        ):
            return node.lineno
    raise AssertionError("configure_cpu_threads() call not found")


def _ast_line_of_platform_compat_import(source: str) -> int:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "_platform_compat":
                    return node.lineno
    raise AssertionError("_platform_compat import not found")


@pytest.mark.parametrize("entry_point", [_RUN_PY, _MAIN_PY])
def test_cpu_thread_configuration_runs_before_backend_imports(entry_point):
    source = entry_point.read_text()
    call_line = _ast_line_of_configure_call(source)
    compat_line = _ast_line_of_platform_compat_import(source)
    assert call_line < compat_line, (
        f"{entry_point.name}: configure_cpu_threads() (line {call_line}) "
        f"must precede import _platform_compat (line {compat_line})"
    )


# ---------------------------------------------------------------------------
# Subprocess: invalid env -> clean stderr, no traceback, exit 1, did not
# proceed past the cap gate (no platform-compat side effects observed).
# ---------------------------------------------------------------------------

def test_invalid_cpu_thread_cap_exits_without_traceback():
    env = os.environ.copy()
    env["UNSLOTH_CPU_THREADS"] = "not-a-count"

    result = subprocess.run(
        [sys.executable, str(_RUN_PY)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert (
        "Error: Invalid UNSLOTH_CPU_THREADS value 'not-a-count': "
        "UNSLOTH_CPU_THREADS must be a positive integer"
    ) in result.stderr
    assert "Traceback" not in result.stderr
    # We must have exited before any heavy backend import.
    assert "_platform_compat" not in result.stderr
