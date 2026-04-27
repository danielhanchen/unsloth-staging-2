"""Behaviour tests for studio/install_python_stack.py:_uv_safe_path.

The helper converts spaced Windows paths to 8.3 short form so uv 0.11.x does
not truncate `-r` / `-c` arguments at the first space. When 8.3 short-name
generation is disabled on the volume, GetShortPathNameW returns the original
long path; the helper must warn (once per path) instead of silently returning
the unsafe value.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_PYTHON_STACK = REPO_ROOT / "studio" / "install_python_stack.py"


def _load_module():
    # The module imports from `backend.utils.wheel_utils`; load with the
    # studio dir on sys.path so those relative imports resolve.
    studio_dir = str(REPO_ROOT / "studio")
    if studio_dir not in sys.path:
        sys.path.insert(0, studio_dir)
    spec = importlib.util.spec_from_file_location(
        "install_python_stack_under_test", INSTALL_PYTHON_STACK
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_non_windows_passes_through_unchanged():
    mod = _load_module()
    mod.IS_WINDOWS = False
    assert mod._uv_safe_path("/tmp/with space/file.txt") == "/tmp/with space/file.txt"


def test_windows_no_space_passes_through_unchanged():
    mod = _load_module()
    mod.IS_WINDOWS = True
    assert mod._uv_safe_path(r"C:\Users\NoSpaces\req.txt") == r"C:\Users\NoSpaces\req.txt"


def test_windows_space_fallback_warns_to_stderr():
    mod = _load_module()
    mod.IS_WINDOWS = True
    mod._UV_SAFE_PATH_WARNED.clear()
    buf = io.StringIO()
    saved = sys.stderr
    sys.stderr = buf
    try:
        out = mod._uv_safe_path(r"C:\Users\First Last\req.txt")
    finally:
        sys.stderr = saved
    # Conversion fails on Linux (no kernel32) so the helper falls back to the
    # original spaced value -- but that fall-through MUST emit a warning.
    assert " " in out
    text = buf.getvalue()
    assert "WARN" in text
    assert "First Last" in text


def test_windows_space_warning_is_idempotent_per_path():
    mod = _load_module()
    mod.IS_WINDOWS = True
    mod._UV_SAFE_PATH_WARNED.clear()
    buf = io.StringIO()
    saved = sys.stderr
    sys.stderr = buf
    try:
        for _ in range(5):
            mod._uv_safe_path(r"C:\Users\First Last\req.txt")
    finally:
        sys.stderr = saved
    assert buf.getvalue().count("WARN") == 1


def test_pip_install_uses_uv_safe_path_for_constraints_and_requirements():
    """Both constraint (-c) and requirements (-r) args must route through the
    uv-safe path on uv invocations, while pip variants stay as long paths."""
    src = INSTALL_PYTHON_STACK.read_text(encoding="utf-8")
    assert 'constraint_args_uv = ["-c", _uv_safe_path(CONSTRAINTS)]' in src
    assert 'req_args_uv = ["-r", _uv_safe_path(actual_req)]' in src
    # pip variants must keep the raw string path
    assert 'constraint_args_pip = ["-c", str(CONSTRAINTS)]' in src
    assert 'req_args_pip = ["-r", str(actual_req)]' in src
