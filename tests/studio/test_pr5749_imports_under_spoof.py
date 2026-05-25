# SPDX-License-Identifier: AGPL-3.0-only
"""Spoof-backed behavioral validation of PR #5749 + review fixes.

Two AST-based regression-lock files already cover the structural shape
of the fix (test_disconnect_watcher.py, test_llama_cpp_windows_launch_gating.py).
This file complements them by actually IMPORTING the patched modules under
the project's existing CUDA spoof harness (tests/_zoo_aggressive_cuda_spoof.py),
so we exercise:

  1. The patched modules import cleanly on a no-GPU runner (ubuntu, macos,
     windows-latest GHA all have no NVIDIA driver).
  2. The function signature exposed to live callers matches the AST view
     (catches refactors where the AST text and the loaded module diverge,
     e.g. a decorator wraps the function and renames params).
  3. Both passthrough streamer functions, when introspected, reference
     `cancel_event` so the watcher receives a non-None argument.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

# Apply the consolidated CPU spoof at module import time, mirroring the
# pattern used in tests/vllm_compat/test_unsloth_zoo_imports.py (lines 42-50)
# so this file works on the project's existing CI runners that ship no GPU.
_SPOOF_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SPOOF_DIR))
import _zoo_aggressive_cuda_spoof as _spoof  # noqa: E402

_spoof.apply()


@pytest.fixture(scope="module")
def inference_module():
    """Import routes.inference under the spoof.

    studio/backend is the package root for the studio runtime (see
    studio/backend/tests/test_anthropic_messages.py:15-16). We add it
    to sys.path then import via the bare ``routes.inference`` path the
    project uses internally."""
    repo_root = Path(__file__).resolve().parents[2]
    backend_root = repo_root / "studio" / "backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    try:
        import importlib
        return importlib.import_module("routes.inference")
    except ImportError as e:
        pytest.skip(f"studio runtime dep missing: {e}")


@pytest.fixture(scope="module")
def llama_cpp_module():
    repo_root = Path(__file__).resolve().parents[2]
    backend_root = repo_root / "studio" / "backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    try:
        import importlib
        return importlib.import_module("core.inference.llama_cpp")
    except ImportError as e:
        pytest.skip(f"studio runtime dep missing: {e}")


# ---------------------------------------------------------------------------
# Live-import structural assertions on the patched signature + call sites.
# ---------------------------------------------------------------------------


def test_disconnect_watcher_signature_under_spoof(inference_module):
    """The patched function must expose (request, resp, cancel_event)
    when actually imported and introspected. AST-only checks can be
    fooled by a runtime decorator; this one cannot."""

    fn = inference_module._await_disconnect_then_close
    sig = inspect.signature(fn)
    params = list(sig.parameters)
    assert params == ["request", "resp", "cancel_event"], (
        f"Expected (request, resp, cancel_event); got {params!r}. "
        "The watcher must accept cancel_event so it can signal "
        "cancellation before closing the upstream response."
    )


def test_passthrough_streamers_reference_cancel_event(inference_module):
    """Both passthrough streamer functions must reference cancel_event
    in their bytecode so the watcher is wired correctly."""

    for fn_name in ("_openai_passthrough_stream", "_anthropic_passthrough_stream"):
        fn = getattr(inference_module, fn_name)
        # The streamer functions are nested-async-def, so check that
        # cancel_event appears in their closure or code names.
        co = fn.__code__
        all_names = set(co.co_names) | set(co.co_freevars) | set(co.co_varnames)
        nested_names = set()
        for const in co.co_consts:
            if hasattr(const, "co_names"):
                nested_names |= set(const.co_names)
                nested_names |= set(const.co_varnames)
        all_names |= nested_names
        assert "cancel_event" in all_names, (
            f"{fn_name}: cancel_event missing from code references. "
            "The disconnect watcher needs cancel_event passed through "
            "from this function."
        )


def test_llama_cpp_imports_under_spoof(llama_cpp_module):
    """Importing the patched llama_cpp module on a no-GPU runner must
    succeed under the spoof. Catches accidental top-level imports of
    GPU-only modules added during the patch."""

    assert hasattr(llama_cpp_module, "LlamaCppBackend"), (
        "LlamaCppBackend missing after import under spoof"
    )


def test_spoof_is_active_during_this_test_module(inference_module):
    """Sanity: confirm the spoof harness is in fact applied. If a future
    refactor breaks the spoof, the upstream import would have failed
    earlier; we double-check the sentinel here so a silent regression
    in the spoof itself surfaces as a clear failure."""

    import torch

    assert getattr(torch.cuda, "_unsloth_consolidated_spoof", False), (
        "Spoof sentinel torch.cuda._unsloth_consolidated_spoof must be "
        "True after _spoof.apply()"
    )
    assert torch.cuda.is_available() is True, "spoof must make is_available True"
    assert torch.cuda.device_count() == 1, "spoof reports 1 device"
