# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Cross-OS import smoke for the Studio backend.

The pytest harness in tests/conftest.py already pre-loads
unsloth_zoo.device_type / unsloth.device_type under a CUDA-on spoof so
the import chain reaches the Studio modules. Here we verify the modules
we touched in the staging branch actually import, and that the regex
constants and helper functions exist as named symbols.

Requires PYTHONPATH=studio/backend so the relative `loggers`,
`utils.subprocess_compat`, and `utils.native_path_leases` imports inside
llama_cpp.py resolve.
"""

from __future__ import annotations

import importlib
import sys


def test_llama_cpp_imports_and_has_constants():
    sys.modules.pop("studio_backend.core.inference.llama_cpp", None)
    mod = importlib.import_module("core.inference.llama_cpp")

    # Mid-plan EOS detectors (added in studio: auto-continue when model stops mid-plan).
    for name in (
        "_INTENT_SIGNAL",
        "_TRAILING_PLAN_INTENT",
        "_TRAILING_PLAN_LIST",
        "_TRAILING_PLAN_COLON",
        "_TRAILING_PLAN_WINDOW",
        "_MAX_REPROMPTS",
        "_MAX_CONTINUES",
        "_trailing_plan_hit",
    ):
        assert hasattr(mod, name), f"llama_cpp missing {name}"

    assert mod._MAX_CONTINUES >= 1
    assert mod._TRAILING_PLAN_WINDOW >= 100

    # Behavioural smoke: the helper must agree with the regex it wraps.
    assert mod._trailing_plan_hit("Let me clone the repo.")
    assert not mod._trailing_plan_hit("Done.")


def test_subprocess_compat_imports():
    # llama_cpp.py imports `windows_hidden_subprocess_kwargs` from this module
    # at the top; if the helper goes missing or renames, Studio fails to boot
    # on Windows. Importing here gives an early signal on all three OSes.
    mod = importlib.import_module("utils.subprocess_compat")
    assert hasattr(mod, "windows_hidden_subprocess_kwargs")


def test_native_path_leases_imports():
    mod = importlib.import_module("utils.native_path_leases")
    assert hasattr(mod, "child_env_without_native_path_secret")
