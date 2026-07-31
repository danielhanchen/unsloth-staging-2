# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_run_setup_script must not hand PowerShell 7's PSModulePath to Windows PowerShell.

setup.ps1 installs uv by running astral's install.ps1, which calls
Get-ExecutionPolicy out of Microsoft.PowerShell.Security. Windows PowerShell 5.1
cannot load that module when it inherits PowerShell 7's PSModulePath, so
`unsloth studio update` failed from a pwsh 7 prompt with

    The 'Get-ExecutionPolicy' command was found in the module
    'Microsoft.PowerShell.Security', but the module could not be loaded.

and succeeded from a 5.1 prompt, on the same runner, in the same job.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


class _Result:
    returncode = 0


def _capture_setup_env(monkeypatch, *, system, verbose = False, environ = None):
    """Run _run_setup_script against a stubbed subprocess and return the child env."""
    studio = _studio()
    seen = {}

    def _run(args, **kwargs):
        seen["args"] = args
        seen["env"] = kwargs.get("env")
        return _Result()

    monkeypatch.setattr(studio, "_find_setup_script", lambda: Path("setup.ps1"))
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(subprocess, "run", _run)
    if environ is not None:
        for key, value in environ.items():
            monkeypatch.setenv(key, value)
    studio._run_setup_script(verbose = verbose)
    return seen


def test_windows_drops_psmodulepath(monkeypatch):
    seen = _capture_setup_env(
        monkeypatch, system = "Windows",
        environ = {"PSModulePath": r"C:\Program Files\PowerShell\7\Modules"},
    )
    assert seen["env"] is not None, "the child would inherit the parent environment wholesale"
    assert "PSModulePath" not in seen["env"]


def test_windows_drops_psmodulepath_under_verbose_too(monkeypatch):
    # --verbose was the only path that built an env dict at all, so a fix
    # applied there alone would leave the default path broken.
    seen = _capture_setup_env(
        monkeypatch, system = "Windows", verbose = True,
        environ = {"PSModulePath": r"C:\Program Files\PowerShell\7\Modules"},
    )
    assert "PSModulePath" not in seen["env"]
    assert seen["env"]["UNSLOTH_VERBOSE"] == "1"


def test_windows_keeps_the_rest_of_the_environment(monkeypatch):
    seen = _capture_setup_env(
        monkeypatch, system = "Windows",
        environ = {
            "PSModulePath": r"C:\Program Files\PowerShell\7\Modules",
            "UNSLOTH_STUDIO_HOME": r"C:\custom\root",
        },
    )
    # Dropping more than PSModulePath would break env-mode installs, which
    # resolve their root from exactly this variable.
    assert seen["env"]["UNSLOTH_STUDIO_HOME"] == r"C:\custom\root"


def test_a_missing_psmodulepath_is_not_an_error(monkeypatch):
    monkeypatch.delenv("PSModulePath", raising = False)
    seen = _capture_setup_env(monkeypatch, system = "Windows")
    assert "PSModulePath" not in seen["env"]


def test_posix_is_untouched(monkeypatch):
    # bash has no module path, and rebuilding the env there would be a silent
    # behaviour change for every Linux and macOS update.
    seen = _capture_setup_env(monkeypatch, system = "Linux")
    assert seen["env"] is None
    assert seen["args"][0] == "bash"
