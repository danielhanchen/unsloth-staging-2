"""Behaviour tests for install.ps1's Get-UvSafePath helper and the no-torch
uv -r call sites.

Pre-PR, install.ps1 passed `$NoTorchReq` directly to `uv pip install ... -r
<path>` even though uv 0.11.x truncates the value at the first space. The PR
adds Get-UvSafePath next to Find-NoTorchRuntimeFile and routes both no-torch
branches through it. This test extracts the helper from install.ps1, runs it
under pwsh, and asserts both no-torch sites consume the safe variable.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_PS1 = REPO_ROOT / "install.ps1"

PWSH = "/usr/bin/pwsh"
PWSH_AVAILABLE = os.path.isfile(PWSH) and os.access(PWSH, os.X_OK)
requires_pwsh = pytest.mark.skipif(not PWSH_AVAILABLE, reason="pwsh not available")


def _extract_function(name: str) -> str:
    """Pull a `function <name> { ... }` block out of install.ps1 by brace-balance."""
    src = INSTALL_PS1.read_text(encoding="utf-8")
    m = re.search(rf"function\s+{re.escape(name)}\s*\{{", src)
    if not m:
        raise AssertionError(f"function {name} not found in install.ps1")
    start = m.start()
    depth = 0
    i = m.end() - 1  # land on the opening '{'
    while i < len(src):
        ch = src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
        i += 1
    raise AssertionError(f"unterminated function block for {name}")


def _run_pwsh(script: str, *, timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_get_uv_safe_path_function_exists():
    src = INSTALL_PS1.read_text(encoding="utf-8")
    assert "function Get-UvSafePath" in src


@requires_pwsh
def test_get_uv_safe_path_no_space_passthrough():
    fn = _extract_function("Get-UvSafePath")
    proc = _run_pwsh(fn + "\nWrite-Output (Get-UvSafePath 'C:\\Users\\NoSpaces\\req.txt')")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == r"C:\Users\NoSpaces\req.txt"


@requires_pwsh
def test_get_uv_safe_path_warns_on_unconvertible_spaced_path():
    """Linux pwsh has no kernel32; the helper must fall back AND warn."""
    fn = _extract_function("Get-UvSafePath")
    proc = _run_pwsh(
        fn + "\n$r = Get-UvSafePath 'C:\\Users\\First Last\\req.txt'\nWrite-Output \"RESULT=$r\""
    )
    assert proc.returncode == 0, proc.stderr
    combined = proc.stdout + proc.stderr
    assert "WARN" in combined
    assert "First Last" in combined
    assert "RESULT=C:\\Users\\First Last\\req.txt" in proc.stdout


def test_no_torch_install_calls_use_safe_variant():
    """Both `-r $NoTorchReq` sites must now go through Get-UvSafePath."""
    src = INSTALL_PS1.read_text(encoding="utf-8")
    # No raw `-r $NoTorchReq` invocations should remain in the no-torch flow.
    raw = re.findall(r"uv pip install [^}]*-r \$NoTorchReq\b", src)
    assert raw == [], f"raw -r $NoTorchReq still present: {raw}"
    # The uv-safe variable must be assigned and consumed in both branches.
    assert src.count("$NoTorchReqUv = Get-UvSafePath $NoTorchReq") >= 2
    assert src.count("-r $NoTorchReqUv") >= 2


def test_tauri_overlay_skipped_for_local_install():
    src = INSTALL_PS1.read_text(encoding="utf-8")
    assert "if ($TauriMode -and -not $StudioLocalInstall) {" in src


def test_tauri_overlay_warns_on_missing_source():
    """Pre-PR silently skipped when the bundled overlay source was missing."""
    src = INSTALL_PS1.read_text(encoding="utf-8")
    # The new code must warn rather than just `continue`.
    assert re.search(
        r"if \(-not \(Test-Path \$src\)\) \{\s*\n\s*Write-Host \"\[WARN\] Overlay source missing",
        src,
    ), "missing-source path no longer warns"


def test_tauri_overlay_warns_when_script_path_unresolved():
    """When $rawPath is empty in TauriMode the user must see a warning."""
    src = INSTALL_PS1.read_text(encoding="utf-8")
    assert "Could not determine script directory; Tauri overlay skipped." in src
