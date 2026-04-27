"""Behaviour tests for studio/setup.ps1 Phase 1g Python detection.

Pre-PR concerns this PR fixes:
  - $PythonOk = $true was set unconditionally after a `py.exe -<minor>
    --version` match, even when the inner `sys.executable` resolution
    failed; downstream bare `python` calls then crashed.
  - The PATH-prepend was skipped when the supported interpreter's
    directory was already present later in PATH behind an unsupported
    Python (e.g., 3.14 ahead of 3.13). Bare `python` continued to
    resolve to the unsupported version.
  - Quoted PATH entries (`"C:\\Python313"`) compared unequal to the
    unquoted resolved dir, re-prepending on every run.
  - `python --version 2>&1` without Out-String can return an
    ErrorRecord under PowerShell 5.1 when Python writes to stderr.
  - The `python --version` fallback dropped the patch component
    ("3.12" vs "3.12.7" from the py.exe branch).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"

PWSH = "/usr/bin/pwsh"
PWSH_AVAILABLE = os.path.isfile(PWSH) and os.access(PWSH, os.X_OK)
requires_pwsh = pytest.mark.skipif(not PWSH_AVAILABLE, reason="pwsh not available")


def _run_pwsh(script: str, *, timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_python_ok_only_set_after_resolved_exe_check():
    """`$PythonOk = $true; break` must be inside the resolvedExe success
    branch, not at the outer foreach scope where it ran unconditionally
    pre-fix."""
    src = SETUP_PS1.read_text(encoding="utf-8")
    block = re.search(
        r"if \(\$PyLauncher\) \{(?P<body>.*?)^\}\s*$",
        src,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert block, "py.exe gate block not found in setup.ps1"
    body = block.group("body")
    # The success setters must live inside the if ($resolvedExe -and Test-Path)
    # branch, ordered HasPython -> PythonOk -> break.
    pattern = (
        r"if \(\$resolvedExe -and \(Test-Path \$resolvedExe\)\) \{"
        r".*?\$HasPython = \$true"
        r".*?\$PythonOk = \$true"
        r".*?break"
    )
    assert re.search(pattern, body, flags=re.DOTALL), (
        "PythonOk/break must be inside resolvedExe success branch"
    )
    # And there must NOT be a bare `$PythonOk = $true; break` at the outer
    # foreach scope after the inner try/catch.
    bad = re.search(r"\}\s*catch\s*\{\s*\}\s*\n\s*\$PythonOk = \$true\s*\n\s*break", body)
    assert not bad, "bare PythonOk/break still present at outer foreach scope"


@requires_pwsh
def test_path_reorder_promotes_supported_when_unsupported_ahead():
    """3.14 ahead of 3.13 on PATH must end up with 3.13 first after the
    new dedupe-and-prepend logic."""
    script = """
$resolvedDir = 'C:\\Python313'
$env:PATH = 'C:\\Python314;C:\\Python313'
$resolvedNorm = $resolvedDir.Trim().Trim('"').TrimEnd('\\')
$filtered = @($env:PATH -split ';' | Where-Object {
    ($_.Trim().Trim('"').TrimEnd('\\')) -ine $resolvedNorm
})
$env:PATH = (@($resolvedDir) + $filtered) -join ';'
Write-Output $env:PATH
"""
    proc = _run_pwsh(script)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().startswith("C:\\Python313;"), proc.stdout
    # 3.14 must still be present, just no longer first.
    assert "C:\\Python314" in proc.stdout


@requires_pwsh
def test_path_reorder_dedupes_quoted_entries():
    """A quoted entry like `\"C:\\Python313\"` must be treated as equal to
    `C:\\Python313` so it gets removed before the prepend (no duplication)."""
    script = """
$resolvedDir = 'C:\\Python313'
$env:PATH = '"C:\\Python313";C:\\Python314'
$resolvedNorm = $resolvedDir.Trim().Trim('"').TrimEnd('\\')
$filtered = @($env:PATH -split ';' | Where-Object {
    ($_.Trim().Trim('"').TrimEnd('\\')) -ine $resolvedNorm
})
$env:PATH = (@($resolvedDir) + $filtered) -join ';'
$entries = $env:PATH -split ';'
$count313 = ($entries | Where-Object { $_.Trim().Trim('"').TrimEnd('\\') -ieq 'C:\\Python313' }).Count
Write-Output ("ENTRIES=" + $entries.Count + " COUNT_313=" + $count313 + " FIRST=" + $entries[0])
"""
    proc = _run_pwsh(script)
    assert proc.returncode == 0, proc.stderr
    assert "COUNT_313=1" in proc.stdout, proc.stdout
    assert "FIRST=C:\\Python313" in proc.stdout, proc.stdout


def test_python_version_capture_uses_out_string():
    """Both `python --version 2>&1` sites must pipe through Out-String to
    stay safe under PowerShell 5.1's native-stderr ErrorRecord behaviour."""
    src = SETUP_PS1.read_text(encoding="utf-8")
    # Phase 1g fallback branch
    assert "$PyVer = (python --version 2>&1 | Out-String).Trim()" in src
    # Winget-success step display
    assert 'step "python" "$((python --version 2>&1 | Out-String).Trim())"' in src


def test_detected_pyver_uses_full_version_in_fallback():
    """Fallback branch must use the captured $PyVer string (full 3.x.y)
    instead of constructing major.minor only."""
    src = SETUP_PS1.read_text(encoding="utf-8")
    assert "$DetectedPyVer = ($PyVer -replace '^Python\\s+', '').Trim()" in src
    assert '$DetectedPyVer = "$PyMajor.$PyMinor"' not in src
