$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# CODEX P2 #1: VS 2026 + CMake < 4.2 should NOT pass through. Baseline has no
# gate, so this test checks for the presence of the gate (which only exists in
# helpers_fixed.ps1). On baseline, expect XFAIL.

# We re-load helpers_fixed if HARNESS_MODE=fixed.
if ($env:HARNESS_MODE -eq 'fixed') {
    . "$PSScriptRoot/../helpers_fixed.ps1"
}

$root = New-Sandbox 't24'
New-FakeVsWhere -ShimName 'vswhere-2026'
Add-ShimToPath -ShimName 'cmake-3' -ExposeAs 'cmake'

if ($env:HARNESS_MODE -eq 'fixed') {
    # The fixed helper must downgrade or fail.
    $r = Find-VsBuildToolsGated
    if ($r -and $r.Generator -eq 'Visual Studio 18 2026') {
        throw "BUG: gated helper accepted VS 2026 with CMake 3.x"
    }
    "PASS t24_codex_p1_cmake_gate_old"
} else {
    $r = Find-VsBuildTools
    if ($r.Generator -eq 'Visual Studio 18 2026') {
        Write-Host "XFAIL t24_codex_p1_cmake_gate_old — baseline lets VS 2026 + CMake 3.x through"
        return
    }
    throw "Unexpected baseline behaviour: $($r.Generator)"
}
