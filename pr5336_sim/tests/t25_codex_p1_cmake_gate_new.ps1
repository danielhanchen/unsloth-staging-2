$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# CMake 4.2 IS supported -> VS 2026 should be accepted under both baseline and fixed.

if ($env:HARNESS_MODE -eq 'fixed') {
    . "$PSScriptRoot/../helpers_fixed.ps1"
}

$root = New-Sandbox 't25'
New-FakeVsWhere -ShimName 'vswhere-2026'
Add-ShimToPath -ShimName 'cmake-4-2' -ExposeAs 'cmake'

if ($env:HARNESS_MODE -eq 'fixed') {
    $r = Find-VsBuildToolsGated
    Assert-True ($null -ne $r) "VS 2026 + CMake 4.2 -> should pass"
    Assert-Eq $r.Generator 'Visual Studio 18 2026' 'should pick VS 2026'
} else {
    $r = Find-VsBuildTools
    Assert-Eq $r.Generator 'Visual Studio 18 2026' 'baseline: VS 2026 always accepted'
}

"PASS t25_codex_p1_cmake_gate_new"
