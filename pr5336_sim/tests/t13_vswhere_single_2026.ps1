$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# Codex P2 #1 baseline path: PR adds VS 2026 detection. This test confirms the
# happy path (vswhere reports '18' -> generator 'Visual Studio 18 2026').
$root = New-Sandbox 't13'
New-FakeVsWhere -ShimName 'vswhere-2026'
$r = Find-VsBuildTools
Assert-True ($null -ne $r) "vswhere -> VS 2026 should resolve (PR feature)"
Assert-Eq $r.Generator 'Visual Studio 18 2026' 'VS 2026 generator string'

"PASS t13_vswhere_single_2026"
