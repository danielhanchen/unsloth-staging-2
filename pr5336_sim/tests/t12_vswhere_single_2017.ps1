$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

$root = New-Sandbox 't12'
New-FakeVsWhere -ShimName 'vswhere-2017'
$r = Find-VsBuildTools
Assert-True ($null -ne $r) "vswhere -> VS 2017 should resolve"
Assert-Eq $r.Generator 'Visual Studio 15 2017' 'VS 2017 generator string'

"PASS t12_vswhere_single_2017"
