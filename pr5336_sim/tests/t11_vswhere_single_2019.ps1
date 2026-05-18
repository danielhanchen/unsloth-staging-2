$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

$root = New-Sandbox 't11'
New-FakeVsWhere -ShimName 'vswhere-2019'
$r = Find-VsBuildTools
Assert-True ($null -ne $r) "vswhere -> VS 2019 should resolve"
Assert-Eq $r.Generator 'Visual Studio 16 2019' 'VS 2019 generator string'

"PASS t11_vswhere_single_2019"
