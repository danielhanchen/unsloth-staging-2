$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

$root = New-Sandbox 't10'
New-FakeVsWhere -ShimName 'vswhere-2022'
$r = Find-VsBuildTools
Assert-True ($null -ne $r) "vswhere -> VS 2022 should resolve"
Assert-Eq $r.Generator 'Visual Studio 17 2022' 'VS 2022 generator string'
Assert-Eq $r.Source 'vswhere' 'source should be vswhere'
Assert-Like $r.InstallPath '*2022*' 'installpath should contain 2022'

"PASS t10_vswhere_single_2022"
