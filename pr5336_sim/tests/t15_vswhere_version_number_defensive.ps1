$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# Defensive: $map has '17'='17'+'2022' for the case where vswhere returns the
# version number instead of the year.
$root = New-Sandbox 't15'
New-FakeVsWhere -ShimName 'vswhere-version-num'   # returns "17\n"
$r = Find-VsBuildTools
Assert-True ($null -ne $r) "defensive fallback should resolve"
Assert-Eq $r.Generator 'Visual Studio 17 2022' '17 -> VS 17 2022 defensive map'

"PASS t15_vswhere_version_number_defensive"
