$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# vswhere registered but returns nothing -> falls through to filesystem scan.
# With nothing on disk either -> $null.
$root = New-Sandbox 't16'
New-FakeVsWhere -ShimName 'vswhere-empty'
$r = Find-VsBuildTools
Assert-True ($null -eq $r) "no vswhere result + empty filesystem -> null"

"PASS t16_vswhere_empty"
