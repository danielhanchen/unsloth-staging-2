$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

$root = New-Sandbox 't22'
$r = Test-AnyVsInstalled
Assert-False $r "no vswhere + empty filesystem -> false"

"PASS t22_test_any_vs_installed_none"
