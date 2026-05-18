$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# Multi-instance vswhere fix: (@($info)[0]).Trim() takes first line cleanly.
$root = New-Sandbox 't14'
New-FakeVsWhere -ShimName 'vswhere-multi'   # returns "2026\n2022\n"
$r = Find-VsBuildTools
Assert-True ($null -ne $r) "multi-line vswhere should still resolve"
# First line of vswhere-multi is "2026", so generator should be VS 18 2026.
Assert-Eq $r.Generator 'Visual Studio 18 2026' 'first-line of multi-instance vswhere = 2026'
Assert-Like $r.InstallPath '*18*BuildTools*' 'install path matches first-line'

"PASS t14_vswhere_multi_instance"
