$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# Filesystem scan for VS 2026 (dir name '18') under ProgramFiles -> Community.
$root = New-Sandbox 't18'
New-FakeVsInstall -Root $env:ProgramFiles -DirName '18' -Edition 'Community'
$r = Find-VsBuildTools
Assert-True ($null -ne $r) "filesystem scan should find VS 2026"
Assert-Eq $r.Generator 'Visual Studio 18 2026' 'VS 2026 generator'
Assert-Like $r.Source '*Community*' 'source should mention edition'

"PASS t18_filesystem_scan_2026"
