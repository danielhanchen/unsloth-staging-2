$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# VS install present but no cl.exe (e.g. workload not installed) -> filesystem
# scan should skip and return null.
$root = New-Sandbox 't19'
New-FakeVsInstall -Root $env:ProgramFiles -DirName '2022' -Edition 'Community' -NoCl
$r = Find-VsBuildTools
Assert-True ($null -eq $r) "VS install without cl.exe should not be selected"

"PASS t19_filesystem_no_cl_exe"
