$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# No vswhere, but VS 2022 BuildTools on disk -> filesystem scan should find it.
$root = New-Sandbox 't17'
New-FakeVsInstall -Root $env:ProgramFiles -DirName '2022' -Edition 'BuildTools'
$r = Find-VsBuildTools
Assert-True ($null -ne $r) "filesystem scan should find VS 2022"
Assert-Eq $r.Generator 'Visual Studio 17 2022' 'VS 2022 generator'
Assert-Like $r.Source 'filesystem*' 'source should report filesystem'
Assert-Like $r.ClExe '*cl.exe' 'ClExe should be populated'

"PASS t17_filesystem_scan_2022"
