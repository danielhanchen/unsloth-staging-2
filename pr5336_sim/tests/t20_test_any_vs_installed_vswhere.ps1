$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# vswhere returns an installationPath but no VC.Tools workload -> Find-VsBuildTools
# returns $null but Test-AnyVsInstalled returns $true.
$root = New-Sandbox 't20'
# Stand up a vswhere shim that returns a path for plain installationPath query
# but nothing for VC.Tools-requiring query.
$shim = "$env:SystemRoot/../bin/vswhere.exe"
New-Item -ItemType Directory -Path (Split-Path -Parent $shim) -Force | Out-Null
@'
#!/usr/bin/env bash
args="$*"
if [[ "$args" == *"VC.Tools"* ]]; then
  exit 0
fi
if [[ "$args" == *"-property installationPath"* ]]; then
  echo "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community"
fi
exit 0
'@ | Set-Content -Path $shim
chmod +x $shim | Out-Null

# Drop into expected vswhere location.
$vsw = (Get-Item env:'ProgramFiles(x86)').Value + '/Microsoft Visual Studio/Installer/vswhere.exe'
New-Item -ItemType Directory -Path (Split-Path -Parent $vsw) -Force | Out-Null
Copy-Item $shim $vsw -Force
chmod +x $vsw | Out-Null

$resultVs = Find-VsBuildTools
Assert-True ($null -eq $resultVs) "Find-VsBuildTools should fail (no VC.Tools)"
$resultAny = Test-AnyVsInstalled
Assert-True $resultAny "Test-AnyVsInstalled should still report true (some VS installed)"

"PASS t20_test_any_vs_installed_vswhere"
