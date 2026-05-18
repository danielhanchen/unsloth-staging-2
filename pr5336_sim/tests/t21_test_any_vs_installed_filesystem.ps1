$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# vswhere absent but VS dir present without VC/Tools/MSVC/cl.exe -> Find returns
# null, Test-AnyVsInstalled returns true.
$root = New-Sandbox 't21'
$vsRoot = "$env:ProgramFiles/Microsoft Visual Studio/2022/Community"
New-Item -ItemType Directory -Path $vsRoot -Force | Out-Null
# Note: no VC subtree, no cl.exe — workload absent.

$resultVs = Find-VsBuildTools
Assert-True ($null -eq $resultVs) "Find-VsBuildTools should fail (no MSVC tree)"
$resultAny = Test-AnyVsInstalled
Assert-True $resultAny "Test-AnyVsInstalled should report true (folder exists)"

"PASS t21_test_any_vs_installed_filesystem"
