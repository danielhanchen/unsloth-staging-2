$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# VS-without-C++-workload + UNSLOTH_FORCE_BUILD_TOOLS unset -> AbortMissingWorkload.
$root = New-Sandbox 't23a'
$vsRoot = "$env:ProgramFiles/Microsoft Visual Studio/2022/Community"
New-Item -ItemType Directory -Path $vsRoot -Force | Out-Null
$d = Invoke-VsSelectionGate
Assert-Eq $d.Decision 'AbortMissingWorkload' 'unset env -> abort'

# Same setup but UNSLOTH_FORCE_BUILD_TOOLS=1 -> InstallBuildTools (proceed).
$root = New-Sandbox 't23b'
$vsRoot = "$env:ProgramFiles/Microsoft Visual Studio/2022/Community"
New-Item -ItemType Directory -Path $vsRoot -Force | Out-Null
$env:UNSLOTH_FORCE_BUILD_TOOLS = '1'
$d = Invoke-VsSelectionGate
Assert-Eq $d.Decision 'InstallBuildTools' 'forced env -> proceed'

# No VS at all + no env -> InstallBuildTools.
$root = New-Sandbox 't23c'
$d = Invoke-VsSelectionGate
Assert-Eq $d.Decision 'InstallBuildTools' 'no VS at all -> install'

# Working VS with C++ workload -> UseExistingVS.
$root = New-Sandbox 't23d'
New-FakeVsInstall -Root $env:ProgramFiles -DirName '2022' -Edition 'BuildTools'
$d = Invoke-VsSelectionGate
Assert-Eq $d.Decision 'UseExistingVS' 'working VS -> use it'

"PASS t23_force_build_tools_bypass"
