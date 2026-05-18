$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# CASE A: ARM64 host with nv_dispi.inf_arm64_* -> detected.
$root = New-Sandbox 't06a'
$env:PROCESSOR_ARCHITECTURE = 'ARM64'
$dch = "$env:SystemRoot/System32/DriverStore/FileRepository/nv_dispi.inf_arm64_xyz9876/nvidia-smi.exe"
New-FakeNvidiaSmiAt -AtPath $dch -ShimName 'nvidia-smi-ok'
Set-Mock-GetCimInstance -Controllers @()
$r = Invoke-NvidiaDetection
Assert-True $r.HasNvidiaSmi "ARM64 DCH nvidia-smi should be detected"
Assert-Like $r.NvidiaSmiExe '*nv_dispi.inf_arm64_*' "path should point at arm64 DCH location"
Clear-Mock-GetCimInstance

# CASE B: ARM64 host but only amd64 DCH folder exists -> NOT detected
# (this is the architecture-awareness assertion; on ARM64 Windows you do not
#  want to find an x64 nvidia-smi binary).
$root = New-Sandbox 't06b'
$env:PROCESSOR_ARCHITECTURE = 'ARM64'
$dch_wrong = "$env:SystemRoot/System32/DriverStore/FileRepository/nv_dispi.inf_amd64_xyz9876/nvidia-smi.exe"
New-FakeNvidiaSmiAt -AtPath $dch_wrong -ShimName 'nvidia-smi-ok'
Set-Mock-GetCimInstance -Controllers @()
$r = Invoke-NvidiaDetection
Assert-False $r.HasNvidiaSmi "On ARM64, amd64 DCH folder must not be picked"
Clear-Mock-GetCimInstance

"PASS t06_dch_arm64"
