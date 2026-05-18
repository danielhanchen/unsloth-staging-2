$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# nvidia-smi only under DriverStore (DCH layout), amd64 -> detected.
$root = New-Sandbox 't05'
$env:PROCESSOR_ARCHITECTURE = 'AMD64'
$dch = "$env:SystemRoot/System32/DriverStore/FileRepository/nv_dispi.inf_amd64_abcd1234/nvidia-smi.exe"
New-FakeNvidiaSmiAt -AtPath $dch -ShimName 'nvidia-smi-ok'
Set-Mock-GetCimInstance -Controllers @()
$r = Invoke-NvidiaDetection
Assert-True $r.HasNvidiaSmi "DCH amd64 nvidia-smi should be detected"
Assert-Like $r.NvidiaSmiExe '*nv_dispi.inf_amd64_*nvidia-smi.exe' "path should point at DCH location"
Clear-Mock-GetCimInstance

"PASS t05_dch_amd64"
