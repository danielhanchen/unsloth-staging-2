$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# Same as t07 in spirit, but using the actual installed Invoke-NvidiaDetection from
# fixed helpers when HARNESS_MODE=fixed.
if ($env:HARNESS_MODE -eq 'fixed') {
    . "$PSScriptRoot/../helpers_fixed.ps1"
}

$root = New-Sandbox 't30'
$env:PROCESSOR_ARCHITECTURE = 'AMD64'

# Only OEM NVIDIA INF folder names, no nv_dispi.inf_*.
$oem = "$env:SystemRoot/System32/DriverStore/FileRepository/nvltsg.inf_amd64_oem01/nvidia-smi.exe"
New-FakeNvidiaSmiAt -AtPath $oem -ShimName 'nvidia-smi-ok'
Set-Mock-GetCimInstance -Controllers @(
    [PSCustomObject]@{ Name = 'NVIDIA RTX 5000 Ada'; Caption = 'NVIDIA RTX 5000 Ada' }
)
$r = Invoke-NvidiaDetection
Clear-Mock-GetCimInstance

if ($env:HARNESS_MODE -eq 'fixed') {
    Assert-True $r.HasNvidiaSmi "fixed: OEM nvltsg.inf folder should be detected"
    Assert-Like $r.NvidiaSmiExe '*nvltsg.inf_amd64_*' 'NvidiaSmiExe should point at OEM folder'
    "PASS t30_codex_p3_oem_glob"
} else {
    if ($r.HasNvidiaSmi) {
        throw "BASELINE PASS: OEM glob already works — codex P2 #3 may already be addressed."
    }
    Write-Host "XFAIL t30_codex_p3_oem_glob — baseline only matches nv_dispi.inf_*"
}
