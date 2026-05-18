$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"
if ($env:HARNESS_MODE -eq 'fixed') {
    . "$PSScriptRoot/../helpers_fixed.ps1"
}

# CODEX P2 #3: DCH glob is currently hardcoded to nv_dispi.inf_*. OEM/notebook
# driver packages use different INF folder names. With the current PR head, the
# script will MISS these and silently fall back to CPU.
#
# Expected BEFORE codex fix: HasNvidiaSmi == false  (this test fails on PR head)
# Expected AFTER  codex fix: HasNvidiaSmi == true   (broader glob picks it up)
#
# We mark this test as "expected to fail on baseline" by writing it normally;
# the runner classifies it.

$root = New-Sandbox 't07'
$env:PROCESSOR_ARCHITECTURE = 'AMD64'

# OEM/notebook INF folder names that exist in the wild:
$oemPaths = @(
    "$env:SystemRoot/System32/DriverStore/FileRepository/nvltsg.inf_amd64_oem01/nvidia-smi.exe",
    "$env:SystemRoot/System32/DriverStore/FileRepository/nvgridswn.inf_amd64_oem02/nvidia-smi.exe",
    "$env:SystemRoot/System32/DriverStore/FileRepository/nvltcs.inf_amd64_oem03/nvidia-smi.exe"
)
foreach ($p in $oemPaths) {
    New-FakeNvidiaSmiAt -AtPath $p -ShimName 'nvidia-smi-ok'
}
# Specifically: NO nv_dispi.inf_* folder exists.
Set-Mock-GetCimInstance -Controllers @(
    [PSCustomObject]@{ Name = 'NVIDIA Quadro RTX 5000'; Caption = 'NVIDIA Quadro RTX 5000' }
)

$r = Invoke-NvidiaDetection
$expectedAfterFix = $true
$actualOnBaseline = $false   # documented expectation for current PR head

if ($env:HARNESS_MODE -eq 'fixed') {
    Assert-True $r.HasNvidiaSmi "After codex fix: OEM INF should be detected"
} else {
    # On baseline, we expect this to fail; mark as XFAIL.
    if ($r.HasNvidiaSmi) {
        throw "UNEXPECTED PASS: OEM INF was detected on baseline. Codex P2 #3 may already be fixed."
    }
    Write-Host "XFAIL t07_dch_oem_inf — codex P2 #3 reproduced (OEM INF not detected)"
    return
}

"PASS t07_dch_oem_inf"
