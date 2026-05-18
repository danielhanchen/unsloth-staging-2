$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"
if ($env:HARNESS_MODE -eq 'fixed') {
    . "$PSScriptRoot/../helpers_fixed.ps1"
}

# Codex P2 #4: x64 PowerShell on ARM64 Windows. The PROCESSOR_ARCHITECTURE
# variable describes the PROCESS arch (AMD64), not the OS (ARM64). The OS arch
# lives in PROCESSOR_ARCHITEW6432 when the process is emulated. Baseline (the
# version of helpers.ps1 the original PR shipped) uses only PROCESSOR_ARCHITECTURE
# and misses the ARM64 folder; fixed mode uses PROCESSOR_ARCHITEW6432 with
# PROCESSOR_ARCHITECTURE as a fallback.

$root = New-Sandbox 't32'
$env:PROCESSOR_ARCHITECTURE = 'AMD64'         # process arch (x64 emulated)
$env:PROCESSOR_ARCHITEW6432 = 'ARM64'         # OS arch (real hardware)

$dch = "$env:SystemRoot/System32/DriverStore/FileRepository/nv_dispi.inf_arm64_abcd/nvidia-smi.exe"
New-FakeNvidiaSmiAt -AtPath $dch -ShimName 'nvidia-smi-ok'
Set-Mock-GetCimInstance -Controllers @()

$r = Invoke-NvidiaDetection
Clear-Mock-GetCimInstance
Remove-Item Env:PROCESSOR_ARCHITEW6432 -ErrorAction SilentlyContinue

if ($env:HARNESS_MODE -eq 'fixed') {
    Assert-True $r.HasNvidiaSmi "fixed: x64-on-ARM64 must still find arm64 DriverStore"
    Assert-Like $r.NvidiaSmiExe '*nv_dispi.inf_arm64_*' 'should pick arm64 folder'
    "PASS t32_arch_wow64"
} else {
    if ($r.HasNvidiaSmi) {
        throw "BASELINE PASS: arch detection already correct?"
    }
    Write-Host "XFAIL t32_arch_wow64 - baseline uses PROCESSOR_ARCHITECTURE, misses ARM64 OS"
}
