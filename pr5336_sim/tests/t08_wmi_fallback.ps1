$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# CASE A: No nvidia-smi anywhere, but WMI sees an NVIDIA GPU -> HasNvidiaSmi
# remains false (we still can't run CUDA) BUT diagnostic messages are emitted.
$root = New-Sandbox 't08a'
Set-Mock-GetCimInstance -Controllers @(
    [PSCustomObject]@{ Name = 'NVIDIA GeForce RTX 4090'; Caption = 'NVIDIA GeForce RTX 4090' }
)
$r = Invoke-NvidiaDetection
Assert-False $r.HasNvidiaSmi "WMI hit alone is not enough to claim HasNvidiaSmi"
$msgs = ($r.Messages -join "`n")
Assert-Like $msgs '*NVIDIA GPU detected via WMI*' "WMI message must be emitted"
Assert-Like $msgs '*nvidia-smi not found*' "must guide user to reinstall drivers"
Assert-Like $msgs '*Continuing in CPU-only*' "must say it will continue in CPU mode"
Clear-Mock-GetCimInstance

# CASE B: WMI sees only non-NVIDIA cards -> no message, no detection.
$root = New-Sandbox 't08b'
Set-Mock-GetCimInstance -Controllers @(
    [PSCustomObject]@{ Name = 'Intel(R) UHD Graphics 770'; Caption = 'Intel(R) UHD Graphics 770' },
    [PSCustomObject]@{ Name = 'AMD Radeon RX 7900 XTX';   Caption = 'AMD Radeon RX 7900 XTX' }
)
$r = Invoke-NvidiaDetection
Assert-False $r.HasNvidiaSmi "no nvidia-smi + no NVIDIA in WMI -> false"
$msgs = ($r.Messages -join "`n")
Assert-False ($msgs -like '*WMI*') "must NOT mention WMI when no NVIDIA found"
Clear-Mock-GetCimInstance

# CASE C: WMI hit but Caption (not Name) carries the NVIDIA string — script
# checks both fields.
$root = New-Sandbox 't08c'
Set-Mock-GetCimInstance -Controllers @(
    [PSCustomObject]@{ Name = ''; Caption = 'NVIDIA RTX A6000' }
)
$r = Invoke-NvidiaDetection
$msgs = ($r.Messages -join "`n")
Assert-Like $msgs '*WMI*' "Caption-only NVIDIA match should still hit"
Clear-Mock-GetCimInstance

"PASS t08_wmi_fallback"
