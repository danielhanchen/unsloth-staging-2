$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# Pure CPU-only host: no nvidia-smi, no NVIDIA in WMI.
# The detection block should quietly return false with no WMI message.
$root = New-Sandbox 't09'
Set-Mock-GetCimInstance -Controllers @()
$r = Invoke-NvidiaDetection
Assert-False $r.HasNvidiaSmi
$msgs = ($r.Messages -join "`n")
Assert-False ($msgs -like '*WMI*')
Clear-Mock-GetCimInstance

"PASS t09_no_gpu_at_all"
