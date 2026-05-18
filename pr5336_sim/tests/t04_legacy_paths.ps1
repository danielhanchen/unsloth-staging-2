$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# CASE A: nvidia-smi at legacy %ProgramFiles%\NVIDIA Corporation\NVSMI\.
$root = New-Sandbox 't04a'
$legacy = "$env:ProgramFiles/NVIDIA Corporation/NVSMI/nvidia-smi.exe"
New-FakeNvidiaSmiAt -AtPath $legacy -ShimName 'nvidia-smi-ok'
Set-Mock-GetCimInstance -Controllers @()
$r = Invoke-NvidiaDetection
Assert-True $r.HasNvidiaSmi "legacy NVSMI path should be detected"
Assert-Like $r.NvidiaSmiExe '*NVSMI*nvidia-smi.exe' "path should be the legacy one"
Clear-Mock-GetCimInstance

# CASE B: nvidia-smi at %SystemRoot%\System32\.
$root = New-Sandbox 't04b'
$sys32 = "$env:SystemRoot/System32/nvidia-smi.exe"
New-FakeNvidiaSmiAt -AtPath $sys32 -ShimName 'nvidia-smi-ok'
Set-Mock-GetCimInstance -Controllers @()
$r = Invoke-NvidiaDetection
Assert-True $r.HasNvidiaSmi "System32 path should be detected"
Assert-Like $r.NvidiaSmiExe '*System32*nvidia-smi.exe' "path should be System32"
Clear-Mock-GetCimInstance

# CASE C: nvidia-smi at legacy location but BROKEN -> not detected.
$root = New-Sandbox 't04c'
$legacy = "$env:ProgramFiles/NVIDIA Corporation/NVSMI/nvidia-smi.exe"
New-FakeNvidiaSmiAt -AtPath $legacy -ShimName 'nvidia-smi-broken'
Set-Mock-GetCimInstance -Controllers @()
$r = Invoke-NvidiaDetection
Assert-False $r.HasNvidiaSmi "broken nvidia-smi at legacy path should not be detected"
Clear-Mock-GetCimInstance

"PASS t04_legacy_paths"
