$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# CASE A: nvidia-smi on PATH, working.
$root = New-Sandbox 't03a'
Add-ShimToPath -ShimName 'nvidia-smi-ok' -ExposeAs 'nvidia-smi'
$r = Invoke-NvidiaDetection
Assert-True $r.HasNvidiaSmi "PATH+working shim should be detected"
Assert-Like $r.NvidiaSmiExe '*nvidia-smi*' "NvidiaSmiExe should point at the shim"

# CASE B: nvidia-smi on PATH, broken (driver dead).
$root = New-Sandbox 't03b'
Add-ShimToPath -ShimName 'nvidia-smi-broken' -ExposeAs 'nvidia-smi'
Set-Mock-GetCimInstance -Controllers @()   # no WMI hit
$r = Invoke-NvidiaDetection
Assert-False $r.HasNvidiaSmi "PATH+broken shim should NOT be detected"
Clear-Mock-GetCimInstance

# CASE C: nvidia-smi NOT on PATH and no other paths exist.
$root = New-Sandbox 't03c'
Set-Mock-GetCimInstance -Controllers @()
$r = Invoke-NvidiaDetection
Assert-False $r.HasNvidiaSmi "missing nvidia-smi -> not detected"
Clear-Mock-GetCimInstance

"PASS t03_path_lookup"
