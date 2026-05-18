$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# CASE A: nvidia-smi shim that exits 0 -> Test-NvidiaSmiExe returns true.
$root = New-Sandbox 't01a'
$ok = Join-Path $env:SystemRoot 'nvidia-smi-ok.exe'
Copy-Item "$PSScriptRoot/../shims/nvidia-smi-ok" $ok -Force
chmod +x $ok | Out-Null
$r = Test-NvidiaSmiExe $ok
Assert-True $r "exit-0 shim should return true"

# CASE B: nvidia-smi shim that exits 1 -> returns false.
$root = New-Sandbox 't01b'
$bad = Join-Path $env:SystemRoot 'nvidia-smi-broken.exe'
Copy-Item "$PSScriptRoot/../shims/nvidia-smi-broken" $bad -Force
chmod +x $bad | Out-Null
$r = Test-NvidiaSmiExe $bad
Assert-False $r "exit-1 shim should return false"

# CASE C: shim that exits 9 (permission) -> returns false.
$root = New-Sandbox 't01c'
$perm = Join-Path $env:SystemRoot 'nvidia-smi-permission.exe'
Copy-Item "$PSScriptRoot/../shims/nvidia-smi-permission" $perm -Force
chmod +x $perm | Out-Null
$r = Test-NvidiaSmiExe $perm
Assert-False $r "exit-9 shim should return false"

# CASE D: path that does not exist -> returns false (caught by try).
$r = Test-NvidiaSmiExe '/this/does/not/exist'
Assert-False $r "nonexistent path should return false"

"PASS t01_test_nvidia_smi_exe"
