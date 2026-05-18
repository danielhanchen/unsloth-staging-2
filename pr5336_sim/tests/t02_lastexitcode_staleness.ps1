$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# This is the bug the PR fixes: pre-PR code used `*> $null` which under PS 5.1 left
# $LASTEXITCODE poisoned by the previous external command's exit code. We can't
# install PS 5.1 here, but we CAN demonstrate the helper's invariant:
# whatever $LASTEXITCODE was before, after calling Test-NvidiaSmiExe its return
# reflects ONLY the shim's actual exit code, not the stale value.

$root = New-Sandbox 't02'
$ok = Join-Path $env:SystemRoot 'nvidia-smi.exe'
Copy-Item "$PSScriptRoot/../shims/nvidia-smi-ok" $ok -Force
chmod +x $ok | Out-Null

# CASE A: poison $LASTEXITCODE = 1 before calling helper that should return true.
& bash -c 'exit 1'
Assert-Eq $LASTEXITCODE 1 "precondition: LASTEXITCODE should be 1 before helper call"
$r = Test-NvidiaSmiExe $ok
Assert-True $r "helper must return true even when LASTEXITCODE was 1 going in"

# CASE B: poison $LASTEXITCODE = 0 before calling helper on a broken shim.
$bad = Join-Path $env:SystemRoot 'nvidia-smi-broken.exe'
Copy-Item "$PSScriptRoot/../shims/nvidia-smi-broken" $bad -Force
chmod +x $bad | Out-Null
& bash -c 'exit 0'
Assert-Eq $LASTEXITCODE 0 "precondition: LASTEXITCODE should be 0 before helper call"
$r = Test-NvidiaSmiExe $bad
Assert-False $r "helper must return false even when LASTEXITCODE was 0 going in"

"PASS t02_lastexitcode_staleness"
