$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# When cmake is missing entirely, the Get-CMakeVersion helper must return $null
# without throwing. (Only meaningful on fixed branch; baseline has no helper.)

if ($env:HARNESS_MODE -eq 'fixed') {
    . "$PSScriptRoot/../helpers_fixed.ps1"
}

$root = New-Sandbox 't26'
# Deliberately do NOT put cmake on PATH.

if ($env:HARNESS_MODE -eq 'fixed') {
    $v = Get-CMakeVersion
    Assert-True ($null -eq $v) "missing cmake -> Get-CMakeVersion returns null"
} else {
    # Nothing to assert on baseline; just pass.
    "Baseline: no Get-CMakeVersion to test."
}

"PASS t26_codex_p1_cmake_missing"
