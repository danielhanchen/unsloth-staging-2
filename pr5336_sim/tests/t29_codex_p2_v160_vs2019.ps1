$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# Latent bug: VS 2019 should map to v160 BuildCustomizations, but baseline
# hardcodes v170. Fixed branch should produce v160.

if ($env:HARNESS_MODE -eq 'fixed') {
    . "$PSScriptRoot/../helpers_fixed.ps1"
    $root = New-Sandbox 't29f'
    New-FakeVsWhere -ShimName 'vswhere-2019'
    $r = Find-VsBuildToolsGated
    Assert-Eq $r.MsbuildToolsetVersion 'v160' 'fixed: VS 2019 -> v160'
    "PASS t29_codex_p2_v160_vs2019"
} else {
    Write-Host "XFAIL t29_codex_p2_v160_vs2019 — baseline has no MsbuildToolsetVersion field (latent bug pre-existing main)"
}
