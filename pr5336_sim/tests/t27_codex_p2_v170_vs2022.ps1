$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# When VS 2022 is selected, the BuildCustomizations toolset version MUST be
# v170. (Baseline hardcodes v170, so passes. Fixed must still produce v170.)

if ($env:HARNESS_MODE -eq 'fixed') {
    . "$PSScriptRoot/../helpers_fixed.ps1"
}

$root = New-Sandbox 't27'
New-FakeVsWhere -ShimName 'vswhere-2022'
$r = if ($env:HARNESS_MODE -eq 'fixed') { Find-VsBuildToolsGated } else { Find-VsBuildTools }

if ($env:HARNESS_MODE -eq 'fixed') {
    Assert-Eq $r.MsbuildToolsetVersion 'v170' 'fixed: VS 2022 -> v170'
} else {
    # Baseline doesn't expose this field; the hardcoded literal v170 lives in
    # the script body. Smoke check: the generator is VS 17 2022.
    Assert-Eq $r.Generator 'Visual Studio 17 2022' 'baseline: VS 17 2022 generator'
}

"PASS t27_codex_p2_v170_vs2022"
