$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../sandbox.ps1"
. "$PSScriptRoot/../helpers.ps1"

# Codex P2 #2: VS 2026 must route to v180, not v170. Baseline has no field; we
# instead grep the actual setup.ps1 file to confirm the hardcoded v170 is still
# the literal at line 1098 (regression detector).

# Walk up from $PSScriptRoot until install.ps1 is found, then anchor.
$repoRoot = $PSScriptRoot
for ($i = 0; $i -lt 6; $i++) {
    if (Test-Path (Join-Path $repoRoot 'install.ps1')) { break }
    $repoRoot = Split-Path -Parent $repoRoot
}
$setupPs1 = Join-Path $repoRoot 'studio/setup.ps1'
$content = Get-Content $setupPs1 -Raw

if ($env:HARNESS_MODE -eq 'fixed') {
    . "$PSScriptRoot/../helpers_fixed.ps1"
    $root = New-Sandbox 't28f'
    New-FakeVsWhere -ShimName 'vswhere-2026'
    # Test the ROUTING field on Find-VsBuildTools (codex P2 #2 is about the
    # field, not about gating). The gate is exercised separately in t24/t25.
    $r = Find-VsBuildTools
    Assert-Eq $r.MsbuildToolsetVersion 'v180' 'fixed: VS 2026 -> v180'
    # And the file must NOT contain a literal v170 at the CUDA repair site.
    $cudaRepairBlock = ($content -split "`n")[1090..1130] -join "`n"
    if ($cudaRepairBlock -match '\\v170\\BuildCustomizations') {
        throw "Fixed branch still has literal \\v170\\BuildCustomizations in CUDA repair block."
    }
    "PASS t28_codex_p2_v180_vs2026"
} else {
    # Baseline regression detector. The real studio/setup.ps1 may be in either:
    #   (a) pre-fix state — still has literal v170\BuildCustomizations
    #   (b) post-fix state — the file has been patched in place
    # Both cases are OK here; the codex-fix helpers test only meaningfully runs
    # under HARNESS_MODE=fixed.
    $cudaRepairBlock = ($content -split "`n")[1090..1135] -join "`n"
    if ($cudaRepairBlock -match '\\v170\\BuildCustomizations' -and $cudaRepairBlock -notmatch '\$VsMsbuildToolsetVersion|\$toolsetVer') {
        Write-Host "XFAIL t28_codex_p2_v180_vs2026 — baseline hardcodes v170 (codex P2 #2 reproduced)"
    } else {
        Write-Host "PASS t28_codex_p2_v180_vs2026 — fix already in source"
    }
}
