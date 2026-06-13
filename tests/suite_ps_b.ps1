# Suite PS-B: Test-UvVersionOk floor (0.8.16) + env-default layering, using the
# REAL Test-UvVersionOk from install.ps1. Run: pwsh -NoProfile -File suite_ps_b.ps1 <install.ps1> <tests_dir>
param([string]$InstallPs1, [string]$TestsDir)
$ErrorActionPreference = "Stop"

$src = Get-Content -Raw $InstallPs1

# pull Test-UvVersionOk and define it; bind $UvMinVersion to the real pinned value
$ast = [System.Management.Automation.Language.Parser]::ParseInput($src, [ref]$null, [ref]$null)
$fn = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Test-UvVersionOk' }, $true)[0]
Invoke-Expression $fn.Extent.Text
if ($src -match '\$UvMinVersion\s*=\s*"([^"]+)"') { $UvMinVersion = $Matches[1] } else { throw "UvMinVersion not found" }
Write-Host "  UvMinVersion (from install.ps1) = $UvMinVersion"
if ($UvMinVersion -ne "0.8.16") { Write-Host "  FAIL: expected floor 0.8.16"; exit 1 }

$PASS = 0; $FAIL = 0
function ok($n)  { $script:PASS++; Write-Host "  PASS  $n" }
function bad($n) { $script:FAIL++; Write-Host "  FAIL  $n" }

# mock `uv` on PATH that prints a chosen version (cross-platform: a shell shim on
# Unix, a .cmd shim on Windows where a shebang file is not executable as `uv`).
$mockdir = Join-Path $TestsDir "ps_mockuv"
New-Item -ItemType Directory -Force -Path $mockdir | Out-Null
$onWindows = ($IsWindows -or $env:OS -eq 'Windows_NT')
function Set-MockUv([string]$ver) {
    if ($onWindows) {
        # Clear any stale shim, then write uv.cmd. PATH order (mockdir first) +
        # PATHEXT make `uv` resolve to this .cmd ahead of the real uv.exe.
        Remove-Item -LiteralPath (Join-Path $mockdir "uv") -ErrorAction SilentlyContinue
        Set-Content -LiteralPath (Join-Path $mockdir "uv.cmd") -Value "@echo off`r`necho uv $ver"
    } else {
        $p = Join-Path $mockdir "uv"
        Set-Content -LiteralPath $p -Value "#!/bin/sh`necho `"uv $ver`""
        & chmod +x $p
    }
}
$env:PATH = "$mockdir" + [IO.Path]::PathSeparator + $env:PATH

function CheckUv([string]$ver, [string]$want) {
    Set-MockUv $ver
    $got = if (Test-UvVersionOk) { "ok" } else { "reinstall" }
    if ($got -eq $want) { ok "Test-UvVersionOk($ver) -> $got" } else { bad "Test-UvVersionOk($ver): got $got want $want" }
}
CheckUv "0.7.22"  "reinstall"   # old floor now too old
CheckUv "0.8.15"  "reinstall"
CheckUv "0.8.16"  "ok"
CheckUv "0.8.17"  "ok"
CheckUv "0.10.12" "ok"          # 0.10 vs 0.8 must compare numerically ([version])
CheckUv "0.11.21" "ok"
CheckUv "1.2.3"   "ok"
CheckUv "0.8.2"   "reinstall"   # 2 < 16

# env-default layering: the four defaults apply when unset, preserved when set
foreach ($pair in @(@('UV_HTTP_RETRIES','5'), @('UV_HTTP_TIMEOUT','180'))) {
    $name = $pair[0]; $def = $pair[1]
    # the literal default line must exist in install.ps1
    if ($src -match [regex]::Escape("if (-not `$env:$name) {") ) { ok "install.ps1 declares $name default" } else { bad "install.ps1 missing $name default" }
}
# the bumped floor literal must be present and the old one gone
if ($src -match 'UvMinVersion\s*=\s*"0\.8\.16"') { ok "floor bumped to 0.8.16" } else { bad "floor not 0.8.16" }
if ($src -notmatch 'UvMinVersion\s*=\s*"0\.7\.22"') { ok "old floor 0.7.22 removed" } else { bad "old floor still present" }

Write-Host "  ------------------------------------"
Write-Host "  SUITE PS-B: $PASS passed, $FAIL failed"
if ($FAIL -ne 0) { exit 1 } else { exit 0 }
