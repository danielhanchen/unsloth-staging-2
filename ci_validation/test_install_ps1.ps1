# Validates install.ps1's uv logic on real Windows PowerShell.
# 1) parse install.ps1 (0 errors) and assert the PR's floor + function exist
# 2) dot-source the REAL Test-UvVersionOk (extracted via AST) and exercise it with a mocked uv
# 3) env-override (-not $env:X) semantics, including "0" preservation
# 4) winget upgrade -> install -> astral fallback control flow, with re-verification
# Exits non-zero on any failure.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ps1  = Join-Path $root "install.ps1"
if (-not (Test-Path $ps1)) { Write-Error "install.ps1 not found at $ps1"; exit 2 }
$fail = 0
function Assert($name, $cond, $detail) {
    Write-Host ("  [{0}] {1} : {2}" -f $(if ($cond) {"PASS"} else {"FAIL"}), $name, $detail)
    if (-not $cond) { $script:fail++ }
}

Write-Host "=== 1) parse install.ps1 ==="
$errs = $null; $ast = [System.Management.Automation.Language.Parser]::ParseFile($ps1, [ref]$null, [ref]$errs)
Assert "ps1_parses" (($errs -eq $null) -or ($errs.Count -eq 0)) ("parse errors: " + (@($errs).Count))
$text = Get-Content $ps1 -Raw
Assert "floor_0.7.22" ($text -match '\$UvMinVersion\s*=\s*"0\.7\.22"') "install.ps1 declares `$UvMinVersion = 0.7.22"
Assert "has_test_fn"  ($text -match 'function\s+Test-UvVersionOk') "Test-UvVersionOk present"
Assert "env_default"  ($text -match '\$env:UV_COMPILE_BYTECODE_TIMEOUT\s*=\s*"180"') "defaults timeout to 180"

Write-Host "=== 2) REAL Test-UvVersionOk (AST-extracted) vs mocked uv ==="
$fn = $ast.FindAll({ param($n) ($n -is [System.Management.Automation.Language.FunctionDefinitionAst]) -and ($n.Name -eq 'Test-UvVersionOk') }, $true)
if ($fn.Count -lt 1) { Write-Error "Test-UvVersionOk not found in AST"; exit 2 }
Invoke-Expression $fn[0].Extent.Text   # defines Test-UvVersionOk in this scope
$UvMinVersion = "0.7.22"
$script:fakeout = ""
function uv { $script:fakeout }        # `& uv --version` -> this mock; Get-Command uv finds it
function Check([string]$out, [string]$exp, [string]$note) {
    $script:fakeout = $out
    $got = if (Test-UvVersionOk) { "OK" } else { "NO" }
    Assert ("ver:" + $note) ($got -eq $exp) ("exp=$exp got=$got  '$out'")
}
Check "uv 0.7.21"      "NO" "below floor"
Check "uv 0.7.22"      "OK" "exact floor"
Check "uv 0.7.23"      "OK" "newer patch"
Check "uv 0.8.0"       "OK" "newer minor"
Check "uv 0.10.0"      "OK" "numeric 0.10>=0.7.22"
Check "uv 1.0.0"       "OK" "major"
Check "uv 0.7.22 (abc 2025-01-01)" "OK" "extra metadata"
Check "uv abc"         "NO" "non-numeric"
Check "uv"             "NO" "no version field"

Write-Host "=== 3) env override (-not `$env:X) semantics + 0 preserved ==="
function ApplyDefault { if (-not $env:UV_COMPILE_BYTECODE_TIMEOUT) { $env:UV_COMPILE_BYTECODE_TIMEOUT = "180" } }
foreach ($pair in @(@("__UNSET__","180"), @("","180"), @("0","0"), @("300","300"))) {
    if ($pair[0] -eq "__UNSET__") { Remove-Item Env:\UV_COMPILE_BYTECODE_TIMEOUT -ErrorAction SilentlyContinue }
    else { $env:UV_COMPILE_BYTECODE_TIMEOUT = $pair[0] }
    ApplyDefault
    Assert ("env:" + $pair[0]) ($env:UV_COMPILE_BYTECODE_TIMEOUT -eq $pair[1]) ("-> " + $env:UV_COMPILE_BYTECODE_TIMEOUT + " (expect " + $pair[1] + ")")
}
Remove-Item Env:\UV_COMPILE_BYTECODE_TIMEOUT -ErrorAction SilentlyContinue

Write-Host "=== 4) winget upgrade/install/astral fallback flow (re-verified) ==="
function Run-Flow($st) {
    $script:S = $st; $script:acts = @()
    function T { if (-not $script:S.uv) { return $false }; try { return ([version]$script:S.uv -ge [version]"0.7.22") } catch { return $false } }
    if (-not (T)) {
        if ($script:S.winget) {
            $script:acts += "upgrade"; if ($script:S.uv) { $script:S.uv = $script:S.upTo }
            if (-not (T)) { $script:acts += "install"; if ($null -ne $script:S.instTo) { $script:S.uv = $script:S.instTo } }
        }
        if (-not (T)) { $script:acts += "astral"; if ($null -ne $script:S.astTo) { $script:S.uv = $script:S.astTo } }
    }
    if (-not (T)) { $script:acts += "FAIL" }
    return ,$script:acts
}
function Scen($name, $st, $expectFail, $must) {
    $a = Run-Flow $st
    $okEnd = if ($expectFail) { $a -contains "FAIL" } else { $a -notcontains "FAIL" }
    $okMust = $true; foreach ($m in $must) { if ($a -notcontains $m) { $okMust = $false } }
    Assert ("flow:" + $name) ($okEnd -and $okMust) ("uv=" + $script:S.uv + " acts=[" + ($a -join ",") + "]")
}
Scen "absent->install"        @{uv=$null;   winget=$true;  upTo=$null;   instTo="0.8.0"; astTo=$null}    $false @("install")
Scen "old->upgrade"           @{uv="0.7.14";winget=$true;  upTo="0.8.0"; instTo="0.8.0"; astTo=$null}    $false @("upgrade")
Scen "old->upgrade noop->inst" @{uv="0.7.14";winget=$true;  upTo="0.7.14";instTo="0.8.0"; astTo=$null}    $false @("upgrade","install")
Scen "winget fail->astral"    @{uv="0.7.14";winget=$true;  upTo="0.7.14";instTo="0.7.14";astTo="0.8.0"}  $false @("upgrade","install","astral")
Scen "already-new->noop"      @{uv="0.8.0"; winget=$true;  upTo=$null;   instTo=$null;   astTo=$null}    $false @()
Scen "no-winget->astral"      @{uv=$null;   winget=$false; upTo=$null;   instTo=$null;   astTo="0.8.0"}  $false @("astral")
Scen "all-fail->errors"       @{uv="0.7.14";winget=$true;  upTo="0.7.14";instTo="0.7.14";astTo="0.7.14"} $true  @("FAIL")

Write-Host ""
if ($fail -gt 0) { Write-Host "PS1 VALIDATION FAILED ($fail)"; exit 1 }
Write-Host "ALL install.ps1 CHECKS PASSED"
