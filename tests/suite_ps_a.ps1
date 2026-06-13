# Suite PS-A: edge-case unit tests for Invoke-InstallCommandRetry against the
# REAL functions extracted from install.ps1 via the PowerShell AST.
# Run: pwsh -NoProfile -File suite_ps_a.ps1 <install.ps1> <tests_dir>
param([string]$InstallPs1, [string]$TestsDir)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $InstallPs1)) { Write-Host "install.ps1 not found"; exit 2 }

# --- pull the two real functions out of install.ps1 and define them here ---
$src = Get-Content -Raw $InstallPs1
$ast = [System.Management.Automation.Language.Parser]::ParseInput($src, [ref]$null, [ref]$null)
$fns = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -in @('Invoke-InstallCommand','Invoke-InstallCommandRetry') }, $true)
foreach ($f in $fns) { Invoke-Expression $f.Extent.Text }

# --- stubs the real functions depend on ---
$script:UnslothVerbose = $false
$script:SUBLOG = Join-Path $TestsDir "_ps_sublog.txt"
function substep { param($m, $c) Add-Content -LiteralPath $script:SUBLOG -Value $m }
$script:FLAKY = Join-Path $TestsDir "flaky.sh"
$env:STATE = Join-Path $TestsDir "_ps_state"
$env:UNSLOTH_INSTALL_RETRY_DELAY = "0"   # keep tests instant

$PASS = 0; $FAIL = 0
function ok($n)  { $script:PASS++; Write-Host "  PASS  $n" }
function bad($n) { $script:FAIL++; Write-Host "  FAIL  $n" }

# run a flaky scriptblock: fails $fails times (via STATE counter) then succeeds.
# returns @{ rc=<int>; retries=<int> }
function RunFlaky([int]$fails, [int]$exitcode = 4) {
    Set-Content -LiteralPath $env:STATE -Value ""             # reset counter
    Set-Content -LiteralPath $script:SUBLOG -Value $null      # reset retry log
    $script:n = $fails; $script:ec = $exitcode
    $rc = Invoke-InstallCommandRetry -Label "L" { & sh $script:FLAKY $script:n $script:ec }
    $retries = 0
    if (Test-Path $script:SUBLOG) {
        $retries = @(Select-String -Path $script:SUBLOG -Pattern 'retrying ' -SimpleMatch).Count
    }
    return @{ rc = [int]$rc; retries = [int]$retries }
}

# 1. success first try -> rc0, 0 retries
$r = RunFlaky 0
if ($r.rc -eq 0 -and $r.retries -eq 0) { ok "success_first_try: rc0, 0 retries" } else { bad "success_first_try (rc=$($r.rc) retries=$($r.retries))" }

# 2-3. fail K then ok (default 3) -> rc0, K retries  (K=1,2)
foreach ($k in 1,2) {
    $r = RunFlaky $k
    if ($r.rc -eq 0 -and $r.retries -eq $k) { ok "fail${k}_then_ok: rc0, $k retries" } else { bad "fail${k}_then_ok (rc=$($r.rc) retries=$($r.retries))" }
}

# 4. always fail (default 3) -> rc4 (REAL code), 2 retries
$r = RunFlaky 99
if ($r.rc -eq 4 -and $r.retries -eq 2) { ok "always_fail: rc=4 (REAL code), 2 retries" } else { bad "always_fail (rc=$($r.rc) retries=$($r.retries))" }

# 5. RETRIES=5, fail 4 then ok -> rc0, 4 retries
$env:UNSLOTH_INSTALL_RETRIES = "5"; $r = RunFlaky 4; $env:UNSLOTH_INSTALL_RETRIES = $null
if ($r.rc -eq 0 -and $r.retries -eq 4) { ok "RETRIES=5 fail4_then_ok: rc0, 4 retries" } else { bad "retries5 (rc=$($r.rc) retries=$($r.retries))" }

# 6. RETRIES=1 always fail -> single attempt, rc4, 0 retries
$env:UNSLOTH_INSTALL_RETRIES = "1"; $r = RunFlaky 99; $env:UNSLOTH_INSTALL_RETRIES = $null
if ($r.rc -eq 4 -and $r.retries -eq 0) { ok "RETRIES=1 always_fail: single attempt" } else { bad "retries1 (rc=$($r.rc) retries=$($r.retries))" }

# 7. exit-code preservation (RETRIES=1, always fail with code EC)
$env:UNSLOTH_INSTALL_RETRIES = "1"
foreach ($ec in 1,2,42,130) {
    $r = RunFlaky 99 $ec
    if ($r.rc -eq $ec) { ok "exit_code_preserved: $ec" } else { bad "exit_code_$ec (got $($r.rc))" }
}
$env:UNSLOTH_INSTALL_RETRIES = $null

# 8. invalid/zero RETRIES -> falls back to DEFAULT 3 (typo must not disable retry)
foreach ($bad in @('', 'abc', '0', '-2', '3.5')) {
    $env:UNSLOTH_INSTALL_RETRIES = $bad; $r = RunFlaky 99; $env:UNSLOTH_INSTALL_RETRIES = $null
    if ($r.rc -eq 4 -and $r.retries -eq 2) { ok "invalid RETRIES='$bad' -> default 3 attempts" } else { bad "invalid_retries_'$bad' (rc=$($r.rc) retries=$($r.retries))" }
}

# 9. huge RETRIES stops on success (fail2 then ok)
$env:UNSLOTH_INSTALL_RETRIES = "99999999"; $r = RunFlaky 2; $env:UNSLOTH_INSTALL_RETRIES = $null
if ($r.rc -eq 0 -and $r.retries -eq 2) { ok "huge RETRIES stops on success" } else { bad "huge_retries (rc=$($r.rc) retries=$($r.retries))" }

Write-Host "  ------------------------------------"
Write-Host "  SUITE PS-A: $PASS passed, $FAIL failed"
if ($FAIL -ne 0) { exit 1 } else { exit 0 }
