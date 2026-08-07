# Executes Clear-WebViewCaches from studio/setup.ps1 against a sandboxed LOCALAPPDATA.
# setup.ps1 as a whole performs a full install, so the function is lifted out by AST
# rather than dot-sourced.
$ErrorActionPreference = 'Stop'

$Pass = 0
$Fail = 0
function Ok($label)  { $script:Pass++; Write-Host "  PASS: $label" }
function Bad($label) { $script:Fail++; Write-Host "  FAIL: $label" -ForegroundColor Red }
function Check($label, $cond) { if ($cond) { Ok $label } else { Bad $label } }

$SetupPs1 = Join-Path $PSScriptRoot "..\..\studio\setup.ps1"
$SetupPs1 = (Resolve-Path $SetupPs1).Path
Write-Host "setup.ps1: $SetupPs1"

$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($SetupPs1, [ref]$null, [ref]$errors)
if ($errors) { $errors | ForEach-Object { Write-Host $_.Message }; exit 1 }

$fnAst = $ast.FindAll(
    { param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Clear-WebViewCaches' },
    $true) | Select-Object -First 1
if (-not $fnAst) { Write-Host "Clear-WebViewCaches not found in setup.ps1" -ForegroundColor Red; exit 1 }
Write-Host "extracted Clear-WebViewCaches ($($fnAst.Extent.EndLineNumber - $fnAst.Extent.StartLineNumber + 1) lines)"

# substep is setup.ps1's logger; capture what the function reports instead of printing it.
$script:Notes = New-Object System.Collections.Generic.List[string]
function substep { param([string]$Message, [string]$Color) $script:Notes.Add($Message) }

# Dot-sourced, not invoked: `&` would define the function in a child scope that is
# discarded the moment the block returns.
. ([scriptblock]::Create($fnAst.Extent.Text))
if (-not (Get-Command Clear-WebViewCaches -ErrorAction SilentlyContinue)) {
    Write-Host "Clear-WebViewCaches did not land in this scope" -ForegroundColor Red; exit 1
}

$BID = 'ai.unsloth.studio'
$Root = Join-Path ([System.IO.Path]::GetTempPath()) ("wvprobe_" + [System.Guid]::NewGuid().ToString('N').Substring(0, 8))

function New-Sandbox {
    $sandbox = Join-Path $Root ([System.Guid]::NewGuid().ToString('N').Substring(0, 8))
    $default = Join-Path $sandbox "$BID\EBWebView\Default"
    foreach ($sub in @('Cache', 'Code Cache', 'GPUCache', 'Service Worker',
                       'Local Storage', 'IndexedDB', 'Network')) {
        New-Item -ItemType Directory -Force -Path (Join-Path $default $sub) | Out-Null
    }
    Set-Content -LiteralPath (Join-Path $default 'Cache\data_1') -Value 'stale asset'
    Set-Content -LiteralPath (Join-Path $default 'Code Cache\index') -Value 'compiled js'
    Set-Content -LiteralPath (Join-Path $default 'GPUCache\data_0') -Value 'shaders'
    Set-Content -LiteralPath (Join-Path $default 'Service Worker\sw') -Value 'worker'
    Set-Content -LiteralPath (Join-Path $default 'Local Storage\leveldb.log') -Value 'user settings'
    Set-Content -LiteralPath (Join-Path $default 'IndexedDB\store.blob') -Value 'user data'
    Set-Content -LiteralPath (Join-Path $default 'Network\Cookies') -Value 'session'
    Set-Content -LiteralPath (Join-Path $default 'Preferences') -Value 'prefs'
    Set-Content -LiteralPath (Join-Path $sandbox "$BID\.webview-cache-cleared") -Value '2026.4.8'
    # Decoys: a prefix-sharing bundle id and an unrelated app.
    New-Item -ItemType Directory -Force -Path (Join-Path $sandbox "${BID}2\EBWebView\Default\Cache") | Out-Null
    Set-Content -LiteralPath (Join-Path $sandbox "${BID}2\EBWebView\Default\Cache\data_1") -Value 'other'
    New-Item -ItemType Directory -Force -Path (Join-Path $sandbox 'Microsoft\EdgeWebView\Default\Cache') | Out-Null
    Set-Content -LiteralPath (Join-Path $sandbox 'Microsoft\EdgeWebView\Default\Cache\data_1') -Value 'edge'
    return $sandbox
}

function Invoke-Clear($sandbox) {
    $script:Notes.Clear()
    $saved = $env:LOCALAPPDATA
    try {
        $env:LOCALAPPDATA = $sandbox
        Clear-WebViewCaches
    } finally {
        $env:LOCALAPPDATA = $saved
    }
}

Write-Host ""
Write-Host "== 1. a populated profile: caches go, user data stays =="
$s = New-Sandbox
$d = Join-Path $s "$BID\EBWebView\Default"
Invoke-Clear $s
foreach ($sub in @('Cache', 'Code Cache', 'GPUCache', 'Service Worker')) {
    Check "$sub removed" (-not (Test-Path -LiteralPath (Join-Path $d $sub)))
}
Check "Local Storage kept"  (Test-Path -LiteralPath (Join-Path $d 'Local Storage\leveldb.log'))
Check "IndexedDB kept"      (Test-Path -LiteralPath (Join-Path $d 'IndexedDB\store.blob'))
Check "cookies kept"        (Test-Path -LiteralPath (Join-Path $d 'Network\Cookies'))
Check "Preferences kept"    (Test-Path -LiteralPath (Join-Path $d 'Preferences'))
Check "version stamp dropped so the app retries" `
    (-not (Test-Path -LiteralPath (Join-Path $s "$BID\.webview-cache-cleared")))
Check "prefix-decoy bundle id untouched" (Test-Path -LiteralPath (Join-Path $s "${BID}2\EBWebView\Default\Cache\data_1"))
Check "unrelated WebView2 host untouched" (Test-Path -LiteralPath (Join-Path $s 'Microsoft\EdgeWebView\Default\Cache\data_1'))
Check "reports what it cleared" ($script:Notes -match 'cleared stale WebView caches')

Write-Host ""
Write-Host "== 2. nothing to clear is a silent no-op =="
$s = New-Sandbox
Remove-Item -LiteralPath (Join-Path $s "$BID") -Recurse -Force
$threw = $false
try { Invoke-Clear $s } catch { $threw = $true; Write-Host "    | $($_.Exception.Message)" }
Check "an absent profile does not throw" (-not $threw)
Check "and says nothing" ($script:Notes.Count -eq 0)

Write-Host ""
Write-Host "== 3. no LOCALAPPDATA is a no-op, not a delete under the cwd =="
$saved = $env:LOCALAPPDATA
$threw = $false
try {
    Remove-Item Env:\LOCALAPPDATA -ErrorAction SilentlyContinue
    $script:Notes.Clear()
    Clear-WebViewCaches
} catch { $threw = $true; Write-Host "    | $($_.Exception.Message)" } finally { $env:LOCALAPPDATA = $saved }
Check "unset LOCALAPPDATA does not throw" (-not $threw)
Check "unset LOCALAPPDATA says nothing" ($script:Notes.Count -eq 0)

Write-Host ""
Write-Host "== 4. the stamp is dropped even when no cache dir exists =="
# The case that matters: the removals failed or there was nothing to remove, and the
# app's clear is the retry. A surviving stamp suppresses that retry forever.
$s = New-Sandbox
Remove-Item -LiteralPath (Join-Path $s "$BID\EBWebView") -Recurse -Force
Invoke-Clear $s
Check "stamp dropped with no caches present" `
    (-not (Test-Path -LiteralPath (Join-Path $s "$BID\.webview-cache-cleared")))
Check "no clear reported when nothing was removed" ($script:Notes.Count -eq 0)

Write-Host ""
Write-Host "== 5. a junction is unlinked, not recursed into =="
$s = New-Sandbox
$d = Join-Path $s "$BID\EBWebView\Default"
$canary = Join-Path $s 'canary'
New-Item -ItemType Directory -Force -Path $canary | Out-Null
Set-Content -LiteralPath (Join-Path $canary 'keep.txt') -Value 'must survive'
Remove-Item -LiteralPath (Join-Path $d 'Cache') -Recurse -Force
New-Item -ItemType Junction -Path (Join-Path $d 'Cache') -Target $canary | Out-Null
Invoke-Clear $s
Check "the junction itself is gone" (-not (Test-Path -LiteralPath (Join-Path $d 'Cache')))
Check "the junction target survives" (Test-Path -LiteralPath (Join-Path $canary 'keep.txt'))

Write-Host ""
Write-Host "== 6. a locked cache file is survivable, not fatal =="
# This is the whole reason the app-side clear exists: during an in-app update the old
# WebView still holds these files open.
$s = New-Sandbox
$d = Join-Path $s "$BID\EBWebView\Default"
$locked = Join-Path $d 'Code Cache\index'
$fs = [System.IO.File]::Open($locked, 'Open', 'ReadWrite', 'None')
$threw = $false
try { Invoke-Clear $s } catch { $threw = $true; Write-Host "    | $($_.Exception.Message)" } finally { $fs.Close() }
Check "a locked file does not abort setup" (-not $threw)
Check "the locked cache is left behind" (Test-Path -LiteralPath $locked)
Check "the other caches are still cleared" (-not (Test-Path -LiteralPath (Join-Path $d 'GPUCache')))
Check "the stamp is still dropped, so the app retries" `
    (-not (Test-Path -LiteralPath (Join-Path $s "$BID\.webview-cache-cleared")))

Write-Host ""
Write-Host "== 7. setup.ps1 clears only after validating the override =="
# Mirrors the shell-side assertion: clearing before the UNSLOTH_STUDIO_HOME check
# would turn a typo into cache loss plus an abort.
$text = Get-Content -LiteralPath $SetupPs1 -Raw
$lines = $text -split "`r?`n"
$callLine = ($lines | Select-String -Pattern '^\s*Clear-WebViewCaches\s*$' | Select-Object -First 1).LineNumber
$validateLine = ($lines | Select-String -Pattern 'does not exist"$' | Select-Object -Last 1).LineNumber
Check "setup.ps1 calls Clear-WebViewCaches" ($null -ne $callLine)
Check "the call sits after the override validation ($callLine > $validateLine)" `
    ($null -ne $callLine -and $null -ne $validateLine -and $callLine -gt $validateLine)

Write-Host ""
Write-Host "Results: $Pass passed, $Fail failed"
if (Test-Path -LiteralPath $Root) { Remove-Item -LiteralPath $Root -Recurse -Force -ErrorAction SilentlyContinue }
if ($Fail -gt 0) { exit 1 }
