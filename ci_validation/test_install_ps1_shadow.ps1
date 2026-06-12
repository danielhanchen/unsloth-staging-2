# Validates the PATH-shadowing recovery added to install.ps1: when an older uv
# sits earlier on PATH than a freshly installed one (active venv, Scoop/pipx
# shim), the installer must still find the new uv instead of aborting.
#
# Uses real uv shims on PATH (uv.cmd on Windows, a uv script elsewhere) and the
# AST-extracted Test-UvVersionOk. The recovery loop mirrors install.ps1; it uses
# the OS path separator here so the same test runs on every OS, and a textual
# assertion confirms the shipped block uses the Windows ';' separator.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ps1  = Join-Path $root "install.ps1"
$fail = 0
function Assert($name, $cond, $detail) {
    Write-Host ("  [{0}] {1} : {2}" -f $(if ($cond) {"PASS"} else {"FAIL"}), $name, $detail)
    if (-not $cond) { $script:fail++ }
}
$onWindows = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)
$sep = [System.IO.Path]::PathSeparator

# The shipped recovery block must be present and use the Windows ';' separator.
$txt = Get-Content $ps1 -Raw
Assert "ps1_has_recovery_loop" ($txt -match 'foreach \(\$d in @\(\$env:UV_INSTALL_DIR') "recovery loop present"
Assert "ps1_uses_semicolon"    ($txt -match '\$env:PATH = "\$d;\$origPath"') "prepends with ';' (Windows)"
Assert "ps1_winget_links"      ($txt -match 'Microsoft\\WinGet\\Links') "includes winget Links candidate"

# AST-extract the real Test-UvVersionOk.
$errs = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($ps1, [ref]$null, [ref]$errs)
$fn = $ast.FindAll({ param($n) ($n -is [System.Management.Automation.Language.FunctionDefinitionAst]) -and ($n.Name -eq 'Test-UvVersionOk') }, $true)
Invoke-Expression $fn[0].Extent.Text
$UvMinVersion = "0.7.22"

$work   = Join-Path ([System.IO.Path]::GetTempPath()) ("uvshadow_" + [guid]::NewGuid().ToString("N"))
$oldDir = Join-Path $work "old"
$newDir = Join-Path $work "new"
$null = New-Item -ItemType Directory -Force -Path $oldDir, $newDir
function Write-Shim($dir, $ver) {
    if ($onWindows) {
        Set-Content -LiteralPath (Join-Path $dir "uv.cmd") -Value "@echo uv $ver" -Encoding ascii
    } else {
        $p = Join-Path $dir "uv"
        Set-Content -LiteralPath $p -Value "#!/bin/sh`necho 'uv $ver'"
        & chmod +x $p
    }
}
Write-Shim $oldDir "0.7.10"   # stale, below floor
Write-Shim $newDir "0.8.0"    # freshly installed, satisfies floor

# Recovery loop copied from install.ps1 (OS-separator so it runs everywhere).
function Invoke-Recovery {
    if (-not (Test-UvVersionOk)) {
        $origPath = $env:PATH
        foreach ($d in @($env:UV_INSTALL_DIR, $env:XDG_BIN_HOME)) {
            if ($d -and (Test-Path $d)) {
                $env:PATH = "$d$sep$origPath"
                if (Test-UvVersionOk) { break }
                $env:PATH = $origPath
            }
        }
    }
}

$basePath = $env:PATH
try {
    # 1) stale uv earlier on PATH shadows everything -> bare check fails (the bug).
    $env:PATH = "$oldDir$sep$basePath"
    Remove-Item Env:\UV_INSTALL_DIR -ErrorAction SilentlyContinue
    Assert "stale_shadows_bare" (-not (Test-UvVersionOk)) "bare uv resolves stale 0.7.10 (< 0.7.22)"

    # 2) recovery points at the freshly installed dir -> new uv now resolves.
    $env:UV_INSTALL_DIR = $newDir
    Invoke-Recovery
    Assert "recovery_finds_new" (Test-UvVersionOk) "after recovery, uv resolves 0.8.0 (no false abort)"

    # 3) no good uv anywhere -> recovery cannot help -> stays failed (abort is correct).
    $env:PATH = "$oldDir$sep$basePath"
    $env:UV_INSTALL_DIR = $oldDir
    Invoke-Recovery
    Assert "no_good_uv_stays_failed" (-not (Test-UvVersionOk)) "recovery does not mask a genuine failure"
} finally {
    $env:PATH = $basePath
    Remove-Item Env:\UV_INSTALL_DIR -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}

Write-Host ""
if ($fail -gt 0) { Write-Host "SHADOW TEST FAILED ($fail)"; exit 1 }
Write-Host "ALL PATH-SHADOW CHECKS PASSED"
