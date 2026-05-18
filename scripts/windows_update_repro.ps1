# SPDX-License-Identifier: AGPL-3.0-only
# Reproduce the Windows `unsloth studio update` regression:
#   1. install Studio at the current checkout (creates Start Menu / Desktop .lnk)
#   2. fingerprint launch-studio.ps1 (the launcher script the shortcuts target)
#   3. overwrite the launcher script with a sentinel
#   4. run `unsloth studio update --local`
#   5. assert the sentinel was overwritten
# Today this fails because `setup.ps1` never recreates the launcher. Passes
# once New-StudioShortcuts is wired into the update path.

$ErrorActionPreference = 'Stop'

$StudioHomeDefault = Join-Path $env:USERPROFILE '.unsloth\studio'
$LocalAppData = $env:LOCALAPPDATA
if (-not $LocalAppData) {
    $LocalAppData = Join-Path $env:USERPROFILE 'AppData\Local'
}
$DataDirDefault = Join-Path $LocalAppData 'Unsloth Studio'
$LauncherPs1 = Join-Path $DataDirDefault 'launch-studio.ps1'
$VenvScripts = Join-Path $StudioHomeDefault 'unsloth_studio\Scripts'
$UnslothExe = Join-Path $VenvScripts 'unsloth.exe'

$RepoRoot = if ($env:REPO_ROOT) { $env:REPO_ROOT } else { (Get-Location).Path }
$InstallScript = Join-Path $RepoRoot 'install.ps1'
if (-not (Test-Path -LiteralPath $InstallScript)) {
    Write-Host "FAIL: install.ps1 not found at $InstallScript"
    exit 1
}

Write-Host "[repro] installing Studio @ $RepoRoot (default mode)"
& $InstallScript --local
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: install.ps1 failed (exit $LASTEXITCODE)"
    exit 1
}

if (-not (Test-Path -LiteralPath $LauncherPs1)) {
    Write-Host "FAIL: launcher script missing: $LauncherPs1"
    exit 1
}
if (-not (Test-Path -LiteralPath $UnslothExe)) {
    Write-Host "FAIL: unsloth.exe missing: $UnslothExe"
    exit 1
}

$Sentinel = '@@WINDOWS_UPDATE_REGRESSION_SENTINEL@@'
Write-Host "[repro] writing sentinel into launcher"
"# $Sentinel`nWrite-Host '$Sentinel'`nexit 17" | Out-File -LiteralPath $LauncherPs1 -Encoding UTF8

$BeforeHash = (Get-FileHash -LiteralPath $LauncherPs1 -Algorithm SHA256).Hash
Write-Host "[repro] sentinel sha256 = $BeforeHash"

Write-Host "[repro] running update via venv python -c unsloth_cli.app"
# Windows refuses to delete a running unsloth.exe (WinError 32 from pip).
# Production users should invoke the CLI through the venv python in that
# scenario. The CI test exercises that path so we validate the launcher
# rebuild logic end-to-end without tripping the OS-level lock.
$VenvPython = Join-Path $VenvScripts 'python.exe'
& $VenvPython -c "from unsloth_cli import app; app(['studio', 'update', '--local'])"
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: unsloth studio update exited $LASTEXITCODE"
    exit 1
}

if (-not (Test-Path -LiteralPath $LauncherPs1)) {
    Write-Host "FAIL: launcher gone after update"
    exit 1
}
$AfterHash = (Get-FileHash -LiteralPath $LauncherPs1 -Algorithm SHA256).Hash
Write-Host "[repro] after update sha256 = $AfterHash"

if ($BeforeHash -eq $AfterHash) {
    Write-Host "FAIL: Bug A reproduced -- 'unsloth studio update' did not rewrite the launcher script."
    Get-Content -LiteralPath $LauncherPs1 | Select-String -SimpleMatch $Sentinel
    exit 1
}
if (Select-String -Path $LauncherPs1 -SimpleMatch $Sentinel -Quiet) {
    Write-Host "FAIL: sentinel still present in launcher after update"
    exit 1
}

Write-Host "PASS: 'unsloth studio update' rewrote the launcher script."
