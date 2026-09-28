$WshShell = New-Object -ComObject WScript.Shell
$targetExe = (Get-Command 'cmd.exe' -ErrorAction SilentlyContinue).Source
if (-not $targetExe) { exit 1 }
# Best-effort: fetch the Unsloth icon to a stable Windows path (shared with a
# native install if one exists) so the WSL shortcut shows the proper icon.
$iconDir = Join-Path $env:LOCALAPPDATA 'Unsloth Studio'
$iconPath = Join-Path $iconDir 'unsloth.ico'
$preIconHash = $null
if (Test-Path -LiteralPath $iconPath) {
    try { $preIconHash = (Get-FileHash -LiteralPath $iconPath -Algorithm SHA256).Hash } catch {}
}
$packagedIcon = 'C:\proof\unsloth.ico'
if (-not (Test-Path -LiteralPath $iconPath) -and $packagedIcon -and (Test-Path -LiteralPath $packagedIcon)) {
    try {
        New-Item -ItemType Directory -Force -Path $iconDir | Out-Null
        Copy-Item -LiteralPath $packagedIcon -Destination $iconPath -Force -ErrorAction Stop
    } catch {}
}
if (-not (Test-Path -LiteralPath $iconPath)) {
    try {
        New-Item -ItemType Directory -Force -Path $iconDir | Out-Null
        Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/unslothai/unsloth/main/studio/frontend/public/unsloth.ico' -OutFile $iconPath -UseBasicParsing -ErrorAction Stop
    } catch {}
}
$hasIcon = $false
if (Test-Path -LiteralPath $iconPath) {
    try { $b = [System.IO.File]::ReadAllBytes($iconPath); if ($b.Length -ge 4 -and $b[0] -eq 0 -and $b[1] -eq 0 -and $b[2] -eq 1 -and $b[3] -eq 0) { $hasIcon = $true } } catch {}
}
$locations = @(
    [Environment]::GetFolderPath('Desktop'),
    (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs')
)
$created = @()
$firstShortcut = $false
foreach ($dir in $locations) {
    if (-not $dir -or -not (Test-Path $dir)) { continue }
    $linkPath = Join-Path $dir 'Unsloth Studio (WSL - Ubuntu).lnk'
    if (-not (Test-Path -LiteralPath $linkPath)) { $firstShortcut = $true }
    $shortcut = $WshShell.CreateShortcut($linkPath)
    $shortcut.TargetPath = $targetExe
    $shortcut.Arguments = '/k echo Unsloth Studio (WSL - Ubuntu)'
    $shortcut.Description = 'Launch Unsloth Studio (WSL)'
    if ($hasIcon) { $shortcut.IconLocation = "$iconPath,0" }
    $shortcut.Save()
    $created += $linkPath
}
$iconChanged = $false
if ($hasIcon) {
    if (-not $preIconHash) {
        $iconChanged = $true
    } else {
        try {
            $postIconHash = (Get-FileHash -LiteralPath $iconPath -Algorithm SHA256).Hash
            $iconChanged = ($postIconHash -ne $preIconHash)
        } catch { $iconChanged = $true }
    }
} elseif ($preIconHash) {
    $iconChanged = $true
}
# Per-item SHCNE_UPDATEITEM so a rewritten same-name .lnk re-reads its icon; the global broadcast
# alone does not. Called through a Windows Python's ctypes, as install.ps1 does, so this script
# defines no native types. Without one the shortcut still works, and the heavier refresh below
# still runs on a first install or an icon change.
if ($created.Count -gt 0) {
    $isAdmin = $true
    try {
        $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    } catch {}
    # Never launch a user-writable interpreter from an elevated shell.
    $pyCandidates = @()
    if (-not $isAdmin) {
        $pyCandidates += Join-Path $env:USERPROFILE '.unsloth\studio\unsloth_studio\Scripts\python.exe'
        foreach ($name in @('python3', 'python')) {
            try {
                foreach ($cmd in @(Get-Command $name -All -CommandType Application -ErrorAction SilentlyContinue)) {
                    if ($cmd -and $cmd.Source) { $pyCandidates += $cmd.Source }
                }
            } catch {}
        }
    }
    # SHCNF_FLUSH (0x1000): the child exits at once and a queued notification would be lost.
    $refreshCode = "import ctypes,os;from ctypes import wintypes as w;f=ctypes.WinDLL('shell32').SHChangeNotify;f.restype=None;f.argtypes=[w.LONG,w.UINT,w.LPCWSTR,w.LPCWSTR];[f(0x2000,0x1005,p,None) for p in os.environ['UNSLOTH_SHORTCUT_PATHS'].split('|') if p];f(0x8000000,0x1000,None,None);print('ok')"
    foreach ($py in @()) {
        # The WindowsApps alias opens the Store instead of running anything.
        if (-not $py -or $py -like '*\Microsoft\WindowsApps\*') { continue }
        if (-not (Test-Path -LiteralPath $py -PathType Leaf)) { continue }
        $proc = $null
        try {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $py
            # -I -S: no user site, no PYTHON* variables, no sitecustomize. -B: write no .pyc.
            $psi.Arguments = '-I -S -B -c "' + $refreshCode + '"'
            # '|' cannot appear in a Windows path, so it separates them safely.
            $psi.EnvironmentVariables['UNSLOTH_SHORTCUT_PATHS'] = ($created -join '|')
            $psi.WorkingDirectory = Split-Path -Parent $py
            $psi.UseShellExecute = $false
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.CreateNoWindow = $true
            $proc = [System.Diagnostics.Process]::Start($psi)
            $out = $proc.StandardOutput.ReadToEndAsync()
            $null = $proc.StandardError.ReadToEndAsync()
            if (-not $proc.WaitForExit(10000)) {
                try { $proc.Kill() } catch {}
                continue
            }
            if ($proc.ExitCode -eq 0 -and $out.Wait(2000) -and "$($out.Result)".Trim() -eq 'ok') { break }
        } catch {
        } finally {
            if ($proc) { try { $proc.Dispose() } catch {} }
        }
    }
}
# Heavier on-disk icon-cache clear + StartMenuExperienceHost tile rebuild
# (preserve start2.bin) only on first install or a real icon change, so a no-op
# WSL reinstall does not purge caches and kill a shell process for nothing.
# See tests/studio/test_installer_av_shapes.py (AV_SHAPES_RECORD)
if ($created.Count -gt 0 -and ($firstShortcut -or $iconChanged)) {
    try { & "$env:SystemRoot\System32\ie4uinit.exe" -ClearIconCache } catch {}
    try { & "$env:SystemRoot\System32\ie4uinit.exe" -show } catch {}
    try {
        $smeh = Join-Path $env:LOCALAPPDATA 'Packages\Microsoft.Windows.StartMenuExperienceHost_cw5n1h2txyewy\TempState'
        if (Test-Path -LiteralPath $smeh) {
            Get-ChildItem -LiteralPath $smeh -Filter 'TileCache_*' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath (Join-Path $smeh 'StartUnifiedTileModelCache.dat') -Force -ErrorAction SilentlyContinue
            Stop-Process -Name StartMenuExperienceHost -Force -ErrorAction SilentlyContinue
        }
    } catch {}
}
