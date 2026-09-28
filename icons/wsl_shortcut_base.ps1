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
# Per-item refresh always (cheap, non-disruptive) so the rewritten .lnk renders
# immediately instead of a stale/blank (generic) icon. The reliable fix (no
# explorer restart) is a PER-ITEM SHChangeNotify(SHCNE_UPDATEITEM, SHCNF_PATHW,
# <lnk>) -- the global SHCNE_ASSOCCHANGED alone does not recover a stale item.
#
# Emitted, not compiled. Add-Type -MemberDefinition writes C# to %TEMP% and runs
# csc.exe on Windows PowerShell 5.1, and security software blocks the DLL that comes
# out. install.ps1 carries the same reflection-emit form for the same reason; this
# copy was missed when that one changed. Reflection emit builds the identical stub in
# memory: no compiler process, no source on disk, no DLL.
# Which product blocked what: tests/studio/test_installer_av_shapes.py (AV_SHAPES_RECORD)
try {
    $refreshType = 'UnslothShellIconRefresh' -as [type]
    if (-not $refreshType) {
        $asmName = New-Object System.Reflection.AssemblyName 'UnslothShellIconRefreshAsm'
        # Both spellings, matching New-StudioDynamicAssembly in install.ps1. The static
        # AssemblyBuilder::DefineDynamicAssembly is documented for .NET Framework 4.5 through
        # 4.8.1, so the 5.1 host this script is launched under should take the first branch; it
        # is tried rather than assumed because the outer catch here is empty, so guessing wrong
        # costs the icon refresh with nothing printed. AppDomain.CurrentDomain is the .NET
        # Framework spelling and is absent on .NET Core, so it is the fallback and not the lead.
        $access = [System.Reflection.Emit.AssemblyBuilderAccess]::Run
        try {
            $asm = [System.Reflection.Emit.AssemblyBuilder]::DefineDynamicAssembly($asmName, $access)
        } catch [System.Management.Automation.MethodException] {
            $asm = [AppDomain]::CurrentDomain.DefineDynamicAssembly($asmName, $access)
        } catch [System.Management.Automation.RuntimeException] {
            # Some hosts surface a missing static as RuntimeException rather than
            # MethodException. Both mean "no such method here", and a real emit failure throws
            # from the AppDomain call too, so a genuine refusal still reaches the outer catch.
            $asm = [AppDomain]::CurrentDomain.DefineDynamicAssembly($asmName, $access)
        }
        $module = $asm.DefineDynamicModule('UnslothShellIconRefreshMod')
        $typeBuilder = $module.DefineType('UnslothShellIconRefresh',
            'Public, Class, AutoClass, AnsiClass, BeforeFieldInit')
        $method = $typeBuilder.DefinePInvokeMethod(
            'SHChangeNotify', 'shell32.dll', 'SHChangeNotify',
            'Public, Static, PinvokeImpl',
            [System.Reflection.CallingConventions]::Standard,
            [System.Void],
            @([int], [uint32], [string], [IntPtr]),
            [System.Runtime.InteropServices.CallingConvention]::Winapi,
            [System.Runtime.InteropServices.CharSet]::Unicode)
        $method.SetImplementationFlags(
            $method.GetMethodImplementationFlags() -bor [System.Reflection.MethodImplAttributes]::PreserveSig)
        $refreshType = $typeBuilder.CreateType()
    }
    foreach ($p in $created) { try { $refreshType::SHChangeNotify(0x00002000, 0x0005, $p, [System.IntPtr]::Zero) } catch {} }
    $refreshType::SHChangeNotify(0x08000000, 0, $null, [System.IntPtr]::Zero)
} catch {}
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
