# helpers_fixed.ps1 — helpers.ps1 with codex P2 fixes layered on top.
# Functions of the same name as helpers.ps1 are OVERRIDDEN by re-definition
# (PowerShell binds to the latest definition). New functions are additive.

# ---- Get-CMakeVersion  (NEW — codex P2 #1) ---------------------------------
function Get-CMakeVersion {
    try {
        $cmd = Get-Command cmake -ErrorAction SilentlyContinue
        if (-not $cmd) { return $null }
        $line = (& cmake --version 2>&1 | Select-Object -First 1)
        if ($LASTEXITCODE -ne 0 -or -not $line) { return $null }
        if ($line -match 'cmake version\s+([0-9]+)\.([0-9]+)(?:\.([0-9]+))?') {
            $patch = if ($Matches[3]) { [int]$Matches[3] } else { 0 }
            return [version]::new([int]$Matches[1], [int]$Matches[2], $patch)
        }
        return $null
    } catch { return $null }
}

# ---- Find-VsBuildTools  (REPLACED — codex P2 #2 adds MsbuildToolsetVersion)
function Find-VsBuildTools {
    $map = @{
        '2022' = @{ N = '17'; Y = '2022' }
        '2019' = @{ N = '16'; Y = '2019' }
        '2017' = @{ N = '15'; Y = '2017' }
        '18'   = @{ N = '18'; Y = '2026' }
        '17'   = @{ N = '17'; Y = '2022' }
        '16'   = @{ N = '16'; Y = '2019' }
        '15'   = @{ N = '15'; Y = '2017' }
    }

    $vsw = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsw) {
        $info = & $vsw -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property catalog_productLineVersion 2>$null
        $path = & $vsw -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($info -and $path) {
            $key = (@($info)[0]).Trim()
            $entry = $map[$key]
            if ($entry) {
                return @{
                    Generator             = "Visual Studio $($entry.N) $($entry.Y)"
                    InstallPath           = (@($path)[0]).Trim()
                    Source                = 'vswhere'
                    MsbuildToolsetVersion = "v$($entry.N)0"
                }
            }
        }
    }

    $roots = @($env:ProgramFiles, ${env:ProgramFiles(x86)})
    $editions = @('BuildTools', 'Community', 'Professional', 'Enterprise')
    $dirEntries = @(
        @{ Dir = '18';   N = '18'; Y = '2026' }
        @{ Dir = '2022'; N = '17'; Y = '2022' }
        @{ Dir = '2019'; N = '16'; Y = '2019' }
        @{ Dir = '2017'; N = '15'; Y = '2017' }
    )
    foreach ($entry in $dirEntries) {
        foreach ($r in $roots) {
            foreach ($ed in $editions) {
                $candidate = Join-Path $r "Microsoft Visual Studio\$($entry.Dir)\$ed"
                if (Test-Path $candidate) {
                    $vcDir = Join-Path $candidate "VC\Tools\MSVC"
                    if (Test-Path $vcDir) {
                        $cl = Get-ChildItem -Path $vcDir -Filter "cl.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
                        if ($cl) {
                            return @{
                                Generator             = "Visual Studio $($entry.N) $($entry.Y)"
                                InstallPath           = $candidate
                                Source                = "filesystem ($ed)"
                                ClExe                 = $cl.FullName
                                MsbuildToolsetVersion = "v$($entry.N)0"
                            }
                        }
                    }
                }
            }
        }
    }
    return $null
}

# ---- Find-VsBuildTools2022  (NEW — codex P2 #1 fallback when VS 2026 + old CMake)
function Find-VsBuildTools2022 {
    $vsw = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsw) {
        $path = & $vsw -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($path) {
            return @{
                Generator             = 'Visual Studio 17 2022'
                InstallPath           = (@($path)[0]).Trim()
                Source                = 'vswhere (2022 fallback)'
                MsbuildToolsetVersion = 'v170'
            }
        }
    }
    # Filesystem fallback: look for VS 2022 dir explicitly.
    $roots = @($env:ProgramFiles, ${env:ProgramFiles(x86)})
    $editions = @('BuildTools', 'Community', 'Professional', 'Enterprise')
    foreach ($r in $roots) {
        foreach ($ed in $editions) {
            $candidate = Join-Path $r "Microsoft Visual Studio\2022\$ed"
            if (Test-Path $candidate) {
                $vcDir = Join-Path $candidate "VC\Tools\MSVC"
                if (Test-Path $vcDir) {
                    $cl = Get-ChildItem -Path $vcDir -Filter "cl.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
                    if ($cl) {
                        return @{
                            Generator             = 'Visual Studio 17 2022'
                            InstallPath           = $candidate
                            Source                = "filesystem (2022 fallback, $ed)"
                            ClExe                 = $cl.FullName
                            MsbuildToolsetVersion = 'v170'
                        }
                    }
                }
            }
        }
    }
    return $null
}

# ---- Find-VsBuildToolsGated  (NEW — codex P2 #1 entry point) ----------------
# Wraps Find-VsBuildTools and downgrades to VS 2022 when the selected generator
# requires a CMake newer than the installed one.
function Find-VsBuildToolsGated {
    $r = Find-VsBuildTools
    if (-not $r) { return $null }
    if ($r.MsbuildToolsetVersion -eq 'v180') {
        $cmakeVersion = Get-CMakeVersion
        if ($null -eq $cmakeVersion -or $cmakeVersion -lt [version]'4.2.0') {
            # Try VS 2022 fallback.
            $fallback = Find-VsBuildTools2022
            if ($fallback) {
                return $fallback
            }
            return $null   # caller decides whether to upgrade CMake or hard-exit
        }
    }
    return $r
}

# ---- Invoke-NvidiaDetection  (REPLACED — codex P2 #3 broadens DriverStore glob)
function Invoke-NvidiaDetection {
    $script:Messages.Clear()
    $HasNvidiaSmi = $false
    $NvidiaSmiExe = $null

    $nvSmiCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvSmiCmd -and (Test-NvidiaSmiExe $nvSmiCmd.Source)) {
        $HasNvidiaSmi = $true
        $NvidiaSmiExe = $nvSmiCmd.Source
    }

    if (-not $HasNvidiaSmi) {
        $nvSmiPaths = [System.Collections.Generic.List[string]]@(
            "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            "$env:SystemRoot\System32\nvidia-smi.exe"
        )
        # Broaden the DCH DriverStore glob to match all nv*.inf_<arch>_* folders
        # (consumer DCH = nv_dispi.inf_*, OEM/notebook variants = nvlt*.inf_*,
        #  vGPU guest = nvgridsw*.inf_*, etc). Enumerate every match (sorted
        #  newest-first) rather than only the first hit.
        try {
            $osArch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
            $arch = if ($osArch -eq 'ARM64') { 'arm64' } else { 'amd64' }
            $driverStoreSmis = @(
                Get-ChildItem -Path "$env:SystemRoot\System32\DriverStore\FileRepository\nv*.inf_${arch}_*\nvidia-smi.exe" -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending |
                    Select-Object -ExpandProperty FullName
            )
            foreach ($p in $driverStoreSmis) { $nvSmiPaths.Add($p) }
        } catch {}
        foreach ($p in $nvSmiPaths) {
            if ((Test-Path $p) -and (Test-NvidiaSmiExe $p)) {
                $HasNvidiaSmi = $true
                $NvidiaSmiExe = $p
                substep "Found nvidia-smi at $(Split-Path $p -Parent)"
                break
            }
        }
    }

    if (-not $HasNvidiaSmi) {
        try {
            $nvidiaGpu = Get-CimInstance -ClassName Win32_VideoController -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match 'NVIDIA' -or $_.Caption -match 'NVIDIA' } |
                Select-Object -First 1
            if ($nvidiaGpu) {
                substep "NVIDIA GPU detected via WMI: $($nvidiaGpu.Name)" "Yellow"
                substep "nvidia-smi not found -- reinstall NVIDIA drivers to enable GPU support." "Yellow"
                substep "Continuing in CPU-only / GGUF mode until drivers are fixed." "Yellow"
            }
        } catch {}
    }

    return [PSCustomObject]@{
        HasNvidiaSmi = $HasNvidiaSmi
        NvidiaSmiExe = $NvidiaSmiExe
        Messages     = @($script:Messages)
    }
}
