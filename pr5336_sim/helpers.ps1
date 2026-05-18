# helpers.ps1 — functions extracted verbatim from PR-head studio/setup.ps1 + install.ps1.
# Used by the simulation harness to unit-test PR #5336 behaviour on Linux.
# Source lines noted next to each block. NO behavioural edits here.

# ---- substep / step / Write-Host capture --------------------------------------
# The real scripts have substep/step. For the harness we capture them to a global
# $script:Messages list so tests can assert which messages fired.
$script:Messages = New-Object 'System.Collections.Generic.List[string]'

function step    { param($k, $v, $c = 'Green')  $script:Messages.Add("STEP[$k]=$v") }
function substep { param($t, $c = 'White')      $script:Messages.Add("SUBSTEP=$t") }

# ---- Test-NvidiaSmiExe  (install.ps1:1189-1192, studio/setup.ps1:670-673) ----
function Test-NvidiaSmiExe {
    param([string]$Path)
    try { $null = & $Path 2>&1; return ($LASTEXITCODE -eq 0) } catch { return $false }
}

# ---- Find-VsBuildTools  (studio/setup.ps1:374-433) --------------------------
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
                return @{ Generator = "Visual Studio $($entry.N) $($entry.Y)"; InstallPath = (@($path)[0]).Trim(); Source = 'vswhere' }
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
                            return @{ Generator = "Visual Studio $($entry.N) $($entry.Y)"; InstallPath = $candidate; Source = "filesystem ($ed)"; ClExe = $cl.FullName }
                        }
                    }
                }
            }
        }
    }

    return $null
}

# ---- Test-AnyVsInstalled  (studio/setup.ps1:437-454) ------------------------
function Test-AnyVsInstalled {
    $vsw = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsw) {
        $path = & $vsw -latest -property installationPath 2>$null
        if ($path) { return $true }
    }
    $roots = @($env:ProgramFiles, ${env:ProgramFiles(x86)})
    $editions = @('BuildTools', 'Community', 'Professional', 'Enterprise')
    $dirs = @('18', '2022', '2019', '2017')
    foreach ($d in $dirs) {
        foreach ($r in $roots) {
            foreach ($ed in $editions) {
                if (Test-Path (Join-Path $r "Microsoft Visual Studio\$d\$ed")) { return $true }
            }
        }
    }
    return $false
}

# ---- Invoke-NvidiaDetection  (synthesised from install.ps1:1193-1230  +
#       studio/setup.ps1:674-718). Returns a structured result so tests can
#       assert HasNvidiaSmi / NvidiaSmiExe / WMI messages without re-running
#       the surrounding script. Mock layer for Get-CimInstance is provided via
#       a script-scope Set-Item function alias before calling. ------------
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
        try {
            $arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'arm64' } else { 'amd64' }
            $driverStoreSmi = Get-Item -Path "$env:SystemRoot\System32\DriverStore\FileRepository\nv_dispi.inf_${arch}_*\nvidia-smi.exe" -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty FullName -First 1
            if ($driverStoreSmi) { $nvSmiPaths.Add($driverStoreSmi) }
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

# ---- Invoke-VsSelectionGate  (synthesises studio/setup.ps1:835-870 hard-exit
#       branch into a function that returns the decision instead of calling
#       exit, so tests can assert each branch). --------------------------------
function Invoke-VsSelectionGate {
    $script:Messages.Clear()
    $vsResult = Find-VsBuildTools

    if (-not $vsResult) {
        if ((Test-AnyVsInstalled) -and -not $env:UNSLOTH_FORCE_BUILD_TOOLS) {
            return [PSCustomObject]@{
                Decision = 'AbortMissingWorkload'
                VsResult = $null
            }
        }
        return [PSCustomObject]@{
            Decision = 'InstallBuildTools'
            VsResult = $null
        }
    }

    return [PSCustomObject]@{
        Decision = 'UseExistingVS'
        VsResult = $vsResult
    }
}
