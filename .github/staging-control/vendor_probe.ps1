# Vendor simulation harness for install.ps1's Get-TorchIndexUrl.
#
# install.ps1's Get-TorchIndexUrl (install.ps1:2074) is reachable from no test in
# the repo today: tests/python/test_cross_platform_parity.py only diffs its
# cu-suffix if-chain against install.sh's ladder statically. This harness
# executes it, using the AST-extraction idiom the repo's own pwsh tests use
# (tests/studio/test_torch_flavor.ps1:14-25) and stubbing the process-spawning
# probe the way tests/studio/test_resolve_cuda_toolkit.ps1:64-88 stubs Find-Nvcc.
#
# Windows GPU vendors:
#   NVIDIA -> Get-TorchIndexUrl walks the CUDA ladder.
#   AMD    -> Get-TorchIndexUrl returns .../cpu; the arch-aware repo.amd.com
#             reroute happens afterwards in a top-level block (install.ps1:2196).
#             That block's gfx->family table is asserted here too.
#   CPU    -> .../cpu.
param([string]$InstallPs1 = 'install.ps1')

$ErrorActionPreference = 'Stop'
$script:Pass = 0
$script:Fail = 0

function Assert-Eq {
    param([string]$Label, [string]$Expected, [string]$Actual)
    if ($Actual -eq $Expected) {
        Write-Host "  PASS: $Label -> $Actual"
        $script:Pass++
    } else {
        Write-Host "  FAIL: $Label (expected '$Expected', got '$Actual')"
        $script:Fail++
    }
}

$installPath = (Resolve-Path $InstallPs1).Path
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPath, [ref]$tokens, [ref]$errors)
if ($errors) {
    $errors | ForEach-Object { Write-Host "::error file=$InstallPs1::$($_.ToString())" }
    throw "install.ps1 has parse errors"
}

foreach ($name in @('Trim-IndexPathSlashes', 'Get-TorchIndexUrl')) {
    $fn = $ast.FindAll({
        param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true)
    if ($fn.Count -ne 1) { throw "expected exactly one $name in install.ps1, found $($fn.Count)" }
    Invoke-Expression $fn[0].Extent.Text
}

# Stubs for the two things Get-TorchIndexUrl reaches out of its scope for.
function substep { param($a, $b) }
function Invoke-NvidiaSmiBounded { param($Exe, $SmiArgs, $TimeoutSec) return $script:FakeSmiOutput }

function Invoke-Route {
    # $SmiOutput is deliberately untyped: a [string] parameter coerces $null to
    # '', which would make the "no nvidia-smi" leg indistinguishable from an
    # nvidia-smi that printed nothing.
    param($SmiOutput, [hashtable]$EnvVars = @{})
    $saved = @{}
    foreach ($k in @('UNSLOTH_TORCH_INDEX_URL', 'UNSLOTH_TORCH_INDEX_FAMILY', 'UNSLOTH_PYTORCH_MIRROR')) {
        $saved[$k] = [Environment]::GetEnvironmentVariable($k)
        [Environment]::SetEnvironmentVariable($k, $null)
    }
    foreach ($k in $EnvVars.Keys) { [Environment]::SetEnvironmentVariable($k, $EnvVars[$k]) }
    if ($null -eq $SmiOutput) {
        $NvidiaSmiExe = $null
    } else {
        $NvidiaSmiExe = 'C:\Windows\System32\nvidia-smi.exe'   # truthy path; never executed
        $script:FakeSmiOutput = $SmiOutput
    }
    try { return (Get-TorchIndexUrl) }
    finally { foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) } }
}

$BASE = 'https://download.pytorch.org/whl'
$SMI_128 = "| NVIDIA-SMI 570.86.10   Driver Version: 570.86.10   CUDA Version: 12.8   |"
$SMI_130 = "| NVIDIA-SMI 610.10.00   Driver Version: 610.10.00   CUDA UMD Version: 13.0 |"
$SMI_126 = "| NVIDIA-SMI 560.35.03   Driver Version: 560.35.03   CUDA Version: 12.6   |"
$SMI_118 = "| NVIDIA-SMI 470.00.00   Driver Version: 470.00.00   CUDA Version: 11.8   |"

Write-Host "=== install.ps1 Get-TorchIndexUrl vendor simulation on $([System.Runtime.InteropServices.RuntimeInformation]::OSDescription.Trim()) ==="

Write-Host "--- vendor: NVIDIA ---"
Assert-Eq "driver CUDA 12.8 -> cu128" "$BASE/cu128" (Invoke-Route $SMI_128)
Assert-Eq "driver CUDA UMD 13.0 -> cu130" "$BASE/cu130" (Invoke-Route $SMI_130)
Assert-Eq "driver CUDA 12.6 -> cu126" "$BASE/cu126" (Invoke-Route $SMI_126)
Assert-Eq "driver CUDA 11.8 -> cu118" "$BASE/cu118" (Invoke-Route $SMI_118)
Assert-Eq "unparseable nvidia-smi -> cu126 default" "$BASE/cu126" (Invoke-Route "no version here")

Write-Host "--- vendor: CPU-only / AMD (no nvidia-smi: Get-TorchIndexUrl yields the cpu leaf the AMD reroute keys off) ---"
Assert-Eq "no nvidia-smi -> cpu" "$BASE/cpu" (Invoke-Route $null)

Write-Host "--- explicit vendor pins ---"
Assert-Eq "FAMILY=cu130 with no GPU" "$BASE/cu130" (Invoke-Route $null @{ UNSLOTH_TORCH_INDEX_FAMILY = 'cu130' })
Assert-Eq "FAMILY=gfx120X-all with no GPU" "$BASE/gfx120X-all" (Invoke-Route $null @{ UNSLOTH_TORCH_INDEX_FAMILY = 'gfx120X-all' })
Assert-Eq "FAMILY overrides an NVIDIA probe" "$BASE/cpu" (Invoke-Route $SMI_128 @{ UNSLOTH_TORCH_INDEX_FAMILY = 'cpu' })
Assert-Eq "INDEX_URL verbatim" "https://mirror.example.test/whl/cu999" (Invoke-Route $SMI_128 @{ UNSLOTH_TORCH_INDEX_URL = 'https://mirror.example.test/whl/cu999' })
Assert-Eq "INDEX_URL beats FAMILY" "https://mirror.example.test/whl/rocm7.2" (Invoke-Route $null @{ UNSLOTH_TORCH_INDEX_URL = 'https://mirror.example.test/whl/rocm7.2'; UNSLOTH_TORCH_INDEX_FAMILY = 'cu128' })
Assert-Eq "PYTORCH_MIRROR rebases cpu" "https://mirror.example.test/whl/cpu" (Invoke-Route $null @{ UNSLOTH_PYTORCH_MIRROR = 'https://mirror.example.test/whl/' })
Assert-Eq "PYTORCH_MIRROR rebases cu128" "https://mirror.example.test/whl/cu128" (Invoke-Route $SMI_128 @{ UNSLOTH_PYTORCH_MIRROR = 'https://mirror.example.test/whl' })

Write-Host "--- vendor: AMD, arch-aware repo.amd.com table (install.ps1 archFamilyMap) ---"
# The reroute is a top-level block, not a function, so lift the literal table.
$src = Get-Content -Raw -LiteralPath $installPath
$m = [regex]::Match($src, '\$archFamilyMap\s*=\s*@\{(?<body>[\s\S]*?)\n\s*\}')
if (-not $m.Success) { Write-Host "  FAIL: could not locate `$archFamilyMap in install.ps1"; $script:Fail++ }
else {
    $map = @{}
    foreach ($mm in [regex]::Matches($m.Groups['body'].Value, '"(?<k>gfx[0-9a-z]+)"\s*=\s*"(?<v>[^"]+)"')) {
        $map[$mm.Groups['k'].Value] = $mm.Groups['v'].Value
    }
    Assert-Eq "gfx1100 (RDNA 3) -> gfx110X-all" "gfx110X-all" $map['gfx1100']
    Assert-Eq "gfx1201 (RDNA 4) -> gfx120X-all" "gfx120X-all" $map['gfx1201']
    Assert-Eq "gfx1151 (Strix Halo) -> gfx1151" "gfx1151" $map['gfx1151']
    Assert-Eq "gfx1030 (RDNA 2) -> gfx103X-all" "gfx103X-all" $map['gfx1030']
    Assert-Eq "gfx90a (MI200) -> gfx90a" "gfx90a" $map['gfx90a']
    $base = if ([regex]::IsMatch($src, 'UNSLOTH_ROCM_WINDOWS_MIRROR')) { 'present' } else { 'absent' }
    Assert-Eq "UNSLOTH_ROCM_WINDOWS_MIRROR override exists" "present" $base
}

Write-Host ""
Write-Host "=== $($script:Pass) passed, $($script:Fail) failed ==="
if ($script:Fail -gt 0) { exit 1 }
