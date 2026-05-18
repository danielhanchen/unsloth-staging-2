# sandbox.ps1 — fixture helpers for the harness. Each test dot-sources this.

$script:SimRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$script:ShimRoot = Join-Path $script:SimRoot 'shims'

function New-Sandbox {
    param([string]$Name)
    $root = Join-Path $script:SimRoot "sandboxes/$Name"
    if (Test-Path $root) { Remove-Item $root -Recurse -Force }
    New-Item -ItemType Directory -Path $root -Force | Out-Null

    # Map Windows-style env vars to subdirectories under $root.
    $env:SystemRoot         = (New-Item -ItemType Directory -Path (Join-Path $root 'SystemRoot') -Force).FullName
    $env:ProgramFiles       = (New-Item -ItemType Directory -Path (Join-Path $root 'ProgramFiles') -Force).FullName
    Set-Item "Env:ProgramFiles(x86)" (New-Item -ItemType Directory -Path (Join-Path $root 'ProgramFilesX86') -Force).FullName
    $env:LOCALAPPDATA       = (New-Item -ItemType Directory -Path (Join-Path $root 'LocalAppData') -Force).FullName
    $env:PROCESSOR_ARCHITECTURE = 'AMD64'

    # Hermetic PATH: a sandbox-private tools dir containing only the basics the
    # shims need (bash, chmod, etc). Deliberately EXCLUDES nvidia-smi / vswhere
    # / cmake so the host system's binaries never leak into the test. Tests
    # opt-in to specific shims via Add-ShimToPath.
    $tools = New-Item -ItemType Directory -Path (Join-Path $root 'tools') -Force
    foreach ($bin in @('bash','sh','chmod','mkdir','rm','cp','ls','grep','sed','awk','cat','tr','dirname','basename','env','printf','readlink','head','tail','sort','wc','find','xargs','tee','which','id','uname','rev','date')) {
        $real = (& /usr/bin/which $bin 2>$null)
        if ($real) {
            $link = Join-Path $tools.FullName $bin
            if (-not (Test-Path $link)) {
                New-Item -ItemType SymbolicLink -Path $link -Value $real.Trim() -ErrorAction SilentlyContinue | Out-Null
            }
        }
    }
    $env:PATH = $tools.FullName

    # Clear UNSLOTH_FORCE_BUILD_TOOLS unless a test sets it
    Remove-Item Env:UNSLOTH_FORCE_BUILD_TOOLS -ErrorAction SilentlyContinue

    return $root
}

function Add-ShimToPath {
    param([string]$ShimName, [string]$ExposeAs)
    # Symlink temp/.../bin/<ExposeAs> -> shims/<ShimName>; prepend bin dir to PATH.
    $shim = Join-Path $script:ShimRoot $ShimName
    if (-not (Test-Path $shim)) { throw "Shim not found: $shim" }
    $bin = New-Item -ItemType Directory -Path (Join-Path $env:SystemRoot '..\bin') -Force
    $link = Join-Path $bin.FullName $ExposeAs
    if (Test-Path $link) { Remove-Item $link -Force }
    # Use New-Item -ItemType SymbolicLink (works on pwsh Linux)
    New-Item -ItemType SymbolicLink -Path $link -Value $shim | Out-Null
    $env:PATH = "$($bin.FullName):$env:PATH"
}

function New-FakeNvidiaSmiAt {
    # Drops a working nvidia-smi shim (or broken/permission) at the requested
    # absolute path. Path may use backslashes; we normalise.
    param([string]$AtPath, [string]$ShimName = 'nvidia-smi-ok')
    $shim = Join-Path $script:ShimRoot $ShimName
    $native = $AtPath -replace '\\', '/'
    $parent = Split-Path -Parent $native
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    if (Test-Path $native) { Remove-Item $native -Force }
    Copy-Item $shim $native -Force
    chmod +x $native | Out-Null
}

function New-FakeVsWhere {
    # Drops a vswhere shim at $env:ProgramFiles(x86)\Microsoft Visual Studio\Installer\vswhere.exe.
    param([string]$ShimName)
    $vsw = (Get-Item env:'ProgramFiles(x86)').Value + '/Microsoft Visual Studio/Installer/vswhere.exe'
    New-FakeNvidiaSmiAt -AtPath $vsw -ShimName $ShimName
}

function New-FakeVsInstall {
    # Builds a fake VS install layout with cl.exe under MSVC.
    param(
        [string]$Root,           # e.g. $env:ProgramFiles
        [string]$DirName,        # '2022' / '18' / etc.
        [string]$Edition = 'BuildTools',
        [switch]$NoCl
    )
    $vsRoot = "$Root/Microsoft Visual Studio/$DirName/$Edition"
    $vcDir = "$vsRoot/VC/Tools/MSVC/14.40.0"
    New-Item -ItemType Directory -Path $vcDir -Force | Out-Null
    if (-not $NoCl) {
        $cl = "$vcDir/cl.exe"
        Set-Content -Path $cl -Value '#!/bin/sh' -NoNewline
    }
    return $vsRoot
}

function Set-Mock-GetCimInstance {
    # Replaces Get-CimInstance in the *caller's* scope so that calls hit our mock.
    # The mock returns a list of fake video controllers, filtered by ClassName.
    param([object[]]$Controllers)
    # Install a script-scope function that overrides the cmdlet within this scope.
    $script:MockControllers = $Controllers
    function global:Get-CimInstance {
        param($ClassName, [Parameter(ValueFromRemainingArguments=$true)]$Rest)
        if ($ClassName -eq 'Win32_VideoController') {
            return $script:MockControllers
        }
        return @()
    }
}

function Clear-Mock-GetCimInstance {
    if (Get-Command Get-CimInstance -CommandType Function -ErrorAction SilentlyContinue) {
        Remove-Item function:Get-CimInstance -ErrorAction SilentlyContinue
    }
}

function Assert-True  { param($Cond, $Msg) if (-not $Cond) { throw "ASSERT: $Msg" } }
function Assert-False { param($Cond, $Msg) if ($Cond)      { throw "ASSERT (expected false): $Msg" } }
function Assert-Eq    { param($A, $B, $Msg) if ($A -ne $B) { throw "ASSERT: $Msg — got '$A', expected '$B'" } }
function Assert-Like  { param($A, $Pat, $Msg) if ($A -notlike $Pat) { throw "ASSERT: $Msg — got '$A', wanted like '$Pat'" } }
