# Probe whether a GitHub-hosted Windows runner can actually run WSL.
#
# Emits `usable=true|false` on $GITHUB_OUTPUT. Every wsl.exe call is bounded so a
# prompt-for-input on a headless runner cannot hang the job (wsl --install is
# known to behave differently when stdin is not a console, see
# microsoft/WSL discussion #12426).
$ErrorActionPreference = 'Continue'

function Invoke-Bounded {
    param([string[]]$WslArgs, [int]$TimeoutSec = 600)
    $out = New-TemporaryFile
    $err = New-TemporaryFile
    try {
        $p = Start-Process -FilePath 'wsl.exe' -ArgumentList $WslArgs -NoNewWindow -PassThru `
            -RedirectStandardOutput $out.FullName -RedirectStandardError $err.FullName
        if (-not $p.WaitForExit($TimeoutSec * 1000)) {
            try { $p.Kill() } catch {}
            return @{ Code = 124; Text = "<timed out after ${TimeoutSec}s>" }
        }
        # wsl.exe writes UTF-16LE on most surfaces.
        $text = (Get-Content -Raw -LiteralPath $out.FullName -Encoding Unicode) + "`n" +
                (Get-Content -Raw -LiteralPath $err.FullName -Encoding Unicode)
        $text = ($text -replace "`0", '').Trim()
        if ([string]::IsNullOrWhiteSpace($text)) {
            $text = ((Get-Content -Raw -LiteralPath $out.FullName) + "`n" +
                     (Get-Content -Raw -LiteralPath $err.FullName)).Trim()
        }
        return @{ Code = $p.ExitCode; Text = $text }
    } finally {
        Remove-Item $out.FullName, $err.FullName -ErrorAction SilentlyContinue
    }
}

function Show {
    param([string]$Label, [string[]]$WslArgs, [int]$TimeoutSec = 600)
    Write-Host "--- wsl $($WslArgs -join ' ') ---"
    $r = Invoke-Bounded -WslArgs $WslArgs -TimeoutSec $TimeoutSec
    Write-Host $r.Text
    Write-Host "exit=$($r.Code)"
    return $r
}

$usable = 'false'
$verdict = ''
$wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue

if (-not $wsl) {
    $verdict = "wsl.exe is not on PATH on this image"
    Write-Host "WSL VERDICT: $verdict"
} else {
    Write-Host "wsl.exe: $($wsl.Source)"
    $optional = Get-WindowsOptionalFeature -Online -FeatureName 'Microsoft-Windows-Subsystem-Linux' -ErrorAction SilentlyContinue
    if ($optional) { Write-Host "Microsoft-Windows-Subsystem-Linux feature state: $($optional.State)" }
    else { Write-Host "Microsoft-Windows-Subsystem-Linux feature: <not queryable>" }
    $vmp = Get-WindowsOptionalFeature -Online -FeatureName 'VirtualMachinePlatform' -ErrorAction SilentlyContinue
    if ($vmp) { Write-Host "VirtualMachinePlatform feature state: $($vmp.State)" }

    Show -Label 'version' -WslArgs @('--version') -TimeoutSec 120 | Out-Null
    Show -Label 'status'  -WslArgs @('--status')  -TimeoutSec 120 | Out-Null
    Show -Label 'list'    -WslArgs @('-l', '-v')  -TimeoutSec 120 | Out-Null

    # Bring up the optional component with no distro first: on windows-2025 the
    # feature is often absent even though wsl.exe exists.
    Show -Label 'install-nodistro' -WslArgs @('--install', '--no-distribution') -TimeoutSec 900 | Out-Null
    Show -Label 'install-ubuntu'   -WslArgs @('--install', '-d', 'Ubuntu-24.04', '--no-launch') -TimeoutSec 1200 | Out-Null

    $smoke = Show -Label 'smoke' -WslArgs @('-d', 'Ubuntu-24.04', '--', 'uname', '-a') -TimeoutSec 300
    if ($smoke.Code -eq 0 -and $smoke.Text -match 'Linux') {
        $usable = 'true'
        $verdict = "usable -- a real Ubuntu-24.04 distro answered uname"
    } else {
        $verdict = "NOT usable -- wsl.exe exists but no distro could be brought up (likely needs a reboot, which a hosted runner cannot do)"
    }
    Write-Host "WSL VERDICT: $verdict"
}

if ($env:GITHUB_OUTPUT) { "usable=$usable" | Out-File -FilePath $env:GITHUB_OUTPUT -Append }
if ($env:GITHUB_STEP_SUMMARY) {
    "### WSL probe on $env:RUNNER_OS / $env:RUNNER_ARCH`n`nusable: **$usable**`n`n$verdict`n" |
        Out-File -FilePath $env:GITHUB_STEP_SUMMARY -Append
}
exit 0
