# Restore Microsoft Defender on a GitHub-hosted Windows runner.
#
# The runner image deliberately guts Defender at build time -- see
# actions/runner-images images/windows/scripts/build/Configure-WindowsDefender.ps1,
# whose header comment is literally "Desc: Disables Windows Defender". It sets
# DisableRealtimeMonitoring, turns MAPS (cloud) reporting off, disables
# block-at-first-sight, and -- the part that silently voids every scan -- adds
# ExclusionPath = @("D:\", "C:\").
#
# Two of those matter more than the rest for this experiment:
#   * the drive-root exclusions, which make every scan a no-op, and
#   * MAPSReporting = 0, because Trojan:Win32/Wacatac.B!ml is a *cloud* machine
#     learning verdict. With the cloud off the classifier that produces "!ml"
#     names is never consulted and the scan cannot reproduce the detection.
#
# Nothing is uninstalled by the image, so no feature install and no reboot are
# needed. Tamper Protection cannot block us: it pins settings toward the secure
# state, and every change here moves that way.

$ErrorActionPreference = 'Continue'

Write-Host '== BEFORE =='
Get-MpComputerStatus | Format-List AMServiceEnabled, AntivirusEnabled, RealTimeProtectionEnabled,
    BehaviorMonitorEnabled, IoavProtectionEnabled, IsTamperProtected, AMRunningMode,
    AMEngineVersion, AntivirusSignatureVersion, AntivirusSignatureLastUpdated
Get-MpPreference | Format-List ExclusionPath, DisableRealtimeMonitoring, DisableIOAVProtection,
    DisableScriptScanning, DisableBehaviorMonitoring, DisableArchiveScanning,
    MAPSReporting, SubmitSamplesConsent, CloudBlockLevel, DisableBlockAtFirstSeen

# Passive mode wins over every preference below, so clear it first. The image
# sets ForceDefenderPassiveMode only when the ATP policy key already exists.
$atp = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows Advanced Threat Protection'
if (Test-Path $atp) { Remove-ItemProperty -Path $atp -Name 'ForceDefenderPassiveMode' -ErrorAction SilentlyContinue }
$fdp = 'HKLM:\SOFTWARE\Microsoft\Windows Advanced Threat Protection'
if (Test-Path $fdp) { Set-ItemProperty -Path $fdp -Name 'ForceDefenderPassiveMode' -Value 0 -Type DWord -ErrorAction SilentlyContinue }

Set-Service  -Name WinDefend -StartupType Automatic -ErrorAction SilentlyContinue
Start-Service -Name WinDefend -ErrorAction SilentlyContinue

# Drop the image's blanket exclusions. Without this every scan below reports
# clean no matter what the file contains.
$pref = Get-MpPreference
foreach ($p in @($pref.ExclusionPath))      { if ($p) { Write-Host "removing exclusion path: $p"; Remove-MpPreference -ExclusionPath $p -ErrorAction SilentlyContinue } }
foreach ($e in @($pref.ExclusionExtension)) { if ($e) { Remove-MpPreference -ExclusionExtension $e -ErrorAction SilentlyContinue } }
foreach ($x in @($pref.ExclusionProcess))   { if ($x) { Remove-MpPreference -ExclusionProcess   $x -ErrorAction SilentlyContinue } }

Set-MpPreference `
    -DisableRealtimeMonitoring        $false `
    -DisableIOAVProtection            $false `
    -DisableScriptScanning            $false `
    -DisableBehaviorMonitoring        $false `
    -DisableArchiveScanning           $false `
    -DisableIntrusionPreventionSystem $false `
    -DisableScanningNetworkFiles      $false `
    -DisableAutoExclusions            $false `
    -MAPSReporting                    Advanced `
    -SubmitSamplesConsent             SendAllSamples `
    -CloudBlockLevel                  High `
    -CloudExtendedTimeout             50 `
    -DisableBlockAtFirstSeen          $false `
    -PUAProtection                    Enabled `
    -ScanAvgCPULoadFactor             50 `
    -ErrorAction Continue

# Fresh definitions. The image sets SignatureDisableUpdateOnStartupWithoutEngine,
# so a runner can otherwise be carrying signatures as old as the image build.
$mp = "$env:ProgramFiles\Windows Defender\MpCmdRun.exe"
$plat = Get-ChildItem 'C:\ProgramData\Microsoft\Windows Defender\Platform' -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
if ($plat -and (Test-Path "$($plat.FullName)\MpCmdRun.exe")) { $mp = "$($plat.FullName)\MpCmdRun.exe" }
Write-Host "MpCmdRun: $mp"
"MPCMDRUN=$mp" | Out-File $env:GITHUB_ENV -Append -Encoding utf8

& $mp -RemoveDefinitions -DynamicSignatures 2>&1 | Write-Host
Update-MpSignature -UpdateSource MicrosoftUpdateServer -ErrorAction Continue
& $mp -SignatureUpdate 2>&1 | Write-Host

# Prove the runner can actually reach MAPS. Without a live cloud connection an
# "!ml" verdict is unreachable and a clean result below would mean nothing.
Write-Host '== ValidateMapsConnection =='
& $mp -ValidateMapsConnection 2>&1 | Write-Host
Write-Host "ValidateMapsConnection exit: $LASTEXITCODE"

# Verify rather than assume: a Set-MpPreference that returns without error can
# still have been dropped on the floor.
$s = Get-MpComputerStatus
$p = Get-MpPreference
Write-Host '== AFTER =='
$s | Format-List AMServiceEnabled, RealTimeProtectionEnabled, BehaviorMonitorEnabled,
    IoavProtectionEnabled, IsTamperProtected, AMRunningMode, AMEngineVersion,
    AntivirusSignatureVersion, AntivirusSignatureLastUpdated

$problems = @()
if ($p.DisableRealtimeMonitoring)      { $problems += 'DisableRealtimeMonitoring is still true' }
# MAPSReporting comes back as the underlying enum value (2 = Advanced), not the
# string, so compare numerically. Same for SubmitSamplesConsent (3 = SendAllSamples)
# and CloudBlockLevel (2 = High).
if ([int]$p.MAPSReporting -ne 2)       { $problems += "MAPSReporting is $($p.MAPSReporting), expected 2 (Advanced)" }
if ([int]$p.SubmitSamplesConsent -ne 3){ $problems += "SubmitSamplesConsent is $($p.SubmitSamplesConsent), expected 3 (SendAllSamples)" }
if ($p.DisableBlockAtFirstSeen)        { $problems += 'block-at-first-sight is still disabled' }
if ($p.ExclusionPath)                  { $problems += "exclusions remain: $($p.ExclusionPath -join ',')" }
if (-not $s.RealTimeProtectionEnabled) { $problems += "RealTimeProtectionEnabled is false (AMRunningMode=$($s.AMRunningMode))" }
if ($s.AMRunningMode -ne 'Normal')     { $problems += "AMRunningMode is $($s.AMRunningMode), not Normal" }

$env_state = if ($problems) { 'unsupported' } else { 'ok' }
Write-Host "DEFENDER_ENV=$env_state"
"DEFENDER_ENV=$env_state" | Out-File $env:GITHUB_ENV -Append -Encoding utf8

if ($problems) {
    Write-Host "::warning::Defender not fully restored: $($problems -join '; '). Verdicts below are not authoritative."
} else {
    Write-Host 'Defender fully restored with cloud-delivered protection.'
}

[pscustomobject]@{
    environment                   = $env_state
    problems                      = $problems
    RealTimeProtectionEnabled     = $s.RealTimeProtectionEnabled
    BehaviorMonitorEnabled        = $s.BehaviorMonitorEnabled
    IoavProtectionEnabled         = $s.IoavProtectionEnabled
    IsTamperProtected             = $s.IsTamperProtected
    AMRunningMode                 = [string]$s.AMRunningMode
    AMEngineVersion               = [string]$s.AMEngineVersion
    AntivirusSignatureVersion     = [string]$s.AntivirusSignatureVersion
    AntivirusSignatureLastUpdated = [string]$s.AntivirusSignatureLastUpdated
    MAPSReporting                 = [string]$p.MAPSReporting
    SubmitSamplesConsent          = [string]$p.SubmitSamplesConsent
    CloudBlockLevel               = [string]$p.CloudBlockLevel
} | ConvertTo-Json -Depth 3 | Out-File "$env:REPORTDIR\00-defender-env.json" -Encoding utf8
