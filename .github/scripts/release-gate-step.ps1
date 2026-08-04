$ErrorActionPreference = 'Continue'

$atp = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows Advanced Threat Protection'
if (Test-Path $atp) { Remove-ItemProperty -Path $atp -Name 'ForceDefenderPassiveMode' -ErrorAction SilentlyContinue }
Start-Service -Name WinDefend -ErrorAction SilentlyContinue

foreach ($x in @((Get-MpPreference).ExclusionPath)) {
  if ($x) { Remove-MpPreference -ExclusionPath $x -ErrorAction SilentlyContinue }
}
Set-MpPreference `
  -DisableRealtimeMonitoring $false -DisableIOAVProtection $false `
  -DisableScriptScanning $false -DisableBehaviorMonitoring $false `
  -DisableArchiveScanning $false -MAPSReporting Advanced `
  -SubmitSamplesConsent SendAllSamples -CloudBlockLevel High `
  -CloudExtendedTimeout 50 -DisableBlockAtFirstSeen $false -ErrorAction Continue

$mp = "$env:ProgramFiles\Windows Defender\MpCmdRun.exe"
$plat = Get-ChildItem 'C:\ProgramData\Microsoft\Windows Defender\Platform' -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
if ($plat -and (Test-Path "$($plat.FullName)\MpCmdRun.exe")) { $mp = "$($plat.FullName)\MpCmdRun.exe" }
Update-MpSignature -UpdateSource MicrosoftUpdateServer -ErrorAction Continue
& $mp -SignatureUpdate 2>&1 | Write-Host

$status = Get-MpComputerStatus
$pref = Get-MpPreference
Write-Host "engine $($status.AMEngineVersion), signatures $($status.AntivirusSignatureVersion) ($($status.AntivirusSignatureLastUpdated))"

# A Set-MpPreference that returns without error can still have been
# dropped. If protection is not genuinely on, a clean scan proves
# nothing, so say so rather than gate the release on a fiction.
$degraded = @()
if (-not $status.RealTimeProtectionEnabled) { $degraded += "RealTimeProtectionEnabled=false (AMRunningMode=$($status.AMRunningMode))" }
if ([int]$pref.MAPSReporting -ne 2)         { $degraded += "MAPSReporting=$($pref.MAPSReporting), expected 2 (Advanced)" }
if ($pref.ExclusionPath)                    { $degraded += "exclusions remain: $($pref.ExclusionPath -join ',')" }

# Positive control. Without it, "clean" and "the scanner is not working"
# are the same output. EICAR is assembled from fragments so this workflow
# file is not itself a sample.
$eicarPath = Join-Path $env:RUNNER_TEMP 'defender-selftest.com'
[System.IO.File]::WriteAllText(
  $eicarPath,
  'X5O!P%@AP[4\PZX54(P^)7CC)7}' + '$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!' + '$H+H*')
$controlPassed = $true
if (Test-Path $eicarPath) {
  $controlOut = (& $mp -Scan -ScanType 3 -File $eicarPath -DisableRemediation 2>&1 | Out-String)
  $controlPassed = [bool](($controlOut -match 'found\s+\d+\s+threat') -or ($controlOut -match 'EICAR'))
  Remove-Item $eicarPath -Force -ErrorAction SilentlyContinue
}
if (-not $controlPassed) { $degraded += 'EICAR positive control did not fire' }

if ($degraded) {
  Write-Host "::warning::Defender is degraded on this runner ($($degraded -join '; ')). Scanning anyway, but a clean result below is not authoritative."
}

$paths = @()
try { $paths = @($env:ARTIFACT_PATHS | ConvertFrom-Json) } catch {
  Write-Host "::warning::could not parse artifactPaths: $($_.Exception.Message)"
}
$paths = @($paths | Where-Object { $_ -and (Test-Path $_ -PathType Leaf) })
if (-not $paths) { Write-Host '::warning::no Windows artifacts to scan'; exit 0 }

$detected = @()
foreach ($f in $paths) {
  # Real users download over HTTPS, which stamps a mark-of-the-web.
  # Block-at-first-sight only consults the cloud for internet-zone
  # files, so without this the release path is not the user path.
  "[ZoneTransfer]`nZoneId=3" | Set-Content -Path $f -Stream Zone.Identifier -Encoding Ascii -ErrorAction SilentlyContinue

  $out = & $mp -Scan -ScanType 3 -File $f -DisableRemediation 2>&1
  $text = ($out | Out-String)
  $out | ForEach-Object { Write-Host "  $_" }

  # Exit codes are ambiguous by design: 0 covers "clean" and "found and
  # remediated", 2 covers "found", "needs user action" and "scan error".
  # Read the output text instead.
  if ($text -match 'found\s+\d+\s+threat') {
    $names = [regex]::Matches($text, '((?:Trojan|Virus|Worm|Backdoor|Ransom|HackTool|PUA|Behavior|Program|Misleading)[:\w\./!\-]+)') |
             ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
    Write-Host "::error::Defender flagged $(Split-Path $f -Leaf): $($names -join ', ')"
    $detected += "$(Split-Path $f -Leaf) [$($names -join ', ')]"
  } else {
    Write-Host "clean: $(Split-Path $f -Leaf)"
  }
}

if ($detected) {
  Write-Host "::error::Refusing to publish a Windows bundle Defender flags: $($detected -join '; ')"
  Write-Host 'If this is a false positive, submit it at https://www.microsoft.com/en-us/wdsi/filesubmission (Software developer -> incorrectly detected) and re-run once cleared.'
  exit 1
}
Write-Host "All $($paths.Count) Windows bundle(s) scanned clean."
