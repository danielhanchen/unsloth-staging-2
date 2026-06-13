<#
.SYNOPSIS
  Track A live behavioral test: install Unsloth Studio with the real installer,
  launch it through the generated Start-Menu/Desktop .lnk -> VBS shortcut (the
  exact chain the Kaspersky report came from), open it via localhost, the
  0.0.0.0 bind, and the Cloudflare tunnel, and drive inference / image-gen /
  training through the UI, all while Microsoft Defender (real-time + behavior +
  cloud) watches. EICAR proves the engine is live; any Studio-correlated
  detection is a blocker. No AV exclusions are added (asserted at the end).
.PARAMETER RepoRoot   cloned unsloth repo (has install.ps1)
.PARAMETER OutDir     per-iteration artifacts
.PARAMETER Iteration  iteration index (for filenames)
.PARAMETER DriverPy   path to drive_studio_flows.py
.PARAMETER NewPassword password to rotate the bootstrap password to
#>
param(
  [Parameter(Mandatory = $true)][string]$RepoRoot,
  [Parameter(Mandatory = $true)][string]$OutDir,
  [int]$Iteration = 1,
  [string]$DriverPy = "tools/av/drive_studio_flows.py",
  [string]$NewPassword = "UnslothAvTest2026!"
)
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$runStart = Get-Date

# ── Defender helpers (shared with run-av-scan.ps1) ────────────────────
function Resolve-MpCmdRun {
  $root = Join-Path $env:ProgramData "Microsoft\Windows Defender\Platform"
  if (Test-Path $root) {
    $n = Get-ChildItem $root -Directory | Sort-Object Name -Descending | Select-Object -First 1
    if ($n) { $p = Join-Path $n.FullName "MpCmdRun.exe"; if (Test-Path $p) { return $p } }
  }
  $fb = Join-Path $env:ProgramFiles "Windows Defender\MpCmdRun.exe"
  if (Test-Path $fb) { return $fb } ; return $null
}
function Enable-Defender {
  try { Set-Service WinDefend -StartupType Manual -ErrorAction SilentlyContinue } catch {}
  try { Start-Service WinDefend -ErrorAction SilentlyContinue } catch {}
  try {
    Set-MpPreference -DisableRealtimeMonitoring $false -DisableBehaviorMonitoring $false `
      -DisableIOAVProtection $false -MAPSReporting Advanced `
      -SubmitSamplesConsent SendAllSamples -ErrorAction SilentlyContinue
  } catch {}
  Update-MpSignature -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 3
}
function Get-Detections {
  @(Get-MpThreatDetection -ErrorAction SilentlyContinue | ForEach-Object {
      [pscustomobject]@{ name = $_.ThreatName; res = ($_.Resources -join ';'); time = $_.InitialDetectionTime } })
}
function Get-DefenderEvents {
  param([datetime]$Since)
  $ids = 1006,1007,1008,1015,1116,1117,1118,1119,1120,1121,1122
  try {
    Get-WinEvent -FilterHashtable @{ LogName = 'Microsoft-Windows-Windows Defender/Operational'; StartTime = $Since } -ErrorAction SilentlyContinue |
      Where-Object { $ids -contains $_.Id } |
      ForEach-Object { [pscustomobject]@{ id = $_.Id; time = $_.TimeCreated.ToString('o'); line = (($_.Message -split "`r?`n") | Select-Object -First 1) } }
  } catch { @() }
}
function Test-EicarControl {
  param([string]$MpCmd)
  $dir = Join-Path $env:RUNNER_TEMP "av_control"; New-Item -ItemType Directory -Force -Path $dir | Out-Null
  $eicar = 'X5O!P%@AP[4\PZX54(P^)7CC)7}' + '$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*'
  $f = Join-Path $dir "eicar.com"
  try { [IO.File]::WriteAllText($f, $eicar) } catch {}
  Start-Sleep -Seconds 3
  $rec = Get-MpThreatDetection -ErrorAction SilentlyContinue | Where-Object { ($_.Resources -join ';') -match 'eicar' } | Select-Object -First 1
  if ($rec) { return @{ detected = $true; threat = $rec.ThreatName } }
  if ($MpCmd) { & $MpCmd -Scan -ScanType 3 -File $f -DisableRemediation 2>&1 | Out-Null }
  $rec = Get-MpThreatDetection -ErrorAction SilentlyContinue | Where-Object { ($_.Resources -join ';') -match 'eicar' } | Select-Object -First 1
  if (-not (Test-Path $f)) { return @{ detected = $true; threat = 'EICAR (real-time removed)' } }
  if ($rec) { return @{ detected = $true; threat = $rec.ThreatName } }
  return @{ detected = $false; threat = $null }
}
function Exclusion-Snapshot {
  try {
    $p = Get-MpPreference
    return @{ paths = @($p.ExclusionPath); procs = @($p.ExclusionProcess); ext = @($p.ExclusionExtension) }
  } catch { return @{ paths=@(); procs=@(); ext=@() } }
}

# ── 1. Defender preflight + liveness ─────────────────────────────────
$mp = Resolve-MpCmdRun
Enable-Defender
$status0 = Get-MpComputerStatus -ErrorAction SilentlyContinue | Select-Object AMServiceEnabled, RealTimeProtectionEnabled, BehaviorMonitorEnabled, IsTamperProtected, AMEngineVersion, AntivirusSignatureVersion
$exBase = Exclusion-Snapshot
$eicar = Test-EicarControl -MpCmd $mp
$baseDet = Get-Detections
Write-Host "Defender RT=$($status0.RealTimeProtectionEnabled) behavior=$($status0.BehaviorMonitorEnabled) EICAR_detected=$($eicar.detected)"

# ── 2. Real install (default mode => persistent shortcuts; NOT env-override) ─
$unslothExe = $null
$installLog = Join-Path $OutDir "install.log"
if (Get-Command unsloth -ErrorAction SilentlyContinue) { $unslothExe = (Get-Command unsloth).Source }
$didInstall = $false
if (-not $unslothExe) {
  Push-Location $RepoRoot
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "install.ps1") --no-torch --local *>&1 | Tee-Object -FilePath $installLog
  Pop-Location
  $didInstall = $true
}
$detAfterInstall = Get-Detections
$evAfterInstall = @(Get-DefenderEvents -Since $runStart)

# ── 3. Find + validate the generated shortcut (-> VBS) ────────────────
$lnks = @()
foreach ($d in @([Environment]::GetFolderPath("Desktop"),
                 (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"))) {
  $p = Join-Path $d "Unsloth Studio.lnk"; if (Test-Path $p) { $lnks += $p }
}
$shortcutOk = $false; $vbsTarget = $null
if ($lnks.Count -gt 0) {
  $wsh = New-Object -ComObject WScript.Shell
  $sc = $wsh.CreateShortcut($lnks[0])
  $vbsTarget = "$($sc.TargetPath) $($sc.Arguments)"
  $shortcutOk = ($sc.Arguments -match 'launch-studio\.vbs')
}
Write-Host "shortcuts=$($lnks.Count) target=$vbsTarget vbsChain=$shortcutOk"

# ── 4. Launch through the REAL shortcut (localhost origin) ────────────
function Wait-Health { param([int]$Port,[int]$TimeoutS=180)
  $t=(Get-Date); while(((Get-Date)-$t).TotalSeconds -lt $TimeoutS){
    try { $r=Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 4
          if($r.status -eq 'healthy'){ return $true } } catch {}
    Start-Sleep 3 } ; return $false }

$launched = $false
if ($lnks.Count -gt 0) { Invoke-Item $lnks[0]; $launched = Wait-Health -Port 8888 -TimeoutS 240 }
Write-Host "shortcut launch healthy=$launched"

# bootstrap password is written under the studio home; discover it.
$pwFile = Get-ChildItem -Path @($env:LOCALAPPDATA, $env:USERPROFILE) -Recurse -Filter ".bootstrap_password" -ErrorAction SilentlyContinue | Select-Object -First 1
$bootPw = if ($pwFile) { (Get-Content $pwFile.FullName -Raw).Trim() } else { "" }

# ── 5. Drive Studio per origin with the Python driver ─────────────────
function Invoke-Driver { param([string]$Base,[string]$Origin,[string]$Actions)
  $o = Join-Path $OutDir $Origin
  & python $DriverPy --base $Base --origin $Origin --mode cpu --actions $Actions `
      --bootstrap-password $bootPw --new-password $NewPassword --out $o `
      --inference-model "unsloth/SmolLM2-135M-Instruct-GGUF" --inference-variant "Q4_K_M" --inference-needle "SmolLM2-135M" `
      --image-model "unsloth/diffusiongemma-26B-A4B-it-GGUF" --image-variant "Q4_K_M" --image-needle "diffusiongemma" `
      *>&1 | Tee-Object -FilePath (Join-Path $OutDir "driver_$Origin.log")
  return $LASTEXITCODE }

# DiffusionGemma is GPU-only and its GGUF is ~17 GB (won't fit a 14 GB hosted
# runner), so it is covered for real on the Linux/ClamAV track; here we drive the
# CPU-capable inference plus a small training attempt (graceful no-GPU on --no-torch).
$driverRc = @{}
if ($launched) { $driverRc["localhost"] = Invoke-Driver -Base "http://127.0.0.1:8888" -Origin "localhost" -Actions "inference,training" }

# ── 6. 0.0.0.0 bind + Cloudflare tunnel ──────────────────────────────
$cfUrl = $null
$cfLog = Join-Path $OutDir "studio_0000.log"
# PATH is not refreshed in this session post-install; fall back to a direct search.
$studioCmd = (Get-Command unsloth -ErrorAction SilentlyContinue).Source
if (-not $studioCmd) {
  $cand = Get-ChildItem -Path @($env:USERPROFILE, $env:LOCALAPPDATA) -Recurse -Filter "unsloth.exe" -ErrorAction SilentlyContinue |
          Where-Object { $_.FullName -match '\\bin\\unsloth\.exe$' } | Select-Object -First 1
  if ($cand) { $studioCmd = $cand.FullName }
}
if ($studioCmd) {
  Start-Process -FilePath $studioCmd -ArgumentList @("studio","-H","0.0.0.0","-p","8890","--cloudflare") `
    -RedirectStandardOutput $cfLog -RedirectStandardError (Join-Path $OutDir "studio_0000.err.log") -WindowStyle Hidden
  if (Wait-Health -Port 8890 -TimeoutS 240) {
    $driverRc["bind0000"] = Invoke-Driver -Base "http://127.0.0.1:8890" -Origin "bind0000" -Actions "inference"
    $t=(Get-Date); while(((Get-Date)-$t).TotalSeconds -lt 90 -and -not $cfUrl){
      if (Test-Path $cfLog) { $m=[regex]::Match((Get-Content $cfLog -Raw),'https://[a-z0-9-]+\.trycloudflare\.com'); if($m.Success){$cfUrl=$m.Value} }
      Start-Sleep 3 }
    if ($cfUrl) { $driverRc["cloudflare"] = Invoke-Driver -Base $cfUrl -Origin "cloudflare" -Actions "inference" }
  }
}
Write-Host "cloudflare_url=$cfUrl"

# ── 7. Defender poll across the full window + classify ────────────────
$finalDet = Get-Detections
$events = @(Get-DefenderEvents -Since $runStart)
$newDet = @($finalDet | Where-Object { $d=$_; -not ($baseDet | Where-Object { $_.time -eq $d.time -and $_.name -eq $d.name }) })
# Studio-correlated detections = anything NOT the EICAR control.
$studioDet = @($newDet | Where-Object { ($_.res -notmatch 'eicar') -and ($_.name -notmatch 'EICAR') })
$hardEvents = @($events | Where-Object { 1006,1015,1116,1117,1121,1122 -contains $_.id -and $_.line -notmatch 'EICAR' })

# ── 8. No-exclusions guard ────────────────────────────────────────────
$exEnd = Exclusion-Snapshot
$addedEx = @()
foreach ($k in 'paths','procs','ext') {
  $addedEx += @($exEnd[$k] | Where-Object { $_ -and ($exBase[$k] -notcontains $_) })
}

# Did we actually exercise Studio (so "no detections" is meaningful)?
$executed = $launched -and ($driverRc.ContainsKey("localhost"))
$noFindings = ($studioDet.Count -eq 0) -and ($hardEvents.Count -eq 0) -and ($addedEx.Count -eq 0)
$result = [ordered]@{
  iteration = $Iteration
  defender_status = $status0
  eicar_control = $eicar
  shortcuts = @($lnks); shortcut_target = $vbsTarget; vbs_chain_ok = $shortcutOk
  installed_now = $didInstall
  shortcut_launch_healthy = $launched
  test_executed = $executed
  cloudflare_url_obtained = [bool]$cfUrl
  driver_rc = $driverRc
  studio_detections = $studioDet
  hard_defender_events = $hardEvents
  added_exclusions = $addedEx
  events_count = $events.Count
  control_valid = [bool]$eicar.detected
  no_av_findings = $noFindings
  clean = ([bool]$eicar.detected) -and $executed -and $noFindings
}
$result | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $OutDir "iteration_$Iteration.json") -Encoding UTF8
Write-Host ("=== iter {0}: clean={1} executed={2} control_valid={3} studio_detections={4} added_exclusions={5} ===" -f `
  $Iteration, $result.clean, $executed, $result.control_valid, $studioDet.Count, $addedEx.Count)
if (-not $result.control_valid) { Write-Host "INVALID: EICAR control did not fire"; exit 3 }
if (-not $noFindings) { Write-Host "AV FINDING during Studio use"; exit 1 }
if (-not $executed) { Write-Host "INCOMPLETE: install/launch did not execute Studio"; exit 4 }
exit 0
