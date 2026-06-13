<#
.SYNOPSIS
  Scan launcher variants on Windows with Microsoft Defender, ClamAV, and
  Kaspersky KVRT. Writes one JSON per variant to <OutDir>, plus a
  _defender_control.json proving whether Defender actually works on this runner.

  IMPORTANT methodology:
   * MpCmdRun exit code 2 is NOT a reliable "malware found" signal -- it is also
     returned on scan/RPC errors (0x800106ba "RPC server is unavailable" when the
     WinDefend service is stopped, which is common on hosted runners). We only
     call a result "detected" when a real threat name / MpThreatDetection record
     exists; RPC/other failures are "error", never "detected".
   * Defender cloud/ML verdicts on never-before-seen files are non-deterministic,
     so each file is scanned -Repeats times and we report a detection RATE.
   * An EICAR positive control + a benign negative control verify the engine is
     live before we trust any "clean".

.PARAMETER ArtifactsDir  Folder containing <variant>/ subfolders
.PARAMETER OutDir        Where to write result JSONs
.PARAMETER KvrtExe       Optional path to a pre-downloaded KVRT.exe
.PARAMETER ClamDb        Optional ClamAV database dir
.PARAMETER Repeats       Defender scan passes per file (default 3)
#>
param(
  [Parameter(Mandatory = $true)][string]$ArtifactsDir,
  [Parameter(Mandatory = $true)][string]$OutDir,
  [string]$KvrtExe = "",
  [string]$ClamDb = "",
  [int]$Repeats = 3
)

$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Resolve-MpCmdRun {
  $root = Join-Path $env:ProgramData "Microsoft\Windows Defender\Platform"
  if (Test-Path $root) {
    $newest = Get-ChildItem $root -Directory | Sort-Object Name -Descending | Select-Object -First 1
    if ($newest) { $p = Join-Path $newest.FullName "MpCmdRun.exe"; if (Test-Path $p) { return $p } }
  }
  $fb = Join-Path $env:ProgramFiles "Windows Defender\MpCmdRun.exe"
  if (Test-Path $fb) { return $fb }
  return $null
}

function Enable-Defender {
  # Best-effort: bring WinDefend up and enable real-time + cloud (MAPS) so cloud
  # heuristics (the thing that flags these launchers) are in play.
  try { Set-Service WinDefend -StartupType Manual -ErrorAction SilentlyContinue } catch {}
  try { Start-Service WinDefend -ErrorAction SilentlyContinue } catch {}
  try {
    Set-MpPreference -DisableRealtimeMonitoring $false -ErrorAction SilentlyContinue
    Set-MpPreference -MAPSReporting Advanced -SubmitSamplesConsent SendAllSamples -ErrorAction SilentlyContinue
    Set-MpPreference -DisableIOAVProtection $false -ErrorAction SilentlyContinue
  } catch {}
  Start-Sleep -Seconds 3
}

function Parse-DefenderThreat {
  param([string]$Text)
  if ($Text -match 'Threat\s+(?:information|name)?\s*:\s*(.+)') { return $Matches[1].Trim() }
  $m = [regex]::Match($Text, '((?:Trojan|Behavior|HackTool|VirTool|PUA|PUADlMaster|Program|Trojan:Script|Trojan:VBS|Trojan:Win32|Trojan:PowerShell|Exploit|Wacatac|Sabsik)[:/!][\w./!\-]+)')
  if ($m.Success) { return $m.Value }
  return $null
}

function Invoke-DefenderFileScan {
  # Returns: @{ verdict='clean|detected|error'; threat=<name|null>; raw=<stdout> }
  param([string]$File, [string]$MpCmd)
  $out = & $MpCmd -Scan -ScanType 3 -File $File -DisableRemediation 2>&1
  $code = $LASTEXITCODE
  $text = ($out | Out-String)
  $threat = Parse-DefenderThreat $text
  # Authoritative cross-check: a recorded detection referencing this file.
  $rec = $null
  try {
    $rec = Get-MpThreatDetection -ErrorAction SilentlyContinue |
           Where-Object { ($_.Resources -join ';') -match [regex]::Escape($File) } |
           Select-Object -First 1
  } catch {}
  if ($rec) { return @{ verdict = 'detected'; threat = $rec.ThreatName; raw = $text } }
  if ($text -match 'RPC server is unavailable|\[Failed\]\[0x|CmdTool: Failed|0x800106ba') {
    return @{ verdict = 'error'; threat = $null; raw = $text }   # scan failed, NOT a detection
  }
  if ($code -eq 0) { return @{ verdict = 'clean'; threat = $null; raw = $text } }
  if ($code -eq 2 -and $threat) { return @{ verdict = 'detected'; threat = $threat; raw = $text } }
  if ($code -eq 2) { return @{ verdict = 'error'; threat = $null; raw = $text } }  # exit2 w/o evidence = ambiguous
  return @{ verdict = 'error'; threat = $null; raw = $text }
}

function Test-DefenderControls {
  # EICAR positive control + benign negative control. Proves the engine is live.
  param([string]$MpCmd)
  $dir = Join-Path $env:RUNNER_TEMP "av_control"
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  # Build EICAR from parts so this script itself is not flagged.
  $eicar = 'X5O!P%@AP[4\PZX54(P^)7CC)7}' + '$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*'
  $eicarFile = Join-Path $dir "eicar.com"
  try { [IO.File]::WriteAllText($eicarFile, $eicar) } catch {}
  Start-Sleep -Seconds 2
  $pos = if (Test-Path $eicarFile) { Invoke-DefenderFileScan -File $eicarFile -MpCmd $MpCmd }
         else { @{ verdict = 'detected'; threat = 'EICAR (removed by real-time)'; raw = 'file gone' } }
  $benFile = Join-Path $dir "benign.txt"
  "hello world, this is a normal text file" | Set-Content $benFile
  $neg = Invoke-DefenderFileScan -File $benFile -MpCmd $MpCmd
  $status = $null
  try { $status = Get-MpComputerStatus | Select-Object AMServiceEnabled, RealTimeProtectionEnabled, BehaviorMonitorEnabled, IsTamperProtected, AMEngineVersion, AntivirusSignatureVersion } catch {}
  $functional = ($pos.verdict -eq 'detected') -and ($neg.verdict -eq 'clean')
  return @{ functional = $functional; eicar = $pos; benign = $neg; status = $status }
}

function Scan-Defender {
  param([string]$VariantDir, [string]$MpCmd, [int]$Repeats)
  $res = @{ verdict = 'error'; threats = @(); perfile = @{}; repeats = $Repeats }
  if (-not $MpCmd) { $res.verdict = 'unavailable'; return $res }
  $files = Get-ChildItem $VariantDir -File | Where-Object { $_.Name -ne 'shortcut.json' }
  $anyDetected = $false; $anyClean = $false; $anyError = $false
  foreach ($f in $files) {
    $passes = @()
    for ($p = 1; $p -le $Repeats; $p++) {
      $r = Invoke-DefenderFileScan -File $f.FullName -MpCmd $MpCmd
      $passes += $r.verdict
      if ($r.verdict -eq 'detected') {
        $anyDetected = $true
        $res.threats += "$($f.Name): $($r.threat)"
      }
    }
    $detN = @($passes | Where-Object { $_ -eq 'detected' }).Count
    $clnN = @($passes | Where-Object { $_ -eq 'clean' }).Count
    $errN = @($passes | Where-Object { $_ -eq 'error' }).Count
    if ($clnN -gt 0) { $anyClean = $true }
    if ($errN -gt 0) { $anyError = $true }
    $res.perfile[$f.Name] = @{ detected = $detN; clean = $clnN; error = $errN; passes = $passes }
  }
  if ($anyDetected) { $res.verdict = 'detected' }
  elseif ($anyClean -and -not $anyError) { $res.verdict = 'clean' }
  elseif ($anyClean) { $res.verdict = 'clean-with-errors' }
  else { $res.verdict = 'error' }
  return $res
}

function Scan-ClamAV {
  param([string]$VariantDir, [string]$ClamDb)
  $res = @{ verdict = 'error'; exit = $null; raw = '' }
  if (-not (Get-Command clamscan -ErrorAction SilentlyContinue)) { $res.verdict = 'unavailable'; return $res }
  $a = @('--recursive', '--infected', '--no-summary')
  if ($ClamDb -and (Test-Path $ClamDb)) { $a += "--database=$ClamDb" }
  $a += $VariantDir
  $out = & clamscan @a 2>&1
  $res.exit = $LASTEXITCODE; $res.raw = ($out | Out-String)
  if ($res.raw -match 'cl_load|No such file or directory.*database|Can.t get file status') { $res.verdict = 'unavailable'; return $res }
  switch ($res.exit) { 0 { $res.verdict = 'clean' } 1 { $res.verdict = 'detected' } default { $res.verdict = 'error' } }
  return $res
}

function Scan-KVRT {
  param([string]$VariantDir, [string]$Kvrt, [string]$ReportRoot)
  $res = @{ verdict = 'inconclusive'; exit = $null; detected = 0; raw = '' }
  if (-not $Kvrt -or -not (Test-Path $Kvrt)) { $res.verdict = 'unavailable'; return $res }
  New-Item -ItemType Directory -Force -Path $ReportRoot | Out-Null
  $a = @('-accepteula', '-silent', '-fixednames', '-dontcryptsupportinfo', '-processlevel', '2', '-noads', '-d', $ReportRoot, '-custom', $VariantDir)
  $out = & $Kvrt @a 2>&1
  $res.exit = $LASTEXITCODE; $res.raw = ($out | Out-String)
  $dirs = @($ReportRoot, "$env:SystemDrive\KVRT_Data", "$env:SystemDrive\KVRT2024_Data", "$env:SystemDrive\KVRT2020_Data") | Where-Object { Test-Path $_ }
  $det = 0; $seen = $false
  foreach ($d in $dirs) {
    foreach ($r in (Get-ChildItem $d -Recurse -File -Include *.txt, *.klr, *.log -ErrorAction SilentlyContinue)) {
      $seen = $true; $t = Get-Content $r.FullName -Raw -ErrorAction SilentlyContinue
      if ($t -match 'Detected:\s*(\d+)') { $det += [int]$Matches[1] }
    }
  }
  if ($res.raw -match 'Detected:\s*(\d+)') { $det = [Math]::Max($det, [int]$Matches[1]) }
  $res.detected = $det
  if ($det -gt 0) { $res.verdict = 'detected' } elseif ($seen) { $res.verdict = 'clean' } else { $res.verdict = 'inconclusive' }
  return $res
}

function New-VariantShortcut {
  param([string]$VariantDir)
  $scJson = Join-Path $VariantDir "shortcut.json"
  if (-not (Test-Path $scJson)) { return }
  $sc = Get-Content $scJson -Raw | ConvertFrom-Json
  try {
    $wsh = New-Object -ComObject WScript.Shell
    $lnk = $wsh.CreateShortcut((Join-Path $VariantDir "Unsloth Studio.lnk"))
    $lnk.TargetPath = [Environment]::ExpandEnvironmentVariables($sc.target)
    if ($sc.args) { $lnk.Arguments = $sc.args }
    $lnk.WorkingDirectory = $VariantDir
    if ($sc.window -eq "Minimized") { $lnk.WindowStyle = 7 }
    $lnk.Save()
  } catch { Write-Host "[warn] .lnk create failed: $($_.Exception.Message)" }
}

# ===== main =====
$mp = Resolve-MpCmdRun
Write-Host "MpCmdRun: $mp"
Enable-Defender
$control = Test-DefenderControls -MpCmd $mp
$control | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $OutDir "_defender_control.json") -Encoding UTF8
Write-Host "Defender functional (EICAR detected + benign clean): $($control.functional)"
Write-Host "  EICAR: $($control.eicar.verdict)  benign: $($control.benign.verdict)"
Update-MpSignature -ErrorAction SilentlyContinue

foreach ($v in (Get-ChildItem $ArtifactsDir -Directory | Where-Object { $_.Name -like 'V*' })) {
  Write-Host "=== scanning $($v.Name) (x$Repeats) ==="
  New-VariantShortcut -VariantDir $v.FullName
  $result = [ordered]@{
    variant = $v.Name; os = "windows-latest"
    files = @(Get-ChildItem $v.FullName -File | ForEach-Object { $_.Name })
    defender_functional = $control.functional
    engines = [ordered]@{
      defender = (Scan-Defender -VariantDir $v.FullName -MpCmd $mp -Repeats $Repeats)
      clamav   = (Scan-ClamAV   -VariantDir $v.FullName -ClamDb $ClamDb)
      kvrt     = (Scan-KVRT     -VariantDir $v.FullName -Kvrt $KvrtExe -ReportRoot (Join-Path $env:RUNNER_TEMP "kvrt_$($v.Name)"))
    }
  }
  $result | ConvertTo-Json -Depth 9 | Set-Content (Join-Path $OutDir "$($v.Name).json") -Encoding UTF8
  $d = $result.engines.defender
  Write-Host "    defender=$($d.verdict) [$($d.threats -join '; ')] clamav=$($result.engines.clamav.verdict) kvrt=$($result.engines.kvrt.verdict)"
}
Write-Host "done."
exit 0
