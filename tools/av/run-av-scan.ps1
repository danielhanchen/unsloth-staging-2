<#
.SYNOPSIS
  Scan launcher variants on Windows with Microsoft Defender, ClamAV, and
  Kaspersky KVRT. Writes one JSON per variant to <OutDir>.

  Keyless engines only (no VirusTotal here -- VT runs locally off-box with the
  private key). Each variant folder is materialised into a real .lnk (from
  shortcut.json) so the shortcut binary is scanned alongside the .vbs / .ps1.
  Defender is scanned per-file so a detection is attributed to launch-studio.vbs
  vs the .lnk vs the .ps1.

.PARAMETER ArtifactsDir  Folder containing <variant>/ subfolders + manifest.json
.PARAMETER OutDir        Where to write <variant>.json result files
.PARAMETER KvrtExe       Optional path to a pre-downloaded KVRT.exe
.PARAMETER ClamDb        Optional ClamAV database dir (Windows choco layout)
#>
param(
  [Parameter(Mandatory = $true)][string]$ArtifactsDir,
  [Parameter(Mandatory = $true)][string]$OutDir,
  [string]$KvrtExe = "",
  [string]$ClamDb = ""
)

$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Resolve-MpCmdRun {
  $root = Join-Path $env:ProgramData "Microsoft\Windows Defender\Platform"
  if (Test-Path $root) {
    $newest = Get-ChildItem $root -Directory | Sort-Object Name -Descending |
              Select-Object -First 1
    if ($newest) {
      $p = Join-Path $newest.FullName "MpCmdRun.exe"
      if (Test-Path $p) { return $p }
    }
  }
  $fallback = Join-Path $env:ProgramFiles "Windows Defender\MpCmdRun.exe"
  if (Test-Path $fallback) { return $fallback }
  return $null
}

function New-VariantShortcut {
  param([string]$VariantDir)
  $scJson = Join-Path $VariantDir "shortcut.json"
  if (-not (Test-Path $scJson)) { return $null }
  $sc = Get-Content $scJson -Raw | ConvertFrom-Json
  $lnkPath = Join-Path $VariantDir "Unsloth Studio.lnk"
  try {
    $wsh = New-Object -ComObject WScript.Shell
    $lnk = $wsh.CreateShortcut($lnkPath)
    $lnk.TargetPath = [Environment]::ExpandEnvironmentVariables($sc.target)
    if ($sc.args) { $lnk.Arguments = $sc.args }
    $lnk.WorkingDirectory = $VariantDir
    if ($sc.window -eq "Minimized") { $lnk.WindowStyle = 7 }
    $lnk.Save()
    return $lnkPath
  } catch {
    Write-Host "[warn] could not create .lnk for $VariantDir : $($_.Exception.Message)"
    return $null
  }
}

function Parse-DefenderThreat {
  param([string]$Text)
  if ($Text -match 'Threat\s+(?:information|name)?\s*:\s*(.+)') { return $Matches[1].Trim() }
  $m = [regex]::Match($Text,
    '((?:Trojan|Behavior|HackTool|VirTool|PUA|PUADlMaster|Trojan:Script|Trojan:VBS|Trojan:Win32|Trojan:PowerShell|Exploit)[:/][\w./!\-]+)')
  if ($m.Success) { return $m.Value }
  return $null
}

function Scan-Defender {
  param([string]$VariantDir, [string]$MpCmd)
  $res = @{ verdict = "error"; threats = @(); perfile = @{}; raw = "" }
  if (-not $MpCmd) { $res.verdict = "unavailable"; return $res }
  $anyDetected = $false; $allClean = $true
  foreach ($f in (Get-ChildItem $VariantDir -File)) {
    if ($f.Name -eq "shortcut.json") { continue }   # our metadata, not an artifact
    $out = & $MpCmd -Scan -ScanType 3 -File $f.FullName -DisableRemediation 2>&1
    $code = $LASTEXITCODE
    $text = ($out | Out-String)
    $res.raw += "`n--- $($f.Name) (exit $code) ---`n$text"
    $threat = Parse-DefenderThreat $text
    $v = if ($code -eq 2) { "detected" } elseif ($code -eq 0) { "clean" } else { "error" }
    $res.perfile[$f.Name] = @{ verdict = $v; exit = $code; threat = $threat }
    if ($v -eq "detected") {
      $anyDetected = $true
      $res.threats += "$($f.Name): $(if ($threat) { $threat } else { 'exit2 (name unparsed)' })"
    }
    if ($v -ne "clean") { $allClean = $false }
  }
  if ($anyDetected) { $res.verdict = "detected" }
  elseif ($allClean)  { $res.verdict = "clean" }
  else                { $res.verdict = "error" }
  try {
    $res.history = @(Get-MpThreatDetection -ErrorAction SilentlyContinue |
      Where-Object { ($_.Resources -join ';') -match [regex]::Escape($VariantDir) } |
      ForEach-Object { @{ name = $_.ThreatName; resources = ($_.Resources -join ';') } })
  } catch {}
  return $res
}

function Scan-ClamAV {
  param([string]$VariantDir, [string]$ClamDb)
  $res = @{ verdict = "error"; exit = $null; raw = "" }
  $clam = (Get-Command clamscan -ErrorAction SilentlyContinue)
  if (-not $clam) { $res.verdict = "unavailable"; return $res }
  $clamArgs = @("--recursive", "--infected", "--no-summary")
  if ($ClamDb -and (Test-Path $ClamDb)) { $clamArgs += "--database=$ClamDb" }
  $clamArgs += $VariantDir
  $out = & clamscan @clamArgs 2>&1
  $res.exit = $LASTEXITCODE
  $res.raw = ($out | Out-String)
  if ($res.raw -match 'cl_load|No such file or directory.*database|Can.t get file status') {
    $res.verdict = "unavailable"; return $res    # DB not provisioned -> don't gate on it
  }
  switch ($res.exit) {
    0 { $res.verdict = "clean" }
    1 { $res.verdict = "detected" }
    default { $res.verdict = "error" }
  }
  return $res
}

function Scan-KVRT {
  param([string]$VariantDir, [string]$Kvrt, [string]$ReportRoot)
  $res = @{ verdict = "inconclusive"; exit = $null; detected = 0; raw = "" }
  if (-not $Kvrt -or -not (Test-Path $Kvrt)) { $res.verdict = "unavailable"; return $res }
  New-Item -ItemType Directory -Force -Path $ReportRoot | Out-Null
  $kvrtArgs = @("-accepteula", "-silent", "-fixednames", "-dontcryptsupportinfo",
               "-processlevel", "2", "-noads", "-d", $ReportRoot, "-custom", $VariantDir)
  $out = & $Kvrt @kvrtArgs 2>&1
  $res.exit = $LASTEXITCODE
  $res.raw = ($out | Out-String)
  # KVRT may write its report to -d, a Reports subdir, or a default data dir.
  $searchDirs = @($ReportRoot, "$env:SystemDrive\KVRT_Data", "$env:SystemDrive\KVRT2024_Data",
                  "$env:SystemDrive\KVRT2020_Data") | Where-Object { Test-Path $_ }
  $detected = 0; $reportSeen = $false
  foreach ($d in $searchDirs) {
    foreach ($r in (Get-ChildItem $d -Recurse -File -Include *.txt,*.klr,*.log -ErrorAction SilentlyContinue)) {
      $reportSeen = $true
      $txt = Get-Content $r.FullName -Raw -ErrorAction SilentlyContinue
      if ($txt -match 'Detected:\s*(\d+)') { $detected += [int]$Matches[1] }
      if ($txt -match 'launch-studio\.(vbs|ps1|lnk)' -and
          $txt -match '(Trojan|HEUR|UDS|VirTool|not-a-virus|Backdoor)') {
        $res.raw += "`n[report threat match] $($r.FullName)"
      }
    }
  }
  # Also parse stdout for inline detections.
  if ($res.raw -match 'Detected:\s*(\d+)') { $detected = [Math]::Max($detected, [int]$Matches[1]) }
  $res.detected = $detected
  if ($detected -gt 0) { $res.verdict = "detected" }
  elseif ($reportSeen) { $res.verdict = "clean" }
  else { $res.verdict = "inconclusive" }
  return $res
}

$mp = Resolve-MpCmdRun
Write-Host "MpCmdRun: $mp"
Write-Host "KVRT: $KvrtExe"
Write-Host "ClamDb: $ClamDb"
Update-MpSignature -ErrorAction SilentlyContinue

$variants = Get-ChildItem $ArtifactsDir -Directory | Where-Object { $_.Name -like 'V*' }
foreach ($v in $variants) {
  Write-Host "=== scanning $($v.Name) ==="
  $null = New-VariantShortcut -VariantDir $v.FullName
  $files = @(Get-ChildItem $v.FullName -File | ForEach-Object { $_.Name })
  $result = [ordered]@{
    variant = $v.Name
    os      = "windows-latest"
    files   = $files
    engines = [ordered]@{
      defender = (Scan-Defender -VariantDir $v.FullName -MpCmd $mp)
      clamav   = (Scan-ClamAV   -VariantDir $v.FullName -ClamDb $ClamDb)
      kvrt     = (Scan-KVRT     -VariantDir $v.FullName -Kvrt $KvrtExe `
                                -ReportRoot (Join-Path $env:RUNNER_TEMP "kvrt_$($v.Name)"))
    }
  }
  $outFile = Join-Path $OutDir "$($v.Name).json"
  $result | ConvertTo-Json -Depth 9 | Set-Content -Path $outFile -Encoding UTF8
  $d = $result.engines.defender
  Write-Host "    defender=$($d.verdict) [$($d.threats -join ', ')] clamav=$($result.engines.clamav.verdict) kvrt=$($result.engines.kvrt.verdict)"
}
Write-Host "done."
exit 0
