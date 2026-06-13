<#
.SYNOPSIS
  Scan launcher variants on Windows with Microsoft Defender, ClamAV, and
  Kaspersky KVRT. Writes one JSON per variant to <OutDir>.

  Keyless engines only (no VirusTotal here -- VT runs locally off-box with the
  private key). Each variant folder is materialised into a real .lnk (from
  shortcut.json) so the shortcut binary is scanned alongside the .vbs / .ps1.

.PARAMETER ArtifactsDir  Folder containing <variant>/ subfolders + manifest.json
.PARAMETER OutDir        Where to write <variant>.json result files
.PARAMETER KvrtExe       Optional path to a pre-downloaded KVRT.exe
#>
param(
  [Parameter(Mandatory = $true)][string]$ArtifactsDir,
  [Parameter(Mandatory = $true)][string]$OutDir,
  [string]$KvrtExe = ""
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

function Scan-Defender {
  param([string]$VariantDir, [string]$MpCmd)
  $res = @{ verdict = "error"; exit = $null; threats = @(); raw = "" }
  if (-not $MpCmd) { $res.verdict = "unavailable"; return $res }
  $before = @(Get-MpThreatDetection -ErrorAction SilentlyContinue)
  $out = & $MpCmd -Scan -ScanType 3 -File $VariantDir -DisableRemediation 2>&1
  $res.exit = $LASTEXITCODE
  $res.raw = ($out | Out-String)
  Start-Sleep -Seconds 2
  $after = @(Get-MpThreatDetection -ErrorAction SilentlyContinue)
  $new = $after | Where-Object {
    $_.Resources -and ($_.Resources -join ';') -match [regex]::Escape($VariantDir)
  }
  if ($new) {
    $res.verdict = "detected"
    $res.threats = @($new | ForEach-Object { $_.ThreatName + " :: " + ($_.Resources -join ';') })
  } elseif ($res.exit -eq 2) {
    # exit 2 without a matched resource: treat as detected-but-unattributed
    $res.verdict = "detected"
    $res.threats = @("MpCmdRun exit 2 (unattributed); see raw")
  } elseif ($res.exit -eq 0) {
    $res.verdict = "clean"
  } else {
    $res.verdict = "error"
  }
  return $res
}

function Scan-ClamAV {
  param([string]$VariantDir)
  $res = @{ verdict = "error"; exit = $null; raw = "" }
  $clam = (Get-Command clamscan -ErrorAction SilentlyContinue)
  if (-not $clam) { $res.verdict = "unavailable"; return $res }
  $out = & clamscan --recursive --infected --no-summary $VariantDir 2>&1
  $res.exit = $LASTEXITCODE
  $res.raw = ($out | Out-String)
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
  # Custom-only scan of just this variant folder; keep reports plaintext.
  $args = @("-accepteula","-silent","-fixednames","-dontcryptsupportinfo",
            "-processlevel","2","-d",$ReportRoot,"-custom",$VariantDir)
  $out = & $Kvrt @args 2>&1
  $res.exit = $LASTEXITCODE
  $res.raw = ($out | Out-String)
  $reports = Get-ChildItem $ReportRoot -Recurse -Include *.txt,*.klr -ErrorAction SilentlyContinue
  $detected = 0
  foreach ($r in $reports) {
    $txt = Get-Content $r.FullName -Raw -ErrorAction SilentlyContinue
    if ($txt -match 'Detected:\s*(\d+)') { $detected += [int]$Matches[1] }
    # Threat lines that reference our files.
    if ($txt -match 'launch-studio\.(vbs|ps1)' -and
        $txt -match '(Trojan|HEUR|UDS|VirTool|not-a-virus|Backdoor)') {
      $res.raw += "`n[report match] " + $r.FullName
    }
  }
  $res.detected = $detected
  if ($detected -gt 0) { $res.verdict = "detected" }
  elseif ($reports.Count -gt 0) { $res.verdict = "clean" }
  else { $res.verdict = "inconclusive" }
  return $res
}

$mp = Resolve-MpCmdRun
Write-Host "MpCmdRun: $mp"
Write-Host "KVRT: $KvrtExe"
& Update-MpSignature -ErrorAction SilentlyContinue 2>&1 | Out-Null

$variants = Get-ChildItem $ArtifactsDir -Directory | Where-Object { $_.Name -like 'V*' }
foreach ($v in $variants) {
  Write-Host "=== scanning $($v.Name) ==="
  $lnk = New-VariantShortcut -VariantDir $v.FullName
  $files = @(Get-ChildItem $v.FullName -File | ForEach-Object { $_.Name })
  $result = [ordered]@{
    variant = $v.Name
    os      = "windows-latest"
    files   = $files
    engines = [ordered]@{
      defender = (Scan-Defender -VariantDir $v.FullName -MpCmd $mp)
      clamav   = (Scan-ClamAV   -VariantDir $v.FullName)
      kvrt     = (Scan-KVRT     -VariantDir $v.FullName -Kvrt $KvrtExe `
                                -ReportRoot (Join-Path $env:RUNNER_TEMP "kvrt_$($v.Name)"))
    }
  }
  $outFile = Join-Path $OutDir "$($v.Name).json"
  $result | ConvertTo-Json -Depth 8 | Set-Content -Path $outFile -Encoding UTF8
  $d = $result.engines.defender.verdict
  $c = $result.engines.clamav.verdict
  $k = $result.engines.kvrt.verdict
  Write-Host "    defender=$d clamav=$c kvrt=$k -> $outFile"
}
Write-Host "done."
