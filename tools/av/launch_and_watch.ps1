<#
.SYNOPSIS
  Runtime Defender check: EXECUTE a launcher variant the way the shortcut would
  (wscript -> hidden PowerShell, or the .lnk target) several times and watch
  Defender, answering "does launching Studio raise anything?". The launcher
  points at a stub unsloth.exe serving one healthy /api/health so the chain
  completes; the behavioral IOC (script engine -> hidden Bypass PowerShell) fires
  either way.
.PARAMETER VariantDir   artifacts/<variant>
.PARAMETER OutFile      result JSON
.PARAMETER Iterations   launch cycles (default 5)
.PARAMETER WatchSeconds watch window per launch (default 12)
.PARAMETER StubExe      prebuilt stub unsloth.exe; else compiled
#>
param(
  [Parameter(Mandatory = $true)][string]$VariantDir,
  [Parameter(Mandatory = $true)][string]$OutFile,
  [int]$Iterations = 5,
  [int]$WatchSeconds = 12,
  [string]$StubExe = ""
)
$ErrorActionPreference = 'Continue'

function New-StubUnslothExe {
  param([string]$Path)
  # Console exe serving one healthy /api/health on -p <port> so the chain exits.
  $src = @'
using System; using System.Net; using System.Text; using System.Threading;
class Stub {
  static void Main(string[] args) {
    int port = 8888;
    for (int i = 0; i < args.Length - 1; i++) if (args[i] == "-p") int.TryParse(args[i+1], out port);
    try {
      var l = new HttpListener(); l.Prefixes.Add("http://127.0.0.1:" + port + "/");
      l.Start();
      var sw = System.Diagnostics.Stopwatch.StartNew();
      while (sw.Elapsed.TotalSeconds < 30) {
        var ar = l.BeginGetContext(null, null);
        if (!ar.AsyncWaitHandle.WaitOne(1000)) continue;
        var ctx = l.EndGetContext(ar);
        byte[] b = Encoding.UTF8.GetBytes("{\"status\":\"healthy\",\"service\":\"Unsloth UI Backend\",\"studio_root_id\":\"\"}");
        ctx.Response.ContentType = "application/json";
        ctx.Response.OutputStream.Write(b, 0, b.Length); ctx.Response.Close();
      }
    } catch {}
  }
}
'@
  Add-Type -TypeDefinition $src -OutputType ConsoleApplication -OutputAssembly $Path -ErrorAction Stop
}

function Get-Detections {
  @(Get-MpThreatDetection -ErrorAction SilentlyContinue | ForEach-Object {
      [pscustomobject]@{ name = $_.ThreatName; res = ($_.Resources -join ';');
                         time = $_.InitialDetectionTime } })
}

function Get-DefenderEvents {
  param([datetime]$Since)
  $ids = 1006,1007,1008,1009,1010,1011,1015,1116,1117,1118,1119,1120,1121,1122,5001,5007,5008
  try {
    Get-WinEvent -FilterHashtable @{ LogName = 'Microsoft-Windows-Windows Defender/Operational'; StartTime = $Since } -ErrorAction SilentlyContinue |
      Where-Object { $ids -contains $_.Id } |
      ForEach-Object { [pscustomobject]@{ id = $_.Id; time = $_.TimeCreated.ToString('o');
                                          line = (($_.Message -split "`r?`n") | Select-Object -First 1) } }
  } catch { @() }
}

# ---- build / reuse stub unsloth.exe and repoint the launcher at it ----
if (-not $StubExe -or -not (Test-Path $StubExe)) {
  $StubExe = Join-Path $env:RUNNER_TEMP "unsloth.exe"
  try { New-StubUnslothExe -Path $StubExe } catch { Write-Host "[warn] stub compile failed: $($_.Exception.Message)" }
}
# Point launch-studio.ps1's $studioExe at the stub and clear the install-id gate
# (empty = accept any healthy backend) so the chain completes deterministically.
$ps1 = Join-Path $VariantDir "launch-studio.ps1"
if (Test-Path $ps1) {
  $txt = Get-Content $ps1 -Raw
  $txt = [regex]::Replace($txt, "(?m)^\s*\`$studioExe\s*=\s*'.*?'", "    `$studioExe = '$($StubExe -replace "'","''")'")
  $txt = [regex]::Replace($txt, "(?m)^\s*\`$_ExpectedStudioRootId\s*=\s*'.*?'", "`$_ExpectedStudioRootId = ''")
  Set-Content -LiteralPath $ps1 -Value $txt -Encoding UTF8
}

# materialize the .lnk and resolve how the shortcut launches
$sc = Get-Content (Join-Path $VariantDir "shortcut.json") -Raw | ConvertFrom-Json
$target = [Environment]::ExpandEnvironmentVariables($sc.target)
$argline = $sc.args

# Ensure Defender is live (real-time + behavior + cloud) so a detection can fire.
try { Set-Service WinDefend -StartupType Manual -ErrorAction SilentlyContinue; Start-Service WinDefend -ErrorAction SilentlyContinue } catch {}
try {
  Set-MpPreference -DisableRealtimeMonitoring $false -ErrorAction SilentlyContinue
  Set-MpPreference -DisableBehaviorMonitoring $false -ErrorAction SilentlyContinue
  Set-MpPreference -MAPSReporting Advanced -SubmitSamplesConsent SendAllSamples -ErrorAction SilentlyContinue
} catch {}
Update-MpSignature -ErrorAction SilentlyContinue
$st = Get-MpComputerStatus -ErrorAction SilentlyContinue
Write-Host "Defender AM service:        $($st.AMServiceEnabled)"
Write-Host "Defender real-time enabled: $($st.RealTimeProtectionEnabled)"
Write-Host "Defender behavior monitor:  $($st.BehaviorMonitorEnabled)"

$runs = @()
for ($i = 1; $i -le $Iterations; $i++) {
  $t0 = Get-Date
  $before = Get-Detections
  try {
    if ($argline) {
      $proc = Start-Process -FilePath $target -ArgumentList $argline -WorkingDirectory $VariantDir -PassThru -ErrorAction Stop
    } else {
      $proc = Start-Process -FilePath $target -WorkingDirectory $VariantDir -PassThru -ErrorAction Stop
    }
  } catch {
    Write-Host "iter ${i}: launch error: $($_.Exception.Message)"
  }
  Start-Sleep -Seconds $WatchSeconds
  # Tear down the (possibly hidden, possibly orphaned) launch tree.
  if ($proc) { taskkill /T /F /PID $proc.Id 2>$null | Out-Null }
  Get-Process wscript, WScript, powershell, pwsh, conhost -ErrorAction SilentlyContinue |
    Where-Object { $_.StartTime -ge $t0 } | Stop-Process -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 3

  $after = Get-Detections
  $events = @(Get-DefenderEvents -Since $t0)
  $newDet = @($after | Where-Object {
    $a = $_; -not ($before | Where-Object { $_.time -eq $a.time -and $_.name -eq $a.name }) })
  $hardEvents = @($events | Where-Object { 1006,1015,1116,1117,1121,1122 -contains $_.id })
  $detected = ($newDet.Count -gt 0) -or ($hardEvents.Count -gt 0)
  $runs += [pscustomobject]@{
    iteration = $i; detected = $detected
    new_detections = $newDet; defender_events = $events
  }
  Write-Host ("iter {0}: detected={1}  new_detections={2}  defender_events={3}" -f `
    $i, $detected, $newDet.Count, $events.Count)
}

$summary = [pscustomobject]@{
  variant = (Split-Path $VariantDir -Leaf)
  target = $target; args = $argline
  iterations = $Iterations
  detected_any = [bool](@($runs | Where-Object { $_.detected }).Count -gt 0)
  detected_count = @($runs | Where-Object { $_.detected }).Count
  runs = $runs
}
$summary | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath $OutFile -Encoding UTF8
Write-Host ("=== {0}: detected in {1}/{2} launches ===" -f `
  $summary.variant, $summary.detected_count, $Iterations)
exit 0
