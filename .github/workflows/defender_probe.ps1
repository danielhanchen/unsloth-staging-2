$ErrorActionPreference = 'Continue'
$url    = 'https://github.com/unslothai/unsloth/releases/download/v0.1.70-beta/Unsloth-Desktop-0_1_70_beta-Windows.exe'
$tmp    = $env:RUNNER_TEMP; if (-not $tmp) { $tmp = $env:TEMP }

function Status($tag) {
    try {
        $p = Get-MpComputerStatus
        "STATE`t$tag`tRTP=$($p.RealTimeProtectionEnabled) OnAccess=$($p.OnAccessProtectionEnabled) Ioav=$($p.IoavProtectionEnabled) Tamper=$($p.IsTamperProtected)"
    } catch { "STATE`t$tag`tunavailable: $($_.Exception.Message)" }
}

function Dl($name, $dest) {
    $line = curl.exe -sS -L -o $dest -w "%{time_total} %{speed_download}" $url
    $f = $line -split '\s+'
    "RESULT`t$name`t$($f[0])s`t$([math]::Round([double]$f[1]/1MB,2)) MB/s"
    Remove-Item $dest -Force -ErrorAction SilentlyContinue
}

Status 'initial'

# ---- A: real-time protection OFF (runner default) ----
foreach ($i in 1..2) { Dl "A_rtpOFF_exe_run$i" "$tmp\a_$i.exe" }

# ---- Turn real-time protection ON (what every real user has) ----
try {
    Set-MpPreference -DisableRealtimeMonitoring $false -ErrorAction Stop
    Set-MpPreference -DisableIOAVProtection $false -ErrorAction SilentlyContinue
    Set-MpPreference -DisableScanningNetworkFiles $false -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 5
} catch { "Set-MpPreference failed: $($_.Exception.Message)" }
Status 'after-enable'

# ---- B: real-time protection ON ----
foreach ($i in 1..3) { Dl "B_rtpON_exe_run$i" "$tmp\b_$i.exe" }
Dl 'B_rtpON_saved_as_.txt' "$tmp\b_control.txt"

# ---- C: real-time ON but the folder excluded (isolates Defender from network) ----
$exdir = "$tmp\excluded"
New-Item -ItemType Directory -Force -Path $exdir | Out-Null
try {
    Add-MpPreference -ExclusionPath $exdir -ErrorAction Stop
    Start-Sleep -Seconds 3
    foreach ($i in 1..2) { Dl "C_rtpON_excluded_run$i" "$exdir\c_$i.exe" }
    Remove-MpPreference -ExclusionPath $exdir -ErrorAction SilentlyContinue
} catch { "Exclusion path setup failed: $($_.Exception.Message)" }

# ---- D: on-access scan cost -- copy + hash the file with RTP on ----
$keep = "$tmp\keep.exe"
curl.exe -sS -L -o $keep $url | Out-Null
$sw = [Diagnostics.Stopwatch]::StartNew(); Copy-Item $keep "$tmp\keep_copy.exe" -Force; $sw.Stop()
"RESULT`tD_copy_exe_rtpON`t$([math]::Round($sw.Elapsed.TotalSeconds,2))s"
$sw = [Diagnostics.Stopwatch]::StartNew(); Get-FileHash $keep -Algorithm SHA256 | Out-Null; $sw.Stop()
"RESULT`tD_hash_exe_rtpON`t$([math]::Round($sw.Elapsed.TotalSeconds,2))s"

# ---- E: Mark-of-the-Web, what a BROWSER download attaches ----
# curl/IWR do not set Zone.Identifier; Edge/Chrome do, and it is what drives the
# SmartScreen reputation check on first launch.
try {
    Set-Content -Path "$keep`:Zone.Identifier" -Value "[ZoneTransfer]`r`nZoneId=3`r`nHostUrl=$url" -ErrorAction Stop
    "MOTW attached: $(Get-Content "$keep`:Zone.Identifier" -ErrorAction SilentlyContinue | Out-String)"
    $sw = [Diagnostics.Stopwatch]::StartNew(); Get-AuthenticodeSignature $keep | Out-Null; $sw.Stop()
    "RESULT`tE_authenticode_with_MOTW`t$([math]::Round($sw.Elapsed.TotalSeconds,2))s"
} catch { "MOTW test failed: $($_.Exception.Message)" }

Status 'final'
