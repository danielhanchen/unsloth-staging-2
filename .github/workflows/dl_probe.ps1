$ErrorActionPreference = 'Continue'
$url    = 'https://github.com/unslothai/unsloth/releases/download/v0.1.70-beta/Unsloth-Desktop-0_1_70_beta-Windows.exe'
$urlLin = 'https://github.com/unslothai/unsloth/releases/download/v0.1.70-beta/Unsloth-Desktop-0_1_70_beta-Linux.AppImage'
$tmp    = $env:RUNNER_TEMP
if (-not $tmp) { $tmp = $env:TEMP }

function Report($name, $seconds, $path) {
    $mb = 0.0
    if (Test-Path $path) { $mb = (Get-Item $path).Length / 1MB }
    $rate = if ($seconds -gt 0) { [math]::Round($mb / $seconds, 2) } else { 0 }
    "RESULT`t$name`t$([math]::Round($seconds,2))s`t$rate MB/s`t$([math]::Round($mb,1)) MB"
}

# --- 1. curl.exe (native Windows client, 3 runs for variance) ---
foreach ($i in 1..3) {
    $out = "$tmp\curl_$i.exe"
    $line = curl.exe -sS -L -o $out -w "%{time_total} %{speed_download} %{time_connect} %{time_starttransfer}" $url
    $f = $line -split '\s+'
    "RESULT`tcurl.exe_run$i`t$($f[0])s`t$([math]::Round([double]$f[1]/1MB,2)) MB/s`tconnect=$($f[2]) ttfb=$($f[3])"
    Remove-Item $out -Force -ErrorAction SilentlyContinue
}

# --- 2. pwsh 7 Invoke-WebRequest, progress bar ON then OFF ---
$ProgressPreference = 'Continue'
$out = "$tmp\pwsh_prog_on.exe"
$sw = [Diagnostics.Stopwatch]::StartNew()
try { Invoke-WebRequest -Uri $url -OutFile $out } catch { "IWR(on) failed: $($_.Exception.Message)" }
$sw.Stop(); Report 'pwsh7_IWR_progress_on' $sw.Elapsed.TotalSeconds $out
Remove-Item $out -Force -ErrorAction SilentlyContinue

$ProgressPreference = 'SilentlyContinue'
$out = "$tmp\pwsh_prog_off.exe"
$sw = [Diagnostics.Stopwatch]::StartNew()
try { Invoke-WebRequest -Uri $url -OutFile $out } catch { "IWR(off) failed: $($_.Exception.Message)" }
$sw.Stop(); Report 'pwsh7_IWR_progress_off' $sw.Elapsed.TotalSeconds $out
Remove-Item $out -Force -ErrorAction SilentlyContinue

# --- 3. .NET HttpClient streamed to disk ---
$out = "$tmp\httpclient.exe"
$sw = [Diagnostics.Stopwatch]::StartNew()
try {
    $c = [System.Net.Http.HttpClient]::new()
    $c.Timeout = [TimeSpan]::FromMinutes(20)
    $s = $c.GetStreamAsync($url).GetAwaiter().GetResult()
    $fs = [System.IO.File]::Create($out)
    $s.CopyTo($fs, 1MB); $fs.Dispose(); $s.Dispose(); $c.Dispose()
} catch { ".NET HttpClient failed: $($_.Exception.Message)" }
$sw.Stop(); Report 'dotnet_HttpClient' $sw.Elapsed.TotalSeconds $out
Remove-Item $out -Force -ErrorAction SilentlyContinue

# --- 4. BITS (what many installers/updaters use) ---
$out = "$tmp\bits.exe"
$sw = [Diagnostics.Stopwatch]::StartNew()
try { Start-BitsTransfer -Source $url -Destination $out -ErrorAction Stop }
catch { "Start-BitsTransfer failed: $($_.Exception.Message)" }
$sw.Stop(); Report 'BITS' $sw.Elapsed.TotalSeconds $out
Remove-Item $out -Force -ErrorAction SilentlyContinue

# --- 5. exe vs non-exe control: same size, different asset/extension ---
$out = "$tmp\control.AppImage"
$line = curl.exe -sS -L -o $out -w "%{time_total} %{speed_download}" $urlLin
$f = $line -split '\s+'
"RESULT`tcurl.exe_LinuxAppImage_control`t$($f[0])s`t$([math]::Round([double]$f[1]/1MB,2)) MB/s"
Remove-Item $out -Force -ErrorAction SilentlyContinue

$out = "$tmp\as_txt.txt"
$line = curl.exe -sS -L -o $out -w "%{time_total} %{speed_download}" $url
$f = $line -split '\s+'
"RESULT`tcurl.exe_saved_as_.txt`t$($f[0])s`t$([math]::Round([double]$f[1]/1MB,2)) MB/s"
Remove-Item $out -Force -ErrorAction SilentlyContinue

# --- 6. Defender on-demand scan cost for this binary ---
$out = "$tmp\scan_target.exe"
curl.exe -sS -L -o $out $url | Out-Null
try {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    Start-MpScan -ScanPath $out -ScanType CustomScan -ErrorAction Stop
    $sw.Stop()
    "RESULT`tdefender_Start-MpScan_on_exe`t$([math]::Round($sw.Elapsed.TotalSeconds,2))s"
} catch { "Start-MpScan unavailable: $($_.Exception.Message)" }

# --- 7. Mark-of-the-Web / signature state of the binary ---
try {
    $sig = Get-AuthenticodeSignature $out
    "AUTHENTICODE status=$($sig.Status) signer=$($sig.SignerCertificate.Subject)"
} catch { "Get-AuthenticodeSignature failed: $($_.Exception.Message)" }
"SHA256: $((Get-FileHash $out -Algorithm SHA256).Hash)"

# --- 8. Where does the CDN send us + TTFB detail ---
$h = curl.exe -sS -o NUL -D - -L -w "`nHDR_time_total=%{time_total} redirects=%{num_redirects} ip=%{remote_ip}`n" $url
$h | Select-String -Pattern '^(HTTP|Location|Server|Content-Length|x-ms-|via|HDR_)' | ForEach-Object { $_.Line }
