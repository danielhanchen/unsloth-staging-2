# A/B: the exact download call install.ps1 / setup.ps1 make, on the real production URLs,
# under Windows PowerShell 5.1 -- progress bar at its default versus SilentlyContinue.
#
# Two things are asserted, not just measured:
#   1. speed   -- the point of the change
#   2. IDENTITY -- the bytes fetched must be SHA256-identical between the two arms, so the
#      change is purely a rendering preference and can never alter what gets installed.
$ErrorActionPreference = 'Continue'
$tmp = $env:RUNNER_TEMP; if (-not $tmp) { $tmp = $env:TEMP }
$fail = 0

"PSVersion: $($PSVersionTable.PSVersion)  (5.1 expected -- what 'unsloth studio update' spawns)"

# The real URLs, from install.ps1:2220 / install.ps1:2553 / setup.ps1:1500.
$targets = @(
    @{ name = 'python.org-installer'; url = 'https://www.python.org/ftp/python/3.13.13/python-3.13.13-amd64.exe' },
    @{ name = 'uv-0.12.1-zip';        url = 'https://github.com/astral-sh/uv/releases/download/0.12.1/uv-x86_64-pc-windows-msvc.zip' },
    @{ name = 'vc_redist.x64.exe';    url = 'https://aka.ms/vs/17/release/vc_redist.x64.exe' }
)

foreach ($t in $targets) {
    $hashes = @{}
    $times  = @{}
    foreach ($arm in @('BEFORE_progress_default', 'AFTER_progress_silent')) {
        $dest = Join-Path $tmp ("ab_" + [guid]::NewGuid().ToString('N') + ".bin")
        $saved = $ProgressPreference
        $ProgressPreference = if ($arm -eq 'AFTER_progress_silent') { 'SilentlyContinue' } else { 'Continue' }

        $sw = [Diagnostics.Stopwatch]::StartNew()
        $err = ''
        try { Invoke-WebRequest -Uri $t.url -OutFile $dest -UseBasicParsing }
        catch { $err = $_.Exception.Message }
        $sw.Stop()
        $ProgressPreference = $saved

        $mb = 0.0; $sha = 'MISSING'
        if (Test-Path $dest) {
            $mb  = (Get-Item $dest).Length / 1MB
            $sha = (Get-FileHash $dest -Algorithm SHA256).Hash
        }
        $secs = [math]::Round($sw.Elapsed.TotalSeconds, 2)
        $rate = if ($secs -gt 0) { [math]::Round($mb / $secs, 2) } else { 0 }
        $hashes[$arm] = $sha; $times[$arm] = $secs
        "AB`t$($t.name)`t$arm`t${secs}s`t$rate MB/s`t$([math]::Round($mb,1)) MB`t$($sha.Substring(0,16))`t$err"
        Remove-Item $dest -Force -ErrorAction SilentlyContinue
    }

    # IDENTITY: same bytes out of both arms, or the change is not safe.
    if ($hashes['BEFORE_progress_default'] -eq $hashes['AFTER_progress_silent'] -and
        $hashes['BEFORE_progress_default'] -ne 'MISSING') {
        "IDENTITY`t$($t.name)`tPASS`tSHA256 identical across arms"
    } else {
        "IDENTITY`t$($t.name)`tFAIL`tbefore=$($hashes['BEFORE_progress_default']) after=$($hashes['AFTER_progress_silent'])"
        $fail = 1
    }
    $b = $times['BEFORE_progress_default']; $a = $times['AFTER_progress_silent']
    if ($a -gt 0) { "SPEEDUP`t$($t.name)`t$([math]::Round($b / $a, 1))x`t(${b}s -> ${a}s)" }
}

# SCOPE LEAK: install.ps1 assigns $ProgressPreference with no scope qualifier inside
# Install-UnslothStudio, so the caller's own preference must be untouched after it returns.
$before = $ProgressPreference
function Test-UnslothScoping { $ProgressPreference = 'SilentlyContinue'; return 'ran' }
$null = Test-UnslothScoping
if ($ProgressPreference -eq $before) { "SCOPE`tPASS`tcaller preference preserved ($before)" }
else { "SCOPE`tFAIL`tcaller preference changed $before -> $ProgressPreference"; $fail = 1 }

if ($fail -ne 0) { exit 1 }
