# Scan every file under -Path with Defender and emit one structured verdict each.
#
# Two scans per file, because neither alone is sufficient:
#
#   MpCmdRun -Scan -DisableRemediation
#       Ignores exclusion paths entirely (documented), so it still works even if
#       Remove-MpPreference was blocked -- but it writes no event log entries and
#       does not populate Get-MpThreat*, so its only output is stdout.
#
#   Start-MpScan -ScanType CustomScan
#       Honours exclusions (hence the removal in defender-enable.ps1) but does
#       populate the object model and the event log.
#
# Exit codes are NOT a clean signal on their own: per Microsoft's MpCmdRun
# reference, 0 means "no malware OR malware remediated" and 2 means "malware
# found and not remediated OR user action required OR the scan failed". Because
# -DisableRemediation is passed, remediation never happens, so 2 collapses to
# "detected or scan error" -- still ambiguous enough that the text output and
# the threat objects are correlated below rather than trusted blindly.
#
# Caveat that shapes the whole experiment: "!ml" verdicts are cloud/FastPath
# detections that fire on real-time write/open/execute events. A clean result
# here is therefore NOT proof of absence -- the download-survival check and the
# execution test are the stronger signals.

param(
    [Parameter(Mandatory = $true)][string] $Path,
    [Parameter(Mandatory = $true)][string] $OutJson,
    [string] $Filter = '*'
)

$ErrorActionPreference = 'Continue'

$mp = $env:MPCMDRUN
if (-not $mp -or -not (Test-Path $mp)) { $mp = "$env:ProgramFiles\Windows Defender\MpCmdRun.exe" }

$targets = if (Test-Path $Path -PathType Leaf) {
    @(Get-Item $Path)
} else {
    @(Get-ChildItem -LiteralPath $Path -Filter $Filter -File -Recurse)
}

$results = @()
foreach ($t in $targets) {
    $f = $t.FullName
    Write-Host ''
    Write-Host "=============================================================="
    Write-Host "SCAN: $f  ($($t.Length) bytes)"
    Write-Host "=============================================================="

    $sig = Get-AuthenticodeSignature $f -ErrorAction SilentlyContinue

    $out  = & $mp -Scan -ScanType 3 -File $f -DisableRemediation 2>&1
    $code = $LASTEXITCODE
    $text = ($out | Out-String)
    $out | ForEach-Object { Write-Host "  $_" }
    Write-Host "  exit=$code"

    $names = [regex]::Matches(
        $text,
        '((?:Trojan|Virus|Worm|Backdoor|Ransom|HackTool|PUA|PUADlManager|Behavior|TrojanDownloader|TrojanDropper|Program|Misleading|Exploit)[:\w\./!\-]+)'
    ) | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique

    $isClean   = [bool]($text -match 'found no threats')
    $isSkipped = [bool]($text -match 'was skipped')
    $isDetect  = [bool](($text -match 'found\s+\d+\s+threat') -or ($names.Count -gt 0))

    $classification =
        if ($isDetect)        { 'detected' }
        elseif ($isSkipped)   { 'skipped' }
        elseif ($isClean)     { 'clean' }
        elseif ($code -eq 2)  { 'scan-error' }
        else                  { 'inconclusive' }

    # Second pass so the detection lands in the object model + event log.
    Start-MpScan -ScanPath $f -ScanType CustomScan -ErrorAction SilentlyContinue

    $r = [pscustomobject]@{
        file            = $t.Name
        fullPath        = $f
        sizeBytes       = $t.Length
        sha256          = (Get-FileHash -Algorithm SHA256 $f).Hash
        signatureStatus = [string]$sig.Status
        signer          = [string]$sig.SignerCertificate.Subject
        mpCmdRunExit    = $code
        classification  = $classification
        threatNames     = @($names)
        isWacatac       = [bool]($text -match 'Wacatac')
        rawOutput       = $text
    }
    $r | Select-Object * -ExcludeProperty rawOutput | Format-List | Out-String | Write-Host
    $results += $r
}

$results | ConvertTo-Json -Depth 5 | Out-File $OutJson -Encoding utf8

Write-Host ''
Write-Host '================ VERDICT TABLE ================'
$results | Format-Table file, sizeBytes, signatureStatus, mpCmdRunExit, classification, isWacatac -AutoSize |
    Out-String | Write-Host

$hits = @($results | Where-Object { $_.classification -eq 'detected' })
if ($hits) {
    Write-Host "::warning::Defender detected: $(($hits | ForEach-Object { "$($_.file) [$($_.threatNames -join ',')]" }) -join '; ')"
}
