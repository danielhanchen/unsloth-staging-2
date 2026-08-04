# Fail fast if any executable inside a Windows bundle is unsigned.
#
# Signing the installer is not the same as signing what the installer drops on
# disk. NSIS extracts its plugin DLLs to $PLUGINSDIR at install time and runs
# them; an unsigned DLL appearing in a temp directory and being loaded by an
# installer is a well-known false-positive trigger for several AV engines
# (tauri-apps/tauri#11673, tauri-apps/nsis-tauri-utils#37).
#
# Shipping 0.1.512-beta, four of the five plugin DLLs were unsigned and nobody
# knew, because nothing looked inside the installer. This does.
#
# Reports EVERY unsigned file rather than stopping at the first, so one run
# tells you the whole list to fix.

param(
    # Bundles to check. Each is unpacked and every PE inside is verified.
    [Parameter(Mandatory = $true)][string[]] $Path,
    # 7-Zip executable. Preinstalled on windows-latest.
    [string] $SevenZip = '7z',
    # Files that are known-unsigned and accepted, by leaf name. Keep this empty
    # if you can: every entry is a file users' AV will see unsigned.
    [string[]] $Allow = @()
)

$ErrorActionPreference = 'Continue'
$exeExtensions = @('.exe', '.dll', '.sys', '.ocx', '.cpl', '.scr')

$unsigned = @()
$checked = 0

foreach ($bundle in $Path) {
    if (-not (Test-Path $bundle -PathType Leaf)) {
        Write-Host "::error::bundle not found: $bundle"
        exit 1
    }
    $name = Split-Path $bundle -Leaf
    Write-Host ''
    Write-Host "=== $name ==="

    # The bundle itself first.
    $sig = Get-AuthenticodeSignature $bundle
    $checked++
    if ($sig.Status -ne 'Valid') {
        Write-Host "  UNSIGNED  $name  ($($sig.Status))"
        $unsigned += [pscustomobject]@{ Bundle = $name; File = $name; Status = [string]$sig.Status }
    } else {
        Write-Host "  signed    $name  <- $($sig.SignerCertificate.Subject)"
    }

    $dest = Join-Path $env:RUNNER_TEMP ("sigcheck-" + [System.IO.Path]::GetFileNameWithoutExtension($name))
    Remove-Item $dest -Recurse -Force -ErrorAction SilentlyContinue
    & $SevenZip x -y "-o$dest" $bundle | Out-Null
    if (-not (Test-Path $dest)) {
        Write-Host "::error::could not unpack $name; cannot verify its contents"
        exit 1
    }

    $inner = Get-ChildItem $dest -Recurse -File |
        Where-Object { $exeExtensions -contains $_.Extension.ToLower() }
    if (-not $inner) {
        Write-Host "::warning::no executable payload found inside $name"
    }

    foreach ($f in ($inner | Sort-Object Name)) {
        $checked++
        $s = Get-AuthenticodeSignature $f.FullName
        if ($s.Status -eq 'Valid') {
            Write-Host ("  signed    {0}" -f $f.Name)
        } elseif ($Allow -contains $f.Name) {
            Write-Host ("  ALLOWED   {0}  ({1}) - explicitly accepted as unsigned" -f $f.Name, $s.Status)
        } else {
            Write-Host ("  UNSIGNED  {0}  ({1})" -f $f.Name, $s.Status)
            $unsigned += [pscustomobject]@{ Bundle = $name; File = $f.Name; Status = [string]$s.Status }
        }
    }
    Remove-Item $dest -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ''
Write-Host "checked $checked file(s) across $($Path.Count) bundle(s)"

if (-not $unsigned) {
    Write-Host 'Every executable in every bundle is validly signed.'
    exit 0
}

Write-Host ''
Write-Host '================ UNSIGNED FILES ================'
$unsigned | Format-Table Bundle, File, Status -AutoSize | Out-String | Write-Host
foreach ($u in $unsigned) {
    Write-Host "::error file=$($u.File)::$($u.File) in $($u.Bundle) is $($u.Status) and needs signing"
}
Write-Host ''
Write-Host 'These ship inside the installer and are written to disk on the user machine.'
Write-Host 'If they are NSIS plugin DLLs, see .github/workflows/release-desktop.yml for the'
Write-Host 'plugin-signing note: tauri-bundler signs its copy of the plugins but no NSIS'
Write-Host 'template references the NSISPLUGINS directory it puts them in, so only'
Write-Host 'nsis_tauri_utils.dll (reached via ADDITIONALPLUGINSPATH) ends up signed.'
exit 1
