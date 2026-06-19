# Windows-native validation of Test-HipinfoIsVenvInternal (from install.ps1).
# Covers the drive-root guard (Fix C) and case-insensitive containment using REAL
# C:\ path semantics (GetFullPath) that Linux pwsh cannot reproduce. Runs on a
# windows-latest CI runner. No real GPU/hipinfo needed -- the helper is pure path logic.
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path "$PSScriptRoot/../..").Path
$ins  = Get-Content -Raw (Join-Path $repo "install.ps1")

# Extract the Test-HipinfoIsVenvInternal function body by brace matching, then define it.
$start = $ins.IndexOf("function Test-HipinfoIsVenvInternal {")
if ($start -lt 0) { Write-Host "FAIL: function not found in install.ps1"; exit 1 }
$depth = 0; $end = -1
for ($i = $ins.IndexOf("{", $start); $i -lt $ins.Length; $i++) {
    if ($ins[$i] -eq '{') { $depth++ }
    elseif ($ins[$i] -eq '}') { $depth--; if ($depth -eq 0) { $end = $i; break } }
}
Invoke-Expression $ins.Substring($start, $end - $start + 1)

$pass = 0; $fail = 0
function Check($l, $c) { if ($c) { Write-Host "  PASS: $l"; $script:pass++ } else { Write-Host "  FAIL: $l"; $script:fail++ } }

# Neutralize the other venv-root seeds so each case is isolated.
$env:UNSLOTH_STUDIO_HOME = $null; $env:STUDIO_HOME = $null; $env:VenvDir = $null; $env:UNSLOTH_SETUP_PYTHON = $null

Write-Host "=== containment + casing (real Windows paths) ==="
$env:VIRTUAL_ENV = "C:\Users\runneradmin\.unsloth\studio\unsloth_studio"
Check "venv-internal hipInfo -> True"  (Test-HipinfoIsVenvInternal "C:\Users\runneradmin\.unsloth\studio\unsloth_studio\Scripts\hipInfo.exe")
Check "external C: SDK -> False"       (-not (Test-HipinfoIsVenvInternal "C:\Program Files\AMD\ROCm\6.2\bin\hipinfo.exe"))
$env:VIRTUAL_ENV = "C:\USERS\RUNNERADMIN\.UNSLOTH\STUDIO\UNSLOTH_STUDIO"
Check "case-insensitive containment -> True" (Test-HipinfoIsVenvInternal "c:\users\runneradmin\.unsloth\studio\unsloth_studio\Scripts\hipInfo.exe")
$env:VIRTUAL_ENV = $null

Write-Host "=== Fix C: bare drive-root UNSLOTH_SETUP_PYTHON must not match the whole drive ==="
# C:\Python312\python.exe -> parent-of-parent = C:\ ; without the guard every C: path matches.
$env:UNSLOTH_SETUP_PYTHON = "C:\Python312\python.exe"
Check "drive-root setup-python: external C: SDK NOT venv-internal" (-not (Test-HipinfoIsVenvInternal "C:\Program Files\AMD\ROCm\6.2\bin\hipinfo.exe"))
# A genuine venv via UNSLOTH_SETUP_PYTHON is still caught.
$env:UNSLOTH_SETUP_PYTHON = "C:\Users\runneradmin\.unsloth\studio\unsloth_studio\Scripts\python.exe"
Check "real venv setup-python: its hipInfo IS venv-internal" (Test-HipinfoIsVenvInternal "C:\Users\runneradmin\.unsloth\studio\unsloth_studio\Scripts\hipInfo.exe")
$env:UNSLOTH_SETUP_PYTHON = $null

Write-Host ""
Write-Host "Results: $pass passed, $fail failed"
if ($fail -ne 0) { exit 1 }
