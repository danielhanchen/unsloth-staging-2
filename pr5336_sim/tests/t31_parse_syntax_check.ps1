$ErrorActionPreference = 'Stop'

# AST parse of the actual .ps1 files in the repo — picks up syntax errors that
# wouldn't surface from unit tests of extracted helpers.

# Walk up from $PSScriptRoot until install.ps1 is found.
$repoRoot = $PSScriptRoot
for ($i = 0; $i -lt 6; $i++) {
    if (Test-Path (Join-Path $repoRoot 'install.ps1')) { break }
    $repoRoot = Split-Path -Parent $repoRoot
}
$files = @(
    Join-Path $repoRoot 'install.ps1'
    Join-Path $repoRoot 'studio/setup.ps1'
)

$failed = $false
foreach ($f in $files) {
    if (-not (Test-Path $f)) { throw "Not found: $f" }
    $errors = $null
    $tokens = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile(
        (Resolve-Path $f), [ref]$tokens, [ref]$errors)
    if ($errors -and $errors.Count -gt 0) {
        $failed = $true
        Write-Host "FAIL parse: $f" -ForegroundColor Red
        $errors | ForEach-Object { Write-Host "  $($_.Message) @ $($_.Extent.StartLineNumber):$($_.Extent.StartColumnNumber)" }
    } else {
        Write-Host "OK parse: $f"
    }
}
if ($failed) { throw "PowerShell parse errors detected" }

"PASS t31_parse_syntax_check"
