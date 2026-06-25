# Cross-OS validation for the UNSLOTH_NPM_REGISTRY change (unsloth PR #6663 / issue #6491).
# Exercises the $NpmRegistryArgs splat and the Show-NpmRegistryHint helper extracted from
# the real studio/setup.ps1, on whatever PowerShell the runner ships (Windows PowerShell
# 5.1 and/or pwsh 7). Does NOT run a full install; it validates the new logic only.
$ErrorActionPreference = 'Continue'
Write-Host "### PowerShell $($PSVersionTable.PSVersion) Edition=$($PSVersionTable.PSEdition) ###"

$script:fail = 0
function ok($m)  { Write-Host "PASS: $m" }
function bad($m) { Write-Host "FAIL: $m"; $script:fail = 1 }

$setupPath = (Resolve-Path 'studio/setup.ps1').Path

# 1. Script parses under this PowerShell engine.
$perr = $null
[void][System.Management.Automation.Language.Parser]::ParseFile($setupPath, [ref]$null, [ref]$perr)
if ($perr -and $perr.Count) { bad "setup.ps1 parse: $($perr[0].Message)" } else { ok "setup.ps1 parses" }

# 2. Extract Show-NpmRegistryHint via the AST and load it with stubbed output helpers.
$ast = [System.Management.Automation.Language.Parser]::ParseFile($setupPath, [ref]$null, [ref]$null)
$fn  = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Show-NpmRegistryHint' }, $true)
if ($fn.Count -eq 1) { ok "found Show-NpmRegistryHint" } else { bad "Show-NpmRegistryHint not found ($($fn.Count))" }
function step    { param($l,$v,$c) Write-Host "  $l $v" }
function substep { param($m,$c)    Write-Host "  $m" }
if ($fn.Count -ge 1) { . ([scriptblock]::Create($fn[0].Extent.Text)) }

# 3a. Opted in (UNSLOTH_NPM_REGISTRY set) -> stays silent.
# Write-Host writes to the Information stream (6); capture it with 6>&1.
$env:UNSLOTH_NPM_REGISTRY = 'https://m/'
$out = (Show-NpmRegistryHint 6>&1 | Out-String)
if ([string]::IsNullOrWhiteSpace($out)) { ok "opted-in: silent" } else { bad "opted-in not silent: $out" }

# 3b. Mirror in NPM_CONFIG_REGISTRY env -> surfaces the mirror.
$env:UNSLOTH_NPM_REGISTRY = ''
$env:NPM_CONFIG_REGISTRY  = 'https://mirror.corp/api/npm/'
$out = (Show-NpmRegistryHint 6>&1 | Out-String)
if ($out -match 'mirror\.corp') { ok "block+env mirror: detected" } else { bad "env mirror not detected: $out" }
$env:NPM_CONFIG_REGISTRY = ''

# 4. $NpmRegistryArgs derivation + splat behavior (the exact pattern used at the call sites).
$env:UNSLOTH_NPM_REGISTRY = ''
$NpmRegistryArgs = @(); if ($env:UNSLOTH_NPM_REGISTRY) { $NpmRegistryArgs = @('--registry', $env:UNSLOTH_NPM_REGISTRY) }
if ($NpmRegistryArgs.Count -eq 0) { ok "unset -> empty args" } else { bad "unset -> $($NpmRegistryArgs.Count) args" }

$env:UNSLOTH_NPM_REGISTRY = 'https://corp/api/npm/'
$NpmRegistryArgs = @(); if ($env:UNSLOTH_NPM_REGISTRY) { $NpmRegistryArgs = @('--registry', $env:UNSLOTH_NPM_REGISTRY) }
function fake { $args -join '|' }
$splat = (fake @NpmRegistryArgs)
if ($splat -eq '--registry|https://corp/api/npm/') { ok "set -> splat '--registry <url>'" } else { bad "splat -> '$splat'" }
$env:UNSLOTH_NPM_REGISTRY = ''

if ($script:fail -ne 0) { Write-Host "=== RESULT: FAILURES ==="; exit 1 } else { Write-Host "=== RESULT: ALL PASS ==="; exit 0 }
