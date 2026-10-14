param([string]$Tag)
$exe = Resolve-Path "dl/$Tag/Unsloth-Desktop-Windows.exe"
$p = Start-Process -FilePath $exe -ArgumentList '/S' -Wait -PassThru
Write-Host "installer $Tag exit: $($p.ExitCode)"
if ($p.ExitCode -ne 0) { exit 1 }
$found = Get-ChildItem -Path "$env:LOCALAPPDATA","$env:ProgramFiles" -Recurse -Filter 'unsloth-studio.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $found) { Write-Host '::error::no installed executable'; exit 1 }
Write-Host "installed: $($found.FullName) $($found.VersionInfo.ProductVersion)"
"APP_EXE=$($found.FullName)" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
$env:APP_EXE = $found.FullName
$ps1 = Get-ChildItem -Path (Split-Path -Parent $found.FullName) -Recurse -Filter 'install.ps1' -ErrorAction SilentlyContinue | Select-Object -First 1
Remove-Item Env:UNSLOTH_STUDIO_HOME -ErrorAction SilentlyContinue
& powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $ps1.FullName --tauri *>&1 |
  Tee-Object -FilePath "out/install-$Tag.log" | Select-Object -Last 15
Write-Host "bundled installer $Tag exit: $LASTEXITCODE"
exit $LASTEXITCODE
