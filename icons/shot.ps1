param([string]$Name)
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
(New-Object -ComObject Shell.Application).MinimizeAll()
Start-Sleep -Seconds 3
$b = [System.Windows.Forms.SystemInformation]::VirtualScreen
$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Left, $b.Top, 0, 0, $bmp.Size)
New-Item -ItemType Directory -Force -Path shots | Out-Null
$bmp.Save((Join-Path (Resolve-Path shots) "$Name.png"), [System.Drawing.Imaging.ImageFormat]::Png)
"SHOT $Name $($b.Width)x$($b.Height)"
