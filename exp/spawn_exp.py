import os, subprocess, sys, tempfile
from pathlib import Path
PS = Path(os.environ["SystemRoot"]) / r"System32\WindowsPowerShell\v1.0\powershell.exe"
DET, NOWIN = 0x8, 0x08000000
probe = r'''$ErrorActionPreference = "Stop"
$c = $true
try { $null = $Host.UI.RawUI.BufferSize } catch { $c = $false }
[Console]::Error.WriteLine("console_attached=" + $c)
Write-Output "stdout_line"
'''
d = tempfile.mkdtemp(); p = Path(d) / "probe.ps1"; p.write_bytes(probe.replace("\n", "\r\n").encode("ascii"))
flags = ["-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "RemoteSigned"]
cases = {
  "DET_file_stdin_inherit": dict(args=[*flags, "-File", str(p)], cf=DET, stdin=None),
  "DET_file_stdin_devnull": dict(args=[*flags, "-File", str(p)], cf=DET, stdin=subprocess.DEVNULL),
  "DET_file_stdin_pipe":    dict(args=[*flags, "-File", str(p)], cf=DET, stdin=subprocess.PIPE),
  "DET_cmd_stdin_devnull":  dict(args=[*flags, "-Command", f"& '{p}'"], cf=DET, stdin=subprocess.DEVNULL),
  "DET_file_noNonInteractive_devnull": dict(args=["-NoLogo","-NoProfile","-ExecutionPolicy","RemoteSigned","-File",str(p)], cf=DET, stdin=subprocess.DEVNULL),
  "DET_inputformat_none":   dict(args=[*flags, "-InputFormat", "None", "-File", str(p)], cf=DET, stdin=subprocess.DEVNULL),
  "NOWIN_file":             dict(args=[*flags, "-File", str(p)], cf=NOWIN, stdin=subprocess.DEVNULL),
}
for name, c in cases.items():
    try:
        r = subprocess.run([str(PS), *c["args"]], stdin=c["stdin"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=c["cf"], timeout=120)
        print(f"EXP {name}: rc={r.returncode} stdout={r.stdout[:200]!r} stderr={r.stderr[:300]!r}")
    except Exception as e:
        print(f"EXP {name}: EXC {e!r}")
