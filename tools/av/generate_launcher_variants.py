#!/usr/bin/env python3
"""Generate Unsloth Studio Windows launcher variants (V0 = current install.ps1
output, V1..V5 = hardening candidates) for antivirus scanning. The .ps1 body is
lifted from install.ps1 so we scan the real launcher, not a paraphrase.

  python generate_launcher_variants.py --install-ps1 <path> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Representative paths (heuristics key on structure, not the concrete path).
APP_DIR = r"C:\Users\runner\AppData\Local\Unsloth Studio"
PS1_PATH = APP_DIR + r"\launch-studio.ps1"
VBS_PATH = APP_DIR + r"\launch-studio.vbs"
EXE_PATH = r"C:\Users\runner\.unsloth\studio\.venv\Scripts\unsloth.exe"
LAUNCHER_EXE = APP_DIR + r"\unsloth-studio-launcher.exe"
SYSROOT = r"%SystemRoot%"
FAKE_ROOT_ID = "a3f1c9d24b6e8071f5029c4d7e1b8a36c0d9e2f4a7b1c3d5e6f80a1b2c3d4e5f6"

# Non-env-mode value of $studioHomeExport (install.ps1:600-602), already expanded.
STUDIO_HOME_EXPORT_DEFAULT = (
    "$portFile = $null\n"
    "$mutexName = 'Local\\UnslothStudioLauncher'\n"
)


def _ps_herestring_unescape(s: str) -> str:
    """Undo PowerShell expandable here-string backtick escapes (`$ -> $, etc.)."""
    out = []
    i = 0
    ctrl = {"n": "\n", "r": "\r", "t": "\t", "0": "\0",
            "a": "\a", "b": "\b", "f": "\f", "v": "\v"}
    while i < len(s):
        c = s[i]
        if c == "`" and i + 1 < len(s):
            nxt = s[i + 1]
            out.append(ctrl.get(nxt, nxt))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def extract_launcher_ps1(install_ps1: str) -> str:
    """Render the runtime launch-studio.ps1 from install.ps1's here-string."""
    lines = install_ps1.splitlines()
    start = end = None
    for idx, line in enumerate(lines):
        if start is None and '$launcherContent = @"' in line:
            start = idx + 1
        elif start is not None and line.strip() == '"@':
            end = idx
            break
    if start is None or end is None:
        raise RuntimeError("could not locate $launcherContent here-string in install.ps1")
    body = "\n".join(lines[start:end]) + "\n"
    # Fill the generation-time interpolations, then unescape the deferred (`$...).
    body = body.replace("$studioHomeExport", STUDIO_HOME_EXPORT_DEFAULT, 1)
    body = body.replace("'$_studioRootId'", f"'{FAKE_ROOT_ID}'")
    body = body.replace("'$SingleQuotedExePath'", f"'{EXE_PATH}'")
    body = _ps_herestring_unescape(body)
    return body


def vbs_content(ps1_path: str, *, bypass: bool, hidden: bool) -> str:
    flags = ["-NoProfile"]
    if bypass:
        flags += ["-ExecutionPolicy", "Bypass"]
    if hidden:
        flags += ["-WindowStyle", "Hidden"]
    flag_str = " ".join(flags)
    # Inner path is wrapped in VBS-doubled quotes ("") exactly as install.ps1 does.
    return (
        'Set shell = CreateObject("WScript.Shell")\r\n'
        f'cmd = "powershell {flag_str} -File ""{ps1_path}"""\r\n'
        "shell.Run cmd, 0, False\r\n"
    )


def write_utf16le_bom(path: Path, text: str) -> None:
    # Matches install.ps1 Set-Content -Encoding Unicode (UTF-16LE + BOM).
    path.write_bytes(b"\xff\xfe" + text.encode("utf-16-le"))


def write_utf8_bom(path: Path, text: str) -> None:
    # Matches install.ps1 New-Object System.Text.UTF8Encoding($true).
    path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))


VARIANTS = {
    "V0-baseline": dict(
        desc="Current install.ps1 output (control; reproduce the Kaspersky flag).",
        vbs=dict(bypass=True, hidden=True),
        shortcut=dict(target=fr"{SYSROOT}\System32\wscript.exe",
                      args=f'//B //Nologo "{VBS_PATH}"', window="Normal"),
    ),
    "V1-vbs-no-windowstyle": dict(
        desc="VBS, drop redundant -WindowStyle Hidden (shell.Run ,0, already hides).",
        vbs=dict(bypass=True, hidden=False),
        shortcut=dict(target=fr"{SYSROOT}\System32\wscript.exe",
                      args=f'//B //Nologo "{VBS_PATH}"', window="Normal"),
    ),
    "V2-vbs-no-bypass": dict(
        desc="VBS, drop -WindowStyle Hidden and -ExecutionPolicy Bypass.",
        vbs=dict(bypass=False, hidden=False),
        shortcut=dict(target=fr"{SYSROOT}\System32\wscript.exe",
                      args=f'//B //Nologo "{VBS_PATH}"', window="Normal"),
    ),
    "V3-lnk-powershell-hidden": dict(
        desc="No VBS; .lnk -> powershell -WindowStyle Hidden -File ps1, link minimized.",
        vbs=None,
        shortcut=dict(
            target=fr"{SYSROOT}\System32\WindowsPowerShell\v1.0\powershell.exe",
            args=f'-NoProfile -WindowStyle Hidden -File "{PS1_PATH}"',
            window="Minimized"),
    ),
    "V4-lnk-conhost-headless": dict(
        desc="No VBS; .lnk -> conhost.exe --headless powershell -File ps1.",
        vbs=None,
        shortcut=dict(
            target=fr"{SYSROOT}\System32\conhost.exe",
            args=f'--headless powershell.exe -NoProfile -File "{PS1_PATH}"',
            window="Normal"),
    ),
    "V5-native-launcher": dict(
        desc="No VBS/PS invoke; .lnk -> signed native GUI-subsystem launcher exe.",
        vbs=None,
        shortcut=dict(target=LAUNCHER_EXE, args="", window="Normal"),
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--install-ps1", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    install_text = args.install_ps1.read_text(encoding="utf-8-sig")
    launcher_ps1 = extract_launcher_ps1(install_text)

    manifest = {}
    for name, spec in VARIANTS.items():
        vdir = args.out / name
        vdir.mkdir(parents=True, exist_ok=True)
        files = []
        write_utf8_bom(vdir / "launch-studio.ps1", launcher_ps1)
        files.append("launch-studio.ps1")
        if spec["vbs"] is not None:
            write_utf16le_bom(vdir / "launch-studio.vbs",
                              vbs_content(PS1_PATH, **spec["vbs"]))
            files.append("launch-studio.vbs")
        (vdir / "shortcut.json").write_text(
            json.dumps(spec["shortcut"], indent=2), encoding="utf-8")
        manifest[name] = {"desc": spec["desc"], "files": files,
                          "shortcut": spec["shortcut"]}

    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(VARIANTS)} variants to {args.out}")
    for name in VARIANTS:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
