"""Harness only: do external programs started from a shell inside MXC run, and does their output come back?"""
import sys
import tempfile
from pathlib import Path

import shell_probe

CASES = [
    ("cmd_echo", 'cmd.exe /c echo EXT_CMD_OK', "EXT_CMD_OK"),
    ("py_print", '"$PY" -c "print(\'EXT_PY_OK\')"; echo rc=$?', "EXT_PY_OK"),
    ("py_file", '"$PY" -c "open(\'py.txt\',\'w\').write(\'EXT_PY_FILE\')"; echo rc=$?; cat py.txt', "EXT_PY_FILE"),
    ("py_stderr", '"$PY" -c "import sys; sys.stderr.write(\'EXT_PY_ERR\')" 2>&1', "EXT_PY_ERR"),
    ("py_capture", 'v=$("$PY" -c "print(7*6)"); echo "got=$v"', "got=42"),
    ("py_unbuffered", '"$PY" -u -c "print(\'EXT_PY_U\')"', "EXT_PY_U"),
    ("git_version", 'git --version', "git version"),
    ("where_exe", 'where.exe cmd', "cmd.exe"),
    ("wrapped_cmd_echo", 'cmd.exe /c echo EXT_CMD_OK\nexit $?', "EXT_CMD_OK"),
    ("wrapped_git", 'git --version\nexit $?', "git version"),
    ("wrapped_py_u", '"$PY" -u -c "print(\'EXT_PY_U\')"\nexit $?', "EXT_PY_U"),
    ("wrapped_py_rc", '"$PY" -c "import sys; print(\'X\'); sys.exit(3)"\nexit $?', "X"),
    ("wrapped_where", 'where.exe cmd\nexit $?', "cmd.exe"),
]

CMD_CASES = [
    ("cmd_py_print", '%PY% -c "print(6*7)"', "42"),
    ("cmd_git_version", "git --version", "git version"),
]

if __name__ == "__main__":
    label, shell, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    if Path(shell).name.casefold() in {"cmd", "cmd.exe"}:
        CASES = CMD_CASES
    passed = 0
    for name, script, expect in CASES:
        wd = tempfile.mkdtemp(prefix = "ext-wd-")
        out, res, secs = shell_probe.run_in_mxc(shell, script, wd, extra)
        ok = expect in out
        passed += ok
        print(f"EXT {label} {name.ljust(14)} {'ok' if ok else 'FAIL'} exit={res.get('exitCode')} s={secs:.1f} | "
              + out.strip().replace(chr(10), ' / ')[-220:], flush = True)
        extra_files = [p.name for p in Path(wd).iterdir()]
        if not ok and extra_files:
            print(f"    files in workdir: {extra_files}", flush = True)
    print(f"EXT_TOTAL {label} {passed}/{len(CASES)}", flush = True)
