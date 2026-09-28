"""Harness only: HTTPS, listing and pip from a shell inside MXC, plus pip from Python directly as the control arm."""
import sys
import tempfile

import shell_probe

SHELL_CASES = [
    ("curl_https", "curl.exe -s https://pypi.org/simple/six/ | head -c 300 | grep -o six | head -1", "six"),
    ("ls_l", "echo abc > f1 && ls -l", "f1"),
    ("pip_target_quiet", '"$PY" -m pip install -q --no-deps --no-cache-dir --disable-pip-version-check --target ./pkgs six==1.16.0; echo rc=$?; ls pkgs | head -3', "six.py"),
]
PY_CASES = [
    ("py_direct_pip", "import subprocess, sys; r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--no-cache-dir', '--disable-pip-version-check', '--target', 'pkgs', 'six==1.16.0']); import os; print('rc', r.returncode, sorted(os.listdir('pkgs'))[:3])", "six.py"),
]

if __name__ == "__main__":
    label, shell, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    cases = PY_CASES if label.startswith("python") else SHELL_CASES
    for name, script, expect in cases:
        wd = tempfile.mkdtemp(prefix = "net-wd-")
        out, res, secs = shell_probe.run_in_mxc(shell, script, wd, extra)
        ok = expect in out
        print(f"NET {label} {name.ljust(16)} {'ok' if ok else 'FAIL'} exit={res.get('exitCode')} s={secs:.1f} | "
              + out.strip().replace(chr(10), " / ")[-300:], flush = True)
