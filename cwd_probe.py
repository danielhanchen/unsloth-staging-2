"""Harness only: why stock Git for Windows cannot read its cwd inside MXC, and the smallest ancestor grant that fixes it.

Usage: python cwd_probe.py <workdir-parent>
For cumulative ancestor grants to ALL APPLICATION PACKAGES (this folder only, no inheritance):
none -> traverse (X) -> +read attributes (RA) -> +list folder (RD), it runs inside MXC:
  a ctypes probe of the calls mingw_getcwd makes, and stock git through cmd.exe.
Payloads are harmless; the grants are removed again at the end.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

from shell_probe import run_in_mxc

AAP = "*S-1-15-2-1"
CTYPES_PROBE = r"""
import ctypes, os
from ctypes import wintypes as w
k = ctypes.WinDLL("kernel32", use_last_error = True)
k.CreateFileW.restype = w.HANDLE
k.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, w.LPVOID, w.DWORD, w.DWORD, w.HANDLE]
k.GetFinalPathNameByHandleW.argtypes = [w.HANDLE, w.LPWSTR, w.DWORD, w.DWORD]
k.GetLongPathNameW.argtypes = [w.LPCWSTR, w.LPWSTR, w.DWORD]
cwd = os.getcwd()
h = k.CreateFileW(cwd, 0, 7, None, 3, 0x02000000, None)
print("CT open", "ok" if h not in (None, w.HANDLE(-1).value) else f"err={ctypes.get_last_error()}")
for name, flags in (("normalized_dos", 0), ("opened_dos", 8), ("normalized_nt", 2), ("normalized_guid", 1), ("normalized_none", 4)):
    buf = ctypes.create_unicode_buffer(1024)
    n = k.GetFinalPathNameByHandleW(h, buf, 1024, flags)
    print("CT final", name, f"ok {buf.value}" if n else f"err={ctypes.get_last_error()}")
buf = ctypes.create_unicode_buffer(1024)
n = k.GetLongPathNameW(cwd, buf, 1024)
print("CT long", f"ok {buf.value}" if n else f"err={ctypes.get_last_error()}")
for extra in filter(None, os.environ.get("PROBE_PATHS", "").split(";")):
    buf = ctypes.create_unicode_buffer(1024)
    n = k.GetLongPathNameW(extra, buf, 1024)
    print("CT long_of", extra, f"ok {buf.value}" if n else f"err={ctypes.get_last_error()}")
"""


def ancestors(path):
    out, p = [], Path(path).parent
    while True:
        out.append(str(p))
        if p.parent == p:
            return out
        p = p.parent


def icacls(path, *args):
    done = subprocess.run(["icacls", path, *args, "/Q"], capture_output = True, text = True)
    return done.returncode


def git_script(prefix = ""):
    return (f"{prefix}git init -q r && cd r && git -c user.name=t -c user.email=t@example.invalid "
            f"commit --allow-empty -qm first && git log --oneline && git status --short && echo GIT_OK")


def report(label, step, wd, prefix = "", paths = ""):
    os.environ["PROBE_PATHS"] = paths
    out, res, secs = run_in_mxc(sys.executable, CTYPES_PROBE, wd, [])
    for line in out.splitlines():
        if line.startswith("CT "):
            print(f"CWD {label} {step} {line[3:]}", flush = True)
    out, res, secs = run_in_mxc(os.path.join(os.environ["SystemRoot"], "System32", "cmd.exe"), git_script(prefix), wd, [])
    ok = "GIT_OK" in out and res.get("exitCode") == 0
    tail = " | ".join(out.strip().splitlines()[-2:])[-240:]
    print(f"GITCWD {label} {step} {'ok' if ok else 'FAIL'} exit={res.get('exitCode')} s={secs:.1f} | {tail}", flush = True)


def fresh_ancestors(path):
    fresh = []
    for a in ancestors(os.path.join(path, "x"))[1:]:
        acl = subprocess.run(["icacls", a], capture_output = True, text = True).stdout
        has = "ALL APPLICATION PACKAGES" in acl
        print(f"ANCESTOR {a} existing_aap={has}", flush = True)
        if not has:
            fresh.append(a)
    return fresh


if __name__ == "__main__":
    parent = sys.argv[1]
    label = Path(parent).name
    os.makedirs(parent, exist_ok = True)
    wd = tempfile.mkdtemp(prefix = "cwdp-", dir = parent)
    fresh = fresh_ancestors(wd)
    # 1. subst: the workdir becomes a drive root, so no component above it needs resolving
    letter = next(d for d in "WVUTSRQP" if not os.path.exists(f"{d}:\\"))
    print(f"SUBST {label} {letter}: exit={subprocess.run(['subst', f'{letter}:', wd]).returncode}", flush = True)
    report(label, "subst", wd, prefix = f"cd /d {letter}:\\ && ", paths = f"{letter}:\\")
    subprocess.run(["subst", f"{letter}:", "/d"])
    # 2. ancestors: list + synchronize (what FindFirstFile opens a directory with), this folder only
    for step, rights in (("RD+S", "(RD,S)"), ("RD+S+RA+X", "(RD,S,RA,X)")):
        codes = [icacls(a, "/grant", f"{AAP}:{rights}") for a in fresh]
        print(f"STEP {label} {step} grant_exit={codes}", flush = True)
        report(label, step, wd, paths = ";".join(fresh))
    for a in fresh:
        icacls(a, "/remove:g", AAP)
