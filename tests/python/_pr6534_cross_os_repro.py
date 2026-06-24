"""Cross-OS validation for PR #6534 (_uv_safe_path space-safety).

Throwaway CI repro (staging fork only; not for upstream). Proves on a real
runner of each OS:

  1. _uv_safe_path no-space passthrough is unchanged (all OS).
  2. POSIX: a space-containing `-c` path makes uv fail (truncates at the space),
     while the _uv_safe_path copy is accepted -> the PR fix works.
  3. POSIX: the SAME failure still happens for UV_OVERRIDE (the gap this PR does
     NOT yet cover), and _uv_safe_path would fix it too -> demonstrates the gap
     on the real target OS (macos-14 is Apple Silicon, the exact platform).

Windows uses the existing 8.3-short-path branch and uv's path handling differs,
so the uv-level checks there are informational (never hard-fail).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STUDIO = REPO / "studio"
sys.path.insert(0, str(STUDIO))
import install_python_stack as ips  # noqa: E402

IS_WINDOWS = platform.system() == "Windows"
UV = shutil.which("uv")


def banner(t: str) -> None:
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70, flush=True)


def uv_dry_run(extra_args, env=None):
    # --python pins a target interpreter so uv does not require an active venv
    # on the CI runner (otherwise the success cases hit "No virtual environment
    # found", which is unrelated to the path being tested).
    cmd = [UV, "pip", "install", "--dry-run", "--python", sys.executable, *extra_args]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    first = (r.stderr.strip().splitlines() or [""])[0]
    return r.returncode, first


failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}", flush=True)
    if not cond:
        failures.append(name)


banner(f"ENV: {platform.platform()} | python {sys.version.split()[0]} | uv={UV}")
print("uv version:", subprocess.run([UV, "--version"], capture_output=True, text=True).stdout.strip() if UV else "MISSING", flush=True)

# ---- 1. no-space passthrough (all OS, hard invariant) ----
banner("1. no-space passthrough returned unchanged (all OS)")
plain = os.path.join(tempfile.gettempdir(), "plain", "constraints.txt")
check("passthrough", ips._uv_safe_path(plain) == plain, f"-> {ips._uv_safe_path(plain)!r}")

# ---- build a space-containing constraints file ----
base = Path(tempfile.mkdtemp(prefix="pr6534_")) / "Open Source"
base.mkdir(parents=True)
con = base / "constraints.txt"
con.write_text("idna==3.10\n")
safe = ips._uv_safe_path(str(con))
print(f"\nsrc : {con}\nsafe: {safe}", flush=True)

if not IS_WINDOWS:
    # ---- 2. POSIX: -c raw fails, safe works (the PR fix) ----
    banner("2. POSIX: uv -c raw space path fails; _uv_safe_path copy works")
    check("helper produced space-free copy", " " not in safe and safe != str(con))
    check("copy is byte-identical", Path(safe).read_bytes() == con.read_bytes())
    if UV:
        rc_raw, e_raw = uv_dry_run(["idna", "-c", str(con)])
        rc_safe, e_safe = uv_dry_run(["idna", "-c", safe])
        check("uv -c raw FAILS (truncates)", rc_raw != 0, f"rc={rc_raw} {e_raw!r}")
        check("uv -c safe WORKS", rc_safe == 0, f"rc={rc_safe} {e_safe!r}")

    # ---- 3. POSIX: UV_OVERRIDE gap (NOT fixed by this PR) ----
    banner("3. POSIX: UV_OVERRIDE raw space path fails (PR GAP); _uv_safe_path copy would fix it")
    ovr = base / "overrides-darwin-arm64.txt"
    ovr.write_text("transformers>=4.57.6\n")
    safe_ovr = ips._uv_safe_path(str(ovr))
    if UV:
        env_raw = os.environ.copy(); env_raw["UV_OVERRIDE"] = str(ovr)
        env_safe = os.environ.copy(); env_safe["UV_OVERRIDE"] = safe_ovr
        rc_oraw, e_oraw = uv_dry_run(["idna"], env=env_raw)
        rc_osafe, e_osafe = uv_dry_run(["idna"], env=env_safe)
        check("UV_OVERRIDE raw FAILS (gap)", rc_oraw != 0, f"rc={rc_oraw} {e_oraw!r}")
        check("UV_OVERRIDE safe-copy WOULD WORK", rc_osafe == 0, f"rc={rc_osafe} {e_osafe!r}")
else:
    banner("2/3. Windows: 8.3 short-path branch (informational, no hard-fail)")
    print(f"  _uv_safe_path(space path) -> {safe!r} (has_space={' ' in safe})", flush=True)
    if UV:
        rc_raw, e_raw = uv_dry_run(["idna", "-c", str(con)])
        rc_safe, e_safe = uv_dry_run(["idna", "-c", safe])
        print(f"  uv -c raw : rc={rc_raw} {e_raw!r}", flush=True)
        print(f"  uv -c safe: rc={rc_safe} {e_safe!r}", flush=True)

banner("RESULT")
if failures:
    print("FAILURES:", failures, flush=True)
    sys.exit(1)
print("All hard invariants passed on this OS.", flush=True)
