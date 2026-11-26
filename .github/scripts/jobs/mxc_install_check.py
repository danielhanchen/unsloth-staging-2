"""Windows MXC install check for Unsloth Studio (staging CI job, Windows legs only).

Runs the branch's own studio/install_mxc_prebuilt.py the way setup.ps1 does, then checks:
  1. fresh install exits 0 into <home>/mxc-runtime/windows-<arch>
  2. a second run is a no-op ("already matches")
  3. both installed binaries carry a valid Authenticode signature from Microsoft (Get-AuthenticodeSignature)
  4. Studio's own verification (mxc_runtime.verified_installation_identity / acquire_runtime) accepts them
  5. `wxc-exec --probe` runs and reports a tier
  6. a corrupted wxc-exec.exe is refused by Studio, then repaired by rerunning the installer

Usage: python .github/scripts/jobs/mxc_install_check.py --tiny --out job_mxc_install.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path.cwd()
BACKEND = REPO / "studio" / "backend"
INSTALLER = REPO / "studio" / "install_mxc_prebuilt.py"


def _arch_dir() -> str:
    machine = (
        os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE") or ""
    ).lower()
    return "windows-arm64" if machine == "arm64" else "windows-x86_64"


def _run(
    argv,
    env = None,
    timeout = 900,
):
    started = time.perf_counter()
    proc = subprocess.run(argv, capture_output = True, text = True, env = env, timeout = timeout)
    return {
        "argv": [str(a) for a in argv],
        "rc": proc.returncode,
        "seconds": round(time.perf_counter() - started, 2),
        "out": (proc.stdout + proc.stderr)[-3000:],
    }


def _signature(path: Path) -> dict:
    script = (
        f"$s = Get-AuthenticodeSignature -LiteralPath '{path}'; "
        "[pscustomobject]@{status = [string]$s.Status; signer = $s.SignerCertificate.Subject; "
        "issuer = $s.SignerCertificate.Issuer; timestamp = [string]$s.TimeStamperCertificate.Subject} "
        "| ConvertTo-Json -Compress"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        timeout = 120,
    )
    try:
        return json.loads(proc.stdout.strip())
    except Exception:  # noqa: BLE001
        return {"status": "unparsed", "out": (proc.stdout + proc.stderr)[-1000:]}


def _studio_check(home: Path) -> dict:
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(BACKEND)!r})\n"
        "from core.inference import mxc_runtime\n"
        "out = {}\n"
        "try:\n"
        "    verify = getattr(mxc_runtime, 'verified_installation_identity', None) or (lambda: mxc_runtime.selected_runtime().identity)\n"
        "    out['identity'] = verify()[:16]\n"
        "    with mxc_runtime.acquire_runtime() as lease:\n"
        "        out['lease_path'] = str(lease.info.path)\n"
        "        out['architecture'] = lease.info.architecture\n"
        "    out['ok'] = True\n"
        "except Exception as exc:\n"
        "    out['ok'] = False\n"
        "    out['error'] = f'{type(exc).__name__}: {exc}'\n"
        "print('STUDIO ' + json.dumps(out))\n"
    )
    env = {**os.environ, "UNSLOTH_STUDIO_HOME": str(home)}
    result = _run([sys.executable, "-c", code], env = env, timeout = 300)
    line = next((l for l in result["out"].splitlines() if l.startswith("STUDIO ")), None)
    result["studio"] = json.loads(line[7:]) if line else None
    return result


def _ensure_deps():
    probe = [
        sys.executable,
        "-c",
        f"import sys; sys.path.insert(0, {str(BACKEND)!r}); import core.inference.tools",
    ]
    if subprocess.run(probe, capture_output = True, timeout = 600).returncode == 0:
        return {"importable": True}
    failed = []
    for line in (BACKEND / "requirements" / "studio.txt").read_text().splitlines():
        spec = line.split("#", 1)[0].strip()
        if spec and not spec.startswith("-"):
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--only-binary", ":all:", spec],
                capture_output = True,
                timeout = 900,
            )
            if r.returncode:
                failed.append(spec)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "pytest",
            "pytest-asyncio",
            "pytest-timeout",
            "psutil",
            "python-multipart",
        ],
        capture_output = True,
        timeout = 900,
    )
    r = subprocess.run(probe, capture_output = True, text = True, timeout = 600)
    return {"importable": r.returncode == 0, "failed": failed, "err": r.stderr[-800:]}


def _pytest(files):
    files = [f for f in files if (BACKEND / f).is_file()]
    if not files:
        return None
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=", *files],
        cwd = BACKEND,
        capture_output = True,
        text = True,
        timeout = 2400,
    )
    tail = (r.stdout + r.stderr).splitlines()
    return {
        "rc": r.returncode,
        "files": files,
        "summary": tail[-1] if tail else "",
        "failed": [l for l in tail if l.startswith(("FAILED", "ERROR"))][:60],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action = "store_true")
    p.add_argument("--out", default = "job_mxc_install.json")
    p.add_argument("--tests", default = "", help = "comma-separated test files under studio/backend")
    args = p.parse_args()
    out_path = Path(args.out).resolve()
    result = {"job": "mxc_install_check", "passed": False, "checks": {}}
    result["host"] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "arch_dir": _arch_dir(),
        "python": platform.python_version(),
    }
    checks = result["checks"]
    try:
        if sys.platform != "win32":
            result["skipped"] = "Windows only"
            result["passed"] = True
            return
        home = Path(tempfile.mkdtemp(prefix = "mxc-home-"))
        install_dir = home / "mxc-runtime" / _arch_dir()
        env = {**os.environ, "UNSLOTH_STUDIO_HOME": str(home)}
        first = _run([sys.executable, str(INSTALLER), "--install-dir", str(install_dir)], env = env)
        checks["fresh_install"] = first
        second = _run([sys.executable, str(INSTALLER), "--install-dir", str(install_dir)], env = env)
        second["already_matches"] = "already matches" in second["out"]
        checks["second_run"] = second
        binaries = sorted(str(p.relative_to(install_dir)) for p in install_dir.rglob("*.exe"))
        checks["installed_files"] = binaries
        checks["signatures"] = {name: _signature(install_dir / name) for name in binaries}
        checks["studio_verify"] = _studio_check(home)
        wxc = install_dir / "wxc-exec.exe"
        checks["probe"] = _run([str(wxc), "--probe"], timeout = 120) if wxc.is_file() else None
        # Corrupt one byte: Studio must refuse it, and the installer must repair it.
        if wxc.is_file():
            data = bytearray(wxc.read_bytes())
            data[len(data) // 2] ^= 0xFF
            wxc.write_bytes(bytes(data))
            checks["studio_verify_corrupt"] = _studio_check(home)
            repair = _run(
                [sys.executable, str(INSTALLER), "--install-dir", str(install_dir)], env = env
            )
            checks["repair"] = repair
            checks["studio_verify_repaired"] = _studio_check(home)
        if args.tests:
            checks["deps"] = _ensure_deps()
            checks["pytest"] = _pytest([t for t in args.tests.split(",") if t])
        sigs = checks["signatures"]
        result["passed"] = bool(
            first["rc"] == 0
            and second["rc"] == 0
            and second["already_matches"]
            and binaries
            and all(
                s.get("status") == "Valid" and "Microsoft Corporation" in (s.get("signer") or "")
                for s in sigs.values()
            )
            and (checks["studio_verify"]["studio"] or {}).get("ok")
            and checks["probe"]
            and checks["probe"]["rc"] == 0
            and not (checks["studio_verify_corrupt"]["studio"] or {}).get("ok", True)
            and checks["repair"]["rc"] == 0
            and (checks["studio_verify_repaired"]["studio"] or {}).get("ok")
            and (not args.tests or (checks["pytest"] or {}).get("rc") == 0)
        )
    except Exception as exc:  # noqa: BLE001
        import traceback
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-4000:]
    finally:
        out_path.write_text(json.dumps(result, indent = 1, default = str))
        print(json.dumps(result, indent = 1, default = str)[:20000], flush = True)
        print(
            "JOB_RESULT " + json.dumps({"passed": result["passed"], "host": result["host"]}),
            flush = True,
        )
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
