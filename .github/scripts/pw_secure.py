"""Patched Chromium for Playwright: the bundled build lags Chrome's security fixes.

Playwright 1.64.1 (latest stable, Nov 2026) bundles Chromium 156.0.8078.4. CVE-2026-118306 / -118331 /
-118344 / -118348 (a crafted page runs code; 118306 escapes the sandbox) are fixed in 156.0.8078.73, and
Playwright launches Chromium with --no-sandbox unless chromium_sandbox=True. Every Studio journey, scene
and probe renders PR-built Studio (remote images, the in-app browser), so the renderer must be patched.

`patched_executable(headless)`: a Chrome for Testing binary >= MIN_SAFE_CHROMIUM when the bundled build is
older (headless shell for headless launches, full Chrome otherwise), reused from <base>/temp/cft/<version>/
or the current Stable downloaded there once under a lock. Offline / unsupported platform: None plus one
warning, and the caller keeps the bundled build, so no flow breaks on this.
Weekly, a launch also starts a detached `pw_secure.py refresh`: it downloads the current Stable when it is
newer than the local build and keeps the newest two, so the cache follows Chrome's security releases
without anyone bumping MIN_SAFE_CHROMIUM (the launch itself never waits on it).
`ensure(headless)` skips the bundled-version check: for the Playwright MCP, whose Node Playwright
pins its own Chromium. `sandbox_ok(exe)`: whether that binary starts with the Chromium sandbox here (Ubuntu 24.04 blocks it for
binaries without an AppArmor userns profile; macOS / Windows allow it). Cached per binary in cft_root().
pw_fast.patch() applies both to every Chromium launch. PW_PATCHED_CHROMIUM=0 turns this off.
CLI: `python pw_secure.py ensure` (prefetch, used by launcher.sh) / `refresh` / `status`.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows: no flock; os.replace below still publishes atomically
    fcntl = None

MIN_SAFE_CHROMIUM = "156.0.8078.73"  # raise on the next Chrome advisory that matters
CFT_JSON = "https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json"
FAIL_TTL_S = 3600  # after a failed download, do not retry (and slow every launch) for an hour
REFRESH_S = 7 * 24 * 3600  # how often a launch checks for a newer Stable
KEEP_VERSIONS = 2  # the newest build plus the previous one (a launch may still be using it)
SANDBOX_NO_TTL_S = (
    3600  # re-probe a refused sandbox hourly (an AppArmor profile may have been added)
)

# CfT platform -> (dir, binary) per kind; Linux arm64 has no CfT build.
_LAYOUT = {
    "linux64": {
        "chrome": ("chrome-linux64", "chrome"),
        "chrome-headless-shell": ("chrome-headless-shell-linux64", "chrome-headless-shell"),
    },
    "mac-arm64": {
        "chrome": (
            "chrome-mac-arm64",
            "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
        ),
        "chrome-headless-shell": ("chrome-headless-shell-mac-arm64", "chrome-headless-shell"),
    },
    "mac-x64": {
        "chrome": (
            "chrome-mac-x64",
            "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
        ),
        "chrome-headless-shell": ("chrome-headless-shell-mac-x64", "chrome-headless-shell"),
    },
    "win64": {
        "chrome": ("chrome-win64", "chrome.exe"),
        "chrome-headless-shell": ("chrome-headless-shell-win64", "chrome-headless-shell.exe"),
    },
}
_warned = False


def enabled() -> bool:
    return os.environ.get("PW_PATCHED_CHROMIUM", "1") != "0"


def vtuple(v) -> tuple:
    try:
        return tuple(int(x) for x in str(v).split("."))
    except ValueError:
        return ()


def is_safe(version) -> bool:
    t = vtuple(version)
    return bool(t) and t >= vtuple(MIN_SAFE_CHROMIUM)


def cft_platform() -> str | None:
    m = platform.machine().lower()
    if sys.platform.startswith("linux"):
        return "linux64" if m in ("x86_64", "amd64") else None
    if sys.platform == "darwin":
        return "mac-arm64" if m in ("arm64", "aarch64") else "mac-x64"
    if sys.platform == "win32":
        return "win64" if m in ("amd64", "x86_64") else None
    return None


def cft_root() -> Path:
    try:
        import pw_env
        base = pw_env.base_dir()
    except Exception:  # noqa: BLE001 - standalone copy without pw_env
        base = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parent.parent)
    return Path(base) / "temp" / "cft"


def bundled_chromium_version() -> str | None:
    """The Chromium build the installed Playwright drives, from its browsers.json (no launch)."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("playwright")
        manifest = json.loads(
            (Path(spec.origin).parent / "driver" / "package" / "browsers.json").read_text()
        )
    except Exception:  # noqa: BLE001
        return None
    for b in manifest.get("browsers", []):
        if b.get("name") == "chromium":
            return b.get("browserVersion")
    return None


def _kind(headless: bool) -> str:
    return "chrome-headless-shell" if headless else "chrome"


def _local(kind: str) -> Path | None:
    """Newest already-extracted safe build of `kind` under cft_root()."""
    plat, root = cft_platform(), cft_root()
    if plat is None or not root.is_dir():
        return None
    d, b = _LAYOUT[plat][kind]
    for ver in sorted(
        (p.name for p in root.iterdir() if is_safe(p.name)), key = vtuple, reverse = True
    ):
        exe = root / ver / d / b
        if exe.is_file() and os.access(exe, os.X_OK):
            return exe
    return None


def _warn(msg: str) -> None:
    global _warned
    if not _warned:
        _warned = True
        print(f"[pw_secure] WARNING: {msg}", file = sys.stderr)


def _extract(zpath: Path, dest: Path) -> None:
    if os.name == "posix" and shutil.which("unzip"):  # keeps exec bits and the macOS .app symlinks
        subprocess.run(["unzip", "-q", "-o", str(zpath), "-d", str(dest)], check = True)
    else:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
            if os.name == "posix":  # extractall drops the exec bits the archive records
                for info in z.infolist():
                    mode = info.external_attr >> 16
                    if mode & 0o111:
                        os.chmod(dest / info.filename, mode & 0o7777)


@contextmanager
def _locked(path: Path):
    with open(path, "w") as fh:
        if fcntl is not None:
            fcntl.flock(fh, fcntl.LOCK_EX)
        yield


def _stable() -> tuple[str, dict]:
    """(version, downloads) of the current Chrome for Testing Stable."""
    with urllib.request.urlopen(CFT_JSON, timeout = 30) as r:
        stable = json.load(r)["channels"]["Stable"]
    if not is_safe(stable["version"]):
        raise RuntimeError(f"CfT Stable {stable['version']} is older than {MIN_SAFE_CHROMIUM}")
    return stable["version"], stable["downloads"]


def _fetch(
    kind: str,
    ver: str,
    downloads: dict,
    timeout: float = 600,
) -> Path:
    """Download + extract one build into cft_root()/ver (caller holds the lock)."""
    plat, root = cft_platform(), cft_root()
    url = next(x["url"] for x in downloads[kind] if x["platform"] == plat)
    d, b = _LAYOUT[plat][kind]
    with tempfile.TemporaryDirectory(dir = root, prefix = ".tmp-") as tmp:
        zpath = Path(tmp) / "cft.zip"
        with urllib.request.urlopen(url, timeout = timeout) as r, open(zpath, "wb") as out:
            shutil.copyfileobj(r, out)
        _extract(zpath, Path(tmp))
        if not os.access(Path(tmp) / d / b, os.X_OK):
            raise RuntimeError(f"{url} has no runnable {d}/{b}")
        (root / ver).mkdir(exist_ok = True)
        if (root / ver / d).exists():  # a broken earlier copy (_local skipped it): replace it
            shutil.rmtree(root / ver / d)
        os.replace(Path(tmp) / d, root / ver / d)  # atomic: a reader sees all or nothing
    return root / ver / d / b


def _download(kind: str) -> Path | None:
    plat, root = cft_platform(), cft_root()
    if plat is None:
        return None
    root.mkdir(parents = True, exist_ok = True)
    fail = root / ".download_failed"
    with _locked(root / ".lock"):
        exe = _local(kind)  # another process may have finished while we waited
        if exe is not None:
            return exe
        if fail.is_file() and time.time() - fail.stat().st_mtime < FAIL_TTL_S:
            return None
        try:
            exe = _fetch(kind, *_stable())
            fail.unlink(missing_ok = True)
            (root / ".stable_checked").touch()
            return exe
        except Exception as e:  # noqa: BLE001 - never break a launch over this
            fail.touch()
            _warn(
                f"could not fetch a patched Chrome for Testing ({type(e).__name__}: {e}); "
                f"using Playwright's bundled Chromium {bundled_chromium_version()}"
            )
            return None


def refresh() -> str | None:
    """Fetch the current Stable for every kind already here when it is newer; keep the newest two.
    Idempotent and locked; returns the Stable version, or None when the check failed."""
    plat, root = cft_platform(), cft_root()
    if plat is None or not root.is_dir():
        return None
    stamp = root / ".stable_checked"
    with _locked(root / ".lock"):
        try:
            ver, downloads = _stable()
            for kind, (d, b) in _LAYOUT[plat].items():
                have = _local(kind)
                if have is not None and vtuple(ver) > vtuple(have.parents[len(Path(b).parts)].name):
                    _fetch(kind, ver, downloads)
        except Exception as e:  # noqa: BLE001 - retry within the hour, not next week
            stamp.touch()
            t = time.time() - REFRESH_S + FAIL_TTL_S
            os.utime(stamp, (t, t))
            _warn(f"weekly Chrome for Testing refresh failed ({type(e).__name__}: {e})")
            return None
        stamp.touch()
        safe = sorted(
            (p for p in root.iterdir() if p.is_dir() and is_safe(p.name)),
            key = lambda p: vtuple(p.name),
        )
        for old in safe[:-KEEP_VERSIONS]:
            shutil.rmtree(old, ignore_errors = True)
        return ver


def _refresh_in_background() -> None:
    """Start a detached `refresh` when the last Stable check is a week old; never waits on it."""
    stamp = cft_root() / ".stable_checked"
    try:
        if stamp.is_file() and time.time() - stamp.stat().st_mtime < REFRESH_S:
            return
        stamp.touch()  # claim the next 10 minutes so parallel launches spawn one refresher
        t = time.time() - REFRESH_S + 600
        os.utime(stamp, (t, t))
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "refresh"],
            stdin = subprocess.DEVNULL,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            start_new_session = True,
            close_fds = True,
        )
    except Exception:  # noqa: BLE001 - a missed check is retried on a later launch
        pass


def ensure(headless: bool = True) -> str | None:
    """A patched build of the kind `headless` needs (reused, else downloaded once), or None."""
    try:
        exe = _local(_kind(headless))
        if exe is not None:
            _refresh_in_background()
        else:
            exe = _download(_kind(headless))
    except Exception as e:  # noqa: BLE001 - unwritable / unreadable cache dir: keep the bundled build
        _warn(
            f"patched Chrome for Testing unavailable ({type(e).__name__}: {e}); "
            f"using Playwright's bundled Chromium {bundled_chromium_version()}"
        )
        return None
    return str(exe) if exe else None


def patched_executable(headless: bool = True) -> str | None:
    """Path to launch instead of the bundled Chromium, or None to keep the bundled one."""
    if not enabled():
        return None
    bundled = bundled_chromium_version()
    if bundled and is_safe(bundled):
        return None
    return ensure(headless)


def sandbox_ok(exe: str) -> bool:
    """Whether `exe` starts with the Chromium sandbox on this host (one ~1 s probe, cached beside it)."""
    if os.name == "posix" and os.geteuid() == 0:
        return False  # Chromium refuses its sandbox as root
    stamp = cft_root() / f".sandbox-{hashlib.sha1(str(exe).encode()).hexdigest()[:16]}"
    try:
        verdict, ts = stamp.read_text().split()
        if verdict == "ok" or time.time() - float(ts) < SANDBOX_NO_TTL_S:
            return verdict == "ok"
    except (OSError, ValueError):
        pass
    try:
        with tempfile.TemporaryDirectory() as prof:
            r = subprocess.run(
                [
                    exe,
                    "--headless",
                    "--disable-gpu",
                    f"--user-data-dir={prof}",
                    "--dump-dom",
                    "about:blank",
                ],
                capture_output = True,
                text = True,
                timeout = 60,
            )
        ok = r.returncode == 0 and "<html" in r.stdout
    except Exception:  # noqa: BLE001
        ok = False
    try:
        stamp.write_text(f"{'ok' if ok else 'no'} {time.time()}")
    except OSError:
        pass
    return ok


def status() -> dict:
    bundled = bundled_chromium_version()
    out = {
        "enabled": enabled(),
        "min_safe": MIN_SAFE_CHROMIUM,
        "bundled": bundled,
        "bundled_safe": bool(bundled) and is_safe(bundled),
        "platform": cft_platform(),
    }
    stamp = cft_root() / ".stable_checked"
    out["stable_checked_h_ago"] = (
        round((time.time() - stamp.stat().st_mtime) / 3600, 1) if stamp.is_file() else None
    )
    for headless in (True, False):
        exe = _local(_kind(headless))
        out[_kind(headless)] = str(exe) if exe else None
        if exe:
            out[_kind(headless) + "_sandbox"] = sandbox_ok(str(exe))
    return out


def main(argv = None) -> int:
    cmd = (argv or sys.argv[1:] or ["status"])[0]
    if cmd == "ensure":  # idempotent prefetch of both kinds; never fails the caller
        if enabled():
            for headless in (True, False):
                ensure(headless)
        cmd = "status"
    if cmd == "refresh":  # what the weekly background check runs
        print(refresh() or "refresh failed")
        return 0
    if cmd == "status":
        print(json.dumps(status(), indent = 2))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
