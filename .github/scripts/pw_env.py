"""Run Playwright drivers on any Python: re-exec under the workspace's dedicated Playwright venv.

The launcher installs only the Node `@playwright/mcp` browser, so most workspace venvs have no
Python `playwright`. `ensure()` returns at once when this interpreter already imports the needed
modules; otherwise it creates `$WORKSPACE/temp/venv_pw` (once, under `temp/venv_pw.lock`, with
Chromium in `temp/pw_browsers`) and re-execs the current command line under it, so
`python driver.py`, `python -m studio_test_kit._smoke_ui` and `python pr_ui_diff.py` all just work.

Called by studio_regress.py, pr_ui_diff.py and `import studio_test_kit`. `PW_ENV_AUTO=0` opts out
(you then get the plain ImportError). Interactive sessions and pytest never re-exec: they get an
ImportError naming the venv to use instead.
"""

from __future__ import annotations

import fcntl
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

# cryptography: studio_regress runs here and seals secrets for Colab / Kaggle offload (cloud_pool.seal);
# in the import probe too, so an existing venv without it is topped up.
PW_MODULES = (
    "playwright",
    "numpy",
    "PIL",
    "httpx",
    "huggingface_hub",
    "yaml",
    "cryptography",
)  # studio_regress
KIT_MODULES = ("playwright", "httpx", "PIL")  # studio_test_kit, pr_ui_scenes, pr_ui_diff
PW_PACKAGES = [
    "playwright",
    "numpy",
    "pillow",
    "httpx",
    "huggingface_hub",
    "pyyaml",
    "pytest",
    "cryptography",
]
REEXEC_ENV = "STUDIO_REGRESS_PW_REEXEC"  # the name studio_regress always used; children inherit it


def base_dir() -> Path:
    """Where temp/venv_pw lives: $WORKSPACE, a per-user cache for a standalone copy
    (identity.json beside this file), else the workspace root these scripts sit in."""
    here = Path(__file__).resolve().parent
    env = os.environ.get("WORKSPACE") or os.environ.get("UNSLOTH_WORKSPACE")
    if env:
        return Path(env).expanduser()
    if (here / "identity.json").is_file():
        return Path.home() / ".cache" / "unsloth_scripts"
    return here.parent if here.name == "scripts" else here


def venv_python(base: Path | None = None) -> Path:
    return (base or base_dir()) / "temp" / "venv_pw" / "bin" / "python"


def has(modules = PW_MODULES) -> bool:
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except (ImportError, ValueError):
        return False


def _relaunch_argv():
    """The command line to replay under the venv, or None when there is no script to replay
    (REPL, stdin, pytest)."""
    if "pytest" in sys.modules or not sys.argv or sys.argv[0] in ("", "-"):
        return None
    orig = list(getattr(sys, "orig_argv", []) or [])
    if len(orig) >= 2:
        return orig[1:]
    if sys.argv[0] == "-c":
        return None
    return [os.path.abspath(sys.argv[0]), *sys.argv[1:]]


def provision(base: Path | None = None, tag: str = "pw_env") -> Path:
    """Create temp/venv_pw with PW_PACKAGES and Chromium if missing. Idempotent, cross-process safe."""
    base = base or base_dir()
    temp = base / "temp"
    venv, browsers = temp / "venv_pw", temp / "pw_browsers"
    py = venv / "bin" / "python"
    temp.mkdir(parents = True, exist_ok = True)
    env = {**os.environ, "PLAYWRIGHT_BROWSERS_PATH": str(browsers)}
    with open(temp / "venv_pw.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        probe = (
            subprocess.run([str(py), "-c", f"import {', '.join(PW_MODULES)}"], capture_output = True)
            if py.exists()
            else None
        )
        if probe is None or probe.returncode:
            print(
                f"[{tag}] {sys.executable} lacks Playwright; setting up {venv}",
                file = sys.stderr,
                flush = True,
            )
            if not py.exists():
                subprocess.run(
                    ["uv", "venv", str(venv), "--python", sys.executable, "--allow-existing", "-q"],
                    check = True,
                )
            subprocess.run(
                ["uv", "pip", "install", "-q", "--python", str(py), *PW_PACKAGES], check = True
            )
        if not any(browsers.glob("chromium*")):
            subprocess.run(
                [str(py), "-m", "playwright", "install", "chromium"], check = True, env = env
            )
    return py


def ensure(
    modules = PW_MODULES,
    tag: str = "pw_env",
    argv = None,
) -> None:
    """Return when this interpreter imports `modules`; otherwise re-exec under temp/venv_pw.
    `argv` overrides the replayed command line (default: this process's own)."""
    if has(modules) or os.environ.get("PW_ENV_AUTO", "1") == "0":
        return
    base = base_dir()
    hint = (
        f"{sys.executable} has no Playwright; run this with {venv_python(base)} "
        f"(created by `python {Path(__file__).resolve()}`)"
    )
    if os.environ.get(REEXEC_ENV):
        raise ImportError(f"[{tag}] still missing {', '.join(modules)} under {sys.executable}")
    replay = argv if argv is not None else _relaunch_argv()
    if replay is None:
        raise ImportError(f"[{tag}] {hint}")
    py = provision(base, tag)
    os.environ[REEXEC_ENV] = "1"
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(base / "temp" / "pw_browsers")
    print(f"[{tag}] running under {py}", file = sys.stderr, flush = True)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(str(py), [str(py), *replay])


if __name__ == "__main__":
    print(provision())
