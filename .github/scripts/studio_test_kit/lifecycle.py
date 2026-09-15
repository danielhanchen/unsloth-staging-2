"""Install + launch Unsloth Studio for arbitrary branches/ports.

The pre-PR vs post-PR test pattern installs Studio twice, once per branch,
each pinned to its own UNSLOTH_STUDIO_HOME so the two installs share
nothing (separate `.venv_t5_*`, `auth/`, `studio.db`, llama.cpp build).

`install_studio(branch=..., home=...)` clones unslothai/unsloth (or
re-uses an existing clone), checks out the branch, then runs
`./install.sh --local` with UNSLOTH_STUDIO_HOME exported.

`launch_studio(install, port=..., log_path=...)` starts `unsloth studio
-p <port>` detached via setsid and recovers the bootstrap password from
`<home>/auth/.bootstrap_password`, falling back to the log for builds
that printed it there instead.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class StudioInstall:
    """Where a Studio install lives + the credentials it minted on first run."""

    home: Path                   # UNSLOTH_STUDIO_HOME
    repo: Path                   # clone of unslothai/unsloth
    branch: str
    bootstrap_password: Optional[str] = None
    port: Optional[int] = None
    pid: Optional[int] = None


def _run(cmd: str | list[str], cwd: Optional[Path] = None, env: Optional[dict] = None,
         check: bool = True, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
    else:
        cmd_list = cmd
    full_env = {**os.environ, **(env or {})}
    return subprocess.run(
        cmd_list, cwd=cwd, env=full_env, check=check, timeout=timeout,
        text=True, capture_output=True,
    )


def install_studio(
    branch: str,
    home: Path,
    repo: Optional[Path] = None,
    remote: str = "https://github.com/unslothai/unsloth",
    reuse_clone: bool = True,
) -> StudioInstall:
    """Clone (or re-use) the repo at `branch`, then run `./install.sh --local`.

    `home` is exported as UNSLOTH_STUDIO_HOME for the install. After this
    returns, `home/.venv_t5_550/`, `home/auth/`, etc. exist.
    """
    home = Path(home).resolve()
    home.mkdir(parents=True, exist_ok=True)
    repo = (repo or (home.parent / f"{home.name}_repo")).resolve()

    if reuse_clone and (repo / ".git").exists():
        _run(["git", "fetch", "origin", branch], cwd=repo)
        _run(["git", "checkout", branch], cwd=repo)
        _run(["git", "reset", "--hard", f"origin/{branch}"], cwd=repo)
    else:
        if repo.exists():
            shutil.rmtree(repo)
        _run(["git", "clone", "--branch", branch, remote, str(repo)])

    install_sh = repo / "install.sh"
    if not install_sh.exists():
        raise FileNotFoundError(f"install.sh missing at {install_sh}")
    _run(
        ["bash", str(install_sh), "--local"],
        cwd=repo,
        env={"UNSLOTH_STUDIO_HOME": str(home)},
        timeout=60 * 30,
    )
    return StudioInstall(home=home, repo=repo, branch=branch)


def _find_unsloth_bin(install: StudioInstall) -> str:
    """Return the absolute path to the `unsloth` CLI inside the install."""
    for candidate in (
        install.home / "bin" / "unsloth",
        install.home / ".venv_t5_550" / "bin" / "unsloth",
        install.home / ".venv_t5_530" / "bin" / "unsloth",
    ):
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"`unsloth` CLI not found under {install.home}")


# The FALLBACK source. A current Studio writes the password to a file instead (see
# _read_bootstrap_password); these are the log line shapes older builds emit.
# Password log line shapes seen in practice:
#   "Bootstrap password: secret"
#   "Initial password = secret"
#   "Generated password is secret"
#   "bootstrap password is: secret"
# The mandatory `\s+` before the value and the EXPLICIT `[:=]?` separator
# (rather than `[:\s]+` greedy class) stop the regex from backtracking
# to capture `=` itself as the password.
_PW_RE = re.compile(
    r"(?i)(?:bootstrap|initial|generated)\s*password"
    r"(?:\s+is)?\s*[:=]?\s+(\S+)"
)


def _read_bootstrap_password(
    home: Path, log_path: Path, deadline: float
) -> Optional[str]:
    """The bootstrap password, from the file first and the log second.

    Studio writes it to ``<home>/auth/.bootstrap_password`` and does NOT inline it in the
    log, so a log-only read returns None on a current Studio and every authenticated route
    then answers ``403 Password change required``. That failure is quiet: launch succeeds,
    /healthz answers, and the caller only finds out one request later.

    Both sources are polled together because the file appears at first-run auth setup and
    the log line is what older builds emit; whichever arrives first wins. Studio DELETES the
    file once the password is rotated, so a reused home legitimately has neither, and None
    stays a valid answer meaning "log in with the password you already set".
    """
    boot_file = home / "auth" / ".bootstrap_password"
    while time.time() < deadline:
        try:
            if boot_file.exists():
                secret = boot_file.read_text(errors="ignore").strip()
                if secret:
                    return secret
        except OSError:
            pass  # mid-write, or a permission we do not have: fall through and retry
        if log_path.exists():
            text = log_path.read_text(errors="ignore")
            m = _PW_RE.search(text)
            if m:
                return m.group(1).strip().strip(".,")
        time.sleep(0.5)
    return None


def launch_studio(
    install: StudioInstall,
    port: int,
    log_path: Path,
    extra_env: Optional[dict] = None,
    wait_for_healthz: bool = True,
    healthz_timeout_s: int = 180,
    password_timeout_s: int = 30,
    timeout_s: Optional[int] = None,
) -> StudioInstall:
    """Start `unsloth studio -p <port>` detached. Updates `install` in place
    with `port`, `pid`, and `bootstrap_password` (parsed from the log).

    Two INDEPENDENT timeouts:
      - `password_timeout_s`: how long to wait for the bootstrap password
        line. Relaunching an existing install often skips reprinting the
        password, so this should be SHORT (default 30s).
      - `healthz_timeout_s`: how long to wait for `/healthz` to return 200.
        Studio cold-start can take a couple of minutes (default 180s).

    `timeout_s` is accepted for backward compatibility and overrides
    `healthz_timeout_s` if set. With the legacy single-deadline behavior
    a quiet log starved the healthz check and raised a spurious
    TimeoutError even when Studio was up.
    """
    if timeout_s is not None:
        healthz_timeout_s = timeout_s
    log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")

    bin_path = _find_unsloth_bin(install)
    env = {"UNSLOTH_STUDIO_HOME": str(install.home), **(extra_env or {})}
    cmd = ["setsid", "-f", "bash", "-c",
           f'{shlex.quote(bin_path)} studio -p {port} '
           f'2>&1 | tee -a {shlex.quote(str(log_path))}']
    subprocess.Popen(
        cmd,
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    install.port = port
    install.bootstrap_password = _read_bootstrap_password(
        install.home, log_path, time.time() + password_timeout_s
    )

    if wait_for_healthz:
        import urllib.request
        url = f"http://127.0.0.1:{port}/healthz"
        healthz_deadline = time.time() + healthz_timeout_s
        while time.time() < healthz_deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise TimeoutError(
                f"Studio on :{port} did not pass /healthz within "
                f"{healthz_timeout_s}s"
            )

    # Best-effort PID capture via `pgrep -f`.
    try:
        out = _run(f"pgrep -f 'unsloth studio.*-p {port}'", check=False).stdout.strip()
        if out:
            install.pid = int(out.splitlines()[0])
    except Exception:
        pass
    return install


def stop_studio(install: StudioInstall) -> None:
    """Best-effort SIGTERM the launched Studio process group."""
    if install.pid:
        try:
            os.killpg(os.getpgid(install.pid), signal.SIGTERM)
        except Exception:
            pass
