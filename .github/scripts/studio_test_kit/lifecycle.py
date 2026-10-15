"""Install + launch Unsloth Studio for arbitrary branches/ports.

The pre-PR vs post-PR test pattern installs Studio twice, once per branch,
each pinned to its own UNSLOTH_STUDIO_HOME so the two installs share
nothing (separate `.venv_t5_*`, `auth/`, `studio.db`, llama.cpp build).

`install_studio(branch=..., home=...)` clones unslothai/unsloth (or
re-uses an existing clone), checks out the branch, then runs
`./install.sh --local` with UNSLOTH_STUDIO_HOME exported.

`launch_studio(install, port=..., log_path=...)` starts `unsloth studio
-p <port>` in its own session and recovers the bootstrap password from
`<home>/auth/.bootstrap_password`, falling back to the log for builds
that printed it there instead. The server outlives the caller, so every
launch needs a matching `stop_studio(install)`, normally in a `finally`.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
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
    pid: Optional[int] = None        # leader of the launch's session
    pid_start: Optional[str] = None  # its start time, so a reused pid is never signalled


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


def health_body_ok(url: str, timeout: float = 2) -> bool:
    """True when ``url`` answers 200 with a JSON object body (Studio's health payload).

    A 200 alone is not enough: Studio's SPA catch-all serves index.html for any unknown GET
    path, so a probe of a route the build does not have (``/healthz`` on current Studio)
    returns 200 text/html from a server whose API may not be up at all.
    """
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            if r.status != 200:
                return False
            body = r.read(65536)
    except Exception:
        return False
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
    except ValueError:
        return False
    return isinstance(data, dict)


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
    # start_new_session rather than `setsid -f`: the same detached session, but the pid of
    # its leader is known here, set before any wait below can raise, so stop_studio can
    # always find the server, its tee and its llama-server children.
    cmd = ["bash", "-c",
           f'{shlex.quote(bin_path)} studio -p {port} '
           f'2>&1 | tee -a {shlex.quote(str(log_path))}']
    proc = subprocess.Popen(
        cmd,
        env={**os.environ, **env},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    install.port = port
    install.pid = proc.pid
    install.pid_start = _start_time(proc.pid)
    # The leader is our child, not init's as under `setsid -f`: reap it as soon as it exits, or
    # it stays a zombie that pid probes (diffusion_bench's stop) read as still running.
    threading.Thread(target=proc.wait, name=f"studio-{port}-reaper", daemon=True).start()
    install.bootstrap_password = _read_bootstrap_password(
        install.home, log_path, time.time() + password_timeout_s
    )

    if wait_for_healthz:
        # Current Studio serves /api/health; older builds answered /healthz. On a current build /healthz is
        # not a route: with a frontend build the SPA catch-all answers it 200 with index.html, so a bare 200
        # would count the HTML shell as healthy. Only a JSON object body counts, on either path.
        urls = [f"http://127.0.0.1:{port}/api/health", f"http://127.0.0.1:{port}/healthz"]
        healthz_deadline = time.time() + healthz_timeout_s
        ready = False
        while not ready and time.time() < healthz_deadline:
            ready = any(health_body_ok(url) for url in urls)
            if not ready:
                time.sleep(1)
        if not ready:
            raise TimeoutError(
                f"Studio on :{port} did not answer /api/health (or /healthz) "
                f"with a JSON health body within {healthz_timeout_s}s"
            )

    return install


_HAVE_PROC = os.path.isdir("/proc/self")


def _start_time(pid: int) -> Optional[str]:
    """Start time of `pid`, or None when it is gone (or on Windows)."""
    if _HAVE_PROC:
        try:
            with open(f"/proc/{pid}/stat") as f:
                return f.read().rsplit(")", 1)[1].split()[19]
        except (OSError, IndexError):
            return None
    if os.name == "nt":
        return None
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True,
                             text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def _is_zombie(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] in ("Z", "X")
    except (OSError, IndexError):
        return True


def _session_members(sid: int) -> list[int]:
    """Live (non-zombie) pids in session `sid`.

    Without /proc (macOS) the leader's process group and its descendants stand in for
    the session.
    """
    if _HAVE_PROC:
        out = []
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:  # a syscall per pid, not a stat read: ~10x cheaper on a busy host
                if os.getsid(int(name)) != sid:
                    continue
            except OSError:
                continue
            if not _is_zombie(int(name)):
                out.append(int(name))
        return out
    try:
        table = subprocess.run(["ps", "-Ao", "pid=,ppid=,pgid=,stat="], capture_output=True,
                               text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = {}
    for line in table.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit() and not parts[3].startswith("Z"):
            rows[int(parts[0])] = (int(parts[1]), int(parts[2]))
    members = {pid for pid, (_, pgid) in rows.items() if pgid == sid}
    grew = True
    while grew:
        kids = {pid for pid, (ppid, _) in rows.items() if ppid in members} - members
        members |= kids
        grew = bool(kids)
    return sorted(members)


def stop_studio(install: StudioInstall, timeout_s: float = 30) -> bool:
    """Stop everything in the launched Studio's session and wait for it to exit.

    The whole session, not just the leader's process group: that is the bash
    wrapper, the backend, its tee and any llama-server it started. SIGTERM
    first, SIGKILL after `timeout_s`. True when nothing is left running.
    Returns at once when the session is already gone, and never signals a
    session whose leader pid now belongs to a different process.
    """
    if not install.pid:
        return True
    if os.name == "nt":  # no sessions: the launch's process tree
        r = subprocess.run(["taskkill", "/PID", str(install.pid), "/T", "/F"], capture_output=True)
        return r.returncode in (0, 128)  # 128: already gone
    now = _start_time(install.pid)
    if install.pid_start and now and now != install.pid_start:
        return True  # the pid was reused; our session ended long ago
    for sig, wait_s in ((signal.SIGTERM, timeout_s), (signal.SIGKILL, 5.0)):
        sent: set[int] = set()
        deadline = time.monotonic() + wait_s
        while True:
            members = _session_members(install.pid)
            if not members:
                return True
            if sent and time.monotonic() >= deadline:
                break
            for pid in set(members) - sent:  # also whatever it spawned since the last pass
                try:
                    os.kill(pid, sig)
                except OSError:
                    pass
                sent.add(pid)
            time.sleep(0.1)
    return False
