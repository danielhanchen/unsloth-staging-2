#!/usr/bin/env python3
"""Run a studio_regress command with NO network except loopback, for journey 5
(full_access_isolated: Full Access on, web search off, code execution on).

The whole run goes inside: Studio, its tool children, Chromium and the journey driver share one
network namespace, so the browser still reaches Studio on 127.0.0.1 while nothing reaches the
internet. Models must be prefetched first (HF_HUB_OFFLINE=1 inside).

Modes (auto picks the first that works):
  bwrap   `bwrap --unshare-net --unshare-pid --die-with-parent`, rootfs read-only, tmpfs /tmp and
          $HOME, rw binds only for the output / Studio-home dirs. Needs unprivileged user
          namespaces: on GitHub ubuntu-latest run
          `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0` first. Blocked on hosts
          whose AppArmor restricts userns (bwrap: "setting up uid map: Permission denied").
  docker  `docker run --network none`, host /usr + a few /etc files bind-mounted read-only over a
          debian:trixie base (same glibc family as the host, so the host's Python / venvs / CUDA
          user-space libs run unchanged), GPU device nodes passed through, only the listed dirs
          bound. Needs the `docker` group.
  weak    NO namespace: HTTP(S)_PROXY / ALL_PROXY pointed at a dead loopback port, NO_PROXY for
          localhost, HF_HUB_OFFLINE=1. Only proxy-honouring clients are cut off, and host files
          stay readable. Reported as isolation=weak; journey 5 records it, never claims it.

A host sentinel file is created OUTSIDE every bound path; its path is passed in
STUDIO_REGRESS_SENTINEL so the journey can prove the tool child cannot read it.

    python -m studio_regress.isolation [--mode auto|bwrap|docker|weak] [--rw DIR ...] [--ro DIR ...]
        [--print] -- <command ...>
"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[4])
DOCKER_IMAGE = os.environ.get("STUDIO_REGRESS_DOCKER_IMAGE", "debian:trixie")
ETC_FILES = ("/etc/passwd", "/etc/group", "/etc/ssl", "/etc/fonts", "/etc/ld.so.cache",
             "/etc/alternatives", "/etc/localtime")
DEAD_PROXY = "http://127.0.0.1:9"      # discard port: connections are refused


def sentinel(root: Path | None = None) -> Path:
    root = root or WORKSPACE / "temp" / "studio_regress" / "sentinel"
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"host_secret_{secrets.token_hex(4)}.txt"
    p.write_text("host-only sentinel: a sandboxed tool child must not read this\n")
    return p


def base_env(sentinel_path: Path, mode: str) -> dict:
    return {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "UNSLOTH_STUDIO_OFFLINE": "1",
            "UNSLOTH_DISABLE_UPDATE_CHECK": "1", "UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK": "1",
            "STUDIO_REGRESS_ISOLATION": mode, "STUDIO_REGRESS_SENTINEL": str(sentinel_path)}


# Credentials never cross into the sandbox: the PR's Studio and its code-execution tool run in
# there, and a secret copied into an output dir would leave with the artifacts. docker_argv
# forwards an allowlist; bwrap and weak inherit this process's environment, minus these.
_SECRET = re.compile(r"TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_KEY|PRIVATE_KEY|CREDENTIAL|AUTH", re.I)


def child_env(environ=None) -> dict:
    environ = os.environ if environ is None else environ
    return {k: v for k, v in environ.items() if not _SECRET.search(k)}


def gpu_devices() -> list[str]:
    return sorted(str(p) for p in Path("/dev").glob("nvidia*") if p.is_char_device()) + (
        ["/dev/dri"] if Path("/dev/dri").exists() else [])


def bwrap_argv(cmd, rw, ro, env, gpu=True):
    a = ["bwrap", "--unshare-net", "--unshare-pid", "--unshare-ipc", "--die-with-parent",
         "--new-session", "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
         "--tmpfs", "/tmp", "--tmpfs", "/run", "--tmpfs", os.path.expanduser("~")]
    # Hide the sentinel's directory (it lives under the workspace, which "/" exposes read-only).
    a += ["--tmpfs", str(Path(env["STUDIO_REGRESS_SENTINEL"]).parent)]
    if gpu:
        for d in gpu_devices():
            a += ["--dev-bind", d, d]
    a += ["--dev-bind", "/dev/shm", "/dev/shm"]
    for d in ro:
        a += ["--ro-bind", str(d), str(d)]
    for d in rw:
        a += ["--bind", str(d), str(d)]
    for k, v in env.items():
        a += ["--setenv", k, v]
    a += ["--setenv", "HOME", os.path.expanduser("~")]
    return a + ["--"] + list(cmd)


def docker_argv(cmd, rw, ro, env, gpu=True, workdir=None):
    uid, gid = os.getuid(), os.getgid()
    a = ["docker", "run", "--rm", "--network", "none", "--ipc", "host", "-u", f"{uid}:{gid}",
         "--tmpfs", "/tmp:exec", "-v", "/usr:/usr:ro"]
    for f in ETC_FILES:
        if Path(f).exists():
            a += ["-v", f"{f}:{f}:ro"]
    if gpu:
        for d in gpu_devices():
            a += ["--device", d]
    for d in ro:
        a += ["-v", f"{d}:{d}:ro"]
    for d in rw:
        a += ["-v", f"{d}:{d}"]
    env = {**env, "HOME": "/tmp/home", "PATH": "/usr/local/bin:/usr/bin:/bin"}
    for k, v in env.items():
        a += ["-e", f"{k}={v}"]
    passthrough = ("CUDA_VISIBLE_DEVICES", "PLAYWRIGHT_BROWSERS_PATH", "HF_HOME", "HF_HUB_CACHE",
                   "HF_XET_CACHE", "UNSLOTH_STUDIO_HOME", "WORKSPACE", "UIDIFF_PORT_START")
    for k in passthrough:
        if k in os.environ and k not in env:
            a += ["-e", f"{k}={os.environ[k]}"]
    a += ["-w", str(workdir or os.getcwd()), DOCKER_IMAGE]
    return a + list(cmd)


def weak_env(env):
    return {**env, "HTTP_PROXY": DEAD_PROXY, "HTTPS_PROXY": DEAD_PROXY, "ALL_PROXY": DEAD_PROXY,
            "http_proxy": DEAD_PROXY, "https_proxy": DEAD_PROXY, "all_proxy": DEAD_PROXY,
            "NO_PROXY": "127.0.0.1,localhost,::1", "no_proxy": "127.0.0.1,localhost,::1"}


def probe(mode: str) -> tuple[bool, str]:
    """Can this mode start a process whose network is cut but loopback works?"""
    code = ("import socket\ns=socket.socket();s.bind(('127.0.0.1',0));s.listen();"
            "socket.create_connection(s.getsockname()).close()\n"
            "try:\n socket.create_connection(('1.1.1.1',53),timeout=2);print('NET')\n"
            "except OSError:\n print('ISOLATED')")
    env = {"STUDIO_REGRESS_SENTINEL": "/nonexistent/x"}
    py = "/usr/bin/python3"      # under /usr, so it exists inside every mode
    if mode == "bwrap":
        if not shutil.which("bwrap"):
            return False, "bwrap not installed"
        argv = bwrap_argv([py, "-c", code], [], [], env, gpu=False)
    elif mode == "docker":
        if not shutil.which("docker"):
            return False, "docker not installed"
        argv = docker_argv([py, "-c", code], [], [], env, gpu=False, workdir="/tmp")
    else:
        return True, "weak (proxy env only)"
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"{type(e).__name__}: {e}"
    ok = r.returncode == 0 and "ISOLATED" in r.stdout
    return ok, (r.stdout + r.stderr).strip()[-300:]


def pick(mode: str) -> str:
    for m in (("bwrap", "docker", "weak") if mode == "auto" else (mode,)):
        ok, why = probe(m)
        print(f"[isolation] {m}: {'ok' if ok else 'unavailable'} {why}", file=sys.stderr)
        if ok:
            return m
    raise SystemExit(f"isolation mode {mode} unavailable")


def build(mode, cmd, rw, ro, sentinel_path):
    env = base_env(sentinel_path, mode)
    if mode == "bwrap":
        return bwrap_argv(cmd, rw, ro, env), None
    if mode == "docker":
        return docker_argv(cmd, rw, ro, env), None
    return list(cmd), weak_env(env)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", default="auto", choices=("auto", "bwrap", "docker", "weak"))
    p.add_argument("--rw", action="append", default=[], help="dir bound read-write (outputs, Studio home)")
    p.add_argument("--ro", action="append", default=[], help="dir bound read-only (code, venvs, models)")
    p.add_argument("--print", action="store_true", help="print the argv / env and exit")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        p.error("no command")
    mode = a.mode if a.print and a.mode != "auto" else pick(a.mode)
    s = sentinel()
    argv_, extra_env = build(mode, cmd, [Path(d).resolve() for d in a.rw],
                             [Path(d).resolve() for d in a.ro], s)
    if a.print:
        print(" ".join(argv_))
        if extra_env:
            print(extra_env)
        return 0
    print(f"[isolation] mode={mode} sentinel={s}", file=sys.stderr, flush=True)
    try:
        return subprocess.call(argv_, env={**child_env(), **(extra_env or {})})
    finally:
        s.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
