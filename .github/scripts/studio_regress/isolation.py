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
  netns   where bwrap cannot set up loopback (a Colab VM): `unshare --user --map-root-user --net --mount`,
          `lo` brought up, a tmpfs over the sentinel's directory. The user namespace is what lets `lo` come up:
          as plain root in a new network namespace Colab refuses SIOCSIFFLAGS (EPERM). No pid namespace (Colab
          refuses --mount-proc, and under the host's /proc its pids name other processes). Network and the
          sentinel are cut off as in bwrap; the rest of the filesystem stays as it is (the VM is the job's own).
  weak    NO namespace: HTTP(S)_PROXY / ALL_PROXY pointed at a dead loopback port, NO_PROXY for
          localhost, HF_HUB_OFFLINE=1. Only proxy-honouring clients are cut off, and host files
          stay readable. Reported as isolation=weak; journey 5 records it, never claims it.

A host sentinel file is created OUTSIDE every bound path; its path is passed in
STUDIO_REGRESS_SENTINEL so the journey can prove the tool child cannot read it.

    python -m studio_regress.isolation [--mode auto|bwrap|netns|docker|weak] [--rw DIR ...] [--ro DIR ...]
        [--print] -- <command ...>

auto tries bwrap, netns, docker, then weak. STUDIO_REGRESS_ISOLATION_REQUIRE=strong (remote_driver sets
it) leaves weak out: no real namespace is then an error, so the journey is VOID instead of weakly passed.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[4])
DOCKER_IMAGE = os.environ.get("STUDIO_REGRESS_DOCKER_IMAGE", "debian:trixie")
ETC_FILES = (
    "/etc/passwd",
    "/etc/group",
    "/etc/ssl",
    "/etc/fonts",
    "/etc/ld.so.cache",
    "/etc/alternatives",
    "/etc/localtime",
)
DEAD_PROXY = "http://127.0.0.1:9"  # discard port: connections are refused


def sentinel(root: Path | None = None) -> Path:
    root = root or WORKSPACE / "temp" / "studio_regress" / "sentinel"
    root.mkdir(parents = True, exist_ok = True)
    p = root / f"host_secret_{secrets.token_hex(4)}.txt"
    p.write_text("host-only sentinel: a sandboxed tool child must not read this\n")
    return p


def base_env(sentinel_path: Path, mode: str) -> dict:
    return {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "UNSLOTH_STUDIO_OFFLINE": "1",
        "UNSLOTH_DISABLE_UPDATE_CHECK": "1",
        "UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK": "1",
        "STUDIO_REGRESS_ISOLATION": mode,
        "STUDIO_REGRESS_SENTINEL": str(sentinel_path),
    }


# Credentials never cross into the sandbox: the PR's Studio and its code-execution tool run in
# there, and a secret copied into an output dir would leave with the artifacts. docker_argv
# forwards an allowlist; bwrap and weak inherit this process's environment, minus these.
_SECRET = re.compile(
    r"TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_KEY|PRIVATE_KEY|CREDENTIAL|AUTH", re.I
)


def child_env(environ = None) -> dict:
    environ = os.environ if environ is None else environ
    return {k: v for k, v in environ.items() if not _SECRET.search(k)}


def gpu_devices() -> list[str]:
    return sorted(str(p) for p in Path("/dev").glob("nvidia*") if p.is_char_device()) + (
        ["/dev/dri"] if Path("/dev/dri").exists() else []
    )


def bwrap_argv(
    cmd,
    rw,
    ro,
    env,
    gpu = True,
):
    a = [
        "bwrap",
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--die-with-parent",
        "--new-session",
        "--ro-bind",
        "/",
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/run",
        "--tmpfs",
        os.path.expanduser("~"),
    ]
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


# Inside the new namespaces (netns mode): hide the sentinel's directory, bring loopback up (a fresh
# network namespace has lo down; SIOCGIFFLAGS / SIOCSIFFLAGS, so no `ip` binary is needed), exec.
NETNS_PRE = r"""
import fcntl, os, socket, struct, subprocess, sys
subprocess.run(["mount", "--make-rprivate", "/"], check=False)
subprocess.run(["mount", "-t", "tmpfs", "tmpfs", sys.argv[1]], check=True)
s = socket.socket()
flags = struct.unpack("16sH", fcntl.ioctl(s, 0x8913, struct.pack("16sH14x", b"lo", 0))[:18])[1]
fcntl.ioctl(s, 0x8914, struct.pack("16sH14x", b"lo", flags | 1))
os.execvp(sys.argv[2], sys.argv[2:])
"""


def netns_argv(
    cmd,
    env,
    python = None,
):
    """unshare into new user (root mapped) / net / mount namespaces; NETNS_PRE does the rest in there.
    Without --user, Colab gives a new network namespace whose `lo` cannot be brought up (EPERM, measured)."""
    py = python or str(Path(sys.executable).resolve())
    # No pid namespace: Colab refuses --mount-proc, and a new pid namespace under the host's /proc made a
    # child's pid name another process there (engine._owns_port took our own Studio for a stranger).
    # The network and the sentinel are what this mode isolates.
    return [
        "unshare",
        "--user",
        "--map-root-user",
        "--net",
        "--mount",
        "--",
        py,
        "-c",
        NETNS_PRE,
        str(Path(env["STUDIO_REGRESS_SENTINEL"]).parent),
    ] + list(cmd)


def docker_argv(
    cmd,
    rw,
    ro,
    env,
    gpu = True,
    workdir = None,
):
    uid, gid = os.getuid(), os.getgid()
    a = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--ipc",
        "host",
        "-u",
        f"{uid}:{gid}",
        "--tmpfs",
        "/tmp:exec",
        "-v",
        "/usr:/usr:ro",
    ]
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
    passthrough = (
        "CUDA_VISIBLE_DEVICES",
        "PLAYWRIGHT_BROWSERS_PATH",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_XET_CACHE",
        "UNSLOTH_STUDIO_HOME",
        "WORKSPACE",
        "UIDIFF_PORT_START",
    )
    for k in passthrough:
        if k in os.environ and k not in env:
            a += ["-e", f"{k}={os.environ[k]}"]
    a += ["-w", str(workdir or os.getcwd()), DOCKER_IMAGE]
    return a + list(cmd)


def weak_env(env):
    return {
        **env,
        "HTTP_PROXY": DEAD_PROXY,
        "HTTPS_PROXY": DEAD_PROXY,
        "ALL_PROXY": DEAD_PROXY,
        "http_proxy": DEAD_PROXY,
        "https_proxy": DEAD_PROXY,
        "all_proxy": DEAD_PROXY,
        "NO_PROXY": "127.0.0.1,localhost,::1",
        "no_proxy": "127.0.0.1,localhost,::1",
    }


def probe(mode: str) -> tuple[bool, str]:
    """Can this mode start a process whose network is cut but loopback works?"""
    code = (
        "import socket\ns=socket.socket();s.bind(('127.0.0.1',0));s.listen();"
        "socket.create_connection(s.getsockname()).close()\n"
        "try:\n socket.create_connection(('1.1.1.1',53),timeout=2);print('NET')\n"
        "except OSError:\n print('ISOLATED')\n"
        # /proc must be this pid namespace's: engine._owns_port matches a child's pid against it
        "import subprocess, sys\nc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(9)'])\n"
        "import time\nmine = False\n"
        "for _ in range(60):\n"  # the child's cmdline is empty until it has exec'd
        " try:\n  mine = b'time.sleep' in open('/proc/%d/cmdline' % c.pid, 'rb').read()\n"
        " except OSError:\n  pass\n"
        " if mine:\n  break\n"
        " time.sleep(0.05)\n"
        "c.kill()\nprint('PROC_OK' if mine else 'PROC_FOREIGN')"
    )
    env = {"STUDIO_REGRESS_SENTINEL": "/nonexistent/x"}
    # the interpreter the wrapped run uses (a VM's venv python, not /usr/bin/python3, which may not
    # exist or differ); docker sees only /usr, so it keeps that one
    py = str(Path(sys.executable).resolve())
    if mode == "bwrap":
        if not shutil.which("bwrap"):
            return False, "bwrap not installed"
        argv = bwrap_argv([py, "-c", code], [], [Path(py).parents[1]], env, gpu = False)
    elif mode == "netns":
        if not shutil.which("unshare"):
            return False, "unshare not installed"
        import tempfile

        probe_dir = tempfile.mkdtemp(
            prefix = "sr_netns_probe_"
        )  # the tmpfs mount point; removed below
        argv = netns_argv(
            [py, "-c", code], {"STUDIO_REGRESS_SENTINEL": str(Path(probe_dir) / "x")}, python = py
        )
    elif mode == "docker":
        if not shutil.which("docker"):
            return False, "docker not installed"
        argv = docker_argv(["/usr/bin/python3", "-c", code], [], [], env, gpu = False, workdir = "/tmp")
    else:
        return True, "weak (proxy env only)"
    try:
        r = subprocess.run(argv, capture_output = True, text = True, timeout = 120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        if mode == "netns":
            with contextlib.suppress(OSError):
                os.rmdir(probe_dir)
    ok = r.returncode == 0 and "ISOLATED" in r.stdout and "PROC_OK" in r.stdout
    return ok, (r.stdout + r.stderr).strip()[-300:]


STRONG = ("bwrap", "netns", "docker")


def pick(mode: str, require = None) -> str:
    require = os.environ.get("STUDIO_REGRESS_ISOLATION_REQUIRE", "") if require is None else require
    order = STRONG + (() if require == "strong" else ("weak",)) if mode == "auto" else (mode,)
    for m in order:
        if require == "strong" and m not in STRONG:
            break
        ok, why = probe(m)
        print(f"[isolation] {m}: {'ok' if ok else 'unavailable'} {why}", file = sys.stderr)
        if ok:
            return m
    if require == "strong":
        raise SystemExit(
            "isolation: no network namespace on this machine (bwrap / unshare / docker all "
            "unavailable) and STUDIO_REGRESS_ISOLATION_REQUIRE=strong: the isolated journeys "
            "cannot be proven here"
        )
    raise SystemExit(f"isolation mode {mode} unavailable")


def build(mode, cmd, rw, ro, sentinel_path):
    env = base_env(sentinel_path, mode)
    if mode == "bwrap":
        return bwrap_argv(cmd, rw, ro, env), None
    if mode == "docker":
        return docker_argv(cmd, rw, ro, env), None
    if mode == "netns":
        return netns_argv(cmd, env), env
    return list(cmd), weak_env(env)


def main(argv = None):
    p = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    p.add_argument("--mode", default = "auto", choices = ("auto", "bwrap", "netns", "docker", "weak"))
    p.add_argument(
        "--rw", action = "append", default = [], help = "dir bound read-write (outputs, Studio home)"
    )
    p.add_argument(
        "--ro", action = "append", default = [], help = "dir bound read-only (code, venvs, models)"
    )
    p.add_argument("--print", action = "store_true", help = "print the argv / env and exit")
    p.add_argument("cmd", nargs = argparse.REMAINDER)
    a = p.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        p.error("no command")
    mode = a.mode if a.print and a.mode != "auto" else pick(a.mode)
    s = sentinel()
    argv_, extra_env = build(
        mode, cmd, [Path(d).resolve() for d in a.rw], [Path(d).resolve() for d in a.ro], s
    )
    if a.print:
        print(" ".join(argv_))
        if extra_env:
            print(extra_env)
        return 0
    print(f"[isolation] mode={mode} sentinel={s}", file = sys.stderr, flush = True)
    try:
        return subprocess.call(argv_, env = {**child_env(), **(extra_env or {})})
    finally:
        s.unlink(missing_ok = True)


if __name__ == "__main__":
    sys.exit(main())
