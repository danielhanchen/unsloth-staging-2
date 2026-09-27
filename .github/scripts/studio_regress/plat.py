"""Cross-platform process and lock helpers for studio_regress (Linux, macOS, Windows).

The fast tier runs on Linux, macOS and Windows staging runners, so nothing on its path may import
fcntl, read /proc or call os.killpg unconditionally.

  lock(fh, exclusive=True, blocking=True)   fcntl.flock on POSIX, msvcrt.locking on Windows (no
                                            shared mode there: a shared lock is exclusive)
  unlock(fh) / locked(path, exclusive)      release / open-lock-release context manager
  pid_alive(pid)                            signal 0 on POSIX; OpenProcess on Windows (os.kill(pid, 0)
                                            on Windows TERMINATES the process)
  process_table()                           {pid: ppid}: /proc, `ps -A -o pid=,ppid=`, or
                                            Get-CimInstance Win32_Process (wmic as a last resort)
  descendants(root)                         root (when alive) plus every process below it
  listening_pids(port)                      pids listening on 127.0.0.1:port (lsof / Get-NetTCPConnection),
                                            None when the platform tool is missing
  process_env(pid)                          another process's environment (Linux /proc only, else None)
  group_kwargs()                            Popen kwargs that put the child in its own group / session
  signal_tree(root, pids, hard)             stop what the suite spawned: killpg on POSIX; CTRL_BREAK, then
                                            taskkill /T /F on Windows; plus every pid in `pids`
  stop_tree(root, grace_s, proc=)           soft signal_tree, wait, then hard; returns the survivors
  link(dest, target)                        symlink (target_is_directory on Windows), else a junction /
                                            a copy when Windows withholds the symlink privilege

Ownership is always "a process this suite spawned, or one of its descendants": nothing here kills
by command-line match.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


PROC = Path("/proc")


def _is_windows():
    return sys.platform == "win32"


# ------------------------------------------------------------------ locks
def lock(fh, exclusive=True, blocking=True):
    """Lock an open file. Non-blocking: returns False when someone else holds it."""
    if _is_windows():
        import msvcrt
        fh.seek(0)
        mode = msvcrt.LK_NBLCK
        while True:
            try:
                msvcrt.locking(fh.fileno(), mode, 1)
                return True
            except OSError:
                if not blocking:
                    return False
                time.sleep(0.2)
    import fcntl
    flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        fcntl.flock(fh, flags | (0 if blocking else fcntl.LOCK_NB))
        return True
    except (BlockingIOError, PermissionError):
        if blocking:
            raise
        return False


def unlock(fh):
    if _is_windows():
        import msvcrt
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return
    import fcntl
    fcntl.flock(fh, fcntl.LOCK_UN)


@contextlib.contextmanager
def locked(path, exclusive=True):
    """Open `path` (created if missing) and hold a lock on it for the block."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        lock(fh, exclusive)
        try:
            yield fh
        finally:
            unlock(fh)


# ------------------------------------------------------------------ processes
def pid_alive(pid):
    pid = int(pid)
    if pid <= 0:
        return False
    if _is_windows():
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return k32.GetLastError() == 5          # ERROR_ACCESS_DENIED: exists, not ours to query
        try:
            code = ctypes.c_ulong()
            ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
            return bool(ok) and code.value == 259   # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _out(argv, timeout=30):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def _pairs(text):
    tab = {}
    for line in (text or "").splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            tab[int(parts[0])] = int(parts[1])
    return tab


def _powershell():
    return shutil.which("powershell") or shutil.which("pwsh")


def process_table():
    """{pid: ppid} for every visible process ({} when the platform gives nothing)."""
    if not _is_windows() and (PROC / "self" / "stat").exists():
        tab = {}
        for p in PROC.iterdir():
            if not p.name.isdigit():
                continue
            try:
                stat = (p / "stat").read_text()
                tab[int(p.name)] = int(stat.rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
        return tab
    if _is_windows():
        ps = _powershell()
        if ps:
            out = _out([ps, "-NoProfile", "-NonInteractive", "-Command",
                        "Get-CimInstance Win32_Process | ForEach-Object "
                        "{ \"$($_.ProcessId) $($_.ParentProcessId)\" }"], timeout=60)
            if out:
                return _pairs(out)
        out = _out(["wmic", "process", "get", "ProcessId,ParentProcessId", "/format:csv"], timeout=60)
        # csv columns: Node,ParentProcessId,ProcessId
        tab = {}
        for line in (out or "").splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                tab[int(parts[2])] = int(parts[1])
        return tab
    return _pairs(_out(["ps", "-A", "-o", "pid=", "-o", "ppid="]))


def descendants(root, table=None):
    """{root (if alive)} plus every process whose parent chain reaches root. A parent that already
    exited still links its children (Windows keeps the dead ParentProcessId)."""
    if not root:
        return set()
    root = int(root)
    table = process_table() if table is None else table
    kids = {}
    for pid, ppid in table.items():
        if pid != ppid:
            kids.setdefault(ppid, []).append(pid)
    out, todo = set(), [root]
    while todo:
        p = todo.pop()
        for c in kids.get(p, ()):
            if c not in out and c != root:
                out.add(c)
                todo.append(c)
    if root in table or (not table and pid_alive(root)):
        out.add(root)
    return out


def listening_pids(port):
    """Pids listening on TCP `port`, or None when this host has no tool to tell."""
    port = int(port)
    if _is_windows():
        ps = _powershell()
        if not ps:
            return None
        out = _out([ps, "-NoProfile", "-NonInteractive", "-Command",
                    f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue "
                    "| ForEach-Object { $_.OwningProcess }"], timeout=60)
        if out is None:
            return None
        return {int(x) for x in out.split() if x.isdigit()}
    if not shutil.which("lsof"):
        return None
    try:
        r = subprocess.run(["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode not in (0, 1):   # 1: nothing listening
        return None
    return {int(x) for x in r.stdout.split() if x.isdigit()}


def process_env(pid):
    """{name: value} of another process (Linux /proc), None where the platform cannot tell."""
    try:
        raw = (PROC / str(int(pid)) / "environ").read_bytes()
    except OSError:
        return None
    env = {}
    for kv in raw.split(b"\0"):
        if b"=" in kv:
            k, v = kv.split(b"=", 1)
            env[k.decode(errors="replace")] = v.decode(errors="replace")
    return env


def group_kwargs():
    """Popen kwargs: own session (POSIX) / own process group (Windows), so the tree can be stopped."""
    if _is_windows():
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)}
    return {"start_new_session": True}


def _kill_one(pid, hard):
    try:
        if _is_windows():
            os.kill(pid, signal.SIGTERM)   # TerminateProcess
        else:
            os.kill(pid, signal.SIGKILL if hard else signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def signal_tree(root=None, pids=(), hard=False, interrupt=False):
    """Signal the group led by `root` (spawned with group_kwargs()) and every pid in `pids`.
    POSIX: killpg SIGTERM / SIGINT / SIGKILL. Windows: CTRL_BREAK to the group when soft, then
    `taskkill /T` (/F when hard) on root, and TerminateProcess on every listed pid when hard."""
    pids = {int(p) for p in pids if p}
    if _is_windows():
        if root:
            if not hard:
                try:
                    os.kill(int(root), signal.CTRL_BREAK_EVENT)
                except (OSError, AttributeError, ValueError):
                    pass
            _out(["taskkill", "/PID", str(int(root)), "/T"] + (["/F"] if hard else []))
        if hard:
            for p in pids:
                _kill_one(p, True)
        return
    sig = signal.SIGKILL if hard else signal.SIGINT if interrupt else signal.SIGTERM
    groups = set()
    if root:
        groups.add(int(root))
    for p in pids:
        try:
            groups.add(os.getpgid(p))
        except (ProcessLookupError, PermissionError, OSError):
            pass
    for g in groups:
        if g <= 1 or g == os.getpgrp():
            continue
        try:
            os.killpg(g, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    for p in pids:
        if p != os.getpid():
            try:
                os.kill(p, sig)
            except (ProcessLookupError, PermissionError, OSError):
                pass


def stop_tree(root, grace_s=20.0, extra=(), proc=None):
    """Soft-stop root's tree, wait up to grace_s, then hard-kill whatever is left. `proc`: root's
    Popen, polled so the exited root is reaped (a zombie still answers signal 0). The tree is
    snapshotted first: on POSIX a child re-parents to init once root exits. Returns the pids still
    alive at the end (should be empty)."""
    def alive():
        if proc is not None:
            proc.poll()
        return {p for p in tree if pid_alive(p)} | descendants(root)
    tree = set()
    tree = alive() | {int(p) for p in extra if p}
    if not tree:
        return set()
    signal_tree(root, tree, hard=False)
    deadline = time.time() + grace_s
    while time.time() < deadline:
        tree = alive()
        if not tree:
            return set()
        time.sleep(0.5)
    signal_tree(root, tree, hard=True)
    for _ in range(20):
        tree = alive()
        if not tree:
            break
        time.sleep(0.5)
    return tree


# ------------------------------------------------------------------ filesystem
def link(dest, target):
    """dest -> target. A directory symlink on Windows needs target_is_directory; without the
    symlink privilege a directory falls back to a junction and a file to a copy."""
    dest, target = Path(dest), Path(target)
    is_dir = target.is_dir()
    try:
        dest.symlink_to(target, target_is_directory=is_dir)
        return
    except OSError:
        if not _is_windows():
            raise
    if is_dir:
        import _winapi
        _winapi.CreateJunction(str(target), str(dest))
    else:
        shutil.copy2(target, dest)

