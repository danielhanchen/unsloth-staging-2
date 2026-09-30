"""Reclaim the per-SHA caches studio_regress leaves in temp/: Studio installs (multi-GB each),
their source worktrees, and the Core A/B trees.

    python studio_regress.py gc [--days 3] [--dry-run]

An entry goes once it has not been used for --days (the `.last_used` marker ensure_install and
core.tree touch on every reuse; the directory mtime when there is none) AND nothing holds it:
its creation lock is free and no live process has the path in its cwd, executable, command line
or environment (a running Studio's python lives inside its install; a regression run carries its
trees on PYTHONPATH). Git worktrees are removed through `git worktree remove` so the parent repo
keeps no dangling entry. outputs/ (evidence) is never touched.

The host-shared install cache (cache.py, $STUDIO_REGRESS_SHARED_DIR) is collected too. There an
entry is also kept while any run, of any user, holds its `<key>.use` lock (run.py holds it shared
for the whole run), since another user's processes are not fully visible in /proc.

Workspace references (shared store only). Every workspace that uses a shared install leaves
`refs/<key>/<id>` holding its absolute path. `launcher.sh cleanup` runs `gc --release WS`, which
drops that workspace's references and removes every entry whose references are all gone (the
recorded workspace no longer exists), whatever its age. The same orphan rule applies to a plain
`gc`, so a workspace deleted without cleanup is caught on the next pass. Entries with no refs dir
(built before references existed) keep the --days rule only. An entry another user's files
still pin stays an orphan, and that user's next cleanup or gc finishes it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

WS = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[3])
ROOT = WS / "temp" / "studio_regress"
MARKER = ".last_used"


def touch(d):
    """Record a use of a cached install / tree (ignored when the dir is gone)."""
    try:
        (Path(d) / MARKER).touch()
    except OSError:
        pass


def last_used(d):
    m = Path(d) / MARKER
    try:
        return m.stat().st_mtime if m.exists() else Path(d).stat().st_mtime
    except OSError:
        return 0.0


def _ref_id(workspace):
    return hashlib.sha1(str(Path(workspace).resolve()).encode()).hexdigest()[:16]


def add_ref(
    root,
    key,
    workspace = None,
):
    """Record that `workspace` ($WORKSPACE by default) uses shared entry `key`. Best effort."""
    workspace = workspace or os.environ.get("WORKSPACE")
    if not workspace:
        return
    d = Path(root) / "refs" / key
    try:
        d.mkdir(parents = True, exist_ok = True)
        f = d / _ref_id(workspace)
        if not f.exists():
            f.write_text(str(Path(workspace).resolve()) + "\n")
    except OSError:
        pass


def _alive(path):
    """The workspace dir still exists. Unreadable (another user's tree) counts as alive."""
    try:
        Path(path).stat()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return True


def orphaned(root, key):
    """True when the entry has references and none of their workspaces exists any more."""
    d = Path(root) / "refs" / key
    if not d.is_dir():
        return False
    for f in d.iterdir():
        try:
            if _alive(f.read_text().strip()):
                return False
        except OSError:
            return False  # unreadable reference: assume it is live
    return True


def release(
    root,
    workspace,
    dry_run = False,
    log = print,
):
    """Drop `workspace`'s references in the shared store (its entries become orphans once no
    other live workspace refers to them)."""
    refs = Path(root) / "refs"
    rid = _ref_id(workspace)
    for d in sorted(refs.iterdir()) if refs.is_dir() else ():
        f = d / rid
        if f.exists():
            log(f"{'would release' if dry_run else 'release'} {d.name}")
            if not dry_run:
                with contextlib.suppress(OSError):
                    f.unlink()


def _proc_refs():
    """Every cwd / exe / cmdline / environ string of this user's live processes. Without /proc
    (macOS, Windows) only command lines are visible: `ps` / Win32_Process. None when nothing can
    be listed, which collect() treats as "everything in use"."""
    from studio_regress import plat

    if plat._is_windows() or not (plat.PROC / "self").exists():
        return _cmdline_refs()
    refs = []
    for p in plat.PROC.iterdir():
        if not p.name.isdigit() or p.name == str(os.getpid()):
            continue
        for link in ("cwd", "exe"):
            try:
                refs.append(os.readlink(p / link))
            except OSError:
                pass
        for f in ("cmdline", "environ"):
            try:
                refs.append((p / f).read_bytes().replace(b"\0", b" ").decode(errors = "replace"))
            except OSError:
                pass
    return refs


def _cmdline_refs():
    from studio_regress import plat

    if plat._is_windows():
        ps = shutil.which("powershell") or shutil.which("pwsh")
        argv = (
            [
                ps,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_Process | ForEach-Object { $_.ExecutablePath; $_.CommandLine }",
            ]
            if ps
            else None
        )
    else:
        argv = ["ps", "-A", "-ww", "-o", "command="]
    try:
        r = subprocess.run(argv, capture_output = True, text = True, timeout = 60) if argv else None
    except (OSError, subprocess.SubprocessError):
        r = None
    if r is None or r.returncode != 0:
        return None
    return [line for line in r.stdout.splitlines() if line.strip()]


def in_use(path, refs):
    if refs is None:  # could not list processes: never reclaim blind
        return True
    s = str(Path(path).resolve())
    return any(s in r for r in refs)


def _lock_free(lock):
    from studio_regress import plat
    if lock is None or not lock.exists():
        return True
    try:
        with open(lock, "a+") as fh:
            if not plat.lock(fh, blocking = False):
                return False
            plat.unlock(fh)
            return True
    except OSError:
        return False


def _remove(d):
    """git worktree remove when `d` is a linked worktree, then make sure the dir is gone."""
    if (d / ".git").is_file():
        common = subprocess.run(
            ["git", "-C", str(d), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output = True,
            text = True,
        ).stdout.strip()
        if common:
            main = Path(common).parent
            subprocess.run(
                ["git", "-C", str(main), "worktree", "remove", "--force", str(d)],
                capture_output = True,
                text = True,
            )
            shutil.rmtree(d, ignore_errors = True)
            subprocess.run(
                ["git", "-C", str(main), "worktree", "prune"], capture_output = True, text = True
            )
            return
    shutil.rmtree(d, ignore_errors = True)


def candidates(root = ROOT):
    """(dir, its creation lock, linked source worktree or None) for every cached entry."""
    out = []
    inst = root / "installs"
    for d in sorted(inst.iterdir()) if inst.is_dir() else ():
        if d.is_dir():
            out.append((d, inst / f"{d.name}.lock", root / "src" / d.name))
    trees = root / "trees"
    for d in sorted(trees.iterdir()) if trees.is_dir() else ():
        if d.is_dir():
            out.append((d, trees / f"{d.name}.lock", None))
    return out


def _use_lock(d):
    """installs/<key>.use (cache.hold: held shared by every run using the install), None for trees."""
    return d.parent / f"{d.name}.use" if d.parent.name == "installs" else None


@contextlib.contextmanager
def _exclusive(lock):
    """Hold `lock` exclusively for the block when nobody else holds it; yields False otherwise
    (also for a lock file this user cannot open). None (trees have no use lock): True."""
    from studio_regress import plat

    if lock is None:
        yield True
        return
    try:  # created when missing (an entry from before .use existed): a run's hold() then blocks
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o666)
        with contextlib.suppress(OSError, AttributeError):
            os.fchmod(fd, 0o666)
        fh = os.fdopen(fd, "a+")
    except OSError:
        yield False
        return
    with fh:
        if not plat.lock(fh, blocking = False):
            yield False
            return
        try:
            yield True
        finally:
            plat.unlock(fh)


def collect(
    days = 3.0,
    dry_run = False,
    root = ROOT,
    refs = None,
    now = None,
    log = print,
    orphans_only = False,
):
    now = time.time() if now is None else now
    refs = _proc_refs() if refs is None else refs
    removed, kept = [], []
    for d, lock, src in candidates(root):
        age_d = (now - last_used(d)) / 86400
        orphan = d.parent.name == "installs" and orphaned(root, d.name)
        why = (
            "referenced"
            if orphans_only and not orphan
            else "recent"
            if age_d < days and not orphan
            else "in use"
            if in_use(d, refs) or (src is not None and src.exists() and in_use(src, refs))
            else "locked"
            if not _lock_free(lock)
            else None
        )
        if why:
            kept.append((d, why))
            continue
        # Held through the removal: a run takes this shared BEFORE it trusts the stamp, so it
        # either keeps gc out or waits and then finds the entry gone (and rebuilds it).
        with _exclusive(_use_lock(d)) as free:
            if not free:
                kept.append((d, "in use"))
                continue
            removed.append(d)
            log(
                f"{'would remove' if dry_run else 'remove'} {d} "
                + ("(no live workspace refers to it)" if orphan else f"(unused {age_d:.1f} d)")
            )
            if dry_run:
                continue
            _remove(d)
            if src is not None and src.exists():
                _remove(src)
            if d.exists():  # another user's files in the shared cache that we may not delete
                log(f"could not fully remove {d}")
                continue
            lock.unlink(missing_ok = True)
            shutil.rmtree(root / "refs" / d.name, ignore_errors = True)
    return removed, kept


# ------------------------------------------------------------------ leftover processes
# run.py exports this (the run's root) before it starts anything, so every Studio, browser, worker
# and suite the run launches carries it in its environment, whatever session it moved to.
RUN_ROOT_ENV = "STUDIO_REGRESS_RUN_ROOT"


def _ancestors(pid, table):
    out = set()
    while pid and pid not in out:
        out.add(pid)
        pid = table.get(pid)
    return out


def run_processes(
    root = None,
    table = None,
    env_of = None,
):
    """{pid: run root} of live processes carrying RUN_ROOT_ENV (only `root`'s when given), minus
    this process and its ancestors (a nested run inherits its parent's value)."""
    from studio_regress import plat

    table = plat.process_table() if table is None else table
    env_of = env_of or plat.process_env
    mine = _ancestors(os.getpid(), table)
    want = str(Path(root).resolve()) if root else None
    out = {}
    for pid in table:
        if pid in mine:
            continue
        env = env_of(pid) or {}
        v = env.get(RUN_ROOT_ENV)
        if env.get(
            "NVSMI_SHIM_ACTIVE"
        ):  # the shared nvidia-smi cache refresher: short-lived, not the run's
            continue
        if v and (want is None or v == want):
            out[pid] = v
    return out


def _cmd(pid):
    try:
        return (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(errors = "replace")
            .strip()[:200]
        )
    except OSError:
        return "?"


def reap_run(
    root,
    log = print,
    stop = None,
):
    """At the end of a run: stop every process still carrying this run's root. Returns
    [{"pid", "cmd", "stopped"}] (empty when the run cleaned up after itself)."""
    from studio_regress import plat

    stop = stop or (lambda pid: not plat.stop_tree(pid, grace_s = 15))
    left = []
    for pid in sorted(run_processes(root)):
        rec = {"pid": pid, "cmd": _cmd(pid)}
        rec["stopped"] = bool(stop(pid))
        left.append(rec)
        log(
            f"leftover process {pid} ({rec['cmd'][:120]}): {'stopped' if rec['stopped'] else 'STILL ALIVE'}"
        )
    return left


def finished_root(root):
    """True when root's report.json says the run finished (its processes are leftovers)."""
    try:
        return bool(json.loads((Path(root) / "report.json").read_text()).get("finished_utc"))
    except (OSError, ValueError):
        return False


def main(argv = None):
    import argparse

    p = argparse.ArgumentParser(description = "Reclaim unused studio_regress installs and trees")
    p.add_argument(
        "--days", type = float, default = 3.0, help = "keep anything used within this many days"
    )
    p.add_argument("--dry-run", action = "store_true")
    p.add_argument(
        "--release",
        metavar = "WORKSPACE",
        help = "drop WORKSPACE's references to shared installs, then remove only the shared "
        "entries no live workspace refers to (launcher.sh cleanup runs this)",
    )
    p.add_argument(
        "--procs",
        action = "store_true",
        help = "list processes a switchboard run started (STUDIO_REGRESS_RUN_ROOT); LEFTOVER when "
        "their run already finished. With --kill, stop the leftovers",
    )
    p.add_argument("--kill", action = "store_true", help = "with --procs: stop leftover processes")
    a = p.parse_args(argv)
    if a.procs:
        procs = run_processes()
        left = {pid: r for pid, r in procs.items() if finished_root(r)}
        for pid, r in sorted(procs.items()):
            print(f"{'LEFTOVER' if pid in left else 'running '} {pid:>8} {r}  {_cmd(pid)[:100]}")
        if a.kill and left:
            from studio_regress import plat
            for pid in left:
                plat.stop_tree(pid, grace_s = 15)
        print(
            f"{len(procs)} run process(es), {len(left)} leftover"
            + (" (stopped)" if a.kill and left else "")
        )
        return 1 if left and not a.kill else 0
    from studio_regress import cache

    shared = cache.shared_root(create = False)
    if a.release:
        if shared is None:
            return 0
        release(shared, a.release, a.dry_run)
        removed, kept = collect(a.days, a.dry_run, root = shared, orphans_only = True)
        print(
            f"{'would remove' if a.dry_run else 'removed'} {len(removed)} orphaned shared install(s)"
        )
        return 0
    removed, kept = collect(a.days, a.dry_run)
    if shared is not None and shared.resolve() != ROOT.resolve():
        r2, k2 = collect(a.days, a.dry_run, root = shared)
        removed, kept = removed + r2, kept + k2
    busy = [f"{d.name} ({w})" for d, w in kept if w != "recent"]
    print(
        f"{'would remove' if a.dry_run else 'removed'} {len(removed)}, kept {len(kept)}"
        + (f"; held: {', '.join(busy)}" if busy else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
