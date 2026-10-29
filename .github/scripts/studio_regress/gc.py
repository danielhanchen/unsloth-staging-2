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
import sys
import time
from pathlib import Path

WS = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[3])
ROOT = WS / "temp" / "studio_regress"
MARKER = ".last_used"
TRASH = ".trash-"
NOT_DELETABLE = "not deletable here"


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


def deletable(d):
    """True when this user can remove `d` and everything under it: write+search on its parent (to
    rename it) and on every directory inside (to unlink each entry), and no sticky dir holding
    another user's file. Checked up front, so an entry is either removed whole or left untouched;
    the shared store holds other users' installs (dirs 2775 / 2770 group `users`), which a user
    outside that group cannot delete at all."""
    import stat

    me = os.geteuid()
    d = Path(d)
    if not os.access(d.parent, os.W_OK | os.X_OK):
        return False
    bad = []
    for dp, dirs, files in os.walk(d, onerror = bad.append):
        if not os.access(dp, os.W_OK | os.X_OK):
            return False
        with contextlib.suppress(OSError):
            st = os.lstat(dp)
            if st.st_mode & stat.S_ISVTX and st.st_uid != me:
                for n in dirs + files:
                    if os.lstat(os.path.join(dp, n)).st_uid != me:
                        return False
    return not bad


def _trash(d):
    """Rename `d` aside in its own dir, then delete it. False (nothing touched) when the rename is
    refused: a live entry is never left half removed. A trash dir another failure leaves behind is
    swept by sweep_trash()."""
    d = Path(d)
    t = d.parent / f"{TRASH}{d.name}-{os.getpid()}"
    try:
        os.rename(d, t)
    except OSError:
        return False
    _purge(t)
    return True


_PURGERS: list = []
# One delete at a time across every user and workspace: a removed entry is ~100k files, and 18
# `rm -rf` at once (8 per gc process, one gc per user) held the XFS inode and buffer locks every
# uv install on the filesystem needed, so those installs sat in D state with no I/O for minutes.
# `ionice -c3` cannot prevent that (the NVMe queues use the `none` scheduler, which ignores it).
PURGE_LOCK = "gc.purge.lock"


def _purge_lock():
    """The host-wide delete slot: `gc.purge.lock` in the shared lock dir (gpu_queue's, 0777, so every
    user's gc takes the same one, for the shared store and workspace trees alike; on one filesystem
    they contend the same way). $STUDIO_REGRESS_PURGE_LOCK overrides. Created 0666; None when it
    cannot be opened (then no slot is held)."""
    lock = os.environ.get("STUDIO_REGRESS_PURGE_LOCK")
    if not lock:
        try:
            import gpu_queue
            lock = gpu_queue.resolve_lock_dir() / PURGE_LOCK
        except ImportError:
            lock = Path("/mnt/disks/unslothai/shared/studio-regress-locks") / PURGE_LOCK
    lock = Path(lock)
    with contextlib.suppress(OSError):
        lock.parent.mkdir(parents = True, exist_ok = True)
    try:
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o666)
        with contextlib.suppress(OSError, AttributeError):
            os.fchmod(fd, 0o666)
        os.close(fd)
    except OSError:
        if not os.access(lock, os.R_OK):
            return None
    return lock


@contextlib.contextmanager
def _purge_slot():
    """Hold the delete slot for an inline delete (blocking until the running one ends)."""
    from studio_regress import plat

    lock = _purge_lock()
    try:
        fh = open(lock, "r") if lock else None
    except OSError:
        fh = None
    if fh is None:
        yield
        return
    with fh:
        plat.lock(fh, blocking = True)
        try:
            yield
        finally:
            plat.unlock(fh)


def _purge(t):
    """Delete a renamed-aside entry. A ~9 GB venv is ~100k files, so an inline rmtree made a gc of
    dozens of entries take an hour; by default the delete is a detached `rm -rf` that
    outlives this process and waits its turn for the store-wide slot (`flock`), so the caller
    moves on at once and deletes run one at a time host-wide. STUDIO_REGRESS_GC_ASYNC_RM=0 deletes
    inline (still in the slot)."""
    if os.environ.get("STUDIO_REGRESS_GC_ASYNC_RM", "1") == "0" or not shutil.which("rm"):
        with _purge_slot():
            shutil.rmtree(t, ignore_errors = True)
        return
    _PURGERS[:] = [p for p in _PURGERS if p.poll() is None]
    cmd = ["rm", "-rf", "--", str(t)]
    # No `nice`: the rm holds XFS metadata locks that uv installs wait on, and at nice 19 on a
    # loaded host it waited for CPU while holding them. One delete at a time needs no CPU cap.
    if shutil.which("ionice"):
        cmd = ["ionice", "-c3"] + cmd
    lock = _purge_lock()
    if lock is not None and shutil.which("flock"):
        cmd = ["flock", str(lock)] + cmd
    try:
        _PURGERS.append(
            subprocess.Popen(
                cmd,
                stdin = subprocess.DEVNULL,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                start_new_session = True,
            )
        )
    except OSError:
        with _purge_slot():
            shutil.rmtree(t, ignore_errors = True)


def sweep_trash(
    root,
    now = None,
    min_age_s = 3600,
):
    """Finish trash dirs a removal left (an rmtree that died, or another user's files inside)."""
    now = time.time() if now is None else now
    n = 0
    for sub in ("installs", "src"):
        base = Path(root) / sub
        for t in base.glob(f"{TRASH}*") if base.is_dir() else ():
            with contextlib.suppress(OSError):
                if now - t.stat().st_mtime >= min_age_s and deletable(t):
                    with _purge_slot():
                        shutil.rmtree(t, ignore_errors = True)
                    n += not t.exists()
    return n


def candidates(root = ROOT):
    """(dir, its creation lock, linked source worktree or None) for every cached entry."""
    out = []
    inst = root / "installs"
    for d in sorted(inst.iterdir()) if inst.is_dir() else ():
        if d.is_dir() and not d.name.startswith(TRASH):
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


def _can_remove(d, src):
    """Every path a removal touches is deletable by this user (an install and its source checkout)."""
    if (
        d / ".git"
    ).is_file():  # a linked worktree (workspace store): git worktree remove handles it
        return True
    return deletable(d) and (
        src is None or not src.exists() or (src / ".git").is_file() or deletable(src)
    )


def _remove_entry(root, d, lock, src, log):
    """Remove one entry whole. False when it cannot be (nothing removed)."""
    if (d / ".git").is_file():
        _remove(d)
    elif not _trash(d):
        return False
    if src is not None and src.exists():
        if (src / ".git").is_file():
            _remove(src)
        elif not _trash(src):
            log(f"could not remove the source checkout {src}")
    if lock is not None:
        lock.unlink(missing_ok = True)
    shutil.rmtree(Path(root) / "refs" / d.name, ignore_errors = True)
    return True


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
        if not _can_remove(d, src):  # another user's install: left whole, never half removed
            kept.append((d, NOT_DELETABLE))
            continue
        with _exclusive(_use_lock(d)) as free:
            if not free:
                kept.append((d, "in use"))
                continue
            if not dry_run and not _remove_entry(root, d, lock, src, log):
                kept.append((d, NOT_DELETABLE))
                continue
            removed.append(d)
            log(
                f"{'would remove' if dry_run else 'remove'} {d} "
                + ("(no live workspace refers to it)" if orphan else f"(unused {age_d:.1f} d)")
            )
    return removed, kept


# ------------------------------------------------------------------ size budget + automatic runs
# The shared store reached ~2 TiB of du (208 installs x ~9 GB) because nothing ran the age rule on
# its own: gc ran only by hand or from `launcher.sh cleanup` (orphans only). auto() runs the same
# collect() plus an LRU size cap, rate-limited host-wide, and runs/builds start it in the background.
SIZE_FILE = ".uidiff_size"


def max_gb():
    try:
        return float(os.environ.get("STUDIO_REGRESS_CACHE_MAX_GB", "600"))
    except ValueError:
        return 600.0


def auto_enabled():
    return os.environ.get("STUDIO_REGRESS_AUTO_GC", "1") != "0"


def auto_interval_s():
    try:
        return float(os.environ.get("STUDIO_REGRESS_AUTO_GC_INTERVAL_S", "3600"))
    except ValueError:
        return 3600.0


def entry_gb(
    d,
    now = None,
    max_age_s = 86400,
):
    """Allocated size of an entry (st_blocks, each inode once), cached in `.uidiff_size` for a day:
    a walk of a ~100k-file venv takes seconds, and the budget pass sizes every entry."""
    now = time.time() if now is None else now
    f = Path(d) / SIZE_FILE
    with contextlib.suppress(OSError, ValueError, TypeError):
        rec = json.loads(f.read_text())
        if now - float(rec["t"]) < max_age_s:
            return float(rec["gb"])
    idx = _size_index(Path(d).parent.parent)
    rec = idx.get(Path(d).name)
    if rec and now - float(rec.get("t", 0)) < max_age_s:
        return float(rec["gb"])
    gb = _du_gb(d)
    if gb is None:
        gb = _walk_gb(d)
    try:
        f.write_text(json.dumps({"gb": round(gb, 3), "t": round(now, 1)}))
    except OSError:  # another user's entry: remember it in this user's own index instead
        idx[Path(d).name] = {"gb": round(gb, 3), "t": round(now, 1)}
        _size_index_write(Path(d).parent.parent, idx)
    return gb


def _size_index_path(root):
    return Path(root) / f"gc.sizes.{_user()}.json"


def _size_index(root):
    with contextlib.suppress(OSError, ValueError):
        v = json.loads(_size_index_path(root).read_text())
        if isinstance(v, dict):
            return v
    return {}


def _size_index_write(root, idx):
    p = _size_index_path(root)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    with contextlib.suppress(OSError):
        tmp.write_text(json.dumps(idx))
        os.replace(tmp, p)


def _du_gb(d):
    """`du -s -B1` (allocated bytes, each inode once): a C walk is several times faster than lstat
    from Python over a ~100k-file venv. None when du is missing or fails."""
    if not shutil.which("du"):
        return None
    try:
        r = subprocess.run(
            ["du", "-s", "-B1", "--", str(d)], capture_output = True, text = True, timeout = 1800
        )
        return int(r.stdout.split()[0]) / 1e9 if r.stdout.strip() else None
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def _walk_gb(d):
    seen, total = set(), 0
    for dp, dirs, files in os.walk(d):
        for n in dirs + files:
            with contextlib.suppress(OSError):
                st = os.lstat(os.path.join(dp, n))
                if (st.st_dev, st.st_ino) not in seen:
                    seen.add((st.st_dev, st.st_ino))
                    total += st.st_blocks * 512
    return total / 1e9


def evict_to_budget(
    root,
    budget_gb = None,
    dry_run = False,
    refs = None,
    now = None,
    log = print,
    min_idle_s = 3600,
    size_of = entry_gb,
):
    """Remove least-recently-used installs until the store's installs fit `budget_gb`. The same
    holds as collect(): an entry in use (a live process, its creation lock, any run's `.use` lock)
    is skipped, and so is one used within `min_idle_s` (a run between its build and first use)."""
    budget_gb = max_gb() if budget_gb is None else budget_gb
    now = time.time() if now is None else now
    if not os.access(Path(root) / "installs", os.W_OK | os.X_OK):
        return (
            [],
            None,
        )  # this user can rename nothing there, so sizing every entry would buy nothing
    entries = [(d, lock, src) for d, lock, src in candidates(root) if d.parent.name == "installs"]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(
        max_workers = 16
    ) as ex:  # du per entry: I/O bound, so parallel walks overlap
        sizes = dict(zip([d for d, _, _ in entries], ex.map(size_of, [d for d, _, _ in entries])))
    total = sum(sizes.values())
    removed = []
    if total <= budget_gb:
        return removed, total
    refs = _proc_refs() if refs is None else refs
    skipped = 0
    for d, lock, src in sorted(entries, key = lambda e: last_used(e[0])):
        if total <= budget_gb:
            break
        if now - last_used(d) < min_idle_s:
            continue
        if (
            in_use(d, refs)
            or (src is not None and src.exists() and in_use(src, refs))
            or not _lock_free(lock)
        ):
            continue
        if not _can_remove(d, src):  # its owner's own pass evicts it
            skipped += 1
            continue
        with _exclusive(_use_lock(d)) as free:
            if not free:
                continue
            if not dry_run and not _remove_entry(root, d, lock, src, log):
                skipped += 1
                continue
            log(
                f"{'would remove' if dry_run else 'remove'} {d} (over the {budget_gb:g} GB budget, "
                f"{sizes[d]:.1f} GB, unused {(now - last_used(d)) / 3600:.1f} h)"
            )
            removed.append(d)
            total -= sizes[d]
    if skipped:
        log(f"budget: skipped {skipped} not deletable by {_user()}")
    return removed, total


def _user():
    import getpass
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        return str(os.geteuid())


def _stamp(root):
    """Per unix user: each user can delete only its own installs, so each runs its own hourly pass."""
    return Path(root) / f"gc.auto.{_user()}.stamp"


def auto_due(
    root,
    now = None,
    interval_s = None,
):
    interval_s = auto_interval_s() if interval_s is None else interval_s
    now = time.time() if now is None else now
    try:
        return now - _stamp(root).stat().st_mtime >= interval_s
    except OSError:
        return True


def auto(
    root,
    days = 3.0,
    budget_gb = None,
    now = None,
    log = print,
    interval_s = None,
    refs = None,
):
    """One rate-limited pass over the shared store: collect() then evict_to_budget(). Host-wide:
    a non-blocking flock means concurrent callers skip, and the stamp keeps it to one per interval.
    Returns None when skipped, else (removed, total_gb_after)."""
    from studio_regress import plat

    if not os.access(Path(root) / "installs", os.W_OK | os.X_OK):
        return None  # this user can rename nothing in the store, so it can remove nothing
    lock = Path(root) / f"gc.auto.{_user()}.lock"
    try:
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o666)
        with contextlib.suppress(OSError):
            os.fchmod(fd, 0o666)
        fh = os.fdopen(fd, "a+")
    except OSError:
        return None
    with fh:
        if not plat.lock(fh, blocking = False):
            return None
        try:
            if not auto_due(root, now, interval_s):
                return None
            stamp = _stamp(root)
            stamp.touch()  # first, so a crash mid-pass does not retrigger on every run
            with contextlib.suppress(OSError):
                os.chmod(stamp, 0o666)
            refs = _proc_refs() if refs is None else refs
            sweep_trash(root, now)
            removed, kept = collect(days, root = root, refs = refs, now = now, log = log)
            more, total = evict_to_budget(root, budget_gb, refs = refs, now = now, log = log)
            skipped = sum(1 for _, w in kept if w == NOT_DELETABLE)
            log(
                f"auto gc ({_user()}): removed {len(removed) + len(more)}"
                + (f", installs now {total:.0f} GB" if total is not None else "")
                + (f"; skipped {skipped} not deletable by {_user()}" if skipped else "")
            )
            return removed + more, total
        finally:
            plat.unlock(fh)


def spawn_auto(root = None, log = None):
    """Start `studio_regress.py gc --auto` detached when the shared store is due. Never blocks the
    caller and never raises: housekeeping must not fail a run or a build."""
    try:
        if not auto_enabled():
            return False
        if root is None:
            from studio_regress import cache
            root = cache.shared_root(create = False)
        if root is None or not auto_due(root):
            return False
        script = Path(__file__).resolve().parents[1] / "studio_regress.py"
        if not os.access(Path(root) / "installs", os.W_OK | os.X_OK):
            return False
        out = open(Path(root) / f"gc.auto.{_user()}.log", "a")
        subprocess.Popen(
            [sys.executable, str(script), "gc", "--auto"],
            stdin = subprocess.DEVNULL,
            stdout = out,
            stderr = subprocess.STDOUT,
            start_new_session = True,
            cwd = str(Path(script).parent),
        )
        out.close()
        return True
    except Exception as e:  # noqa: BLE001
        if log:
            log(f"auto gc not started: {type(e).__name__}: {e}")
        return False


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
    descendants_only = False,
):
    """{pid: run root} of live processes carrying RUN_ROOT_ENV (only `root`'s when given), minus
    this process and its ancestors (a nested run inherits its parent's value). descendants_only:
    just this process's own subtree, so a nested isolated run never stops its parent's Studios."""
    from studio_regress import plat

    table = plat.process_table() if table is None else table
    env_of = env_of or plat.process_env
    me = os.getpid()
    mine = _ancestors(me, table)
    want = str(Path(root).resolve()) if root else None
    out = {}
    for pid in table:
        if pid in mine:
            continue
        if descendants_only and me not in _ancestors(pid, table):
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
    nested = bool(os.environ.get("STUDIO_REGRESS_ISOLATION"))
    for pid in sorted(run_processes(root, descendants_only = nested)):
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
    p.add_argument(
        "--max-gb",
        type = float,
        default = None,
        help = "also evict least-recently-used shared installs until they fit this many GB "
        "(default $STUDIO_REGRESS_CACHE_MAX_GB or 600)",
    )
    p.add_argument(
        "--auto",
        action = "store_true",
        help = "the rate-limited background pass runs and builds start: skip unless due "
        "($STUDIO_REGRESS_AUTO_GC_INTERVAL_S, default 3600), then age/orphan + budget",
    )
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
    if a.auto:
        if shared is None:
            return 0
        res = auto(
            shared, a.days, a.max_gb, log = lambda m: print(time.strftime("%F %T"), m, flush = True)
        )
        if res is None:
            print("auto gc: not due or another pass is running")
        return 0
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
    if shared is not None:
        more, total = evict_to_budget(shared, a.max_gb, a.dry_run)
        removed += more
        if total is not None:
            print(
                f"shared installs: {total:.0f} GB after the budget pass ({max_gb() if a.max_gb is None else a.max_gb:g} GB)"
            )
    skipped = sum(1 for _, w in kept if w == NOT_DELETABLE)
    busy = [f"{d.name} ({w})" for d, w in kept if w not in ("recent", NOT_DELETABLE)]
    print(
        f"{'would remove' if a.dry_run else 'removed'} {len(removed)}, kept {len(kept)}"
        + (f"; skipped {skipped} not deletable by {_user()}" if skipped else "")
        + (f"; held: {', '.join(busy)}" if busy else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
