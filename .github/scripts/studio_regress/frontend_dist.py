"""Studio frontend builds shared by every install whose studio/frontend is the same git tree.

`vite build` keeps no cache between builds and every install gets a fresh checkout, so each SHA
rebuilt the frontend (about 4 minutes on an idle host, 45+ with dozens of builds at once) even
when studio/frontend was byte-identical to the one built minutes earlier for its merge base.
studio/setup.sh skips the build when frontend/dist is newer than every frontend input, so a copy
of an earlier build's dist, placed in the checkout just before install.sh runs, is all a reuse
takes. pr_ui_diff.install_side wraps every install in Reuse.

    frontend_dist/<key>/dist/        a published build (renamed in whole, never edited after)
    frontend_dist/<key>/meta.json    the key's inputs, the SHA it was built at, files, bytes, sha256
    frontend_dist/<key>/.used        touched on every hit (prune keeps the most recently used)
    frontend_dist/<key>.lock         held by the one install building <key>; the others wait

key = sha1 over the studio/frontend git tree (every source file, package.json and the lockfile),
the Node pins (studio/setup.sh, studio/node_prebuilt_pins.json), the VITE_* / NODE_ENV /
UNSLOTH_NODE_VERSION build environment and the platform. A checkout with an untracked .env* in
studio/frontend (Vite reads those) never uses or feeds the cache.

Every install gets its own copy (a reflink on XFS), checked file by file against meta.json before
it is used, so no run's edits reach what another run serves. The digest catches corrupt or partial
entries; it is not an authenticity check (every group member can write the cache, as the installs). A frontend checkout with local
changes, a checkout that already has a dist, or anything unexpected falls back to the plain
build; the cache never fails an install. Only a build this install actually ran is published.

Root: $STUDIO_FRONTEND_DIST_CACHE (a path; 0 / off disables), else frontend_dist/ in the
host-shared studio_regress cache root, else in the workspace one.
STUDIO_FRONTEND_DIST_WAIT_S (2700): how long a miss waits for another install building the same
key before building it too. STUDIO_FRONTEND_DIST_KEEP (60) / _DAYS (14): prune limits.

    python -m studio_regress.frontend_dist status | adopt | prune
"""

from __future__ import annotations

import contextlib
import getpass
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from studio_regress import cache, plat

VERSION = 1
_OFF = ("0", "off", "none", "false", "no")
_NODE_PIN = re.compile(rb"nodejs\.org/dist/v(\d+\.\d+\.\d+)")
_BUILD_MARK = ".unsloth-build.local"  # setup.sh's mark of a frontend build that never finished
POLL_S = 10.0  # how often a waiting install looks for the build it waits on


def _log(msg):
    print(f"[frontend_dist] {msg}", file = sys.stderr, flush = True)


def _int_env(name, default):
    try:
        return max(0, int(os.environ.get(name, default)))
    except ValueError:
        return default


def _usable_dir(d, create):
    if not create:
        return d if d.is_dir() else None
    try:
        cache._mkdir_shared(d)
    except OSError:
        return None
    return d if cache._writable(d) else None


def root(create = True):
    """The cache dir, or None (disabled, or not writable: every install then builds as before).
    The shared one when this user can write it, else the workspace one (same rule as the installs)."""
    v = os.environ.get("STUDIO_FRONTEND_DIST_CACHE", "").strip()
    if v.lower() in _OFF:
        return None
    if v and v.lower() not in ("1", "true", "yes", "on"):
        return _usable_dir(Path(v).expanduser(), create)
    r = cache.shared_root(create = True) if create else None
    if not create:  # where an install would have published: the shared root only when writable
        r = cache.shared_root(create = False)
        r = r if r is not None and cache._writable(r / "installs") else None
    for base in ([r] if r is not None else []) + [cache.LOCAL_ROOT]:
        d = _usable_dir(base / "frontend_dist", create)
        if d is not None:
            return d
    return None


def _git(worktree, *args):
    """(returncode, stdout) of a read-only git call in the checkout."""
    try:
        r = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(worktree), *args],
            capture_output = True,
            text = True,
            timeout = 120,
        )
    except (OSError, subprocess.SubprocessError):
        return 1, ""
    return r.returncode, r.stdout.strip()


def spec_for(worktree, env):
    """(key, spec) for the checkout's frontend, or (None, reason) when it must build as before."""
    worktree = Path(worktree)
    rc, tree = _git(worktree, "rev-parse", "HEAD:studio/frontend")
    if rc or not re.fullmatch(r"[0-9a-f]{40,64}", tree):
        return None, "no studio/frontend tree at HEAD"
    rc, dirty = _git(
        worktree, "status", "--porcelain", "--untracked-files=all", "--", "studio/frontend"
    )
    if rc:
        return None, "git status failed"
    if dirty:
        return None, "studio/frontend has local changes"
    # Vite reads .env* files as build inputs; ignored ones never show in git status
    local_env = [
        p.name
        for p in (worktree / "studio" / "frontend").glob(".env*")
        if _git(worktree, "ls-files", "--error-unmatch", "--", f"studio/frontend/{p.name}")[0]
    ]
    if local_env:
        return None, f"untracked {', '.join(sorted(local_env))} (a Vite build input)"
    try:
        pins = sorted(
            {m.decode() for m in _NODE_PIN.findall((worktree / "studio" / "setup.sh").read_bytes())}
        )
    except OSError:
        pins = []
    try:  # the managed Node setup.sh installs (newer setups pin it here, not in the script)
        manifest = hashlib.sha256(
            (worktree / "studio" / "node_prebuilt_pins.json").read_bytes()
        ).hexdigest()
    except OSError:
        manifest = None
    build_env = sorted(
        (k, v)
        for k, v in (env or {}).items()
        if k.startswith("VITE_") or k in ("NODE_ENV", "UNSLOTH_NODE_VERSION")
    )
    spec = {
        "v": VERSION,
        "tree": tree,
        "node": pins,
        "env": build_env,
        "platform": f"{sys.platform}-{platform.machine()}",
    }
    if manifest:
        spec["node_manifest"] = manifest
    key = hashlib.sha1(json.dumps(spec, sort_keys = True).encode()).hexdigest()
    return key, spec


def digest(d):
    """{files, bytes, sha256} over every file under d (paths included). A build holds no symlinks;
    one would point outside what the digest covers, so it is refused (OSError)."""
    d = Path(d)
    if d.is_symlink():
        raise OSError(f"{d} is a symlink")
    h, n, size = hashlib.sha256(), 0, 0
    for p in sorted(d.rglob("*")):
        rel = p.relative_to(d).as_posix()
        if p.is_symlink():
            raise OSError(f"symlink in a frontend build: {rel}")
        if p.is_file():
            data = p.read_bytes()
            h.update(f"F {rel} {len(data)} ".encode() + hashlib.sha256(data).digest() + b"\0")
            n, size = n + 1, size + len(data)
    return {"files": n, "bytes": size, "sha256": h.hexdigest()}


def _copy(src, dst):
    """Copy the tree src to the new path dst: a reflink where the filesystem has them. Fresh
    mtimes, this user's ownership and umask modes, never the source's."""
    try:
        r = subprocess.run(
            ["cp", "-R", "--reflink=auto", str(src), str(dst)], capture_output = True, text = True
        )
        if r.returncode == 0:
            return
    except OSError:
        pass
    shutil.rmtree(dst, ignore_errors = True)  # no GNU cp (macOS) or no cp at all: a plain copy
    shutil.copytree(src, dst, symlinks = True, copy_function = shutil.copyfile)


def _read_meta(entry):
    try:
        meta = json.loads((Path(entry) / "meta.json").read_text())
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) and meta.get("sha256") else None


def _usable(entry):
    """meta.json of a published entry that can be served (its dist holds index.html), else None."""
    meta = _read_meta(entry)
    return meta if meta is not None and (Path(entry) / "dist" / "index.html").is_file() else None


def _discard(path):
    """Rename aside, then remove: a reader either sees the whole entry or none. False when the
    rename failed (the entry stays)."""
    aside = path.parent / f".trash.{path.name}.{os.getpid()}.{time.monotonic_ns()}"
    try:
        os.rename(path, aside)
    except OSError:
        return False
    shutil.rmtree(aside, ignore_errors = True)
    return True


def seed(
    cache_dir,
    key,
    dist,
    log = _log,
):
    """Copy the published build of `key` to `dist` (which must not exist) and verify it. Returns the
    entry's meta on success, None on a miss or a bad entry (then nothing is left at `dist`)."""
    entry = Path(cache_dir) / key
    meta = _usable(entry)
    if meta is None:
        if _read_meta(entry) is not None:  # published but unservable: let the next build replace it
            _discard(entry)
        return None
    dist = Path(dist)
    tmp = dist.parent / f".dist.seed.{os.getpid()}.{time.monotonic_ns()}"
    try:
        _copy(entry / "dist", tmp)
        got = digest(tmp)
    except OSError as e:  # pruned while we copied: a miss
        shutil.rmtree(tmp, ignore_errors = True)
        log(f"copy of {key[:12]} failed ({e}); building")
        return None
    want = {k: meta.get(k) for k in ("files", "bytes", "sha256")}
    if got != want:
        shutil.rmtree(tmp, ignore_errors = True)
        log(
            f"entry {key[:12]} does not match its meta.json ({got['files']} files vs {want['files']}); "
            "discarding it and building"
        )
        _discard(entry)
        return None
    try:
        os.rename(tmp, dist)
        now = time.time()
        os.utime(dist, (now, now))  # newer than every input: setup.sh reports "frontend up to date"
    except OSError as e:
        shutil.rmtree(tmp, ignore_errors = True)
        log(f"placing {key[:12]} failed ({e}); building")
        return None
    with contextlib.suppress(OSError):
        (entry / ".used").touch()
    return meta


def publish(
    cache_dir,
    key,
    spec,
    dist,
    sha,
    log = _log,
):
    """Publish the checkout's built dist as `key` (no-op when it is already published)."""
    cache_dir = Path(cache_dir)
    entry = cache_dir / key
    if _usable(entry) is not None:
        return False
    tmp = cache_dir / f".{key}.tmp.{os.getpid()}.{time.monotonic_ns()}"
    try:
        cache._mkdir_shared(cache_dir)
        tmp.mkdir()
        _copy(dist, tmp / "dist")
        meta = {
            **spec,
            **digest(tmp / "dist"),
            "key": key,
            "sha": sha,
            "built_at": time.time(),
            "user": getpass.getuser(),
        }
        (tmp / "meta.json").write_text(json.dumps(meta, indent = 1))
        (tmp / ".used").touch()  # before share_perms: every group member's hit can touch it
        cache.share_perms(tmp)
        if entry.exists() and _usable(entry) is None:  # broken or unservable, never a good one
            _discard(entry)
        os.rename(tmp, entry)  # refused when another publisher's entry landed meanwhile
    except OSError as e:  # lost a race to another publisher, or no space: never an install failure
        if _usable(entry) is None:
            log(f"publishing {key[:12]} failed ({e})")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors = True)
    log(
        f"published {key[:12]} ({meta['files']} files, {meta['bytes'] // 1048576} MB) from {sha[:9]}"
    )
    prune(cache_dir)
    return True


def _used_at(entry):
    t = 0.0
    for name in (".used", "meta.json"):
        with contextlib.suppress(OSError):
            t = max(t, (entry / name).stat().st_mtime)
    return t


def prune(
    cache_dir,
    keep = None,
    days = None,
    now = None,
):
    """Drop entries beyond the `keep` most recently used or unused for `days`, and leftovers of
    dead publishes / removals. One pruner at a time (others skip). Returns the names removed."""
    cache_dir = Path(cache_dir)
    keep = _int_env("STUDIO_FRONTEND_DIST_KEEP", 60) if keep is None else keep
    days = _int_env("STUDIO_FRONTEND_DIST_DAYS", 14) if days is None else days
    now = time.time() if now is None else now
    fh = cache._open_lock(cache_dir / ".prune.lock")
    removed = []
    try:
        if not plat.lock(fh, blocking = False):
            return removed
        entries = []
        for p in cache_dir.iterdir():
            if p.name.startswith(".trash.") or (p.name.startswith(".") and ".tmp." in p.name):
                with contextlib.suppress(OSError):
                    if now - p.stat().st_mtime > 86400:
                        shutil.rmtree(p, ignore_errors = True)
                continue
            if p.is_dir() and not p.name.startswith(".") and _read_meta(p) is not None:
                entries.append((_used_at(p), p))
        entries.sort(reverse = True)
        for i, (t, p) in enumerate(entries):
            if (i >= keep or now - t > days * 86400) and _discard(p):
                removed.append(p.name)
    finally:
        with contextlib.suppress(OSError):
            plat.unlock(fh)
        fh.close()
    return removed


class Reuse:
    """Around one install.sh run of `worktree`: seed its frontend/dist from the cache, or become
    the one install building this key (others wait up to STUDIO_FRONTEND_DIST_WAIT_S for it).
    After a successful install, call publish(). Never raises out of enter / publish / exit."""

    def __init__(
        self,
        worktree,
        sha,
        env = None,
        log = _log,
        wait_s = None,
        poll_s = None,
    ):
        self.fe = Path(worktree) / "studio" / "frontend"
        self.dist = self.fe / "dist"
        self.sha, self.env, self.log = sha or "", env or {}, log
        self.wait_s = _int_env("STUDIO_FRONTEND_DIST_WAIT_S", 2700) if wait_s is None else wait_s
        self.poll_s = POLL_S if poll_s is None else poll_s
        self.dir = self.key = self.spec = self._lock = None
        self.result, self.detail, self.waited_s = "off", "", 0.0
        self.t0 = time.time()

    def __enter__(self):
        try:
            self._start()
        except Exception as e:  # the cache is an optimisation: anything odd is a plain build
            self.result, self.detail = "error", f"{type(e).__name__}: {e}"
            self.log(f"skipped ({self.detail})")
            self._release()
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self._release()
        return False

    def summary(self):
        s = f"[frontend_dist] {self.result}"
        if self.key:
            s += f" key {self.key[:12]}"
        if self.detail:
            s += f": {self.detail}"
        if self.waited_s >= 1:
            s += f" (waited {self.waited_s:.0f}s)"
        return s

    def _start(self):
        if not self.fe.is_dir():
            return
        self.dir = root()
        if self.dir is None:
            return
        if self.dist.exists() or (self.fe / _BUILD_MARK).exists():
            self.result, self.detail = (
                "present",
                "checkout already has a frontend build; setup.sh decides",
            )
            return
        self.key, spec = spec_for(self.fe.parent.parent, self.env)
        if self.key is None:
            self.result, self.detail = "skip", spec
            return
        self.spec = spec
        if self._seed():
            return
        t, said = time.time(), False
        while True:
            fh = cache._open_lock(self.dir / f"{self.key}.lock")
            if plat.lock(fh, blocking = False):
                self._lock = fh
                if self._seed():  # published while we were taking the lock
                    self._release()
                    return
                self.result = "build"
                return
            fh.close()
            self.waited_s = time.time() - t
            if self.waited_s >= self.wait_s:
                self.result, self.detail = (
                    "build",
                    f"still being built elsewhere after {self.wait_s}s",
                )
                return
            if not said:
                self.log(
                    f"another install is building frontend {self.key[:12]}; waiting for it "
                    f"(up to {self.wait_s}s)"
                )
                said = True
            time.sleep(self.poll_s)
            if self._seed():
                return

    def _seed(self):
        meta = seed(self.dir, self.key, self.dist, self.log)
        if meta is None:
            return False
        self.waited_s = time.time() - self.t0
        self.result = "hit"
        self.detail = f"built at {str(meta.get('sha'))[:9]} by {meta.get('user')}"
        self.log(f"reusing frontend {self.key[:12]} ({self.detail})")
        return True

    def publish(self):
        """After install.sh exited 0: publish the dist it built. Only a build this run made."""
        if self.result != "build":
            return False
        try:
            idx = self.dist / "index.html"
            if (
                not idx.is_file()
                or idx.stat().st_mtime < self.t0 - 2
                or (self.fe / _BUILD_MARK).exists()
            ):
                self.log("install did not build the frontend; nothing to publish")
                return False
            key, _spec = spec_for(
                self.fe.parent.parent, self.env
            )  # inputs unchanged during the build
            if key != self.key:
                self.log("frontend inputs changed during the install; not publishing")
                return False
            return publish(self.dir, self.key, self.spec, self.dist, self.sha, self.log)
        except Exception as e:
            self.log(f"publish skipped ({type(e).__name__}: {e})")
            return False

    def _release(self):
        fh, self._lock = self._lock, None
        if fh is not None:
            with contextlib.suppress(OSError):
                plat.unlock(fh)
            fh.close()


# ------------------------------------------------------------------ CLI
def _fresh(fe):
    """setup.sh's own rule: dist exists and no top-level frontend file, src/ or public/ file is newer."""
    dist = Path(fe) / "dist"
    try:
        t = dist.stat().st_mtime
    except OSError:
        return False
    for p in [
        *Path(fe).iterdir(),
        *(Path(fe) / "src").rglob("*"),
        *(Path(fe) / "public").rglob("*"),
    ]:
        with contextlib.suppress(OSError):
            if p.is_file() and p.name != "bun.lock" and p.stat().st_mtime > t:
                return False
    return True


def adopt(
    cache_dir = None,
    store_roots = None,
    log = _log,
):
    """Publish the frontend builds already sitting in published installs' source checkouts (the
    shared store and this workspace's): only a published install whose checkout is still at the
    stamped SHA, with a clean studio/frontend, no local .env, no unfinished-build mark, and a dist
    newer than every input (setup.sh's rule). Those installs ran with no VITE_* build env (no
    switchboard / pr_ui_diff path sets one), so they get the default key. Returns the count."""
    cache_dir = cache_dir or root()
    if store_roots is None:
        shared = cache.shared_root(create = False)
        store_roots = [r for r in (shared, cache.LOCAL_ROOT) if r is not None]
    if cache_dir is None:
        return 0
    n = 0
    for store_root in store_roots:
        installs = Path(store_root) / "installs"
        if not installs.is_dir():
            continue
        for home in sorted(installs.iterdir()):
            sha = home.name.split("-")[0]
            try:
                if not home.is_dir() or (home / cache.STAMP).read_text().strip() != sha:
                    continue
            except OSError:
                continue
            src = cache.source_dir(home)
            fe = src / "studio" / "frontend" if src else None
            if (
                fe is None
                or not (fe / "dist" / "index.html").is_file()
                or (fe / _BUILD_MARK).exists()
            ):
                continue
            if _git(src, "rev-parse", "HEAD")[1] != sha or not _fresh(fe):
                continue
            key, spec = spec_for(src, {})
            if key is None or _usable(Path(cache_dir) / key) is not None:
                continue
            n += publish(cache_dir, key, {**spec, "adopted": True}, fe / "dist", sha, log)
    return n


def status(cache_dir = None):
    cache_dir = cache_dir or root(create = False)
    if cache_dir is None:
        print("frontend dist cache: off")
        return
    rows = []
    for p in cache_dir.iterdir() if cache_dir.is_dir() else ():
        m = _read_meta(p) if p.is_dir() and not p.name.startswith(".") else None
        if m:
            rows.append((_used_at(p), p.name, m))
    rows.sort(reverse = True)
    print(f"{cache_dir}: {len(rows)} builds")
    for t, name, m in rows:
        print(
            f"  {name[:12]} tree {str(m.get('tree'))[:10]} from {str(m.get('sha'))[:9]} "
            f"{int(m.get('bytes') or 0) // 1048576} MB, used {time.strftime('%m-%d %H:%M', time.localtime(t))}"
        )


def main(argv = None):
    import argparse

    ap = argparse.ArgumentParser(
        prog = "python -m studio_regress.frontend_dist", description = __doc__.split("\n")[0]
    )
    ap.add_argument("cmd", choices = ("status", "adopt", "prune"))
    a = ap.parse_args(argv)
    os.umask(0o002)
    if a.cmd == "status":
        status()
    elif a.cmd == "adopt":
        print(f"adopted {adopt()} builds")
    else:
        d = root()
        print(f"removed {len(prune(d)) if d else 0}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
