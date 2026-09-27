"""Studio installs cached per SHA, shared by every session and unix user on the host.

    installs/<key>/        UNSLOTH_STUDIO_HOME built by `install.sh --local`; key = SHA + a hash of
                           the build spec (flags, torch index, unsloth-zoo commit), bare SHA for a
                           legacy spec (non-NVIDIA host, nothing pinned)
    installs/<key>.lock    build lock: one builder per key, everyone else waits and reuses
    installs/<key>.use     held SHARED by every run using the entry; gc needs it EXCLUSIVE
    installs/<key>/.uidiff_build.json   the spec plus the torch actually installed
    src/<key>/             the source checkout the install is editable from

Root: $STUDIO_REGRESS_SHARED_DIR (off / 0 / none disables), else a sibling of the host-wide GPU
lease dir (/mnt/disks/unslothai/shared/studio-regress-cache) when its parent exists. The root is
setgid with group `users` (when the creator belongs to it) and builds run under umask 002, so any
member can reuse, touch and gc an entry. When the root cannot be created or written, the
per-workspace layout under $WORKSPACE/temp/studio_regress is used as before.

Pins. The torch index is pr_ui_diff.expected_torch_cuda()'s (the host driver's; install.sh's own
probe falls back to cu126 on a congested driver) and is part of the key; an entry whose torch does
not fit the host (install_torch_mismatch) is never a hit and is rebuilt. install.sh --local also
takes unsloth-zoo from git main at install time, so the zoo commit is pinned too (UNSLOTH_ZOO_REF):
a head reuses its cached base's zoo, so both sides compare one zoo.

Publishing is a stamp, not a rename: a venv bakes its absolute path into every script and
pyvenv.cfg, so an install cannot be built elsewhere and moved. Readers trust an entry only when
`.uidiff_sha` (written with an atomic replace, after install.sh exits 0 with the right torch) holds
the SHA; a build that died leaves no stamp and the next builder wipes and rebuilds it.

Shared source trees are standalone shallow checkouts (`git fetch --depth 1` of the one commit
from this workspace's clone, hydrating a partial clone first), not linked worktrees: another user
cannot use a worktree whose .git lives in someone else's workspace. Outside the workspace tree no
ancestor `.gitignore` holds `*`, so those builds need no workspace-wide install lock.

Clone + delta (STUDIO_REGRESS_CLONE_DELTA=1, opt-in): reflink-clone a published install of a
donor SHA, rewrite its absolute paths, reuse the frontend dist when studio/frontend is the same
git tree, and run install.sh on top; any leftover donor path makes it a clean build.
`python -m studio_regress.cache compare A B` diffs two installs (freeze, dist, llama / whisper).
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

from studio_regress import gc, plat

WS = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[3])
LOCAL_ROOT = WS / "temp" / "studio_regress"
DEFAULT_SHARED = Path("/mnt/disks/unslothai/shared/studio-regress-cache")
SHARED_GROUP = os.environ.get("STUDIO_REGRESS_SHARED_GROUP", "users")
DEFAULT_FLAGS = ("--local",)
STAMP, SRC_STAMP, BUILD_META = ".uidiff_sha", ".uidiff_src", ".uidiff_build.json"
_OFF = ("", "0", "off", "none", "false", "no")


def _log(msg):
    print(f"[studio_regress] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ roots
def _group_id():
    """gid of SHARED_GROUP when this user belongs to it, else None."""
    try:
        import grp   # POSIX only
        g = grp.getgrnam(SHARED_GROUP)
    except (ImportError, KeyError):
        return None
    return g.gr_gid if g.gr_gid in os.getgroups() or g.gr_gid == os.getegid() else None


def _mkdir_shared(d: Path):
    """mkdir -p; dirs this user creates get setgid + group rwx (and group `users` when possible)."""
    missing = []
    p = d
    while not p.exists():
        missing.append(p)
        p = p.parent
    for m in reversed(missing):
        try:
            m.mkdir()
        except FileExistsError:
            continue
        gid = _group_id()
        try:
            if gid is not None:
                os.chown(m, -1, gid)
            os.chmod(m, 0o2775 if gid is not None else 0o2777)
        except OSError:
            pass


def _writable(d: Path) -> bool:
    try:
        probe = d / f".probe.{os.getpid()}.{time.monotonic_ns()}"
        probe.write_text("")
        probe.unlink()
        return True
    except OSError:
        return False


def shared_root(create=True):
    """The host-shared cache root, or None (disabled, Windows, or not writable). create=False
    (gc) never makes it: an existing root or None."""
    env = os.environ.get("STUDIO_REGRESS_SHARED_DIR")
    if env is not None and env.strip().lower() in _OFF:
        return None
    if env is None:
        if plat._is_windows() or not DEFAULT_SHARED.parent.is_dir():
            return None
        d = DEFAULT_SHARED
    else:
        d = Path(env).expanduser()
    # one spelling for every user (a symlinked root would defeat clone+delta's path rewrite)
    d = Path(os.path.realpath(d))
    if not create:
        return d if (d / "installs").is_dir() else None
    try:
        _mkdir_shared(d / "installs")
        _mkdir_shared(d / "src")
    except OSError:
        return None
    return d if _writable(d / "installs") and _writable(d / "src") else None


class Store:
    """One cache root: shared (cross-user, standalone src checkouts) or the workspace-local one."""

    def __init__(self, root: Path, shared: bool):
        self.root, self.shared = Path(root), shared
        self.installs, self.src = self.root / "installs", self.root / "src"

    def __repr__(self):
        return f"Store({self.root}, shared={self.shared})"

    def home(self, key):
        return self.installs / key

    def published(self, key, sha):
        stamp = self.home(key) / STAMP
        try:
            return stamp.read_text().strip() == sha
        except OSError:
            return False

    def usable(self, key, sha):
        """Published, and its torch fits this host (a cu126 install on a CUDA 13 driver does not)."""
        if not self.published(key, sha):
            return False
        bad = torch_mismatch(self.home(key))
        if bad:
            _log(f"cached install {key[:12]} unusable: {bad}")
        return not bad

    def meta(self, key):
        """The build record of a published entry; a legacy one (no record) was built with the
        default flags, install.sh's own torch probe and whatever zoo main was then."""
        try:
            return json.loads((self.home(key) / BUILD_META).read_text())
        except (OSError, ValueError):
            return {"flags": list(DEFAULT_FLAGS), "torch_index": None, "zoo": None, "legacy": True}

    def find(self, sha, match):
        """Newest published entry of `sha` whose build record agrees with every item of `match`
        -> (key, meta), or None."""
        best = None
        for d in [self.installs / sha, *self.installs.glob(f"{sha}-*")]:
            if not d.is_dir() or not self.usable(d.name, sha):
                continue
            m = self.meta(d.name)
            if all(m.get(k) == v for k, v in match.items()):
                try:
                    t = (d / STAMP).stat().st_mtime
                except OSError:   # gc removed it just now
                    continue
                if best is None or t > best[0]:
                    best = (t, d.name, m)
        return best and best[1:]


def stores():
    """[shared, local] when the shared root is usable, else [local]."""
    out = []
    r = shared_root()
    if r is not None:
        out.append(Store(r, True))
    out.append(Store(LOCAL_ROOT, False))
    return out


def build_spec(flags=DEFAULT_FLAGS, torch_index=None, zoo=None):
    """What an install is a function of, besides its SHA."""
    return {"flags": list(flags), "torch_index": torch_index, "zoo": zoo}


def install_key(sha, spec=None):
    """SHA alone for a legacy spec (default flags, installer-chosen torch, zoo main: the layout
    every existing install uses), else SHA + a hash of the spec (all keys one length, which
    clone+delta's path rewrite relies on)."""
    if isinstance(spec, (tuple, list)):   # the older signature: install_key(sha, flags)
        spec = build_spec(spec)
    spec = spec or build_spec()
    if tuple(spec["flags"]) == DEFAULT_FLAGS and not spec.get("torch_index") and not spec.get("zoo"):
        return sha
    return f"{sha}-{hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]}"


# ------------------------------------------------------------------ pins
ZOO_REPO = "https://github.com/unslothai/unsloth-zoo"


def expected_index():
    """The torch wheel index every install on this host is pinned to: $UNSLOTH_TORCH_INDEX_URL when
    the caller sets one, else pr_ui_diff.expected_torch_cuda() (libcuda's driver version; install.sh's
    own nvidia-smi probe times out on a congested driver and falls back to cu126). None without an
    NVIDIA driver: install.sh decides, as before."""
    pinned = os.environ.get("UNSLOTH_TORCH_INDEX_URL", "").strip()
    if pinned:
        return pinned.rstrip("/")
    from pr_ui_diff import expected_torch_cuda
    exp = expected_torch_cuda()
    return exp[1] if exp else None


def torch_mismatch(home):
    """pr_ui_diff.install_torch_mismatch: why an install's torch does not fit this host, or None."""
    from pr_ui_diff import install_torch_mismatch
    return install_torch_mismatch(Path(home))


def resolve_zoo_main(timeout=60):
    """unsloth-zoo main's current commit (install.sh --local installs zoo from git; pinning it is
    what makes two installs made at different times comparable). None when GitHub is unreachable."""
    try:
        r = subprocess.run(["git", "ls-remote", ZOO_REPO, "refs/heads/main"], capture_output=True, text=True,
                           timeout=timeout)
        out = r.stdout.split()
        return out[0] if r.returncode == 0 and out and len(out[0]) == 40 else None
    except (OSError, subprocess.SubprocessError):
        return None


def installed_zoo(home):
    """The unsloth-zoo commit an install actually holds (its direct_url.json), or None."""
    for d in Path(home, "unsloth_studio").glob("lib/python*/site-packages/unsloth_zoo-*.dist-info/direct_url.json"):
        try:
            return json.loads(d.read_text()).get("vcs_info", {}).get("commit_id")
        except (OSError, ValueError):
            pass
    return None


# ------------------------------------------------------------------ locks
def _open_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        os.fchmod(fd, 0o666)
    except (OSError, AttributeError):
        pass   # another user's file: its mode is already 0666
    return os.fdopen(fd, "r+")


@contextlib.contextmanager
def _flock(path: Path, exclusive=True):
    fh = _open_lock(path)
    try:
        plat.lock(fh, exclusive)
        yield fh
    finally:
        try:
            plat.unlock(fh)
        finally:
            fh.close()


_HELD = []   # (use-lock path, open file) for every entry this process uses: released at exit


def hold(store, key):
    """Take the entry's use lock SHARED for the rest of this process (gc then leaves it alone).
    Once per entry per process. Not on Windows, where plat.lock has no shared mode (two runs of
    one SHA would serialise for a whole run); the shared store is POSIX-only anyway."""
    if plat._is_windows():
        return None
    path = str(store.installs / f"{key}.use")
    for held_path, fh in _HELD:
        if held_path == path:
            return fh
    fh = _open_lock(Path(path))
    plat.lock(fh, exclusive=False)
    _HELD.append((path, fh))
    return fh


def release_all():
    while _HELD:
        _p, fh = _HELD.pop()
        with contextlib.suppress(OSError):
            plat.unlock(fh)
            fh.close()


# ------------------------------------------------------------------ builds
def standalone_checkout(repo: Path, sha: str, dest: Path):
    """dest = a self-contained shallow git checkout of `sha` fetched from the local clone `repo`."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    def git(*a):
        r = subprocess.run(["git", "-C", str(dest), *a], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"git {' '.join(a)} failed in {dest}: {r.stderr.strip()[-500:]}")
    git("init", "-q")
    url = Path(repo).resolve().as_uri()
    try:
        git("fetch", "-q", "--depth", "1", url, sha)
    except RuntimeError:
        # A partial (blob:none) clone cannot serve blobs it never fetched ("lazy fetching
        # disabled" on the upload-pack side). A checkout in the clone itself batch-fetches them
        # from its promisor remote; then the same fetch works.
        hydrate(repo, sha)
        git("fetch", "-q", "--depth", "1", url, sha)
    git("-c", "advice.detachedHead=false", "checkout", "-q", "--detach", "FETCH_HEAD")
    return dest


def hydrate(repo: Path, sha: str):
    """Make every blob of `sha` local in `repo` (a partial clone) via a throwaway worktree."""
    tmp = LOCAL_ROOT / "hydrate" / f"{sha[:12]}.{os.getpid()}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    # one worktree add / prune at a time on the clone (base and head build in parallel threads)
    with plat.locked(tmp.parent / "hydrate.lock"):
        _hydrate(repo, sha, tmp)


def _hydrate(repo, sha, tmp):
    try:
        r = subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", "--detach", str(tmp), sha],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"could not check {sha[:12]} out in {repo}: {r.stderr.strip()[-300:]}")
    finally:
        subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(tmp)], capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(["git", "-C", str(repo), "worktree", "prune"], capture_output=True)


def share_perms(root: Path):
    """Group rw (+x where the owner has x, setgid on dirs) on everything this user owns under root,
    so every group member can read, run, touch and gc it. install.sh makes some entries 0700 /
    0600 (llama.cpp, share/studio_install_id)."""
    me = os.geteuid()

    def fix(p, is_dir):
        try:
            st = os.lstat(p)
        except OSError:
            return
        if st.st_uid != me or (st.st_mode & 0o170000) == 0o120000:   # not ours, or a symlink
            return
        mode = st.st_mode & 0o7777
        want = mode | 0o070 | 0o2000 if is_dir else mode | 0o060 | (0o010 if mode & 0o100 else 0)
        if want != mode:
            with contextlib.suppress(OSError):
                os.chmod(p, want)
    fix(root, True)
    for d, dirs, files in os.walk(root):
        for n in dirs:
            fix(os.path.join(d, n), True)
        for n in files:
            fix(os.path.join(d, n), False)


def _default_installer(src: Path, home: Path, sha: str, log_dir: Path, serialize, uv_cache, env=None):
    from pr_ui_diff import Side, install_side
    install_side(Side(label=sha[:9], sha=sha, worktree=src, home=home), log_dir,
                 serialize=serialize, uv_cache=uv_cache, env=env)


def _install_env(spec):
    env = {}
    if spec.get("torch_index"):
        env["UNSLOTH_TORCH_INDEX_URL"] = spec["torch_index"]
    if spec.get("zoo"):
        env["UNSLOTH_ZOO_REF"] = spec["zoo"]
    return env


def _write_stamp(home: Path, name: str, text: str):
    tmp = home / f"{name}.tmp.{os.getpid()}"
    tmp.write_text(text)
    os.replace(tmp, home / name)


# ------------------------------------------------------------------ clone + delta
def clone_delta_enabled():
    return os.environ.get("STUDIO_REGRESS_CLONE_DELTA") == "1"


# Per-run state the installer writes into a home; a clone must not inherit the donor's.
_NOT_CLONED = (STAMP, SRC_STAMP, BUILD_META, "auth", "studio.db", "studio.db-wal", "studio.db-shm")


def _git_out(repo, *a):
    r = subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def _files_with(root: Path, needles):
    """Files (and symlinks) under root whose bytes / target mention any needle."""
    hits = set()
    r = subprocess.run(["grep", "-rlZF", *sum((["-e", n] for n in needles), []), "--", str(root)],
                       capture_output=True)
    if r.returncode not in (0, 1):
        raise RuntimeError(f"grep over {root} failed: {r.stderr.decode(errors='replace')[-300:]}")
    hits.update(Path(x.decode()) for x in r.stdout.split(b"\0") if x)
    for d, dirs, files in os.walk(root):
        for n in dirs + files:
            p = os.path.join(d, n)
            if os.path.islink(p) and any(x in os.readlink(p) for x in needles):
                hits.add(Path(p))
    return sorted(hits)


def rewrite_paths(root: Path, mapping):
    """Replace every old absolute path (mapping old -> new) inside files and symlink targets under
    root. A length change inside a binary file (NUL bytes: .pyc, .so) would corrupt it: refused."""
    olds = list(mapping)
    changed = 0
    for p in _files_with(root, olds):
        if p.is_symlink():
            t = os.readlink(p)
            for o, n in mapping.items():
                t = t.replace(o, n)
            p.unlink()
            p.symlink_to(t)
            changed += 1
            continue
        data = p.read_bytes()
        new = data
        for o, n in mapping.items():
            if len(o) != len(n) and b"\0" in data and o.encode() in data:
                raise RuntimeError(f"{p}: binary file references {o} and the new path has another length")
            new = new.replace(o.encode(), n.encode())
        if new != data:
            mode = p.stat().st_mode & 0o7777
            tmp = p.with_name(p.name + f".rw{os.getpid()}")
            tmp.write_bytes(new)
            os.chmod(tmp, mode)
            os.replace(tmp, p)
            changed += 1
    return changed


def _reflink_copy(src: Path, dest: Path):
    r = subprocess.run(["cp", "-a", "--reflink=always", str(src), str(dest)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"reflink copy {src} -> {dest} failed: {r.stderr.strip()[-300:]}")


def clone_delta(store, repo, sha, key, donor_key, donor_sha, log_dir, installer, uv, info=None, env=None):
    """Build `key` by reflink-cloning the published donor install, rewriting its absolute paths,
    and running install.sh --local on top (its own up-to-date checks redo only what changed).
    Raises on anything unexpected; the caller wipes and builds clean."""
    home, src = store.home(key), store.src / key
    dhome = store.home(donor_key)
    dsrc = source_dir(dhome)
    if not store.published(donor_key, donor_sha) or dsrc is None:
        raise RuntimeError(f"donor {donor_key[:12]} is not a published install")
    info = {} if info is None else info
    t = time.time()
    standalone_checkout(repo, sha, src)
    # Frontend dist: reused only when studio/frontend is byte-identical (same git tree) in both
    # commits. setup.sh rebuilds when any input file is newer than dist/, so dist is stamped now.
    t_new, t_old = _git_out(src, "rev-parse", "HEAD:studio/frontend"), _git_out(dsrc, "rev-parse", "HEAD:studio/frontend")
    fe = src / "studio" / "frontend"
    if t_new and t_new == t_old and (dsrc / "studio/frontend/dist").is_dir():
        for sub in ("dist", "node_modules"):
            if (dsrc / "studio/frontend" / sub).is_dir():
                _reflink_copy(dsrc / "studio/frontend" / sub, fe / sub)
        if (fe / "node_modules").is_dir():
            rewrite_paths(fe / "node_modules", {str(dsrc): str(src)})
        os.utime(fe / "dist")
        info["frontend_reused"] = True
    home.mkdir(parents=True)
    for e in dhome.iterdir():
        if e.name in _NOT_CLONED:
            continue
        _reflink_copy(e, home / e.name)
    mapping = {str(dhome): str(home), str(dsrc): str(src)}
    info["clone_s"] = round(time.time() - t, 1)
    t = time.time()
    info["rewritten_files"] = rewrite_paths(home, mapping)
    info["rewrite_s"] = round(time.time() - t, 1)
    t = time.time()
    installer(src, home, sha, log_dir, None, uv, env)
    info["delta_install_s"] = round(time.time() - t, 1)
    t = time.time()
    left = _files_with(home, [str(dhome), str(dsrc)])
    info["verify_s"] = round(time.time() - t, 1)
    if left:
        raise RuntimeError(f"{len(left)} files still reference the donor, e.g. {left[0]}")
    _log(f"clone+delta {sha[:9]} from {donor_sha[:9]}: clone {info['clone_s']}s, rewrite "
         f"{info['rewritten_files']} files {info['rewrite_s']}s, install.sh {info['delta_install_s']}s, "
         f"verify {info['verify_s']}s")


def _build_once(store, repo, sha, key, spec, log_dir, installer, make_worktree, donors, info):
    home, src = store.home(key), store.src / key
    env = _install_env(spec)
    if store.shared:
        uv = store.root / "uv_cache" if os.environ.get("STUDIO_REGRESS_SHARED_UV_CACHE") == "1" else None
        done = False
        if clone_delta_enabled():
            for dsha in donors:
                found = store.find(dsha, {"flags": spec["flags"], "torch_index": spec["torch_index"]})
                if not found or found[0] == key:
                    continue
                dkey = found[0]
                for d in (home, src):
                    if d.exists():
                        shutil.rmtree(d)
                try:
                    hold(store, dkey)
                    clone_delta(store, repo, sha, key, dkey, dsha, log_dir, installer, uv, info, env)
                    info["clone_from"] = dsha
                    done = True
                except Exception as e:
                    _log(f"clone+delta from {dsha[:9]} failed ({type(e).__name__}: {e}); clean install")
                    info["clone_failed"] = f"{type(e).__name__}: {e}"[:300]
                break
        if not done:
            for d in (home, src):   # a build that died before its stamp: never reuse half of one
                if d.exists():
                    shutil.rmtree(d)
            standalone_checkout(repo, sha, src)
            installer(src, home, sha, log_dir, None, uv, env)   # None: lock only if an ancestor .gitignore is `*`
    else:
        src = make_worktree(repo, sha, src)
        # a home without a usable stamp is a dead build or a wrong torch; install.sh keeps an
        # existing venv's torch ("keeping it"), so reinstalling over it would keep the wrong one
        shutil.rmtree(home, ignore_errors=True)
        installer(src, home, sha, log_dir, True, None, env)
    return home, src


def _build(store, repo, sha, key, log_dir, installer, make_worktree, donors=(), info=None, spec=None):
    """Build and publish (stamps last). install_side refuses an install whose torch does not fit
    the host (pr_ui_diff.install_torch_mismatch), so nothing wrong gets a stamp."""
    info = {} if info is None else info
    spec = spec or build_spec()
    home, src = _build_once(store, repo, sha, key, spec, log_dir, installer, make_worktree, donors, info)
    if store.shared:
        share_perms(home)
        share_perms(src)
    from pr_ui_diff import torch_cuda_of
    meta = {**spec, "sha": sha, "key": key, "requested_zoo": spec.get("zoo"),
            "zoo": installed_zoo(home) or spec.get("zoo"), "torch_cuda": torch_cuda_of(home),
            "built_at": time.time(), "clone_from": info.get("clone_from")}
    _write_stamp(home, BUILD_META, json.dumps(meta, indent=1))
    _write_stamp(home, SRC_STAMP, str(src))
    _write_stamp(home, STAMP, sha)
    info["meta"] = meta


# ------------------------------------------------------------------ host-wide build slots
def max_builds():
    try:
        return max(0, int(os.environ.get("STUDIO_REGRESS_MAX_BUILDS", "3")))
    except ValueError:
        return 3


@contextlib.contextmanager
def build_slot(n=None, poll_s=10.0, log=None):
    """One of n host-wide install slots (lock files in the GPU lease dir, so every user shares
    them); n = 0 means no cap. Blocks until a slot frees."""
    n = max_builds() if n is None else n
    if n <= 0:
        yield None
        return
    from studio_regress import gpu_pack
    d = gpu_pack.lock_dir_default()
    gpu_pack._mkdir_shared(d)
    said = False
    while True:
        for i in range(n):
            fh = _open_lock(d / f"install_slot{i}.lock")
            if plat.lock(fh, blocking=False):
                try:
                    yield i
                finally:
                    plat.unlock(fh)
                    fh.close()
                return
            fh.close()
        if not said and log:
            log(f"all {n} host-wide install slots busy; waiting")
            said = True
        time.sleep(poll_s)


def lookup(sha, torch_index=None, zoo=None, flags=DEFAULT_FLAGS, store_list=None):
    """(store, key, meta) of the newest published install of `sha` for this spec (any zoo when
    zoo is None), in any store, or None. Nothing is locked: ensure_install does that."""
    match = {"flags": list(flags), "torch_index": torch_index}
    if zoo:
        match["zoo"] = zoo
    for store in store_list or stores():
        found = store.find(sha, match) if store.installs.is_dir() else None
        if found:
            return store, found[0], found[1]
    return None


def ensure_install(repo: Path, sha: str, log_dir: Path, flags=DEFAULT_FLAGS, installer=None,
                   make_worktree=None, store_list=None, info=None, donors=(), wait=True, capped=False,
                   zoo=None, torch_index=False):
    """Install `sha` once per host (shared store) or per workspace (fallback) and return its home.

    The build is pinned: torch from expected_index() (the host driver's, carried in the key),
    unsloth-zoo at `zoo` (the other side's commit, so base and head compare the same zoo; default:
    main now). Any usable published install of `sha` with that torch index (and that zoo, when
    given) is a hit.

    The caller keeps a SHARED use lock on the entry for the rest of the process. `info` receives
    {"store", "shared", "result": hit|built|waited|busy, "s", "lock_wait_s", "clone_from", "meta"}.
    donors: SHAs a clone+delta build may start from (STUDIO_REGRESS_CLONE_DELTA=1). wait=False:
    return None when another process is building it. capped: take a host-wide build slot first."""
    installer = installer or _default_installer
    if make_worktree is None:
        from pr_ui_diff import make_worktree
    index = expected_index() if torch_index is False else torch_index
    info = {} if info is None else info
    t0 = time.time()
    errors = []
    store_list = list(store_list or stores())
    found = lookup(sha, index, zoo, flags, store_list)
    if found:   # a published install in ANY store (e.g. one the workspace built before) wins
        store, key, meta = found
        with contextlib.suppress(OSError):
            hold(store, key)   # before trusting the stamp again: gc cannot delete it from here on
            if store.shared:
                # Studio writes __pycache__ into the shared venv / src as the user running it:
                # keep those group-writable so other users' gc and rebuilds can remove them
                os.umask(0o002)
            if store.usable(key, sha):
                gc.touch(store.home(key))
                info.update(store=str(store.root), shared=store.shared, result="hit", key=key, meta=meta,
                            s=round(time.time() - t0, 1))
                return store.home(key)
    spec = build_spec(flags, index, zoo or resolve_zoo_main())
    key = install_key(sha, spec)
    info["key"] = key
    for store in store_list:
        try:
            store.installs.mkdir(parents=True, exist_ok=True)
            if store.shared:
                os.umask(0o002)   # group-writable everything this run creates in the shared store
            hold(store, key)
            tl = time.time()
            # capped (prebuild): the host-wide slot FIRST, the per-key lock after it and without
            # waiting, so a `run` needing this SHA never queues behind prebuilds waiting for a slot
            with (build_slot(log=_log) if capped else contextlib.nullcontext()):
                fh = _open_lock(store.installs / f"{key}.lock")
                try:
                    if not plat.lock(fh, blocking=wait and not capped):
                        info.update(store=str(store.root), shared=store.shared, result="busy",
                                    s=round(time.time() - t0, 1))
                        return None
                    waited = round(time.time() - tl, 1)
                    if store.usable(key, sha):   # another session built it while we waited
                        result = "waited"
                        info["meta"] = store.meta(key)
                    else:
                        (store.home(key) / STAMP).unlink(missing_ok=True)   # unpublish a wrong-torch entry
                        _build(store, repo, sha, key, log_dir, installer, make_worktree, donors, info, spec)
                        result = "built"
                    gc.touch(store.home(key))
                finally:
                    with contextlib.suppress(OSError):
                        plat.unlock(fh)
                    fh.close()
            info.update(store=str(store.root), shared=store.shared, result=result,
                        lock_wait_s=waited, s=round(time.time() - t0, 1))
            return store.home(key)
        except PermissionError as e:   # the shared store turned out unusable for us: fall back
            errors.append(f"{store.root}: {e}")
            if not store.shared:
                raise
            _log(f"shared install cache unusable ({e}); falling back to the workspace cache")
    raise RuntimeError("; ".join(errors) or "no install store")


def source_dir(install_home):
    """The source checkout an install home was built from (`.uidiff_src`, else the old layout)."""
    if not install_home:
        return None
    home = Path(install_home)
    try:
        d = Path((home / SRC_STAMP).read_text().strip())
        if d.is_dir():
            return d
    except OSError:
        pass
    try:
        sha = (home / STAMP).read_text().strip()
    except OSError:
        return None
    for d in (home.parent.parent / "src" / home.name, LOCAL_ROOT / "src" / sha):
        if d.is_dir():
            return d
    return None


def is_shared(install_home):
    """Whether an install home lives in the host-shared cache (built by any user)."""
    r = shared_root(create=False)
    if r is None or not install_home:
        return False
    try:
        Path(install_home).resolve().relative_to(r.resolve())
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------ equality check (clone+delta vs clean)
# What a prebuilt IS (release, source commit, backend). Left out on purpose: installed_at, and the
# bundle profile / asset / host_profile, which install.sh picks from its own GPU probe at install
# time (on a congested driver it saw no compute caps and chose "portable" over "newer": two clean
# installs of one SHA differ there, so it cannot tell a good clone from a bad one).
_PREBUILT_VERSION_KEYS = ("requested_tag", "tag", "release_tag", "published_repo", "source_commit",
                          "ggml_tree", "backend", "upstream_tag", "paired_llama_tag")


def fingerprint(home):
    """What must match between two installs of one SHA: `uv pip freeze` of the main venv and of
    every transformers sidecar, the frontend build (sha256 over dist/), the llama.cpp and
    whisper.cpp releases (_PREBUILT_VERSION_KEYS), with the install's own paths normalised."""
    import hashlib
    import json
    home = Path(home).resolve()
    src = source_dir(home)
    subs = [(str(home), "<HOME>")] + ([(str(src), "<SRC>")] if src else [])

    def norm(t):
        for o, n in subs:
            t = t.replace(o, n)
        return t

    def freeze(*args):
        r = subprocess.run(["uv", "pip", "freeze", *args], capture_output=True, text=True)
        return norm(r.stdout).splitlines() if r.returncode == 0 else [f"ERROR {r.stderr.strip()[-200:]}"]
    out = {"venv": freeze("--python", str(home / "unsloth_studio" / "bin" / "python"))}
    for sc in sorted(home.glob(".venv_t5_*")):
        out[sc.name] = freeze("--target", str(sc))
    dist = src / "studio" / "frontend" / "dist" if src else None
    if dist and dist.is_dir():
        h = hashlib.sha256()
        for f in sorted(p for p in dist.rglob("*") if p.is_file()):
            h.update(str(f.relative_to(dist)).encode() + b"\0" + f.read_bytes())
        out["frontend_dist"] = h.hexdigest()
    for name in ("llama.cpp", "whisper.cpp"):
        for cand in (home / name / "UNSLOTH_PREBUILT_INFO.json", home / name / "UNSLOTH_WHISPER_PREBUILT_INFO.json"):
            if cand.is_file():
                try:
                    info = json.loads(norm(cand.read_text()))
                except ValueError:
                    info = norm(cand.read_text())
                if isinstance(info, dict):   # the release, not the install-time host probe
                    info = {k: info.get(k) for k in _PREBUILT_VERSION_KEYS if k in info}
                out[f"{name}:{cand.name}"] = info
    return out


def compare(a, b):
    """{key: (a_value, b_value)} for every fingerprint entry that differs (empty = equal)."""
    fa, fb = fingerprint(a), fingerprint(b)
    diffs = {}
    for k in sorted(set(fa) | set(fb)):
        va, vb = fa.get(k), fb.get(k)
        if va != vb:
            if isinstance(va, list) and isinstance(vb, list):
                va, vb = sorted(set(va) - set(vb)), sorted(set(vb) - set(va))
            diffs[k] = (va, vb)
    return diffs


def main(argv=None):
    import argparse
    import json
    p = argparse.ArgumentParser(description="Inspect / compare cached Studio installs")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("where", help="print the cache roots in use")
    f = sub.add_parser("fingerprint")
    f.add_argument("home")
    c = sub.add_parser("compare", help="exit 1 when two installs differ")
    c.add_argument("a")
    c.add_argument("b")
    a = p.parse_args(argv)
    if a.cmd == "where":
        for s in stores():
            print(f"{'shared' if s.shared else 'workspace'}  {s.root}")
        return 0
    if a.cmd == "fingerprint":
        print(json.dumps(fingerprint(a.home), indent=1))
        return 0
    d = compare(a.a, a.b)
    print(json.dumps(d, indent=1) if d else "equal")
    return 1 if d else 0


if __name__ == "__main__":
    sys.exit(main())
