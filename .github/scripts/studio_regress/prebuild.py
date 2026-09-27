"""Build Studio installs into the shared cache before any session needs them.

    python studio_regress.py prebuild --pr 11889 [--pr 11890] [--gh-repo unslothai/unsloth]
    python studio_regress.py prebuild --sha <sha> [--sha ...] [--main]
    python studio_regress.py prebuild --pr 11889 --main --detach     # setsid, log to logs/

Resolves each PR's merge base and head (the pair `run --pr N` installs) and builds them with
cache.ensure_install: a SHA already built, or being built by anyone (its per-SHA flock is held),
is skipped, and at most $STUDIO_REGRESS_MAX_BUILDS (default 3) prebuild installs run at once
across every user (host-wide slot locks next to the GPU leases). --main also builds current
origin/main. Merge bases go first, then heads, so a head can clone its base when
STUDIO_REGRESS_CLONE_DELTA=1. Only unslothai/unsloth has Studio installs; other repos are skipped.
A PR head that moves after this ran is simply a cache miss for `run`, which builds it then.
Exit 0 unless the arguments are wrong: nothing here may fail the caller.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from studio_regress import cache

HERE = Path(__file__).resolve().parent
WS = cache.WS
STUDIO_REPO = "unslothai/unsloth"


def _log(msg):
    print(f"[prebuild {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main_sha(repo: Path):
    subprocess.run(["git", "-C", str(repo), "fetch", "-q", "origin", "main"], capture_output=True)
    r = subprocess.run(["git", "-C", str(repo), "rev-parse", "origin/main"], capture_output=True, text=True)
    return r.stdout.strip() or None


def plan(prs, shas, repo, gh_repo, with_main, resolve=None):
    """[(sha, donors, why)] in build order: main, explicit SHAs and merge bases, then heads."""
    if resolve is None:
        from pr_ui_diff import resolve_shas as resolve
    first, heads = [], []
    m = main_sha(repo) if with_main else None
    if m:
        first.append((m, (), "origin/main"))
    for s in shas:
        first.append((s, (m,) if m else (), "sha"))
    for pr in prs:
        try:
            mb, head, _ref = resolve(repo, pr, gh_repo)
        except Exception as e:
            _log(f"#{pr}: could not resolve SHAs ({type(e).__name__}: {str(e)[:200]})")
            continue
        first.append((mb, (m,) if m else (), f"#{pr} merge base"))
        heads.append((head, (mb,), f"#{pr} head"))
    out, seen = [], set()
    for sha, donors, why in first + heads:
        if sha and sha not in seen:
            seen.add(sha)
            out.append((sha, tuple(d for d in donors if d), why))
    return out


def build_one(repo, sha, donors, why, log_dir, index=None):
    """A head pins unsloth-zoo to its merge base's install (what `run` does), a base takes main."""
    info = {}
    t0 = time.time()
    try:
        zoo = None
        if why.endswith("head") and donors:
            hit = cache.lookup(donors[0], index)
            zoo = hit[2].get("zoo") if hit else None
        home = cache.ensure_install(repo, sha, log_dir, info=info, donors=donors, wait=False, capped=True,
                                    zoo=zoo, torch_index=index)
    except Exception as e:
        _log(f"{sha[:9]} ({why}): FAILED after {time.time() - t0:.0f}s: {type(e).__name__}: {str(e)[:300]}")
        return "failed"
    res = info.get("result")
    if home is None:
        _log(f"{sha[:9]} ({why}): skipped, another process is building it")
    else:
        extra = f", cloned from {info['clone_from'][:9]}" if info.get("clone_from") else ""
        _log(f"{sha[:9]} ({why}): {res} in {info.get('s')}s{extra} -> {home}")
    return res


def run(items, repo, log_dir, workers=None, index=False):
    """Two waves (bases, then heads) so a head's base (zoo pin, clone donor) is published first."""
    workers = workers or max(1, cache.max_builds() or 3)
    if index is False:
        index = cache.expected_index()
    results = {}
    heads = [i for i in items if i[2].endswith("head")]
    first = [i for i in items if i not in heads]
    for wave in (first, heads):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {sha: pool.submit(build_one, repo, sha, donors, why, log_dir, index) for sha, donors, why in wave}
            results.update({sha: f.result() for sha, f in futs.items()})
    return results


def detach(argv, log_path: Path):
    """Re-run this command in its own session (setsid), output to log_path; returns at once."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    args = [a for a in argv if a != "--detach"]
    with open(log_path, "a") as fh:
        subprocess.Popen([sys.executable, str(HERE.parent / "studio_regress.py"), "prebuild", *args],
                         stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True, close_fds=True, cwd=str(HERE.parent))
    return log_path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description="Prebuild Studio installs into the shared cache")
    p.add_argument("--pr", type=int, action="append", default=[])
    p.add_argument("--sha", action="append", default=[])
    p.add_argument("--main", action="store_true", help="also build current origin/main")
    p.add_argument("--gh-repo", default=STUDIO_REPO)
    p.add_argument("--repo", default=str(WS / "unsloth"), help="local unsloth clone")
    p.add_argument("--detach", action="store_true", help="run in the background (setsid), log to --log")
    p.add_argument("--log", default=None)
    a = p.parse_args(argv)
    if not (a.pr or a.sha or a.main):
        p.error("nothing to build: --pr, --sha or --main")
    if a.detach:
        log = Path(a.log or WS / "logs" / f"studio_regress_prebuild_{time.strftime('%Y%m%d_%H%M%S')}.log")
        print(f"prebuild detached; log: {detach(argv, log)}")
        return 0
    if a.gh_repo != STUDIO_REPO:
        _log(f"{a.gh_repo}: no Studio install to prebuild")
        return 0
    repo = Path(a.repo)
    if not (repo / ".git").exists():
        _log(f"{repo} is not a git clone of {STUDIO_REPO}; nothing built")
        return 0
    r = cache.shared_root()
    _log(f"cache: {r or cache.LOCAL_ROOT} ({'shared' if r else 'workspace only'}), "
         f"max {cache.max_builds() or 'unlimited'} concurrent builds host-wide")
    items = plan(a.pr, a.sha, repo, a.gh_repo, a.main)
    for sha, donors, why in items:
        _log(f"plan {sha[:9]} ({why})" + (f" donor {donors[0][:9]}" if donors else ""))
    log_dir = WS / "logs" / "studio_regress_prebuild"
    index = cache.expected_index()
    _log(f"torch index: {index or 'install.sh decides (no NVIDIA driver)'}")
    res = run(items, repo, log_dir, index=index)
    _log("done: " + ", ".join(f"{s[:9]}={v}" for s, v in res.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
