#!/usr/bin/env python3
"""Remote half of remote_studio.py: run switchboard Studio targets on THIS machine (Colab / Kaggle).

    python remote_driver.py --pr N --gh-repo unslothai/unsloth --base-sha B --head-sha H \
        [--head-ref REF] --targets auth,diffusion_gen --bundle sr_bundle.tar.xz

Unpacks the scripts bundle into ./scripts, checks the repo out into ./unsloth (its own clone over the
VM's shared blobless object cache, prepare_repo) with both SHAs fetched, installs Chromium for
Playwright once per VM, then runs studio_regress.py with the SHAs pinned (the VM
has no gh credentials) and the root at ./sr_out, which the job wrapper ships back. Exit 0 when the run
wrote a report (whatever its verdict), 2 otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/: cpu_split

KEEP_HOMES_FREE_GB = 80  # two Studio homes are ~15 GB; below this free, build per job and drop


def log(msg):
    print(f"remote_driver: {msg}", flush = True)


def side_gpus(env = None, count = None):
    """ "before=<g0>,after=<g1>" on a VM with two or more visible GPUs (Kaggle T4x2), else "": base on
    the first card, head on the second, never left to whichever card a lease would pick."""
    env = os.environ if env is None else env
    vis = [x for x in (env.get("CUDA_VISIBLE_DEVICES") or "").replace(" ", "").split(",") if x]
    if not vis:
        if count is None:
            r = subprocess.run(["nvidia-smi", "-L"], capture_output = True, text = True)
            count = (
                len([x for x in r.stdout.splitlines() if x.startswith("GPU ")])
                if r.returncode == 0
                else 0
            )
        vis = [str(i) for i in range(count)]
    return f"before={vis[0]},after={vis[1]}" if len(vis) >= 2 else ""


def side_cpus(pinned):
    """ "before=0,2;after=1,3" when the sides are pinned to two GPUs and the machine has at least two
    physical cores (cpu_split: whole cores, equal disjoint sets), else "" (one card, or one core)."""
    if not pinned:
        return ""
    import cpu_split

    sets = cpu_split.split(2)
    if not sets:
        return ""
    return f"before={cpu_split.fmt_list(sets[0])};after={cpu_split.fmt_list(sets[1])}"


PARALLEL_SIDES_MIN_RAM_GB = 24  # two Studios + two Chromiums with tiny models, plus the harness


SIDE_RAM_GB = 6.0  # a Studio + Chromium with a tiny model (scheduler GB_PER_UNIT)


def side_ram_gb(targets, registry = None):
    """Host RAM one side needs for the heaviest target of the batch: its `ram_gb`, else for a GPU
    target 1.5 x gpu_mem_gb (weights pass through host memory on load), at least SIDE_RAM_GB."""
    if registry is None:
        try:
            from studio_regress import selection
            registry = {t["name"]: t for t in selection.load().get("target") or []}
        except Exception:  # noqa: BLE001 - no registry: the plain per-arm default
            registry = {}
    need = SIDE_RAM_GB
    for name in targets or ():
        t = registry.get(name) or {}
        est = t.get("ram_gb") or 1.5 * float(t.get("gpu_mem_gb") or 0)
        need = max(need, float(est or 0))
    return need


def parallel_sides_env(
    side_cpus_spec,
    mem_gb = None,
    targets = None,
    registry = None,
):
    """Let base and head run AT ONCE on this VM when each side has its own GPU and cores (a Kaggle
    T4x2): the scheduler's host-wide sizing (8 cores and 6 GB per arm, 16 GB reserved for a shared
    host) would make a 4 vCPU / 29 GB kernel serial. Here the VM is ours, so one arm per side's CPU
    set, a small RAM reserve, at most 2 arms; still adaptive (load and free RAM are re-read on every
    admission pass, so a busy or tight VM drops back to one arm). {} when the sides are not pinned or
    the VM has under PARALLEL_SIDES_MIN_RAM_GB."""
    if not side_cpus_spec:
        return {}
    if mem_gb is None:
        from studio_regress import scheduler
        mem_gb = scheduler.mem_available_gb()
    if mem_gb is not None and mem_gb < PARALLEL_SIDES_MIN_RAM_GB:
        return {}
    ram = side_ram_gb(targets, registry) if targets is not None else SIDE_RAM_GB
    if mem_gb is not None and mem_gb - 4 < 2 * ram:
        return {}  # two heavy sides do not fit at once: one side at a time
    import cpu_split

    per = max(
        1, min(len(cpu_split.parse_list(x.split("=", 1)[1])) for x in side_cpus_spec.split(";"))
    )
    return {
        "STUDIO_REGRESS_CPU_PER_UNIT": str(per),
        "STUDIO_REGRESS_OVERSUB": "2",
        "STUDIO_REGRESS_RESERVE_GB": "4",
        "STUDIO_REGRESS_MAX_AUTO_JOBS": "2",
        "STUDIO_REGRESS_GB_PER_UNIT": f"{ram:g}",
    }


def sh(argv, **kw):
    log("$ " + " ".join(str(a) for a in argv)[:300])
    return subprocess.run([str(a) for a in argv], **kw)


@contextlib.contextmanager
def flock(path):
    """An exclusive lock on `path` (co-tenant jobs on one VM share the caches it guards)."""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _rev(repo, rev, run):
    r = run(
        ["git", "-C", repo, "rev-parse", "--verify", "-q", rev + "^{commit}"],
        capture_output = True,
        text = True,
    )
    return (r.stdout or "").strip() if r.returncode == 0 else ""


def _same(full, want):
    return (
        bool(full) and bool(want) and (full == want or (len(want) >= 7 and full.startswith(want)))
    )


def prepare_repo(
    cache_repo,
    job_repo,
    url,
    refs,
    head_sha,
    run = None,
    must_have = (),
):
    """A checkout of `head_sha` that is this job's alone, on objects shared with every job of the VM.

    The cache (a blobless, no-checkout clone kept in SB_CACHE) is fetched under a lock, every ref into
    its own refs/sr/* name so a gc never prunes what a job clone relies on (gc.auto off too). The job
    then clones it with --shared (objects through alternates, refs / HEAD / index its own) and points
    origin back at GitHub as the promisor, so blobs the cache never fetched download lazily. Two jobs
    on one VM may check out different SHAs at once. Returns origin/main's SHA or "" (2 on failure)."""
    run = run or sh  # looked up at call time: tests stub the module's sh
    cache_repo, job_repo = Path(cache_repo), Path(job_repo)
    main_sha = ""
    with flock(str(cache_repo) + ".lock"):
        if not (cache_repo / ".git").exists():
            r = run(["git", "clone", "--filter=blob:none", "--no-checkout", url, cache_repo])
            if r.returncode:
                return 2
            run(["git", "-C", cache_repo, "config", "gc.auto", "0"])
        for ref in refs:
            dest = (
                "refs/remotes/origin/main" if ref == "main" else "refs/sr/" + ref.replace("/", "_")
            )
            run(["git", "-C", cache_repo, "fetch", "--quiet", "origin", "+%s:%s" % (ref, dest)])
        r = run(
            ["git", "-C", cache_repo, "rev-parse", "--verify", "-q", "refs/remotes/origin/main"],
            capture_output = True,
            text = True,
        )
        main_sha = (r.stdout or "").strip() if r.returncode == 0 else ""
    if not (job_repo / ".git").exists():
        r = run(["git", "clone", "--quiet", "--shared", "--no-checkout", cache_repo, job_repo])
        if r.returncode:
            return 2
    for cmd in (
        ["remote", "set-url", "origin", url],
        ["config", "remote.origin.promisor", "true"],
        ["config", "remote.origin.partialclonefilter", "blob:none"],
    ):
        run(["git", "-C", job_repo] + cmd)
    r = run(["git", "-C", job_repo, "checkout", "--quiet", "--detach", head_sha])
    if r.returncode:
        return 2
    # the exact revisions, checked: a fetch that failed quietly must not run the suite on another tree
    if not _same(_rev(job_repo, "HEAD", run), head_sha):
        log(f"REMOTE_DRIVER_INFRA: HEAD is not {head_sha} after checkout")
        return 2
    for want in must_have:
        if not _same(_rev(job_repo, want, run), want):
            log(f"REMOTE_DRIVER_INFRA: revision {want} is missing after fetch")
            return 2
    if main_sha:
        run(["git", "-C", job_repo, "update-ref", "refs/remotes/origin/main", main_sha])
    return main_sha


def main(argv = None):
    p = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    p.add_argument("--pr", type = int, required = True)
    p.add_argument("--gh-repo", required = True)
    p.add_argument("--base-sha", required = True)
    p.add_argument("--head-sha", required = True)
    p.add_argument("--head-ref", default = "")
    p.add_argument("--targets", required = True)
    p.add_argument("--bundle", default = "sr_bundle.tar.xz")
    a = p.parse_args(argv)
    work = Path.cwd()
    out = work / "sr_out"
    out.mkdir(exist_ok = True)
    (out / "remote_root.txt").write_text(str(out.resolve()))  # remote_studio rewrites paths from it
    scripts = work / "scripts"
    with tarfile.open(work / a.bundle) as tar:
        if sys.version_info >= (3, 12):
            tar.extractall(scripts, filter = "data")
        else:
            tar.extractall(scripts)
    sys.path.insert(
        0, str(scripts)
    )  # shipped flat next to the bundle: cpu_split, studio_regress live here
    # Kept on a reused worker (the job wrapper's SB_CACHE survives the per-job rollback): the clone,
    # and the built Studio homes keyed by SHA + build spec (studio_regress/cache.py's shared store),
    # while the scratch disk has room for them.
    cache = Path(os.environ.get("SB_CACHE") or work)
    cache_repo = cache / "sr_src" / a.gh_repo.replace("/", "__")
    repo = work / "unsloth"  # this job's own checkout (prepare_repo): co-tenants never share it
    st = os.statvfs(cache)
    keep_homes = st.f_bavail * st.f_frsize / 2**30 > KEEP_HOMES_FREE_GB
    store = cache / "studio" if keep_homes else None
    log(
        f"cache {cache}: objects {cache_repo.name}, Studio homes {'kept in ' + str(store) if store else 'not kept (disk)'}"
    )
    if (
        prepare_repo(
            cache_repo,
            repo,
            f"https://github.com/{a.gh_repo}.git",
            (a.base_sha, a.head_sha, f"refs/pull/{a.pr}/head", "main"),
            a.head_sha,
            must_have = (a.base_sha,),
        )
        == 2
    ):
        log("REMOTE_DRIVER_INFRA: clone / fetch / checkout of the requested revisions failed")
        return 2
    # Chromium once per VM, shared read-only by every job (installed under a lock: two co-tenants
    # must not run apt / unpack into one directory at once); studio_regress / engine read the path
    browsers = cache / "pw_browsers"
    with flock(str(browsers) + ".lock"):
        browsers.mkdir(parents = True, exist_ok = True)
        r = sh(
            [sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"],
            capture_output = True,
            text = True,
            env = dict(os.environ, PLAYWRIGHT_BROWSERS_PATH = str(browsers)),
        )
    if r.returncode:
        log(f"playwright install failed: {(r.stdout + r.stderr)[-800:]}")
    env = dict(
        os.environ,
        WORKSPACE = str(work),
        STUDIO_REGRESS_SHARED_DIR = str(store) if store else "off",
        STUDIO_REGRESS_OFFLOAD = "0",
        PLAYWRIGHT_BROWSERS_PATH = str(browsers),
        STUDIO_REGRESS_PIN_SHAS = f"{a.base_sha}:{a.head_sha}:{a.head_ref}",
        STUDIO_REGRESS_REMOTE = "1",
        STUDIO_REGRESS_SIDE_GPUS = side_gpus(),
    )
    if os.environ.get("SB_PORT_BASE") and not os.environ.get("UIDIFF_PORT_START"):
        env["UIDIFF_PORT_START"] = os.environ[
            "SB_PORT_BASE"
        ]  # this job's own port band (engine.PortPool)
    env["STUDIO_REGRESS_SIDE_CPUS"] = side_cpus(env["STUDIO_REGRESS_SIDE_GPUS"])
    env.update(
        parallel_sides_env(
            env["STUDIO_REGRESS_SIDE_CPUS"], targets = [t for t in a.targets.split(",") if t]
        )
    )
    if env["STUDIO_REGRESS_SIDE_GPUS"]:
        log(
            f"sides pinned: GPUs {env['STUDIO_REGRESS_SIDE_GPUS']}, CPUs "
            f"{env['STUDIO_REGRESS_SIDE_CPUS'] or 'shared (fewer than 2 physical cores)'}, "
            f"{'base and head at once' if env.get('STUDIO_REGRESS_MAX_AUTO_JOBS') == '2' else 'one side at a time'}"
        )
    rc = sh(
        [
            sys.executable,
            "-u",
            scripts / "studio_regress.py",
            "run",
            "--pr",
            a.pr,
            "--gh-repo",
            a.gh_repo,
            "--repo",
            repo,
            "--only",
            a.targets,
            "--side",
            "both",
            "--no-confirm",
            "--root",
            out,
        ],
        cwd = scripts,
        env = env,
    ).returncode
    log(f"studio_regress exit {rc}")
    return 0 if (out / "report.json").exists() else 2


if __name__ == "__main__":
    raise SystemExit(main())
