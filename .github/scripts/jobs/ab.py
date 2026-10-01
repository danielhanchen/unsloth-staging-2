#!/usr/bin/env python3
"""Deterministic base-vs-head A/B of one jobs/*.py run for a PR, ending in compare.py's verdict.

    python jobs/ab.py --pr 5123 [--repo unslothai/unsloth] --job "sft:--tiny --max-steps 3"
        [--python PY] [--companion-sha auto|SHA|none] [--expect-fail-on-base CHECK]
        [--compare-args "--perf-gate"] [--outdir DIR] [--dry-run] [--json]

Arms: base = merge base of the PR head and its base branch (temp worktree, detached), head =
pr_review_status.ensure_worktree (wt_r<N>). Both run in ONE interpreter (--python, default this
one), so every dependency is byte-identical: only the repo under test is swapped with
`uv pip install --no-deps -e <arm>` plus PYTHONPATH=<arm>, and the resolved import path is verified
to sit inside that arm (else VOID). VOID up front when base == head or head does not touch the
package directory. The companion (unsloth_zoo for unsloth, and vice versa) is pinned once to the
same SHA (auto = currently installed git commit, else upstream main); an editable companion is left
alone and its checked-out HEAD is recorded as the pin. Each arm gets a fresh
UNSLOTH_COMPILE_LOCATION and cwd; job args are identical. At exit both the package and the companion
are restored to their snapshots (uninstalled if they were not installed before). --head-sha uses a
private detached worktree temp/ab_head_<repo>_<N>, never the shared wt_r<N>.

Outputs: outputs/jobs_ab/<owner>__<repo>/pr<N>/{base,head}.json, *.log, verdict.json.
Exit = compare.py's: 0 NO_REGRESSION / FIX_CONFIRMED, 1 REGRESSION, 2 VOID / NOT_RUN / setup error.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pr_review"))

WS = Path(os.environ.get("WORKSPACE") or HERE.parent)
MODULE = {"unslothai/unsloth": ("unsloth", "unsloth"), "unslothai/unsloth-zoo": ("unsloth_zoo", "unsloth_zoo"),
          "unslothai/unsloth_zoo": ("unsloth_zoo", "unsloth_zoo")}
COMPANION = {"unsloth": ("unsloth_zoo", "https://github.com/unslothai/unsloth-zoo"),
             "unsloth_zoo": ("unsloth", "https://github.com/unslothai/unsloth")}


def _log(msg):
    print(f"ab: {msg}", file=sys.stderr, flush=True)


def _run(argv, **kw):
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    return subprocess.run([str(x) for x in argv], **kw)


def parse_job(spec):
    """'sft:--tiny --max-steps 3' -> (script path, [args])."""
    name, _, args = spec.partition(":")
    script = HERE / f"{name}.py"
    if name.startswith("_") or name in ("ab", "compare") or not script.is_file():
        raise SystemExit(f"unknown job {name!r}; one of: " + ", ".join(
            sorted(p.stem for p in HERE.glob("*.py") if not p.stem.startswith(("_", "test_")) and p.stem not in ("ab", "compare"))))
    return script, shlex.split(args)


def installed(py, dist):
    """{'version', 'editable', 'path', 'commit', 'url'} of dist in interpreter py, or None."""
    code = ("import importlib.metadata as m, json\n"
            f"try: d = m.distribution({dist!r})\nexcept Exception: print('null'); raise SystemExit\n"
            "u = d.read_text('direct_url.json'); u = json.loads(u) if u else {}\n"
            "print(json.dumps({'version': d.version, 'url': u.get('url'), "
            "'editable': (u.get('dir_info') or {}).get('editable', False), "
            "'commit': (u.get('vcs_info') or {}).get('commit_id')}))")
    r = _run([py, "-c", code])
    return json.loads(r.stdout.strip() or "null") if r.returncode == 0 else None


def module_path(py, module, env=None):
    """Directory `import module` resolves to, without executing it (find_spec)."""
    r = _run([py, "-c", f"import importlib.util as u, os; s = u.find_spec({module!r}); "
                        "print(os.path.realpath(os.path.dirname(s.origin)) if s and s.origin else '')"],
             cwd=str(WS / "temp"), env=env)
    return r.stdout.strip() or None


def reinstall_spec(info, dist):
    """uv pip install args that restore a previously installed dist."""
    if not info:
        return None
    url = info.get("url") or ""
    if info.get("editable") and url.startswith("file://"):
        return ["-e", url[len("file://"):]]
    if info.get("commit") and url:
        return [f"git+{url}@{info['commit']}"]
    return [f"{dist}=={info['version']}"]


def uv_install(py, args):
    r = _run(["uv", "pip", "install", "-q", "--python", py, "--no-deps", *args])
    if r.returncode:
        raise RuntimeError(f"uv pip install {' '.join(args)}: {r.stderr.strip()[-400:]}")


def editable_head(info):
    """git HEAD of an editable install's source dir, else None."""
    url = (info or {}).get("url") or ""
    if not (info and info.get("editable") and url.startswith("file://")):
        return None
    r = _run(["git", "-C", url[len("file://"):], "rev-parse", "HEAD"])
    return r.stdout.strip() or None if r.returncode == 0 else None


def companion_plan(py, module, want):
    """{dist, url, sha, before, reinstall}. An editable companion is never reinstalled: its checked
    out HEAD is the pinned SHA for both arms (an explicit --companion-sha that disagrees is an error)."""
    comp, url = COMPANION[module]
    before = installed(py, comp)
    plan = {"dist": comp, "url": url, "sha": None, "before": before, "reinstall": False}
    if before and before.get("editable"):
        head = editable_head(before)
        if want not in ("auto", "none") and not (head and head.startswith(want)):
            raise RuntimeError(f"{comp} is an editable install at {head or '?'}; check out {want} there "
                               "or pass --companion-sha auto")
        plan["sha"], plan["editable"] = head, True
        return plan
    if want == "none":  # not pinned: whatever is installed stays
        return plan
    if want != "auto":
        plan.update(sha=want, reinstall=True)
        return plan
    if before and before.get("commit"):
        plan.update(sha=before["commit"], reinstall=True)
        return plan
    r = _run(["git", "ls-remote", url, "HEAD"])
    plan.update(sha=(r.stdout.split() or [None])[0])
    plan["reinstall"] = bool(plan["sha"])
    return plan


def companion_sha(py, module, want):
    c = companion_plan(py, module, want)
    return c["dist"], c["url"], c["sha"]


def restore(py, dist, before):
    """Put dist back as it was: reinstall the snapshot, or uninstall when it was absent. -> error str."""
    if before is None:
        r = _run(["uv", "pip", "uninstall", "-q", "--python", py, dist])
        return None if r.returncode == 0 else f"uv pip uninstall {dist}: {r.stderr.strip()[-300:]}"
    spec = reinstall_spec(before, dist)
    try:
        uv_install(py, spec)
    except RuntimeError as e:
        return f"restore failed, reinstall by hand: uv pip install --no-deps {' '.join(spec)} ({e})"
    return None


def detached_worktree(primary, wt, sha):
    """(Re)point a private detached worktree at sha; RuntimeError on any git failure."""
    if _run(["git", "-C", primary, "cat-file", "-e", f"{sha}^{{commit}}"]).returncode:
        _run(["git", "-C", primary, "fetch", "-q", "origin", sha])
    if not (Path(wt) / ".git").exists():
        r = _run(["git", "-C", primary, "worktree", "add", "-f", "--detach", wt, sha])
    else:
        r = _run(["git", "-C", wt, "checkout", "-q", "--detach", "-f", sha])
    if r.returncode:
        raise RuntimeError(f"worktree {wt} at {sha[:12]}: {r.stderr.strip()[:300]}")
    return Path(wt)


def base_worktree(primary, repo, n, head_sha, base_ref):
    """Detached worktree at merge-base(head, origin/<base_ref>), reused per PR."""
    _run(["git", "-C", primary, "fetch", "-q", "origin", base_ref])
    mb = _run(["git", "-C", primary, "merge-base", head_sha, f"origin/{base_ref}"]).stdout.strip()
    if not mb:
        raise RuntimeError(f"no merge base between {head_sha[:9]} and origin/{base_ref}")
    wt = WS / "temp" / f"ab_base_{repo.replace('/', '__')}_{n}"
    return detached_worktree(primary, wt, mb), mb


def arms_differ(wt, base_sha, head_sha, module):
    """None when head changes the package under test vs base, else the VOID reason."""
    if base_sha == head_sha:
        return f"base and head are the same commit {head_sha[:12]}: nothing to compare"
    r = _run(["git", "-C", wt, "diff", "--quiet", base_sha, head_sha, "--", module])
    if r.returncode == 0:
        return f"no change under {module}/ between base {base_sha[:12]} and head {head_sha[:12]}"
    if r.returncode != 1:
        return f"git diff {base_sha[:12]} {head_sha[:12]}: {r.stderr.strip()[:200]}"
    return None


def ensure_primary(repo):
    import pr_review_status as prs
    primary = prs._resolve_primary(WS, repo)
    if not (primary / ".git").exists():
        _log(f"cloning {repo} (blobless) into {primary}")
        r = _run(["git", "clone", "-q", "--filter=blob:none", f"https://github.com/{repo}", primary])
        if r.returncode:
            raise RuntimeError(f"clone {repo}: {r.stderr.strip()[:300]}")
    return primary


def run_arm(arm, wt, py, module, script, args, outdir):
    # Editable install keeps dist metadata (version checks) on the arm; PYTHONPATH makes the arm win
    # imports even when another copy sits on sys.path (setuptools' editable finder runs AFTER PathFinder).
    uv_install(py, ["-e", str(wt)])
    cache = outdir / f"{arm}_compiled_cache"
    env = dict(os.environ, UNSLOTH_COMPILE_LOCATION=str(cache), PYTHONHASHSEED="0",
               PYTHONPATH=os.pathsep.join(x for x in (str(wt), os.environ.get("PYTHONPATH")) if x))
    got = module_path(py, module, env)
    root = os.path.realpath(wt)
    if not got or not (got == root or got.startswith(root + os.sep)):
        return {"arm": arm, "void": f"`import {module}` resolves to {got}, not the {arm} worktree {wt}"}
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True)
    out, log = outdir / f"{arm}.json", outdir / f"{arm}.log"
    out.unlink(missing_ok=True)
    cmd = [py, "-u", script, *args, "--out", out]
    _log(f"{arm}: {module} from {got}; {' '.join(map(str, cmd))} > {log}")
    with open(log, "w") as fh:
        rc = subprocess.run([str(x) for x in cmd], stdout=fh, stderr=subprocess.STDOUT, env=env,
                            cwd=str(outdir)).returncode
    return {"arm": arm, "rc": rc, "out": str(out), "log": str(log), "module_path": got}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pr", required=True)
    p.add_argument("--repo", default="unslothai/unsloth")
    p.add_argument("--job", required=True, metavar="NAME[:ARGS]")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--companion-sha", default="auto")
    p.add_argument("--expect-fail-on-base", metavar="CHECK")
    p.add_argument("--compare-args", default="", help="extra compare.py flags, e.g. --perf-gate")
    p.add_argument("--head-sha", help="override head (default: live PR head)")
    p.add_argument("--outdir", help="output dir (default outputs/jobs_ab/<owner>__<repo>/pr<N>); set it when "
                                    "several jobs of one PR run, so they do not overwrite each other")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    # SIGTERM (a caller's timeout) unwinds like an exception, so the finally below still restores
    # the package and companion this run swapped into --python.
    signal.signal(signal.SIGTERM, lambda sig, _f: sys.exit(128 + sig))
    a.python = os.path.abspath(a.python)  # arms run with cwd = outdir

    import pr_review_status as prs
    repo, n = prs.resolve_pr(a.pr, a.repo)
    if repo not in MODULE:
        _log(f"{repo}: only {sorted(MODULE)} are supported")
        return 2
    module, dist = MODULE[repo]
    script, args = parse_job(a.job)
    if "--out" in args:
        _log("--out is set per arm; drop it from --job")
        return 2
    outdir = (Path(a.outdir).resolve() if a.outdir
              else WS / "outputs" / "jobs_ab" / repo.replace("/", "__") / f"pr{n}")
    result = {"pr": f"{repo}#{n}", "job": a.job, "python": a.python, "outdir": str(outdir)}
    try:
        meta = prs._gh_obj(["pr", "view", str(n), "--repo", repo, "--json", "headRefOid,baseRefName"]) or {}
        head_sha = a.head_sha or meta.get("headRefOid")
        if not head_sha:
            raise RuntimeError("could not resolve the PR head")
        primary = ensure_primary(repo)
        if a.head_sha:  # private detached worktree: never move the shared wt_r<N>
            _run(["git", "-C", primary, "fetch", "-q", "origin", f"pull/{n}/head"])
            head_wt = detached_worktree(primary, WS / "temp" / f"ab_head_{repo.replace('/', '__')}_{n}", a.head_sha)
            hw = {"action": "detached_head_sha"}
        else:
            hw = prs.ensure_worktree(repo, n, root=WS, primary=primary)
            if hw.get("action") in ("error", "skipped_dirty", "diverged"):
                raise RuntimeError(f"head worktree {hw['wt']}: {hw['action']} {hw.get('warning') or ''}")
            head_wt = Path(hw["wt"])
        head_sha = _run(["git", "-C", head_wt, "rev-parse", "HEAD"]).stdout.strip()
        if a.head_sha and not head_sha.startswith(a.head_sha):
            raise RuntimeError(f"head worktree is at {head_sha[:12]}, not --head-sha {a.head_sha}")
        base_wt, base_sha = base_worktree(primary, repo, n, head_sha, meta.get("baseRefName") or "main")
        why = arms_differ(head_wt, base_sha, head_sha, module)
        if why:
            raise RuntimeError(why)
        cp = companion_plan(a.python, module, a.companion_sha)
        comp, comp_url, comp_sha = cp["dist"], cp["url"], cp["sha"]
        result.update(base={"sha": base_sha, "wt": str(base_wt)},
                      head={"sha": head_sha, "wt": str(head_wt), "worktree_action": hw.get("action")},
                      companion={"dist": comp, "sha": comp_sha, "editable": bool(cp.get("editable")),
                                 "reinstalled": cp["reinstall"]})
        if a.dry_run:
            print(json.dumps(result, indent=2))
            return 0
        outdir.mkdir(parents=True, exist_ok=True)
        before = installed(a.python, dist)
        try:
            if cp["reinstall"]:
                uv_install(a.python, [f"git+{comp_url}@{comp_sha}"])
            arms = [run_arm(arm, wt, a.python, module, script, args, outdir)
                    for arm, wt in (("base", base_wt), ("head", head_wt))]
        finally:
            for d, snap, touched in ((dist, before, True), (comp, cp["before"], cp["reinstall"])):
                err = restore(a.python, d, snap) if touched else None
                if err:
                    _log(err)
        result["arms"] = arms
    except RuntimeError as e:
        _log(str(e))
        result.update(verdict="VOID", reason=str(e))
        print(json.dumps(result, indent=2) if a.json else f"VERDICT VOID {e}")
        return 2
    void = [x["void"] for x in arms if "void" in x]
    if void:
        result.update(verdict="VOID", reason="; ".join(void))
        rc, text = 2, f"VERDICT VOID {result['reason']}"
    else:
        cmp = [sys.executable, HERE / "compare.py", outdir / "base.json", outdir / "head.json",
               *shlex.split(a.compare_args)]
        if a.expect_fail_on_base:
            cmp += ["--expect-fail-on-base", a.expect_fail_on_base]
        r = _run(cmp)
        rc, text = r.returncode, r.stdout.rstrip()
        last = text.splitlines()[-1] if text else "VERDICT NOT_RUN compare.py printed nothing"
        parts = last.split(" ", 2)
        result.update(verdict=parts[1] if len(parts) > 1 else "NOT_RUN", reason=parts[2] if len(parts) > 2 else "",
                      table=text)
    (outdir / "verdict.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2) if a.json else text)
    return rc


if __name__ == "__main__":
    sys.exit(main())
