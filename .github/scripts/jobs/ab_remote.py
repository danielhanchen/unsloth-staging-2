#!/usr/bin/env python3
"""Remote half of a switchboard job A/B: run one jobs/*.py on the PR merge base and head on THIS
machine (a Colab / Kaggle worker), back to back in one venv, and leave the metrics for compare.py.

    python ab_remote.py --repo unslothai/unsloth --pr 5123 --base-sha B --head-sha H \
        [--companion-repo unslothai/unsloth-zoo --companion-sha C] --job "sft:--tiny" [--outdir ab_out]

Shipped flat next to the jobs/*.py it runs (studio_regress/offload.py builds the job). The comparison
itself runs locally with the same compare.py as jobs/ab.py, so a remote run and a local one are judged
by one rule. Steps:
  1. clone the public repo (blobless) once into $SB_CACHE (kept on a reused Colab VM), fetch both SHAs (and refs/pull/<N>/head for a fork PR), and add
     detached worktrees for base and head;
  2. `pip install unsloth` (dependencies from PyPI, torch held at the version the cell installed), then
     the companion at the run's SHA (else its default branch) with --no-deps --force-reinstall;
  3. per arm (base first): `--no-deps -e <tree>` + PYTHONPATH=<tree>, verify the import resolves inside
     the tree, run the job with a fresh UNSLOTH_COMPILE_LOCATION and cwd, `--out <outdir>/<arm>.json`.
Outputs: <outdir>/{base,head}.json, {base,head}.log, remote.json (SHAs, import paths, rc, GPU).
Exit 0 when both arms wrote metrics, 2 otherwise (setup failure or an arm wrote nothing).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPANION = {
    "unslothai/unsloth": "unslothai/unsloth-zoo",
    "unslothai/unsloth-zoo": "unslothai/unsloth",
}
MODULE = {
    "unslothai/unsloth": "unsloth",
    "unslothai/unsloth-zoo": "unsloth_zoo",
    "unslothai/unsloth_zoo": "unsloth_zoo",
}


def log(msg):
    print(f"ab_remote: {msg}", flush = True)


def run(argv, **kw):
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    return subprocess.run([str(a) for a in argv], **kw)


def uv_cmd():
    uv = os.environ.get("SB_UV") or shutil.which("uv")
    return [uv, "pip"] if uv and os.path.exists(uv) else [sys.executable, "-m", "pip"]


def pip(args, py = sys.executable):
    base = uv_cmd()
    argv = (
        base
        + (["install", "--python", py] if base[0] != sys.executable else ["install"])
        + list(args)
    )
    r = run(argv)
    if r.returncode:
        log(f"pip {' '.join(map(str, args))[:200]} failed: {(r.stdout + r.stderr)[-1500:]}")
    return r.returncode == 0


def trees(repo, pr, base_sha, head_sha, root):
    # The clone lives in the worker's shared cache (SB_CACHE, kept across jobs on a reused Colab VM):
    # only the per-job worktrees are made here, and pruned clones from earlier jobs are forgotten.
    cache = Path(os.environ.get("SB_CACHE") or root)
    src = cache / "ab_src" / repo.replace("/", "__")
    src.parent.mkdir(parents = True, exist_ok = True)
    if (src / ".git").exists():
        run(["git", "-C", src, "worktree", "prune"])
    else:
        r = run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                f"https://github.com/{repo}.git",
                src,
            ]
        )
        if r.returncode:
            raise RuntimeError(f"clone {repo}: {r.stderr[-500:]}")
    refs = [base_sha, head_sha] + ([f"refs/pull/{pr}/head"] if pr else [])
    for ref in refs:
        run(["git", "-C", src, "fetch", "--quiet", "origin", ref])
    out = {}
    for arm, sha in (("base", base_sha), ("head", head_sha)):
        wt = root / arm
        if not wt.exists():
            r = run(["git", "-C", src, "worktree", "add", "--detach", wt, sha])
            if r.returncode:
                raise RuntimeError(f"{arm} worktree at {sha}: {r.stderr[-500:]}")
        out[arm] = wt
    return out


def torch_constraint(root):
    r = run([sys.executable, "-c", "import torch; print(torch.__version__)"])
    if r.returncode or not r.stdout.strip():
        return []
    c = root / "constraints.txt"
    c.write_text(f"torch=={r.stdout.strip()}\n")
    return ["--constraint", str(c)]


def gpu_name():
    r = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None


def main(argv = None):
    p = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    p.add_argument("--repo", required = True)
    p.add_argument("--pr", type = int, default = 0)
    p.add_argument("--base-sha", required = True)
    p.add_argument("--head-sha", required = True)
    p.add_argument("--companion-repo", default = "")
    p.add_argument("--companion-sha", default = "")
    p.add_argument("--job", required = True, help = '"<stem>:<args>", as jobs/ab.py takes it')
    p.add_argument("--outdir", default = "ab_out")
    p.add_argument("--arm-timeout", type = int, default = 5400)
    a = p.parse_args(argv)

    out = Path(a.outdir).resolve()
    out.mkdir(parents = True, exist_ok = True)
    root = Path("ab_src").resolve()
    root.mkdir(exist_ok = True)
    stem, _, args = a.job.partition(":")
    script = HERE / f"{stem.strip().removesuffix('.py').split('/')[-1]}.py"
    module = MODULE.get(a.repo)
    rec = {
        "repo": a.repo,
        "pr": a.pr,
        "job": a.job,
        "gpu": gpu_name(),
        "companion": None,
        "base": {"sha": a.base_sha},
        "head": {"sha": a.head_sha},
    }

    def finish(code, why = ""):
        rec["exit"], rec["reason"] = code, why
        (out / "remote.json").write_text(json.dumps(rec, indent = 2))
        log(f"done exit={code} {why}")
        return code

    if not module or not script.is_file():
        return finish(2, f"unknown repo {a.repo} or job script {script.name}")
    try:
        tw = trees(a.repo, a.pr, a.base_sha, a.head_sha, root)
    except Exception as e:  # noqa: BLE001 - every setup failure is the run's VOID, never a regression
        return finish(2, f"source setup failed: {e}")
    cons = torch_constraint(root)
    # the usual install: `pip install unsloth` (PyPI, with its dependencies: zoo, bitsandbytes, trl,
    # peft, ...) under the torch the cell installed, then the companion at the run's SHA (else its
    # default branch) --no-deps; each arm's tree goes in --no-deps below
    comp = a.companion_repo or COMPANION.get(a.repo, "")
    ref = f"@{a.companion_sha}" if a.companion_sha else ""
    if not pip([*cons, "unsloth"]):
        return finish(2, "pip install unsloth failed")
    if comp and not pip(
        ["--no-deps", "--force-reinstall", f"git+https://github.com/{comp}.git{ref}"]
    ):
        return finish(2, f"installing {comp}{ref[:13]} --no-deps failed")
    if comp:
        rec["companion"] = {"repo": comp, "sha": a.companion_sha or "default branch"}

    import shlex

    job_args = shlex.split(args)
    for arm in ("base", "head"):
        tree = tw[arm]
        if not pip(["--no-deps", "-e", tree]):
            return finish(2, f"{arm}: editable install failed")
        env = dict(
            os.environ, PYTHONPATH = str(tree), UNSLOTH_COMPILE_LOCATION = str(root / f"compile_{arm}")
        )
        r = run(
            [
                sys.executable,
                "-c",
                f"import importlib.util as u; print(u.find_spec({module!r}).origin)",
            ],
            env = env,
        )
        origin = r.stdout.strip()
        rec[arm]["import"] = origin
        if not origin.startswith(str(tree)):
            return finish(2, f"{arm}: {module} resolves to {origin or 'nothing'}, not {tree}")
        cwd = root / f"cwd_{arm}"
        cwd.mkdir(exist_ok = True)
        dest = out / f"{arm}.json"
        dest.unlink(missing_ok = True)
        t0 = time.time()
        with open(out / f"{arm}.log", "w") as fh:
            try:
                p_ = subprocess.run(
                    [sys.executable, "-u", str(script), *job_args, "--out", str(dest)],
                    cwd = cwd,
                    env = env,
                    stdout = fh,
                    stderr = subprocess.STDOUT,
                    timeout = a.arm_timeout,
                )
                rc = p_.returncode
            except subprocess.TimeoutExpired:
                fh.write(f"\nab_remote: {arm} timed out after {a.arm_timeout}s\n")
                rc = -9
        rec[arm].update(rc = rc, secs = round(time.time() - t0, 1), wrote = dest.exists())
        log(f"{arm}: rc={rc} {rec[arm]['secs']}s metrics={'yes' if dest.exists() else 'no'}")
    ok = all(rec[arm].get("wrote") for arm in ("base", "head"))
    return finish(0 if ok else 2, "" if ok else "an arm wrote no metrics")


if __name__ == "__main__":
    raise SystemExit(main())
