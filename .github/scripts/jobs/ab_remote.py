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

--parallel-arms auto|on|off (default off; studio_regress/offload.py passes auto): with two GPUs
visible (a Kaggle T4x2 kernel) base runs on GPU0 and head on GPU1 AT ONCE, each in its own copy of
the venv (hardlinked, so the one shared install is paid once and the two editable installs do not
race). Same card model, same kernel, CPU and disk shared by both arms. --swap (with parallel arms)
runs a second pass with the GPUs swapped, recorded in remote.json under "swap" (timing control only;
compare.py reads the first pass).

CPU isolation for parallel arms (cpu_split.py): the machine's physical cores are split into two equal,
disjoint sets (hyperthread siblings together), each arm is pinned to its set (sched_setaffinity in
the child; a cgroup v2 cpuset group too when one is writable) and its thread pools (OpenMP, MKL,
OpenBLAS, Rayon, torch, tokenizers) are capped at its CPU count. --swap swaps the CPU sets with the
GPUs. Fewer than 2 physical cores: base then head, sequentially, said in remote.json.
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
# cpu_split.py: flat next to this file on a remote worker, one level up in the scripts repo
sys.path[:0] = [str(HERE), str(HERE.parent)]
import cpu_split  # noqa: E402

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
    # without uv, `py`'s own pip: never this interpreter's, which would install into the shared venv
    argv = (
        base + ["install", "--python", py]
        if base[0] != sys.executable
        else [py, "-m", "pip", "install"]
    ) + list(args)
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


def gpu_names():
    r = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return [x.strip() for x in r.stdout.splitlines() if x.strip()] if r.returncode == 0 else []


def gpu_name():
    names = gpu_names()
    return names[0] if names else None


def arm_venv(arm, root):
    """A private copy of this interpreter's venv for one arm: hardlinks (cp -al), so the shared
    install is not repeated; pip / uv replace files rather than rewrite them in place, so an arm's
    editable install never touches the other's. None when the copy fails (caller runs serially)."""
    src = Path(sys.prefix)
    dest = root / f"venv_{arm}"
    if not dest.exists():
        for flags in (["-al"], ["-a"]):
            r = run(["cp", *flags, src, dest])
            if r.returncode == 0:
                break
            shutil.rmtree(dest, ignore_errors = True)
        else:
            log(f"{arm}: venv copy failed: {(r.stdout + r.stderr)[-300:]}")
            return None
    py = dest / "bin" / "python"
    return py if py.exists() else None


def run_arm(
    arm,
    tree,
    py,
    script,
    job_args,
    out,
    root,
    module,
    env_extra,
    timeout,
    tag = "",
    cpus = None,
    cgroup = None,
):
    """Editable-install `tree` into `py`'s venv and run the job -> (record dict) or raise RuntimeError.
    `cpus`: this arm's CPU set (the job and its children are pinned to it; thread pools capped)."""
    rec = {}
    if cpus:
        env_extra = dict(env_extra, **cpu_split.thread_env(len(cpus)))
    if not pip(["--no-deps", "-e", tree], py = str(py)):
        raise RuntimeError(f"{arm}: editable install failed")
    env = dict(
        os.environ,
        PYTHONPATH = str(tree),
        UNSLOTH_COMPILE_LOCATION = str(root / f"compile_{arm}{tag}"),
        **env_extra,
    )
    r = run(
        [py, "-c", f"import importlib.util as u; print(u.find_spec({module!r}).origin)"], env = env
    )
    origin = r.stdout.strip()
    rec["import"] = origin
    if not origin.startswith(str(tree)):
        raise RuntimeError(f"{arm}: {module} resolves to {origin or 'nothing'}, not {tree}")
    if cpus:
        # what the arm itself sees, from inside its own pinned process (the evidence for remote.json)
        r = subprocess.run(
            [
                str(py),
                "-c",
                "import os, json; print(json.dumps({'affinity': sorted(os.sched_getaffinity(0)), "
                "'omp': os.environ.get('OMP_NUM_THREADS'), 'cuda': os.environ.get('CUDA_VISIBLE_DEVICES')}))",
            ],
            env = env,
            capture_output = True,
            text = True,
            preexec_fn = cpu_split.isolate(cpus, cgroup),
        )
        try:
            rec["inside"] = json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            rec["inside"] = {"error": (r.stdout + r.stderr)[-200:]}
        log(f"{arm}{tag}: inside the arm: {rec['inside']}")
    cwd = root / f"cwd_{arm}{tag}"
    cwd.mkdir(exist_ok = True)
    dest = out / f"{arm}{tag}.json"
    dest.unlink(missing_ok = True)
    t0 = time.time()
    with open(out / f"{arm}{tag}.log", "w") as fh:
        try:
            p_ = subprocess.run(
                [str(py), "-u", str(script), *job_args, "--out", str(dest)],
                cwd = cwd,
                env = env,
                stdout = fh,
                stderr = subprocess.STDOUT,
                timeout = timeout,
                preexec_fn = cpu_split.isolate(cpus, cgroup) if cpus else None,
            )
            rc = p_.returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\nab_remote: {arm} timed out after {timeout}s\n")
            rc = -9
    rec.update(
        rc = rc,
        secs = round(time.time() - t0, 1),
        wrote = dest.exists(),
        gpu = env_extra.get("CUDA_VISIBLE_DEVICES"),
    )
    if cpus:
        rec.update(
            cpus = cpu_split.fmt_list(cpus),
            threads = len(cpus),
            cpu_isolation = "cgroup" if cgroup else "affinity",
        )
    log(
        f"{arm}{tag}: rc={rc} {rec['secs']}s metrics={'yes' if dest.exists() else 'no'}"
        + (f" GPU {rec['gpu']}" if rec["gpu"] is not None else "")
        + (
            f" CPUs {rec['cpus']} ({rec['threads']} threads, {rec['cpu_isolation']})"
            if cpus
            else ""
        )
    )
    return rec


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
    p.add_argument(
        "--parallel-arms",
        choices = ["auto", "on", "off"],
        default = "off",
        help = "base on GPU0 and head on GPU1 at once (auto: when two GPUs are visible)",
    )
    p.add_argument(
        "--swap", action = "store_true", help = "with parallel arms: a second pass on swapped GPUs"
    )
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
    gpus = gpu_names()
    rec["gpus"] = gpus
    parallel = a.parallel_arms == "on" or (a.parallel_arms == "auto" and len(gpus) >= 2)
    pys = {}
    if parallel:
        for arm in ("base", "head"):
            pys[arm] = arm_venv(arm, root)
        if not all(pys.values()):
            log("parallel arms unavailable (venv copy failed); running base then head")
            parallel = False
    sets = cpu_split.split(2) if parallel else None
    if parallel and sets is None:
        rec["sequential_reason"] = (
            f"fewer than 2 physical cores in this machine's {len(cpu_split.allowed())} "
            "allowed CPUs: base then head, sequentially"
        )
        log(rec["sequential_reason"])
        parallel = False
    rec["parallel_arms"] = parallel
    if parallel:
        import concurrent.futures as cf

        cg_base = cpu_split.probe_cgroup()
        cgroups = {i: cpu_split.make_cgroup(cg_base, f"sb_arm{i}", sets[i]) for i in (0, 1)}
        rec["cpu_isolation"] = {
            "sets": [cpu_split.fmt_list(x) for x in sets],
            "method": "cgroup" if all(cgroups.values()) else "affinity",
            "allowed": cpu_split.fmt_list(cpu_split.allowed()),
        }

        def one_pass(order, tag = ""):
            # order: arm -> slot; GPU slot i and CPU set i travel together, so --swap swaps both
            with cf.ThreadPoolExecutor(2) as ex:
                futs = {
                    arm: ex.submit(
                        run_arm,
                        arm,
                        tw[arm],
                        pys[arm],
                        script,
                        job_args,
                        out,
                        root,
                        module,
                        {"CUDA_VISIBLE_DEVICES": str(order[arm])},
                        a.arm_timeout,
                        tag,
                        sets[order[arm]],
                        cgroups[order[arm]],
                    )
                    for arm in ("base", "head")
                }
                return {arm: f.result() for arm, f in futs.items()}

        try:
            got = one_pass({"base": 0, "head": 1})
        except RuntimeError as e:
            return finish(2, str(e))
        for arm in ("base", "head"):
            rec[arm].update(got[arm])
        if a.swap:
            try:
                sw = one_pass({"base": 1, "head": 0}, tag = "_swap")
                rec["swap"] = {
                    arm: {
                        k: sw[arm].get(k) for k in ("rc", "secs", "gpu", "cpus", "threads", "wrote")
                    }
                    for arm in sw
                }
            except RuntimeError as e:
                rec["swap"] = {"error": str(e)}
    else:
        for arm in ("base", "head"):
            try:
                rec[arm].update(
                    run_arm(
                        arm,
                        tw[arm],
                        sys.executable,
                        script,
                        job_args,
                        out,
                        root,
                        module,
                        {},
                        a.arm_timeout,
                    )
                )
            except RuntimeError as e:
                return finish(2, str(e))
    ok = all(rec[arm].get("wrote") for arm in ("base", "head"))
    return finish(0 if ok else 2, "" if ok else "an arm wrote no metrics")


if __name__ == "__main__":
    raise SystemExit(main())
