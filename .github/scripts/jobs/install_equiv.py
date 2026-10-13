"""Install Unsloth Studio from source at two commits and prove the results are byte-identical.

Each arm: fresh clone state (git clean), fresh UNSLOTH_STUDIO_HOME at the same path, the
platform's installer with --local, then a sha256 manifest of the Studio home and the built
frontend (studio/frontend/dist). Arms default to base, head, base: the base-vs-base diff is
the run-to-run noise floor (timestamps, caches), and only base-vs-head differences outside it
count. Writes --out JSON, prints JOB_RESULT, exit 0 when nothing unexplained differs.

    python install_equiv.py --base SHA --head SHA [--no-torch] [--arms base,head,base]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/unslothai/unsloth"
# Paths whose content is expected to vary run to run; reported, never compared.
VOLATILE_PARTS = ("__pycache__", "logs")
VOLATILE_SUFFIXES = (".pyc", ".log")
# Download caches (uv archive ids are random per run), not installed content.
VOLATILE_PREFIXES = ("home/cache/",)


def sh(
    cmd,
    cwd = None,
    env = None,
    log = None,
    check = True,
):
    t0 = time.time()
    p = subprocess.run(
        cmd,
        cwd = cwd,
        env = env,
        stdin = subprocess.DEVNULL,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        errors = "replace",
    )
    if log is not None:
        log.write_text(p.stdout, encoding = "utf-8")
    if check and p.returncode != 0:
        sys.stdout.write(p.stdout[-4000:])
        raise SystemExit(f"command failed ({p.returncode}): {cmd}")
    return p.returncode, time.time() - t0, p.stdout


def rmtree(path: Path):
    def onerr(fn, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            fn(p)
        except OSError:
            pass

    if path.exists():
        shutil.rmtree(path, onerror = onerr)


def manifest(root: Path, prefix: str, out: dict):
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks = False):
        d = Path(dirpath)
        for name in list(dirnames):
            p = d / name
            if p.is_symlink():
                out[f"{prefix}/{p.relative_to(root).as_posix()}"] = "link:" + os.readlink(p)
                dirnames.remove(name)
        for name in filenames:
            p = d / name
            rel = f"{prefix}/{p.relative_to(root).as_posix()}"
            if p.is_symlink():
                out[rel] = "link:" + os.readlink(p)
                continue
            h = hashlib.sha256()
            try:
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                out[rel] = h.hexdigest()
            except OSError as e:
                out[rel] = f"unreadable:{e.errno}"


def volatile(path: str) -> bool:
    parts = path.split("/")
    return (
        any(p in VOLATILE_PARTS for p in parts)
        or path.endswith(VOLATILE_SUFFIXES)
        or path.startswith(VOLATILE_PREFIXES)
    )


def diff(a: dict, b: dict) -> set:
    return {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required = True)
    ap.add_argument("--head", required = True)
    ap.add_argument("--arms", default = "base,head,base")
    ap.add_argument("--no-torch", action = "store_true")
    ap.add_argument("--work", default = None)
    ap.add_argument("--out", default = "install_equiv.json")
    ap.add_argument(
        "--tiny", action = "store_true", help = "accepted for the staging job contract; unused"
    )
    a = ap.parse_args()

    work = Path(a.work or (Path.cwd() / "equiv_work")).resolve()
    work.mkdir(parents = True, exist_ok = True)
    repo, home = work / "unsloth", work / "home"
    arts = Path("artifacts")
    arts.mkdir(exist_ok = True)
    if not (repo / ".git").exists():
        sh(["git", "clone", "-q", "--filter=blob:none", REPO_URL, str(repo)])
    sh(["git", "-C", str(repo), "fetch", "-q", "origin", a.base, a.head])
    windows = platform.system() == "Windows"
    result = {
        "job": "install_equiv",
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "cpus": os.cpu_count(),
        },
        "base": a.base,
        "head": a.head,
        "no_torch": a.no_torch,
        "arms": [],
        "checks": {},
    }
    manifests = []
    for i, arm in enumerate(a.arms.split(",")):
        sha = a.base if arm == "base" else a.head
        sh(["git", "-C", str(repo), "checkout", "-q", "-f", sha])
        sh(["git", "-C", str(repo), "clean", "-q", "-ffdx"])
        rmtree(home)
        env = dict(os.environ, UNSLOTH_STUDIO_HOME = str(home), UNSLOTH_SKIP_AUTOSTART = "1")
        env.pop("UNSLOTH_STUDIO_SERIAL_SETUP", None)
        flags = ["--local"] + (["--no-torch"] if a.no_torch else [])
        if windows:
            cmd = [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                "install.ps1",
                *flags,
            ]
        else:
            cmd = ["bash", "install.sh", *flags]
        log = arts / f"{i}_{arm}.log"
        rc, wall, out = sh(cmd, cwd = repo, env = env, log = log, check = False)
        m: dict = {}
        manifest(home, "home", m)
        manifest(repo / "studio" / "frontend" / "dist", "dist", m)
        leftovers = [p.name for p in (repo / "studio" / "frontend").glob(".unsloth-build*")]
        overlapped = (
            "waiting for the frontend build" in out
            or "frontend build running in the background" in out
        )
        arm_rec = {
            "arm": arm,
            "sha": sha,
            "rc": rc,
            "wall_s": round(wall, 1),
            "files": len(m),
            "overlapped": overlapped,
            "frontend_built": "frontend       built" in out,
            "marker_left": leftovers,
        }
        result["arms"].append(arm_rec)
        manifests.append((arm, m))
        print("ARM", json.dumps(arm_rec), flush = True)

    bases = [m for arm, m in manifests if arm == "base"]
    heads = [m for arm, m in manifests if arm == "head"]
    noise = set()
    for x in bases[1:]:
        noise |= diff(bases[0], x)
    bh = diff(bases[0], heads[0]) if bases and heads else set()
    unexplained = sorted(k for k in bh - noise if not volatile(k))
    import gzip

    for j, (arm, m) in enumerate(manifests):
        with gzip.open(arts / f"manifest_{j}_{arm}.json.gz", "wt") as f:
            json.dump(m, f)
    result["summary"] = {
        "files_base": len(bases[0]) if bases else 0,
        "files_head": len(heads[0]) if heads else 0,
        "base_vs_head_diff": len(bh),
        "base_vs_base_noise": len(noise),
        "volatile_skipped": len([k for k in bh if volatile(k)]),
        "unexplained": unexplained[:200],
        "unexplained_count": len(unexplained),
    }
    (arts / "diff.json").write_text(
        json.dumps({"base_vs_head": sorted(bh)[:3000], "noise": sorted(noise)[:3000]}, indent = 0)
    )
    result["checks"]["all_installs_rc0"] = {
        "ok": all(r["rc"] == 0 for r in result["arms"]),
        "detail": str([r["rc"] for r in result["arms"]]),
    }
    result["checks"]["head_identical_to_base"] = {
        "ok": not unexplained,
        "detail": f"{len(unexplained)} unexplained differing paths",
    }
    result["checks"]["no_marker_left"] = {
        "ok": not any(r["marker_left"] for r in result["arms"]),
        "detail": str([r["marker_left"] for r in result["arms"]]),
    }
    result["passed"] = all(c["ok"] for c in result["checks"].values())
    result["finished"] = True
    Path(a.out).write_text(json.dumps(result, indent = 1))
    (arts / "install_equiv.json").write_text(json.dumps(result, indent = 1))
    print(
        "JOB_RESULT "
        + json.dumps(
            {"passed": result["passed"], "summary": result["summary"], "arms": result["arms"]}
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
