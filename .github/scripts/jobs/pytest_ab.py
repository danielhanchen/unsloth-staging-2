"""Run the same pytest selection at two commits on this runner and diff the failing test IDs.

    python pytest_ab.py --base SHA --head SHA --tests a.py b.py [-k EXPR]

Head failing a test that passed at base = regression (exit 1). Writes --out JSON + JOB_RESULT.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/unslothai/unsloth"


def run(
    cmd,
    cwd = None,
    env = None,
):
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
    return p.returncode, p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required = True)
    ap.add_argument("--head", required = True)
    ap.add_argument("--tests", nargs = "+", required = True)
    ap.add_argument("-k", dest = "k", default = None)
    ap.add_argument("--out", default = "pytest_ab.json")
    ap.add_argument("--work", default = None)
    ap.add_argument("--tiny", action = "store_true")
    a = ap.parse_args()
    arts = Path("artifacts")
    arts.mkdir(exist_ok = True)
    work = Path(a.work or (Path.cwd() / "pytest_ab_work")).resolve()
    work.mkdir(parents = True, exist_ok = True)
    repo = work / "unsloth"
    if not (repo / ".git").exists():
        rc, out = run(["git", "clone", "-q", "--filter=blob:none", REPO_URL, str(repo)])
        if rc:
            raise SystemExit(out[-2000:])
    run(["git", "-C", str(repo), "fetch", "-q", "origin", a.base, a.head])
    result = {
        "job": "pytest_ab",
        "host": {"system": platform.system(), "machine": platform.machine()},
        "base": a.base,
        "head": a.head,
        "tests": a.tests,
        "arms": {},
        "checks": {},
    }
    for arm, sha in (("base", a.base), ("head", a.head)):
        run(["git", "-C", str(repo), "checkout", "-q", "-f", sha])
        run(["git", "-C", str(repo), "clean", "-q", "-ffdx"])
        tests = [t for t in a.tests if (repo / t).exists()]
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            *tests,
            "-q",
            "--tb=line",
            "-p",
            "no:cacheprovider",
            "-rfE",
        ]
        if a.k:
            cmd += ["-k", a.k]
        env = dict(os.environ, PYTHONPATH = str(repo / "studio" / "backend"))
        rc, out = run(cmd, cwd = repo, env = env)
        (arts / f"pytest_{arm}.log").write_text(out, encoding = "utf-8")
        failed = sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", out, re.M)))
        m = re.search(r"(\d+) passed", out)
        result["arms"][arm] = {
            "sha": sha,
            "rc": rc,
            "passed": int(m.group(1)) if m else 0,
            "failed": failed,
            "tests": tests,
        }
        print("ARM", arm, rc, len(failed), flush = True)
    b, h = set(result["arms"]["base"]["failed"]), set(result["arms"]["head"]["failed"])
    result["new_on_head"] = sorted(h - b)
    result["fixed_on_head"] = sorted(b - h)
    ran = all(
        result["arms"][x]["rc"] in (0, 1) and result["arms"][x]["passed"] > 0
        for x in ("base", "head")
    )
    result["checks"]["both_arms_ran"] = {
        "ok": ran,
        "detail": str({x: result["arms"][x]["rc"] for x in ("base", "head")}),
    }
    result["checks"]["no_new_failures"] = {
        "ok": not result["new_on_head"],
        "detail": str(result["new_on_head"][:20]),
    }
    result["passed"] = all(c["ok"] for c in result["checks"].values())
    result["finished"] = True
    Path(a.out).write_text(json.dumps(result, indent = 1))
    (arts / "pytest_ab.json").write_text(json.dumps(result, indent = 1))
    print(
        "JOB_RESULT "
        + json.dumps(
            {
                "passed": result["passed"],
                "new_on_head": result["new_on_head"],
                "fixed_on_head": result["fixed_on_head"],
                "failed": {x: len(result["arms"][x]["failed"]) for x in ("base", "head")},
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
