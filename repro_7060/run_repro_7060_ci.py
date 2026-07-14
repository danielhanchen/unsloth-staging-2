# SPDX-License-Identifier: AGPL-3.0-or-later
"""
CI driver for the #7060 cross-platform reproduction.

Prepares every code state as its own studio/backend source tree (all sharing the
one venv the workflow already installed), then runs scripts/repro_7060.py for one
target and writes a JSON report. Pure git + pip + Python so it runs identically on
ubuntu / macOS / windows runners.

Code states:
  main        upstream main            (2026.7.2; has #7031, not #7033/#7113 -- buggy)
  pr7113      pull/7113/head           (the no-symlink size-identity fix for #7060)
  pr7033      pull/7033/head           (companion + packed-Q2 label fix)
  pr7113_7033 main + both PRs merged   (do the fixes compose?)
  v2026_6_9   pip wheel unsloth==2026.6.9 (pre-feature "downloads work" baseline)

Each prep step is best-effort: a state that fails to prepare is recorded and
skipped so the rest of the matrix still produces results.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

UPSTREAM = "https://github.com/unslothai/unsloth"
_HERE = Path(__file__).resolve().parent


def _run(cmd, cwd=None, check=True, timeout=1800):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, check=check, timeout=timeout,
                          capture_output=True, text=True)


def _git(args, cwd, check=True):
    return _run(["git", *args], cwd=cwd, check=check)


def prepare_code_states(checkout: Path, work: Path) -> list[dict]:
    """Return [{name, backend_path, note?}], best-effort."""
    work.mkdir(parents=True, exist_ok=True)
    states: list[dict] = []

    # A remote pointing at real upstream so we can fetch main + PR heads even when
    # `checkout` is a staging fork.
    try:
        _git(["remote", "add", "upstream", UPSTREAM], checkout, check=False)
        _git(["fetch", "--no-tags", "upstream",
              "main:refs/repro/main",
              "pull/7033/head:refs/repro/pr7033",
              "pull/7113/head:refs/repro/pr7113"], checkout)
    except Exception as e:  # noqa: BLE001
        print("WARN upstream fetch failed:", e)

    def worktree(name: str, ref: str, merges: list[str] | None = None) -> None:
        wt = work / name
        try:
            if wt.exists():
                _git(["worktree", "remove", "--force", str(wt)], checkout, check=False)
            _git(["worktree", "add", "--detach", str(wt), ref], checkout)
            for m in (merges or []):
                _git(["-c", "user.email=ci@local", "-c", "user.name=ci",
                      "merge", "--no-edit", m], wt)
            states.append({"name": name, "backend_path": str(wt / "studio" / "backend")})
        except Exception as e:  # noqa: BLE001
            states.append({"name": name, "backend_path": "", "note": f"prep failed: {e}"})

    worktree("main", "refs/repro/main")
    worktree("pr7113", "refs/repro/pr7113")
    worktree("pr7033", "refs/repro/pr7033")
    worktree("pr7113_7033", "refs/repro/main", merges=["refs/repro/pr7113", "refs/repro/pr7033"])

    # v2026.6.9 from the released wheel (contains studio/backend; no heavy deps).
    try:
        wheeldir = work / "_wheel"
        wheeldir.mkdir(exist_ok=True)
        _run([sys.executable, "-m", "pip", "download", "unsloth==2026.6.9",
              "--no-deps", "-d", str(wheeldir)])
        whl = next(p for p in wheeldir.glob("unsloth-2026.6.9*.whl"))
        dst = work / "v2026_6_9"
        with zipfile.ZipFile(whl) as z:
            z.extractall(dst)
        states.append({"name": "v2026_6_9", "backend_path": str(dst / "studio" / "backend")})
    except Exception as e:  # noqa: BLE001
        states.append({"name": "v2026_6_9", "backend_path": "", "note": f"wheel prep failed: {e}"})

    ready = [s for s in states if s["backend_path"] and Path(s["backend_path"]).is_dir()]
    print("code states ready:", [s["name"] for s in ready])
    print("code states skipped:", [(s["name"], s.get("note")) for s in states if s not in ready])
    return ready


def load_repro():
    spec = importlib.util.spec_from_file_location("repro_7060", _HERE / "repro_7060.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--filename", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkout", default=".")
    ap.add_argument("--work", default="_repro_work")
    ap.add_argument("--hf-home", default="_repro_hf")
    args = ap.parse_args()

    checkout = Path(args.checkout).resolve()
    work = Path(args.work).resolve()
    states = prepare_code_states(checkout, work)
    if not states:
        print("FATAL: no code states could be prepared")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"error": "no code states"}))
        return 1

    repro = load_repro()

    class A:  # minimal args object for role_orchestrate
        pass
    a = A()
    a.role = "orchestrate"
    a.hf_home = str(Path(args.hf_home).resolve())
    a.repo = args.repo
    a.quant = args.quant
    a.filename = args.filename
    a.companions = ""
    a.code_states = json.dumps(states)
    a.out = args.out
    return repro.role_orchestrate(a)


if __name__ == "__main__":
    raise SystemExit(main())
