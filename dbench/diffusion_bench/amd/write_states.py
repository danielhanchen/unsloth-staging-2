#!/usr/bin/env python3
"""states.json for amd_ci/lib/differential.py when the head is a local patch on a pinned main commit, not a PR:
base = <work>/states/base at --base-sha, head = <work>/states/head at the same commit with --patch applied (the
workflow's previous step did both). Same shape amd_ci/lib/states.py writes ({pr, merged, commits, paths})."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required = True, type = Path)
    ap.add_argument("--base-sha", required = True)
    ap.add_argument("--patch", required = True)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    paths = {"base": str(args.work / "states" / "base"), "head": str(args.work / "states" / "head")}
    for name, path in paths.items():
        rev = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output = True, text = True).stdout.strip()
        if rev != args.base_sha:
            raise SystemExit(f"{name} worktree is at {rev!r}, expected {args.base_sha}")
    dirty = subprocess.run(["git", "-C", paths["head"], "diff", "--cached", "--name-only"], capture_output = True,
                           text = True).stdout.split()
    if not dirty:
        raise SystemExit("head has no staged patch: the differential would compare main with itself")
    doc = {"pr": None, "merged": False, "patch": args.patch, "head_patched_files": dirty,
           "commits": {"base": args.base_sha, "head": f"{args.base_sha[:12]}+{args.patch}"}, "paths": paths}
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(doc, indent = 2), encoding = "utf-8")
    print(f"base {args.base_sha[:9]}  head {args.base_sha[:9]} + {args.patch} ({len(dirty)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
