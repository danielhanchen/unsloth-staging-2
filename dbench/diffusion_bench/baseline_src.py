#!/usr/bin/env python3
"""Check out the baseline UI's source (and optionally its GGUF loader node) for a generated workflow, without the
workflow text naming either: a workflow pushed to a public fork / staging repo carries only neutral names.

  python diffusion_bench/baseline_src.py --dir "$WORK/baseline" --ref "$BASELINE_REF" \
      [--gguf-sha SHA] [--commit-out FILE] [--reqs-out FILE]

The repository URLs come from $BASELINE_REPO / $BASELINE_GGUF_REPO (extnames), never from a literal. Prints "baseline <sha> <subject>" (and writes it to
--commit-out); --reqs-out writes a pip requirements file: the source's requirements without torch / torchvision /
torchaudio (the venv's own build stays), plus the GGUF node's. Exit 0 = checked out, 1 = clone / checkout failed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

GGUF_NODE_DIR = ("custom_nodes", "gguf_loader")
TORCH_LINE = re.compile(r"^(torch|torchvision|torchaudio)\b")


def git(*cmd) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *cmd], capture_output = True, text = True)


def checkout(url: str, dest: Path, ref: str) -> bool:
    for cmd in (("clone", "-q", url, str(dest)), ("-C", str(dest), "checkout", "-q", ref)):
        r = git(*cmd)
        if r.returncode != 0:
            print(f"checkout failed ({'clone' if cmd[0] == 'clone' else 'checkout'} rc={r.returncode})")
            return False
    return True


def main(argv = None) -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required = True)
    ap.add_argument("--ref", required = True)
    ap.add_argument("--gguf-sha", default = "", help = "also check out the GGUF loader node at this sha")
    ap.add_argument("--commit-out", default = None)
    ap.add_argument("--reqs-out", default = None)
    a = ap.parse_args(argv)
    import extnames as X  # noqa: PLC0415

    try:  # $BASELINE_REPO (vars.BASELINE_REPO in a workflow); no literal URL in a public copy
        repo, node_repo = X.baseline_repo(True), (X.gguf_node_repo(True) if a.gguf_sha else None)
    except RuntimeError as exc:
        print(f"baseline checkout: {exc}")
        return 1
    root = Path(a.dir)
    if not checkout(repo, root, a.ref):
        return 1
    line = "baseline " + git("-C", str(root), "log", "-1", "--format=%H %s").stdout.strip()
    node = root.joinpath(*GGUF_NODE_DIR)
    if a.gguf_sha:
        if not checkout(node_repo, node, a.gguf_sha):
            return 1
        line += " | gguf node " + git("-C", str(node), "rev-parse", "HEAD").stdout.strip()
    print(line)
    if a.commit_out:
        Path(a.commit_out).write_text(line + "\n", encoding = "utf-8")
    if a.reqs_out:
        reqs = [ln for ln in (root / "requirements.txt").read_text().splitlines() if not TORCH_LINE.match(ln.strip())]
        if (node / "requirements.txt").exists():
            reqs.append(f"-r {node / 'requirements.txt'}")
        Path(a.reqs_out).write_text("\n".join(reqs) + "\n", encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
