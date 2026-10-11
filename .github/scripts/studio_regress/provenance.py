"""Run provenance and staleness: which PR head a switchboard result is evidence for.

    python studio_regress.py status --pr 12289 [--root DIR] [--gh-repo R] [--repo CLONE] [--json]

`run` stamps pr, base_sha, head_sha, head_ref, the selected target names and started / finished
UTC into manifest.json, report.json and the summary.md header (stamp_start / finish / header_lines).

`status` reads that record, asks GitHub once for the PR's current head and commit list
(`gh pr view --json headRefOid,headRefName,commits`) and classifies every commit since the recorded
head:
  format   pre-commit.ci (or any) commit whose changed .py files are AST-equal before and after and
           whose other files differ only in whitespace: the tested behaviour did not move
  code     anything else, including a base-branch merge, a file whose parent is unreadable, or a
           commit the local clone cannot fetch (unknown counts as code)
Verdict: CURRENT (head unchanged), STALE(format-only) (only format commits since), STALE.
Exit 0 for CURRENT and STALE(format-only), 1 for STALE, 2 when there is no recorded run or the PR
cannot be read.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = Path(os.environ.get("WORKSPACE") or HERE.parent.parent.parent.parent)
PRECOMMIT_AUTHORS = ("pre-commit-ci[bot]", "pre-commit-ci")
EXIT = {"CURRENT": 0, "STALE(format-only)": 0, "STALE": 1}


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp_start(meta):
    meta.setdefault("started_utc", utc_now())
    return meta


def selection_names(meta):
    sel = meta.get("selection") or {}
    return [s.get("name") for s in sel.get("selected", []) if isinstance(s, dict)]


def record(meta, targets = None):
    """The provenance subset of a run's meta (manifest.json / report.json / summary.md)."""
    rec = {
        k: meta.get(k)
        for k in ("pr", "repo", "base_sha", "head_sha", "head_ref", "started_utc", "finished_utc")
    }
    rec["selection"] = (
        list(targets) if targets is not None else (meta.get("targets") or selection_names(meta))
    )
    return rec


def finish(meta):
    meta["finished_utc"] = utc_now()
    return meta


def header_lines(meta):
    """Lines under the summary.md title: which SHAs and when."""
    short = lambda s: (s or "?")[:10]  # noqa: E731
    lines = [
        f"pr: {meta.get('repo') or '?'}#{meta.get('pr') or 'A/A'}  head: `{short(meta.get('head_sha'))}`"
        f" ({meta.get('head_ref') or '?'})  base: `{short(meta.get('base_sha'))}`",
        f"started: {meta.get('started_utc') or '?'}  finished: {meta.get('finished_utc') or '?'}",
    ]
    tg = meta.get("targets") or selection_names(meta)
    if tg:
        lines.append(f"targets: {', '.join(tg)}")
    return lines


def update_manifest(root, meta):
    """Merge the provenance record into <root>/manifest.json (keeps journeys / keys)."""
    path = Path(root) / "manifest.json"
    try:
        old = json.loads(path.read_text()) if path.exists() else {}
    except ValueError:
        old = {}
    old.update(provenance = record(meta))
    path.write_text(json.dumps(old, indent = 1))
    return old


# ------------------------------------------------------------------ status
def recorded(root):
    """Provenance of the run in `root`: report.json first (finished run), else manifest.json."""
    root = Path(root)
    for name in ("report.json", "manifest.json"):
        p = root / name
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        prov = d.get("provenance") if name == "manifest.json" else d
        if prov and prov.get("head_sha"):
            return {**record(prov), "source": str(p), "functional": d.get("functional")}
    return None


def _git(
    repo,
    *args,
    check = True,
):
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output = True, text = True)
    if check and p.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()[:200]}")
    return p.stdout if p.returncode == 0 else None


def ast_equal(a, b):
    """True when two Python sources parse to the same AST (formatting / comments only)."""
    try:
        return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    except (SyntaxError, ValueError):
        return False


def _ws_norm(t):
    # blank lines and trailing / repeated inner spaces only: indentation (YAML, Makefiles) and the
    # presence of a space between tokens (`a b` vs `ab`) stay significant
    out = []
    for line in t.splitlines():
        if line.strip():
            ind = len(line) - len(line.lstrip())
            out.append(line[:ind] + " ".join(line.split()))
    return out


def _ws_equal(a, b):
    return _ws_norm(a) == _ws_norm(b)


def classify_commit(
    repo,
    sha,
    show = None,
):
    """("format" | "code", reason) for one commit in a local clone. `show(rev, path)` returns the
    file text at rev or None (tests inject it)."""
    parents = (_git(repo, "rev-list", "--parents", "-n", "1", sha, check = False) or "").split()[1:]
    if len(parents) != 1:
        return "code", f"{len(parents)} parents (merge or unreadable)"
    files = _git(repo, "diff", "--name-only", parents[0], sha, check = False)
    if files is None:
        return "code", "diff unreadable"
    show = show or (lambda rev, path: _git(repo, "show", f"{rev}:{path}", check = False))
    for f in files.split():
        before, after = show(parents[0], f), show(sha, f)
        if before is None or after is None:
            return "code", f"{f} added or removed"
        same = ast_equal(before, after) if f.endswith(".py") else _ws_equal(before, after)
        if not same:
            return "code", f"{f} {'AST' if f.endswith('.py') else 'content'} changed"
    return "format", f"{len(files.split())} file(s) AST / whitespace equal"


def pr_state(pr, gh_repo):
    """One `gh pr view` call: head sha, head ref and the PR's commits (oldest first)."""
    out = subprocess.run(
        ["gh", "pr", "view", str(pr), "-R", gh_repo, "--json", "headRefOid,headRefName,commits"],
        capture_output = True,
        text = True,
    )
    if out.returncode:
        raise RuntimeError(out.stderr.strip()[:300])
    return json.loads(out.stdout)


def since(commits, recorded_head):
    """Commits after `recorded_head` in the PR's list; None when it is no longer on the branch."""
    oids = [c.get("oid") for c in commits]
    if recorded_head not in oids:
        return None
    return commits[oids.index(recorded_head) + 1 :]


def is_precommit(c):
    authors = [a.get("login") or a.get("name") or "" for a in c.get("authors") or []]
    return any(a in PRECOMMIT_AUTHORS for a in authors) or (
        c.get("messageHeadline") or ""
    ).startswith("[pre-commit.ci]")


def verdict(rows, current_head, recorded_head):
    if current_head == recorded_head:
        return "CURRENT"
    if rows is None:
        return "STALE"
    return "STALE(format-only)" if rows and all(r["kind"] == "format" for r in rows) else "STALE"


def status(
    pr,
    root,
    gh_repo = None,
    repo = None,
    state = None,
    classify = None,
):
    rec = recorded(root)
    if rec is None:
        return {"verdict": "NO_RUN", "root": str(root)}
    # the run's own repo (an unsloth-zoo run asks about unsloth-zoo #N), else the default
    rr = rec.get("repo") or ""
    gh_repo = gh_repo or (rr if re.fullmatch(r"[\w.-]+/[\w.-]+", rr) else "unslothai/unsloth")
    state = state or pr_state(pr, gh_repo)
    head = state["headRefOid"]
    new = since(state.get("commits") or [], rec["head_sha"])
    rows = None
    if head != rec["head_sha"] and new is not None:
        repo = Path(repo or WS / "unsloth")
        if classify is None and repo.is_dir():
            _git(repo, "fetch", "-q", "origin", f"pull/{pr}/head", check = False)
        rows = []
        for c in new:
            kind, why = (
                classify
                or (
                    lambda sha: classify_commit(repo, sha)
                    if repo.is_dir()
                    else ("code", "no local clone")
                )
            )(c["oid"])
            rows.append(
                {
                    "sha": c["oid"],
                    "headline": c.get("messageHeadline", ""),
                    "precommit": is_precommit(c),
                    "kind": kind,
                    "why": why,
                }
            )
    return {
        "verdict": verdict(rows, head, rec["head_sha"]),
        "root": str(root),
        "recorded": rec,
        "current_head": head,
        "current_ref": state.get("headRefName"),
        "commits": rows,
        "force_pushed": new is None and head != rec["head_sha"],
    }


def main(argv = None):
    p = argparse.ArgumentParser(
        description = "Is a switchboard run still evidence for the PR's current head?"
    )
    p.add_argument("--pr", type = int, required = True)
    p.add_argument("--root", help = "run output dir (default outputs/studio_regress/pr<N>)")
    p.add_argument("--gh-repo", help = "default: the run's recorded repo, else unslothai/unsloth")
    p.add_argument("--repo", default = str(WS / "unsloth"), help = "local clone to classify commits in")
    p.add_argument("--json", action = "store_true")
    a = p.parse_args(argv)
    root = Path(a.root or WS / "outputs" / "studio_regress" / f"pr{a.pr}")
    try:
        res = status(a.pr, root, a.gh_repo, a.repo)
    except RuntimeError as e:
        print(f"status: cannot read PR #{a.pr}: {e}", file = sys.stderr)
        return 2
    if a.json:
        print(json.dumps(res, indent = 1))
    elif res["verdict"] == "NO_RUN":
        print(f"NO_RUN: no report.json / manifest.json with a head SHA under {root}")
    else:
        rec = res["recorded"]
        print(
            f"{res['verdict']}: run {rec['head_sha'][:10]} ({rec.get('finished_utc') or 'unfinished'}, "
            f"{rec.get('functional') or '?'}) vs PR head {res['current_head'][:10]}"
        )
        if res["force_pushed"]:
            print("  recorded head is no longer on the PR branch (force-push or rebase)")
        for r in res["commits"] or []:
            tag = " pre-commit.ci" if r["precommit"] else ""
            print(f"  {r['sha'][:10]} {r['kind']:6}{tag}  {r['headline'][:70]}  ({r['why']})")
    return EXIT.get(res["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
