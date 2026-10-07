"""Switchboard selection: which journeys / jobs a PR needs, with the reason for each.

    python studio_regress.py select --pr 11606 [--gh-repo unslothai/unsloth|unslothai/unsloth-zoo] [--json]
    python studio_regress.py select --files a.py,b.tsx [--labels "affects UI#"]
    python studio_regress.py select --list

Deterministic: glob triggers over changed paths + labels + "always". Paths no trigger
matches (docs / tests excepted) are reported as `unmatched` so the registry can be widened.
`[[workflow]]` entries route the same way to playbook workflows (claude/workflows/<name>.md) the
session must follow, reported as `workflows` (pr_extra_workflows.py inlines them).
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "switchboard.toml"
_IGNORED = re.compile(
    r"\.(md|mdx|rst|txt)$|(^|/)(LICENSE|COPYING)|(^|/)tests?/|(^|/)test_[^/]*\.py$|"
    r"\.(test|spec)\.[cm]?[jt]sx?$|^\.github/ISSUE_TEMPLATE/"
)


KINDS = ("journey", "job", "external", "regression")
CORE_KINDS = ("job", "regression")
# Which repos' PRs a target applies to, unless it sets `repos`: Studio journeys and the
# diffusion_bench suites exist only in unslothai/unsloth; Core targets run for both repos.
DEFAULT_REPOS = {
    "journey": ("unslothai/unsloth",),
    "external": ("unslothai/unsloth",),
    "job": ("unslothai/unsloth", "unslothai/unsloth-zoo"),
    "regression": ("unslothai/unsloth", "unslothai/unsloth-zoo"),
}


def target_repos(t):
    return tuple(t.get("repos") or DEFAULT_REPOS[t["kind"]])


def load(path = REGISTRY):
    data = tomllib.loads(Path(path).read_text())
    targets = data.get("target") or []
    names = [t["name"] for t in targets]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"duplicate switchboard targets: {sorted(dup)}")
    for w in data.get("workflow") or []:
        if not (w.get("name", "").endswith("_workflow") and w.get("triggers") and w.get("repos")):
            raise ValueError(f"workflow entry needs name *_workflow, repos and triggers: {w}")
    for t in targets:
        if t.get("kind") not in KINDS:
            raise ValueError(f"{t['name']}: kind must be one of {KINDS}")
        if t["kind"] == "external" and not (isinstance(t.get("cmd"), list) and t["cmd"]):
            raise ValueError(f"{t['name']}: external target needs a non-empty cmd argv list")
        if t["kind"] == "job" and not (
            t["name"].startswith("jobs/") and t["name"].split()[0].endswith(".py")
        ):
            raise ValueError(f"{t['name']}: job target name must be 'jobs/<file>.py [args]'")
        if t.get("repos") and not set(t["repos"]) <= set(DEFAULT_REPOS["job"]):
            raise ValueError(f"{t['name']}: repos must be among {DEFAULT_REPOS['job']}")
        for d in t.get("deps") or []:
            if d not in names:
                raise ValueError(f"{t['name']}: unknown dep {d}")
        for m in t.get("models") or []:
            if m not in (data.get("models") or {}):
                raise ValueError(f"{t['name']}: unknown model key {m}")
    return data


def _glob_match(path, pattern):
    # fnmatch "*" crosses "/" already; "**/" also matches zero dirs.
    return fnmatch.fnmatch(path, pattern) or (
        "/**/" in pattern and fnmatch.fnmatch(path, pattern.replace("/**/", "/"))
    )


def staging_commands(
    sel,
    pr = None,
    repo = None,
):
    """{target: command} for selected targets that only run on a staging runner (Apple Silicon,
    WebDriver). Without this a staging_only Core job was dropped locally with nothing saying where
    it must run, so MLX PRs merged with no Mac coverage."""
    out = {}
    for s in sel.get("selected") or []:
        cmd = s.get("staging")
        if s.get("staging_only") and cmd:
            out[s["name"]] = cmd.format(
                pr = pr if pr is not None else "<PR>", repo = repo or "unslothai/unsloth"
            )
    return out


def select(
    files,
    labels = (),
    data = None,
    tiers = None,
    kinds = None,
    repo = None,
):
    """`repo`: the PR's repo; targets that do not apply to it are never picked (None: no filter)."""
    data = data or load()
    labels = set(labels or ())
    picked, matched_paths = {}, set()
    for t in data["target"]:
        if repo and repo not in target_repos(t):
            continue
        why = []
        for trig in t.get("triggers") or []:
            if trig == "always":
                why.append("always")
            elif trig.startswith("label:"):
                if trig[6:] in labels:
                    why.append(trig)
            else:
                hits = [f for f in files if _glob_match(f, trig)]
                if hits:
                    matched_paths.update(hits)
                    why.append(f"{trig} ({len(hits)}: {hits[0]}{'…' if len(hits) > 1 else ''})")
        if why:
            picked[t["name"]] = {"target": t, "reasons": why}
    # dependencies come along, flagged as such
    changed = True
    while changed:
        changed = False
        for name in list(picked):
            for d in picked[name]["target"].get("deps") or []:
                if d not in picked:
                    dt = next(x for x in data["target"] if x["name"] == d)
                    picked[d] = {"target": dt, "reasons": [f"dep of {name}"]}
                    changed = True
    if tiers:
        picked = {k: v for k, v in picked.items() if v["target"]["tier"] in tiers}
    if kinds:
        picked = {k: v for k, v in picked.items() if v["target"]["kind"] in kinds}
    unmatched = sorted(f for f in files if f not in matched_paths and not _IGNORED.search(f))
    return {
        "workflows": select_workflows(files, labels, data, repo),
        "selected": [
            {
                "name": k,
                "kind": v["target"]["kind"],
                "tier": v["target"]["tier"],
                "est_s": v["target"].get("est_s"),
                "gpu_mem_gb": v["target"].get("gpu_mem_gb", 0),
                "perf_sensitive": v["target"].get("perf_sensitive", False),
                "staging_only": bool(v["target"].get("staging_only")),
                "staging": v["target"].get("staging"),
                "reasons": v["reasons"],
            }
            for k, v in picked.items()
        ],
        "unmatched": unmatched,
        "labels": sorted(labels),
        "files": len(files),
        "repo": repo,
    }


def select_workflows(
    files,
    labels = (),
    data = None,
    repo = None,
):
    """[{name, reasons}] for `[[workflow]]` entries whose repo + triggers match (docs / tests never do)."""
    data = data or load()
    out = []
    for w in data.get("workflow") or []:
        if repo and repo not in w["repos"]:
            continue
        why = []
        for trig in w["triggers"]:
            if trig.startswith("label:"):
                if trig[6:] in set(labels or ()):
                    why.append(trig)
                continue
            hits = [f for f in files if not _IGNORED.search(f) and _glob_match(f, trig)]
            if hits:
                why.append(f"{trig} ({len(hits)}: {hits[0]}{'…' if len(hits) > 1 else ''})")
        if why:
            prev = next((o for o in out if o["name"] == w["name"]), None)
            if prev:
                prev["reasons"] += why
            else:
                out.append({"name": w["name"], "reasons": why})
    return out


def pr_files_and_labels(pr, repo = "unslothai/unsloth"):
    files = subprocess.run(
        ["gh", "api", "--paginate", f"repos/{repo}/pulls/{pr}/files", "--jq", ".[].filename"],
        capture_output = True,
        text = True,
        check = True,
    ).stdout.split()
    labels = json.loads(
        subprocess.run(
            ["gh", "pr", "view", str(pr), "-R", repo, "--json", "labels"],
            capture_output = True,
            text = True,
            check = True,
        ).stdout
    )["labels"]
    return files, [l["name"] for l in labels]


def apply_overrides(
    sel,
    only = None,
    exclude = None,
    all_targets = False,
    data = None,
):
    """--only / --exclude / --all on top of a selection (names or fnmatch patterns)."""
    data = data or load()
    names = [t["name"] for t in data["target"]]
    if all_targets:
        chosen = names
    elif only:
        chosen = [n for n in names if any(fnmatch.fnmatch(n, o) or n == o for o in only)]
        unknown = [o for o in only if not any(fnmatch.fnmatch(n, o) or n == o for n in names)]
        if unknown:
            raise ValueError(f"unknown targets: {unknown}")
    else:
        chosen = [s["name"] for s in sel["selected"]]
    if exclude:
        chosen = [n for n in chosen if not any(fnmatch.fnmatch(n, e) for e in exclude)]
    return chosen


def main(argv = None):
    import argparse

    p = argparse.ArgumentParser(description = "Switchboard selection for a PR")
    p.add_argument("--pr")
    p.add_argument(
        "--repo",
        "--gh-repo",
        dest = "repo",
        default = None,
        help = "PR repo (default unslothai/unsloth); filters targets to that repo",
    )
    p.add_argument("--files", help = "comma-separated changed paths (offline)")
    p.add_argument("--labels", default = "", help = "comma-separated labels (offline)")
    p.add_argument("--tier", action = "append", choices = ("fast", "model", "gpu"))
    p.add_argument("--kind", action = "append", choices = KINDS)
    p.add_argument("--list", action = "store_true")
    p.add_argument("--json", action = "store_true")
    a = p.parse_args(argv)
    data = load()
    if a.list:
        for t in data["target"]:
            print(
                f"{t['name']:44} {t['kind']:10} {t['tier']:6} ~{t.get('est_s', '?')}s "
                f"{t.get('gpu_mem_gb', 0)}GB owner={t.get('owner', '-')} repos={','.join(target_repos(t))}"
            )
        return 0
    if a.files is not None:
        files, labels = [f for f in a.files.split(",") if f], [l for l in a.labels.split(",") if l]
    elif a.pr:
        files, labels = pr_files_and_labels(a.pr, a.repo or "unslothai/unsloth")
    else:
        p.error("--pr or --files required")
    repo = a.repo or ("unslothai/unsloth" if a.pr else None)
    out = select(files, labels, data, tiers = a.tier, kinds = a.kind, repo = repo)
    out["staging_commands"] = staging_commands(out, a.pr, repo)
    if a.json:
        print(json.dumps(out, indent = 1))
    else:
        print(
            f"SELECT {len(out['selected'])} targets ({out['files']} files, labels={out['labels']})"
        )
        for s in out["selected"]:
            print(f"  {s['name']:44} {s['tier']:6} {'; '.join(s['reasons'])[:150]}")
        for w in out["workflows"]:
            print(f"  WORKFLOW {w['name']:35} {'; '.join(w['reasons'])[:150]}")
        for name, cmd in out["staging_commands"].items():
            print(f"  STAGING {name:36} {cmd}")
        if out["unmatched"]:
            print(f"  UNMATCHED ({len(out['unmatched'])}): " + ", ".join(out["unmatched"][:15]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
