#!/usr/bin/env python3
"""Deterministic UI gate for a PR: run the FULL paired UI suite, or only the cheap PROBE.

    python -m studio_regress.gate --pr N [--repo unslothai/unsloth] [--json]
    python -m studio_regress.gate upgrade --report outputs/studio_regress/prN/report.json

FULL  when the `affects UI#` label is present, ui_impact says YES / MAYBE, or classification
      is unknown / failed (conservative).
PROBE otherwise: fast tier + crawl still run, so a backend-only visible change (#11604) is
      caught; `upgrade` turns a PROBE report with any VISUAL_DIFF / DOM_ONLY_DIFF into FULL.

Prints one line `UI_GATE {FULL|PROBE} reasons=[...]`. Exit: FULL 0, PROBE 10, error 2.
No LLM anywhere: label + file globs + ui_impact's static verdict only.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "pr_review"))

UI_LABEL = "affects UI#"
FULL, PROBE = "FULL", "PROBE"
EXIT = {FULL: 0, PROBE: 10}
# Paths whose change needs the GPU tier (training, export, diffusion, inference backends).
# The switchboard owns per-journey triggers; this only widens FULL to include `gpu`.
GPU_AREAS = (
    "studio/backend/core/training/*", "studio/backend/core/inference/diffusion*",
    "studio/backend/core/inference/llama_cpp*", "studio/backend/core/export/*",
    "studio/backend/routes/training*", "studio/backend/routes/export*",
    "studio/backend/routes/video*", "unsloth/*", "unsloth_zoo/*",
)
DESKTOP_AREAS = ("studio/src-tauri/*", "studio/frontend/*")
UPGRADE_VERDICTS = {"VISUAL_DIFF", "DOM_ONLY_DIFF", "DIVERGED"}


def _gh_read(cmd):
    try:
        from gh_read_env import run_gh_read
        return run_gh_read(cmd, label="studio_regress.gate")
    except ImportError:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def pr_meta(pr, repo):
    """labels, files (paginated past gh's 100-file cap), head sha: 1 call, +1 when >= 100 files."""
    r = _gh_read(["gh", "pr", "view", str(pr), "-R", repo, "--json", "labels,files,headRefOid"])
    if r.returncode != 0:
        raise RuntimeError(f"gh pr view failed: {(r.stderr or '').strip()[:200]}")
    d = json.loads(r.stdout)
    files = [f["path"] for f in d.get("files") or []]
    if len(files) >= 100:
        r2 = _gh_read(["gh", "api", "--paginate", f"repos/{repo}/pulls/{pr}/files", "--jq", ".[].filename"])
        if r2.returncode == 0 and r2.stdout.strip():
            files = r2.stdout.split()
    return {"labels": [l["name"] for l in d.get("labels") or []], "files": files,
            "head_sha": d.get("headRefOid")}


def ui_impact_verdict(pr, repo, timeout=300):
    """YES / MAYBE / NO from ui_impact.py, or None when it could not decide."""
    cmd = [sys.executable, str(_HERE.parent / "pr_review" / "ui_impact.py"), "--pr", str(pr),
           "--repo", repo, "--json"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        v = json.loads(r.stdout).get("verdict")
        return (v, json.loads(r.stdout).get("reasons", [])) if v in ("YES", "MAYBE", "NO") else (None, [])
    except Exception as e:  # noqa: BLE001  unknown -> FULL
        return None, [f"ui_impact failed: {e}"]


def _match(files, globs):
    return sorted({f for f in files for g in globs if fnmatch.fnmatch(f, g)})


def decide(labels, files, impact, impact_reasons=()):
    """Pure decision: (verdict, reasons, tiers, desktop)."""
    reasons = []
    if UI_LABEL in labels:
        reasons.append(f"label `{UI_LABEL}`")
    if impact in ("YES", "MAYBE"):
        reasons.append(f"ui_impact {impact}" + (f": {impact_reasons[0]}" if impact_reasons else ""))
    if impact is None:
        reasons.append("ui_impact unknown (conservative FULL)")
    if not files:
        reasons.append("no changed files listed (conservative FULL)")
    verdict = FULL if reasons else PROBE
    if verdict == PROBE:
        reasons.append("static triage NO and no UI label: fast tier + crawl probe only")
    gpu = _match(files, GPU_AREAS)
    tiers = ["fast", "model"] if verdict == FULL else ["fast"]
    if gpu:
        tiers.append("gpu")
        reasons.append(f"gpu tier: {len(gpu)} file(s) in training/inference/export/diffusion areas")
    desktop = bool(_match(files, DESKTOP_AREAS)) and verdict == FULL
    return verdict, reasons, tiers, desktop


def upgrade(report):
    """PROBE report whose steps show a UI change -> FULL, with the reason spelled out."""
    changed = [s["key"] for s in report.get("steps", []) if s.get("verdict") in UPGRADE_VERDICTS]
    if report.get("gate", {}).get("verdict", PROBE) == FULL or not changed:
        return report.get("gate", {}).get("verdict", PROBE), []
    return FULL, [f"static triage said NO, screenshots differ: {', '.join(changed[:5])}"
                  + (f" (+{len(changed) - 5})" if len(changed) > 5 else "")]


def line(verdict, reasons):
    return f"UI_GATE {verdict} reasons={json.dumps(reasons)}"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["upgrade"]:
        p = argparse.ArgumentParser(prog="gate upgrade")
        p.add_argument("--report", required=True)
        a = p.parse_args(argv[1:])
        v, reasons = upgrade(json.loads(Path(a.report).read_text()))
        print(line(v, reasons))
        return EXIT[v]
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pr", required=True, type=lambda s: int(str(s).lstrip("#").rsplit("/", 1)[-1]))
    p.add_argument("--repo", default="unslothai/unsloth")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    try:
        meta = pr_meta(a.pr, a.repo)
    except Exception as e:  # noqa: BLE001
        print(line(FULL, [f"pr metadata unavailable: {e}"]))
        print(f"gate: {e}", file=sys.stderr)
        return 2
    impact, impact_reasons = ui_impact_verdict(a.pr, a.repo)
    v, reasons, tiers, desktop = decide(meta["labels"], meta["files"], impact, impact_reasons)
    out = {"pr": a.pr, "repo": a.repo, "head_sha": meta["head_sha"], "verdict": v, "reasons": reasons,
           "tiers": tiers, "desktop": desktop, "labels": meta["labels"], "ui_impact": impact,
           "files": len(meta["files"])}
    print(json.dumps(out, indent=2) if a.json else line(v, reasons))
    return EXIT[v]


if __name__ == "__main__":
    sys.exit(main())
