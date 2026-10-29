"""Auto-confirm: rerun what would block (FAIL_HEAD) or change the UI verdict (VISUAL_DIFF) once, on
both arms, before believing it.

After the main run, `run` collects the targets owning a FAIL_HEAD / VISUAL_DIFF step (plus VOIDs
marked `retryable`: GPU contention, a harness hang) and reruns only those targets in a nested
`run --only <targets> --root <root>/confirm_<root name> --fresh-base --no-prefetch --no-confirm`: a fresh Studio /
worker per arm and fresh GPU leases. Then per step:

  FAIL_HEAD     confirm FAIL_HEAD -> stays FAIL_HEAD, "reproduced" only when a check that failed
                the first time fails again (external suites with per-check results; a step without
                them compares verdicts only); other checks failing instead -> stays FAIL_HEAD, noted
                "not reproduced" with both check lists (each failed 1 of 2); confirm FAIL_BOTH (base fails
                too) -> FLAKY; confirm SAME -> stays FAIL_HEAD ("intermittent": head failed 1 of 2, base
                never did, so a race the PR added still blocks)
  VISUAL_DIFF   confirm VISUAL_DIFF / DOM_ONLY_DIFF / DIVERGED -> stays ("reproduced"); SAME -> FLAKY
  VOID (retryable)  confirm SAME or FAIL_HEAD -> that verdict (the first attempt proved nothing)
  anything else on confirm (VOID, FAIL_BOTH, missing) -> the first verdict stays, noted "unconfirmed"

Both attempts' evidence stays: the step keeps its own paths and gains `first_attempt` (verdict,
note, statuses) and `confirm` (the confirm run's record, whose paths point under the confirm root). The confirm root is named
per run (`confirm_<root name>`): the nested run keys its Studio state and ports on its root's name,
so a bare `confirm` would share them across PRs.
`run --no-confirm` (or STUDIO_REGRESS_CONFIRM=1, set in the nested run) skips all of this.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLI = HERE.parent / "studio_regress.py"
CONFIRM = ("FAIL_HEAD", "VISUAL_DIFF")
REPRODUCED = {
    "FAIL_HEAD": ("FAIL_HEAD",),
    "VISUAL_DIFF": ("VISUAL_DIFF", "DOM_ONLY_DIFF", "DIVERGED"),
}
CLEARED = {"FAIL_HEAD": ("FAIL_BOTH",), "VISUAL_DIFF": ("SAME", "FLAKY")}
INTERMITTENT = {"FAIL_HEAD": ("SAME", "FLAKY")}


def candidates(steps):
    return [
        s
        for s in steps
        if s.get("verdict") in CONFIRM or (s.get("verdict") == "VOID" and s.get("retryable"))
    ]


def target_of(step):
    return step.get("journey") or step["key"].rsplit("/", 1)[0]


def targets(steps):
    out = []
    for s in candidates(steps):
        t = target_of(s)
        if t not in out:
            out.append(t)
    return out


def confirm_root(root):
    return Path(root) / f"confirm_{Path(root).name}"


def enabled(a):
    return (
        not getattr(a, "no_confirm", False)
        and bool(a.pr)
        and a.side == "both"
        and not a.base_url
        and not a.record_flaky
        and not os.environ.get("STUDIO_REGRESS_CONFIRM")
        and not os.environ.get("STUDIO_REGRESS_ISOLATION")
    )


def argv(a, root, names):
    cmd = [
        sys.executable,
        str(CLI),
        "run",
        "--pr",
        str(a.pr),
        "--gh-repo",
        a.gh_repo,
        "--repo",
        str(a.repo),
        "--only",
        ",".join(names),
        "--root",
        str(confirm_root(root)),
        "--fresh-base",
        "--no-prefetch",
        "--no-confirm",
    ]
    if a.online:
        cmd.append("--online")
    if a.core_python:
        cmd += ["--core-python", a.core_python]
    if a.scheduler:
        cmd += ["--scheduler", a.scheduler]
    return cmd


def run_confirm(
    a,
    root,
    names,
    log = print,
    runner = subprocess.run,
):
    """The nested run; returns its report steps by key ({} when it wrote no report)."""
    croot = confirm_root(root)
    report = croot / "report.json"
    report.unlink(missing_ok = True)  # never merge an older confirm's steps
    (Path(root) / "logs").mkdir(parents = True, exist_ok = True)
    logp = Path(root) / "logs" / "confirm.log"
    log(f"confirm: rerunning {names} on both arms -> {croot} ({logp})")
    with open(logp, "w") as fh:
        rc = runner(
            argv(a, root, names),
            stdout = fh,
            stderr = subprocess.STDOUT,
            env = {**os.environ, "STUDIO_REGRESS_CONFIRM": "1"},
        ).returncode
    try:
        steps = json.loads(report.read_text()).get("steps") or []
    except (OSError, ValueError):
        log(f"confirm: no report (exit {rc})")
        return {}
    return {s["key"]: s for s in steps}


def _reproduced(first, again, cv):
    """FAIL_HEAD twice is a reproduction only when a check that failed the first time fails again."""
    if cv != "FAIL_HEAD":
        return f"reproduced on confirm ({cv})"
    from studio_regress.external import failed_checks

    a, b = failed_checks(first), failed_checks(again)
    if not a or not b:
        return f"reproduced on confirm ({cv})"
    both = [k for k in a if k in b]
    if both:
        return f"reproduced on confirm ({cv}: {', '.join(both[:5])} failed again)"
    first["confirm_checks"] = {"first": a, "confirm": b}
    return (
        f"not reproduced: confirm {cv} on other checks ({', '.join(b[:5])}); "
        f"{', '.join(a[:5])} did not fail on confirm, so each failed 1 of 2"
    )


def merge(
    steps,
    confirmed,
    croot = None,
):
    """Apply the confirm run's verdicts to the candidate steps in place; returns the changes."""
    changes = []
    for s in candidates(steps):
        c = confirmed.get(s["key"])
        first = {
            k: s.get(k)
            for k in ("verdict", "note", "status_before", "status_after", "pixels_changed")
        }
        s["first_attempt"] = first
        s["confirm"] = {**c, "root": str(croot)} if c else {"verdict": None, "root": str(croot)}
        cv = c.get("verdict") if c else None
        was = first["verdict"]
        if was == "VOID":
            if cv in ("SAME", "FAIL_HEAD"):
                s.update(verdict = cv, note = f"{cv} on confirm; first attempt VOID ({first['note']})")
                s.pop("retryable", None)
            else:
                s["note"] = f"{first['note']} | confirm {cv or 'missing'}: unconfirmed"
        elif cv in REPRODUCED[was]:
            s["note"] = f"{first['note']} | {_reproduced(s, c, cv)}"
        elif cv in CLEARED[was]:
            s.update(
                verdict = "FLAKY",
                note = f"{was} did not reproduce on confirm ({cv}); first: {first['note']}",
            )
        elif cv in INTERMITTENT.get(was, ()):
            s["note"] = (
                f"{first['note']} | intermittent: confirm {cv}, head failed 1 of 2, base never"
            )
        else:
            s["note"] = f"{first['note']} | confirm {cv or 'missing'}: unconfirmed"
        if s["verdict"] != was:
            changes.append((s["key"], was, s["verdict"]))
    return changes


def confirm_steps(
    a,
    root,
    steps,
    meta,
    log = print,
    runner = subprocess.run,
):
    if not enabled(a):
        return []
    names = targets(steps)
    if not names:
        return []
    confirmed = run_confirm(a, root, names, log = log, runner = runner)
    changes = merge(steps, confirmed, confirm_root(root))
    meta["confirm"] = {
        "targets": names,
        "root": str(confirm_root(root)),
        "changes": [{"key": k, "from": f, "to": t} for k, f, t in changes],
    }
    for k, f, t in changes:
        log(f"confirm: {k} {f} -> {t}")
    return changes
