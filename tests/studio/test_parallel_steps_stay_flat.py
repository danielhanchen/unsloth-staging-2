# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""
Concurrent steps are spelled flat: `background: true` plus `wait`, never `parallel:`.

GitHub Actions can run steps of one job concurrently (changelog 2026-06-25): a step
with `background: true` starts and the job moves on, a later `- wait: <id>` or
`- wait-all:` step joins it, `- cancel: <id>` stops it, and `parallel:` is sugar for
"these steps in the background, then a wait".

`parallel:` is the one spelling this repository cannot afford. Its members sit one
level down, under a key that no guard here reads: scripts/lint_workflow_triggers.py
and some forty tests walk `job["steps"]` and look at each step's `run`, `uses`,
`with` and `env`. A step moved into a `parallel:` group drops out of every one of
them without turning anything red. Measured: an unbounded `apt-get` and a
`playwright install --with-deps` inside a group pass test_apt_steps_are_bounded.py and
test_playwright_install_avoids_with_deps.py, and the same two steps written flat fail
both. The flat spelling runs the same steps the same way and leaves each one where
those guards look.

The other rules keep the flat spelling honest:

  * a `wait` or `cancel` names a `background: true` step earlier in the same job, so a
    typo cannot leave a step that nothing joins;
  * every background step is joined by a later `wait` naming it, a `wait-all`, or a
    `cancel`, so the point where its failure surfaces is written in the file rather
    than left to the end of the job;
  * no more than 10 background steps are outstanding at once, GitHub's limit.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
ACTIONS = REPO_ROOT / ".github" / "actions"

# GitHub refuses an eleventh concurrent background step in one job.
MAX_BACKGROUND = 10


def _documents() -> list[tuple[str, dict]]:
    paths = sorted(WORKFLOWS.glob("*.yml")) + sorted(ACTIONS.glob("*/action.y*ml"))
    assert paths, "no workflows found; this guard would pass vacuously"
    return [
        (str(path.relative_to(REPO_ROOT)), yaml.safe_load(path.read_text(encoding = "utf-8")))
        for path in paths
    ]


def _step_lists(doc: dict):
    """(scope, steps) for every job, plus a composite action's `runs.steps`."""
    if not isinstance(doc, dict):
        return
    runs = doc.get("runs")
    if isinstance(runs, dict) and isinstance(runs.get("steps"), list):
        yield "runs", runs["steps"]
    for job_id, job in (doc.get("jobs") or {}).items():
        if isinstance(job, dict) and isinstance(job.get("steps"), list):
            yield job_id, job["steps"]


def _ids(value) -> list[str]:
    """The step ids a `wait:` / `cancel:` value names; `${{ }}` is opaque, so skipped."""
    names = value if isinstance(value, list) else [value]
    return [n for n in names if isinstance(n, str) and "${{" not in n]


def _problems(name: str, doc: dict) -> list[str]:
    out = []
    for scope, steps in _step_lists(doc):
        where = f"{name} {scope}"
        pending = {}  # id (or "#index" when unnamed) -> step label, in start order
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            label = step.get("name") or step.get("id") or step.get("uses") or f"step {index}"
            if "parallel" in step:
                out.append(
                    f"{where}: `parallel:` group at step {index}; its members are invisible to "
                    f"every guard that walks job steps. Spell each member as its own step with "
                    f"`background: true` and join them with `- wait-all:`"
                )
                continue
            if step.get("background") is True:
                pending[step.get("id") or f"#{index}"] = label
                if len(pending) > MAX_BACKGROUND:
                    out.append(
                        f"{where}: {len(pending)} background steps outstanding at {label!r}; "
                        f"GitHub allows {MAX_BACKGROUND}"
                    )
                continue
            if "wait-all" in step and step["wait-all"] is not False:
                pending.clear()
                continue
            for key in ("wait", "cancel"):
                if key not in step:
                    continue
                for target in _ids(step[key]):
                    if target in pending:
                        del pending[target]
                    else:
                        out.append(
                            f"{where}: `{key}: {target}` names no background step started "
                            f"earlier in this job"
                        )
        for label in pending.values():
            out.append(f"{where}: background step {label!r} is never joined by a wait or cancel")
    return out


def test_every_concurrent_step_stays_where_the_guards_look():
    problems = [p for name, doc in _documents() for p in _problems(name, doc)]
    assert not problems, "\n".join(problems)


def _check(text: str) -> list[str]:
    return _problems("fixture.yml", yaml.safe_load(text))


FLAT = """
jobs:
  j:
    steps:
      - uses: actions/checkout@0000000000000000000000000000000000000000
      - id: a
        background: true
        run: sudo apt-get install -y foo
      - id: b
        background: true
        run: pytest tests/b
      - wait: a
      - name: needs a
        run: echo a
      - id: server
        background: true
        run: python -m http.server
      - wait-all:
      - id: svc
        background: true
        run: sleep 600
      - cancel: svc
"""


def test_the_flat_spelling_passes():
    assert _check(FLAT) == []


def test_a_parallel_group_is_refused():
    """The members below are exactly what the step-walking guards exist to inspect."""
    problems = _check(
        """
jobs:
  j:
    steps:
      - parallel:
          - run: sudo apt-get install -y foo
          - uses: actions/checkout@v4
"""
    )
    assert len(problems) == 1 and "`parallel:` group" in problems[0], problems


def test_a_parallel_group_in_a_composite_action_is_refused():
    problems = _check(
        """
runs:
  using: composite
  steps:
    - parallel:
        - run: echo a
          shell: bash
"""
    )
    assert len(problems) == 1 and "`parallel:` group" in problems[0], problems


@pytest.mark.parametrize(
    "steps, expected",
    [
        ("- id: a\n        background: true\n        run: x\n", "never joined"),
        ("- run: x\n        background: true\n", "never joined"),
        ("- wait: a\n", "names no background step"),
        ("- cancel: a\n", "names no background step"),
        # A wait before the step it names joins nothing.
        (
            "- wait: a\n      - id: a\n        background: true\n        run: x\n      - wait-all:\n",
            "names no background step",
        ),
        (
            "- id: a\n        background: true\n        run: x\n      - wait-all: false\n",
            "never joined",
        ),
    ],
)
def test_a_background_step_must_be_joined_by_name(steps, expected):
    problems = _check(f"jobs:\n  j:\n    steps:\n      {steps}")
    assert any(expected in p for p in problems), problems


def test_a_wait_may_name_several_steps():
    problems = _check(
        """
jobs:
  j:
    steps:
      - {id: a, background: true, run: x}
      - {id: b, background: true, run: y}
      - wait: [a, b]
"""
    )
    assert problems == []


def test_more_than_ten_outstanding_background_steps_is_refused():
    steps = "".join(
        f"      - {{id: s{i}, background: true, run: x}}\n" for i in range(MAX_BACKGROUND + 1)
    )
    problems = _check(f"jobs:\n  j:\n    steps:\n{steps}      - wait-all:\n")
    assert any("GitHub allows" in p for p in problems), problems
    steps = "".join(
        f"      - {{id: s{i}, background: true, run: x}}\n" for i in range(MAX_BACKGROUND)
    )
    assert _check(f"jobs:\n  j:\n    steps:\n{steps}      - wait-all:\n") == []
