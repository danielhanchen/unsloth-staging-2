"""Shared contract for studio_regress: journeys, steps, evidence and the report schema.

Every journey module in journeys/ exposes

    JOURNEY = Journey(name="auth", tier="fast", steps=[Step(...), ...], needs=("gguf_270m",))

and the runner calls each step's `action(ctx)` in order on ONE side (before or after), then
captures evidence. Both sides run the identical, frozen step list; pairs are matched by
`<journey>/<step.id>`, never by position.

Step.action(ctx) -> dict | None   facts (JSON-safe) for this step; raise StepFailed on a failed
                                   positive assertion, StepUnreachable when the control / route
                                   does not exist on this side (reported DIVERGED, never skipped).
ctx: Ctx                           page (Playwright async Page), base_url, api (httpx.AsyncClient
                                   with auth), side, out_dir, models (resolved fixture paths),
                                   state (dict shared across steps of one journey on one side).

Evidence written per step: <out>/<side>/<journey>/<step>.png (lossless, only if shot=True),
<step>.dom.json (normalised a11y tree + visible text), <step>.facts.json.

report.json (written by run.py, consumed by diff.py / publish.py):
    {"pr", "repo", "base_sha", "head_sha", "merge_base", "suite_version", "tier", "gate",
     "coverage": {"overall", "per_route": {route: pct}, "inventory": n},
     "steps": [{"key": "auth/login_ok", "journey", "step", "status_before", "status_after",
                "verdict": SAME|VISUAL_DIFF|DOM_ONLY_DIFF|DIVERGED|FLAKY|FAIL_HEAD|FAIL_BOTH|VOID|PLAN_DIFF,
                "pixels_changed", "dom_delta", "facts_delta", "png_before", "png_after",
                "note"}],
     "functional": "NO_REGRESSION|REGRESSION|VOID", "timings": {journey: {side: seconds}}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

TIERS = ("fast", "model", "gpu")
# VOID: a Core target (core.py) that could not compare the two sides (setup failure, compare.py
# VOID / NOT_RUN, regression suite infra or invalid run); never counted as clean.
# SKIPPED_NOT_TOUCHED: the step's suite was deliberately not run because test_select.py proved the
# diff cannot reach it. Neither a pass nor a failure; a run of only these is VOID.
# PLAN_DIFF: a `compare` external target (planner_matrix) whose two sides decide differently; listed
# with its moved cells for a reader, never a regression.
VERDICTS = (
    "SAME",
    "VISUAL_DIFF",
    "DOM_ONLY_DIFF",
    "DIVERGED",
    "FLAKY",
    "FAIL_HEAD",
    "FAIL_BOTH",
    "VOID",
    "SKIPPED_NOT_TOUCHED",
    "PLAN_DIFF",
)
SKIPPED_NOT_TOUCHED = "skipped_not_touched"  # the step's facts `_status`
SUITE_VERSION = 1


class StepFailed(AssertionError):
    """Positive assertion for the step's intended consequence failed."""


class StepUnreachable(RuntimeError):
    """Control / route absent on this side: DIVERGED, not skipped."""


class StepNotTouched(Exception):
    """The diff provably cannot affect what this step tests, so it was not run: recorded as
    `skipped_not_touched` (verdict SKIPPED_NOT_TOUCHED), never as a pass."""


@dataclass
class Ctx:
    page: Any
    base_url: str
    api: Any
    side: str
    out_dir: str
    models: dict = field(default_factory = dict)
    state: dict = field(default_factory = dict)


@dataclass(frozen = True)
class Step:
    id: str
    action: Callable[[Ctx], Awaitable[dict | None]]
    shot: bool = True
    masks: tuple = ()  # CSS selectors masked in the screenshot (volatile, never the subject)
    mask_reason: str = ""
    full_page: bool = False
    timeout_s: float = 60.0


@dataclass(frozen = True)
class Journey:
    name: str
    tier: str
    steps: tuple
    needs: tuple = ()  # fixture keys from switchboard [models]
    routes: tuple = ()  # UI routes this journey covers (coverage accounting)
    serial: bool = False  # must not share a Studio with other journeys (e.g. auth rotates pw)
    # Additive (fork A): run every step even after one fails (independent suites), and an
    # async teardown(ctx) always awaited after the steps (kill helper processes, etc.).
    independent: bool = False
    teardown: Optional[Callable[["Ctx"], Awaitable[None]]] = None

    def keys(self):
        return [f"{self.name}/{s.id}" for s in self.steps]
