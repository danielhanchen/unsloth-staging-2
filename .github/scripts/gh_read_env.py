#!/usr/bin/env python3
"""
gh_read_env.py — rate-aware token pool for the summarisers' READ-ONLY gh calls.

GitHub rate limits are per USER, not per token, so every `gh pr view`,
`gh pr diff`, `gh pr list` and `gh issue list` a rebuild makes competes with
everything else the session does on one account's 5000/hr budget.

That budget is not big enough for a catch-up. Observed 2026-09-13: a full
rebuild over ~2500 PRs running alongside a few review agents exhausted it, the
PR build died with 1652 "Could not fetch metadata" failures at 857/2512, and
the issue build then enumerated 0 of 888 open issues and printed it as an
ordinary count line.

So reads draw from a pool, in this order:

    1. GH_TOKEN_shimmyshimmer   -- spend this one down to the floor first
    2. GH_TOKEN                 -- the session's own token, used only after

and BOTH stop being used once they fall to RESERVE_FRACTION (20%) of their
limit. That reserve is the point: a rebuild is a background chore and must
never be the reason an interactive `gh` call, a CI script or a review agent
gets a 403. Draining shimmyshimmer to its floor before touching GH_TOKEN keeps
the session's own headroom intact for as long as possible.

When every token is at its floor, callers BLOCK until the earliest reset
rather than hammering GitHub with calls that would fail anyway. GitHub's
/rate_limit endpoint does not itself count against the limit, so polling it is
free; results are cached briefly so that parallel workers share one snapshot
instead of each making their own probe.

WRITES DELIBERATELY STAY ON THE DEFAULT TOKEN. The lock ref, the branch push
and the PR creation all run as whoever GH_TOKEN is: attributing automated
commits and PRs to shimmyshimmer would misrepresent who made the change.
This module is for reads only.

Usage:
    from gh_read_env import gh_read_env
    subprocess.run(["gh", "pr", "view", ...], env=gh_read_env(), ...)
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import gh_budget  # noqa: E402  -- the host-wide, cross-user half of the budget

try:
    import gh_graphql as _gq  # noqa: E402  (REST reads served from GraphQL's separate points bucket)
except ImportError:  # pragma: no cover
    _gq = None

# Spend order. shimmyshimmer first so the session's own token keeps its
# headroom for interactive work.
TOKEN_ORDER = ("GH_TOKEN_shimmyshimmer", "GH_TOKEN")

# Stop drawing on a token once it drops to this fraction of its limit.
RESERVE_FRACTION = 0.20

# Resources a summariser actually touches: `gh pr diff` is REST (core),
# `gh pr list` / `gh pr view --json` are GraphQL. Gate on the scarcer of the
# two, since either one running dry breaks the rebuild.
_RESOURCES = ("core", "graphql")

# Parallel workers share one probe rather than each making their own.
_SNAPSHOT_TTL_S = 30.0

# Never block a single call longer than this; past it, hand back the best
# token available and let the caller's own error handling deal with it.
MAX_BLOCK_S = 3600.0

_lock = threading.Lock()
_snapshots: dict[str, tuple[float, dict]] = {}  # var -> (fetched_at, snapshot)
_last_choice: str | None = None


def _log(msg: str) -> None:
    print(f"[gh_read_env] {msg}", file = sys.stderr, flush = True)


def _fetch_rate(token: str, var: str | None = None) -> dict | None:
    """Return {resource: (remaining, limit, reset_epoch)} for a token.

    DO NOT reinstate `gh api rate_limit` here. It is not a working sensor for
    these credentials: measured 2026-09-15, it returned remaining=5000, used=0
    for all three tokens in the same second that a real request to a real
    resource reported remaining=4280, used=720 on the very same token. The
    endpoint answers about a bucket of its own -- even its own response headers
    say used=0 -- so it cannot be sanity-checked against itself either.

    Because it always read "full", every consumer of this function believed
    there was nothing to ration: RESERVE_FRACTION never engaged, the
    TOKEN_ORDER spend-down never switched tokens, and the block-until-reset
    path never once ran. The 2026-09-13 incident recorded below -- 2961
    rejections while /rate_limit reported 5000/5000 -- was read at the time as
    "secondary limits are not reported here", which is true but incomplete: the
    PRIMARY number was wrong too.

    The truth only appears in the x-ratelimit headers of real responses, so
    that is what gh_budget collects and shares across every process and unix
    user on the host. See gh_budget.calibrate().
    """
    # TTL-gated inside calibrate(): ~12 calls/hour, not one per invocation.
    gh_budget.calibrate(var, token)
    state = gh_budget.read_state()
    key = var or "_ambient"
    res = (state.get("resources") or {}).get(key) or {}
    if not res:
        return None
    spent = int((state.get("calls") or {}).get(key, 0))
    out = {}
    for name, triple in res.items():
        if name not in _RESOURCES and name != "search":
            continue
        try:
            remaining, limit, reset = int(triple[0]), int(triple[1]), int(triple[2])
        except (TypeError, ValueError, IndexError):
            continue
        if limit <= 0:
            continue
        # Subtract what we have spent since the reading was taken.
        out[name] = (max(0, remaining - spent), limit, reset)
    return out or None


def _snapshot(
    var: str,
    token: str,
    force: bool = False,
) -> dict | None:
    now = time.time()
    cached = _snapshots.get(var)
    if not force and cached and now - cached[0] < _SNAPSHOT_TTL_S:
        return cached[1]
    snap = _fetch_rate(token, var)
    if snap is not None:
        _snapshots[var] = (now, snap)
    return snap


def _headroom(snap: dict) -> float:
    """Fraction remaining on the scarcest tracked resource. 0.0 if unknown."""
    if not snap:
        return 0.0
    return min(rem / lim for (rem, lim, _) in snap.values())


def _earliest_reset(snap: dict) -> int:
    """Epoch seconds at which the scarcest resource refills."""
    if not snap:
        return int(time.time()) + 60
    scarcest = min(snap.values(), key = lambda v: v[0] / v[1])
    return scarcest[2]


def _available_tokens() -> list[tuple[str, str]]:
    return [(v, os.environ[v]) for v in TOKEN_ORDER if os.environ.get(v)]


def pick_read_token(block: bool = True) -> tuple[str | None, str | None]:
    """Return (env_var_name, token) for the next read call.

    Walks TOKEN_ORDER and returns the first token still above the reserve
    floor. If none is, blocks until the earliest reset (bounded by
    MAX_BLOCK_S) and tries again. Returns (None, None) when no token is
    configured at all, which means "use the ambient credentials".
    """
    global _last_choice
    candidates = _available_tokens()
    if not candidates:
        return None, None

    deadline = time.time() + MAX_BLOCK_S
    while True:
        with _lock:
            best: tuple[str, str] | None = None
            soonest_reset = None
            for var, token in candidates:
                if _in_cooldown(var) and len(candidates) > 1:
                    # Recently burst-limited: prefer a different user,
                    # whose secondary limit is tracked separately.
                    continue
                snap = _snapshot(var, token)
                if snap is None:
                    # Cannot see this token's budget. Try it rather than
                    # stall: a failed call is recoverable, a deadlock is not.
                    best = (var, token)
                    break
                head = _headroom(snap)
                if head > RESERVE_FRACTION:
                    best = (var, token)
                    break
                reset = _earliest_reset(snap)
                soonest_reset = reset if soonest_reset is None else min(soonest_reset, reset)

            if best is not None:
                if best[0] != _last_choice:
                    _log(f"read calls now using {best[0]}")
                    _last_choice = best[0]
                return best

        if not block or time.time() >= deadline:
            # Out of patience: hand back the first token and let the caller's
            # error path handle the 403 rather than blocking forever.
            _log(
                "all read tokens at their reserve floor and not blocking; "
                "returning first token, calls may fail"
            )
            return candidates[0]

        # Every token is at its floor. Sleep until the earliest refill, with
        # a little jitter so parallel workers do not all wake together and
        # stampede the moment the window rolls over.
        now = time.time()
        wait = max(5.0, (soonest_reset or now + 60) - now) + random.uniform(0, 5)
        wait = min(wait, max(5.0, deadline - now))
        _log(
            f"all read tokens at the {RESERVE_FRACTION:.0%} reserve floor; "
            f"sleeping {wait:.0f}s for the rate window to reset"
        )
        time.sleep(wait)
        with _lock:
            _snapshots.clear()


# ---------------------------------------------------------------------------
# Secondary rate limits
#
# GitHub enforces TWO limits. The primary one (5000/hr) is what /rate_limit
# reports and what the pool above rations. The secondary one -- concurrency
# and burst -- is NOT reported anywhere, and it is the one that actually
# breaks a rebuild.
#
# Observed 2026-09-13, the run that motivated this code: 2961 calls rejected
# with "API rate limit already exceeded for user ID 107991372" while
# /rate_limit simultaneously reported that same token at core 5000/5000 and
# graphql 5000/5000. The pool never failed over because, by the only number
# it could see, nothing was wrong. A single sequential call succeeded
# throughout. The trigger was 10 worker threads each firing 2-3 GraphQL
# calls at once.
#
# So rationing the primary budget is necessary but not sufficient. Three
# further things are needed, and all three live here so every read path gets
# them without having to remember:
#   1. a global cap on CONCURRENT gh reads, independent of how many worker
#      threads the builder runs (codex calls are the slow part and should
#      stay parallel; gh calls are what trips the burst limit),
#   2. retry with exponential backoff when a burst rejection does happen,
#   3. a short cooldown on the offending token so retries prefer the other
#      one -- a different user has a separate secondary limit.
# ---------------------------------------------------------------------------

# Concurrent `gh` reads allowed process-wide. Deliberately much lower than the
# builder's --parallel: 10 threads doing codex work is fine, 10 threads doing
# GraphQL at once is what got us rejected.
GH_READ_CONCURRENCY = int(os.environ.get("GH_READ_CONCURRENCY", "3"))
_gh_sem = threading.BoundedSemaphore(GH_READ_CONCURRENCY)

# Minimum spacing between consecutive read calls, to smooth bursts.
GH_READ_MIN_SPACING_S = float(os.environ.get("GH_READ_MIN_SPACING_S", "0.15"))
_spacing_lock = threading.Lock()
_last_call_at = 0.0

# A token that just hit a secondary limit is skipped for this long.
BURST_COOLDOWN_S = 90.0
_burst_cooldown: dict[str, float] = {}

_SECONDARY_MARKERS = (
    "rate limit already exceeded",
    "secondary rate limit",
    "abuse detection",
    "submitted too quickly",
    "exceeded a secondary",
    "retry your request again later",
)


def looks_rate_limited(text: str) -> bool:
    """True if stderr indicates a rate limit (primary or secondary)."""
    low = (text or "").lower()
    return any(m in low for m in _SECONDARY_MARKERS)


def _mark_burst_limited(var: str | None) -> None:
    if not var:
        return
    with _lock:
        _burst_cooldown[var] = time.time() + BURST_COOLDOWN_S
        _snapshots.pop(var, None)
    _log(
        f"{var} hit a secondary rate limit; cooling it down for "
        f"{BURST_COOLDOWN_S:.0f}s and preferring the other token"
    )


def _in_cooldown(var: str) -> bool:
    return _burst_cooldown.get(var, 0.0) > time.time()


def _throttle() -> None:
    """Space consecutive calls out a little."""
    global _last_call_at
    if GH_READ_MIN_SPACING_S <= 0:
        return
    with _spacing_lock:
        now = time.time()
        wait = (_last_call_at + GH_READ_MIN_SPACING_S) - now
        if wait > 0:
            time.sleep(wait)
            now = time.time()
        _last_call_at = now


def run_gh_read(
    cmd: list[str],
    *,
    timeout: int = 60,
    attempts: int = 5,
    label: str = "",
) -> subprocess.CompletedProcess:
    """Run a read-only `gh` command through the pool, throttled and retried.

    Returns the final CompletedProcess. Callers keep their existing
    returncode handling; this only ensures a burst rejection is retried
    rather than counted as a hard failure that burns the record.
    """
    # A REST path GraphQL can serve goes there first (its own points bucket, REST shape), before the
    # pool picks a token: that pick probes REST headroom, which this read then never spends.
    if _gq is not None and cmd[:2] == ["gh", "api"]:
        got = _gq.try_api(cmd, timeout = timeout)
        if got is not None:
            return got
    last: subprocess.CompletedProcess | None = None
    for attempt in range(1, attempts + 1):
        var, token = pick_read_token(block = True)
        env = dict(os.environ)
        if token:
            env["GH_TOKEN"] = token
            env.pop("GITHUB_TOKEN", None)

        _throttle()
        # _gh_sem caps THIS process; gh_budget.slot caps the host. Both are
        # kept: the semaphore is free and still bounds one interpreter's
        # threads, but it was never the binding constraint -- four unix users
        # at "concurrency 3" is concurrency 12 to GitHub, which is what
        # actually tripped the secondary limit.
        with _gh_sem, gh_budget.slot(var, label = label):
            try:
                last = subprocess.run(
                    cmd,
                    capture_output = True,
                    text = True,
                    timeout = timeout,
                    env = env,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                last = subprocess.CompletedProcess(cmd, 1, "", str(e))
            gh_budget.note_call(var)

        if last.returncode == 0:
            return last
        if not looks_rate_limited(last.stderr):
            return last  # a real error; let the caller handle it

        _mark_burst_limited(var)
        # Park every other process on the host too, rather than letting all 40
        # of them discover the same limit one at a time.
        gh_budget.note_secondary_limit(var)
        if attempt < attempts:
            delay = min(60.0, 2.0**attempt) + random.uniform(0, 3)
            _log(
                f"{label or cmd[:3]} rate-limited (attempt {attempt}/{attempts}); "
                f"retrying in {delay:.0f}s"
            )
            time.sleep(delay)
    return last  # type: ignore[return-value]


def gh_read_env(base: dict | None = None, block: bool = True) -> dict:
    """Env for a read-only `gh` invocation, using the rate-aware pool.

    Falls back to the ambient credentials when no pool token is configured, so
    nothing breaks on a machine that only has the one token.
    """
    env = dict(base if base is not None else os.environ)
    var, token = pick_read_token(block = block)
    if token:
        # gh resolves GH_TOKEN ahead of GITHUB_TOKEN; clear the latter so a
        # stale value cannot win on some other code path.
        env["GH_TOKEN"] = token
        env.pop("GITHUB_TOKEN", None)
    return env


def rate_report() -> str:
    """Human-readable budget across the pool, for logs and CLI."""
    lines = []
    for var, token in _available_tokens():
        snap = _snapshot(var, token, force = True)
        if snap is None:
            lines.append(f"  {var}: probe failed")
            continue
        parts = [
            f"{name} {rem}/{lim} ({rem / lim:.0%})" for name, (rem, lim, _) in sorted(snap.items())
        ]
        floor = "BELOW FLOOR" if _headroom(snap) <= RESERVE_FRACTION else "ok"
        lines.append(f"  {var}: {', '.join(parts)} -> {floor}")
    return "\n".join(lines) or "  (no pool tokens configured)"


if __name__ == "__main__":
    print(
        f"Read-token pool (order: {' -> '.join(TOKEN_ORDER)}, "
        f"reserve floor {RESERVE_FRACTION:.0%}):"
    )
    print(rate_report())
    var, _ = pick_read_token(block = False)
    print(f"\nNext read call would use: {var or '(ambient credentials)'}")
