#!/usr/bin/env python3
"""
gh_budget.py -- one host-wide, cross-user admission controller for `gh` reads.

WHY THIS EXISTS, AND WHY IT IS NOT gh_read_env.py
-------------------------------------------------
gh_read_env.py rations the PRIMARY budget per token. Two measured facts made
that insufficient:

1. Its sensor is dead. `gh api rate_limit` returns remaining=5000, used=0 for
   every token we hold, while the x-ratelimit headers on a REAL request to the
   same resource report the truth. Observed 2026-09-15, same second, same
   token:

       gh api rate_limit          -> remaining=5000  used=0
       gh api -i repos/<o>/<r>    -> remaining=4280  used=720

   So RESERVE_FRACTION, the TOKEN_ORDER spend-down and block-until-reset never
   engaged: by the only number the pool could see, nothing was ever wrong.
   Note the /rate_limit endpoint reports a bucket of its own -- its own
   response headers also say used=0 -- so it cannot be calibrated against
   itself. Only a request to a real resource tells the truth.

2. Its throttle is per-PROCESS. `_gh_sem` and `_throttle()` are module globals,
   so they cap one Python process. The deployment is ~40 concurrent launches
   spread over four unix users (daniel1, daniel2, daniel3, ubuntu), and
   GitHub's limits are per USER, not per token or per process. Four users at
   "concurrency 3" is concurrency 12 as far as GitHub is concerned.

This module holds the part that must be shared by every process on the host: a
cap on concurrent requests, a host-wide rate floor, a usage ledger fed from real
response headers, and a cooldown that one process can trip on behalf of all of
them. It costs the sessions nothing real -- they still do all their local and
model work concurrently, and only the GitHub calls queue. Measured with
scripts/bench_gh_budget.py at the defaults below: 40 sessions making 8 requests
each finish in ~54s wall, the unluckiest waiting ~54s, at ~5.9 req/s. Confirmed
against live GitHub with 320 genuine requests: zero secondary-limit rejections.

CROSS-USER FILESYSTEM CONSTRAINTS (verified on this host, do not "simplify")
---------------------------------------------------------------------------
* /mnt/disks/unslothai/daniel{1,2,3} and .../ubuntu are 0700. The four users
  genuinely cannot read each other's trees, which is why none of the existing
  state (~/.claude/summary-db, the account pools) can be reused here. The
  shared directory has to live outside all of them.
* It must be setgid with a group all four users share (2775, group `users`),
  NOT sticky. In a sticky (1777) directory you cannot rename over or unlink a
  file owned by another user, which breaks any replace-based write.
* setgid propagates the PARENT's group. If the parent is owned by one user's
  private group, the auto-created child inherits it and silently locks out
  everyone else. state_dir() warns when it ends up on the fallback path for
  exactly this reason.
* The ledger is ONE long-lived file mutated in place under flock. Do not
  os.replace() it and do not delete it to "reset": the atomic_write pattern in
  account_pool.py:204 is correct for a private dir and wrong here.
* Files are created 0666 via os.open(..., 0o666) + fchmod, because the caller's
  umask would otherwise strip group/other write and lock out the next user.
  This is also what lets a file created before the group was fixed keep
  working.

CLI:
    python3 gh_budget.py status        # show the shared ledger
    python3 gh_budget.py calibrate     # force a real header reading (1 call)
    python3 gh_budget.py reset-cooldown
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import random
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

# --------------------------------------------------------------------------
# Location of the shared state
# --------------------------------------------------------------------------

# The 1777 parent is the only place all four users can write. Overridable so
# tests, sandboxes and non-shared hosts can point somewhere private.
DEFAULT_STATE_DIR = "/mnt/disks/unslothai/shared/gh-budget"
_FALLBACK_STATE_DIR = "/tmp/unsloth-gh-budget"

# The two knobs GitHub's secondary limits actually care about, and they are
# independent:
#
#   MIN_SPACING_S bounds the host-wide REQUEST RATE. Spacing is enforced
#   between call starts across every process, so the global ceiling is
#   1/MIN_SPACING_S regardless of how many slots exist. At 0.2s that is 5/s =
#   300/min, against a documented REST limit of 900 points/min.
#
#   MAX_INFLIGHT bounds CONCURRENCY, which is the separate limit (100
#   concurrent). 3 is far under it, and nothing like the shape that caused the
#   trouble: that was 10 worker threads per process times four unix users, with
#   no global cap and no spacing at all.
#
# These were 1 and 0.35s in the first version, which measured at 1.5 req/s --
# ten times more conservative than the documented ceiling, and slow enough to
# matter: 40 sessions doing 8 calls each took 208s wall, with the unluckiest
# session waiting the whole 208s before its GitHub work finished. Serialising
# was the right instinct and the wrong magnitude. Benchmark before changing
# these again: scripts/bench_gh_budget.py.
MAX_INFLIGHT = int(os.environ.get("GH_BUDGET_INFLIGHT", "3"))
MIN_SPACING_S = float(os.environ.get("GH_BUDGET_SPACING_S", "0.2"))

# Stop handing out slots once the scarcest resource drops to this fraction, so
# background chores can never starve an interactive call.
RESERVE_FRACTION = float(os.environ.get("GH_BUDGET_RESERVE", "0.20"))

# After a secondary-limit rejection, every process on the host waits this long.
COOLDOWN_S = float(os.environ.get("GH_BUDGET_COOLDOWN_S", "120"))

# How often to spend one real request learning the true remaining count.
CALIBRATE_INTERVAL_S = float(os.environ.get("GH_BUDGET_CALIBRATE_S", "300"))

# Cheap public endpoint used only for calibration. Any real resource works;
# /rate_limit does NOT (see the module docstring).
CALIBRATE_PATH = os.environ.get("GH_BUDGET_CALIBRATE_PATH", "repos/unslothai/unsloth")

# Never block a single acquisition longer than this.
MAX_BLOCK_S = float(os.environ.get("GH_BUDGET_MAX_BLOCK_S", "900"))

LEDGER_SCHEMA_VERSION = 1

_SECONDARY_MARKERS = (
    "rate limit already exceeded",
    "secondary rate limit",
    "abuse detection",
    "submitted too quickly",
    "exceeded a secondary",
    "retry your request again later",
)


def _log(msg: str) -> None:
    print(f"[gh_budget] {msg}", file = sys.stderr, flush = True)


def looks_rate_limited(text: str) -> bool:
    """True if stderr indicates a rate limit (primary or secondary).

    Kept identical in behaviour to gh_read_env.looks_rate_limited so the two
    modules cannot disagree about what a rejection looks like.
    """
    low = (text or "").lower()
    return any(m in low for m in _SECONDARY_MARKERS)


_warned_fallback = False


def state_dir() -> Path:
    """The shared directory, created 2775 if missing.

    Falls back to a private directory when the shared path is not usable, so a
    laptop or CI box still works -- but SAYS SO, loudly, once per process.

    The warning is the important part. Silent per-user state is the exact bug
    this module exists to fix: the old refresh job's 24h TTL marker lived in
    $HOME, so four unix users enforced "once a day" four times over and nobody
    could see that happening. A budget that quietly degrades to per-user
    coordination fails the same way, and would look identical from inside any
    one process -- every one of them convinced it was the only caller.

    Note the parent may be owned by whoever provisioned it, so a user who is
    not the owner cannot create the directory even when they could write inside
    it. That is why this warns rather than shrugging.
    """
    global _warned_fallback
    override = os.environ.get("GH_BUDGET_DIR")
    default = DEFAULT_STATE_DIR
    try:
        from local_identity import shared_dir

        # A host without the shared mount: a per-user dir is the design there, not a fallback.
        default = str(shared_dir("gh-budget", DEFAULT_STATE_DIR))
    except ImportError:
        pass
    candidates = ([override] if override else []) + [default, _FALLBACK_STATE_DIR]
    for candidate in candidates:
        p = Path(candidate)
        try:
            p.mkdir(parents = True, exist_ok = True)
            # setgid so files land in the shared group; NOT sticky, so a
            # ledger can be replaced by whoever needs to. Best effort: only the
            # owner may chmod, and whoever gets there first opens it up.
            try:
                os.chmod(p, 0o2775)
            except OSError:
                pass
            if os.access(p, os.W_OK | os.X_OK):
                if candidate == _FALLBACK_STATE_DIR and not _warned_fallback:
                    _warned_fallback = True
                    _log(
                        f"WARNING: falling back to {p} -- the shared budget at "
                        f"{override or DEFAULT_STATE_DIR} is not usable. Rate "
                        f"limiting is now coordinated PER UNIX USER, not per "
                        f"host, so concurrent users will not see each other's "
                        f"traffic or cooldowns. Fix with:\n"
                        f"    install -d -m 2775 -g users {DEFAULT_STATE_DIR}"
                    )
                return p
        except OSError:
            continue
    raise RuntimeError("no writable directory for the shared gh budget")


def _shared_open(path: Path) -> int:
    """Open (creating if needed) a file every user on the host can lock.

    0o666 is passed to os.open AND re-applied with fchmod: the open-time mode
    is masked by the caller's umask (often 0022 or 0077), which would leave the
    file unwritable by the next user and wedge the whole host.
    """
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        os.fchmod(fd, 0o666)
    except OSError:
        # Not ours to chmod; it already exists with someone else's ownership.
        # If the mode is right this still works, and if it is not, the caller
        # gets a clear PermissionError rather than silent serialisation loss.
        pass
    return fd


@contextmanager
def _locked(name: str, exclusive: bool = True):
    """flock a stable file in the shared dir. The inode is never replaced."""
    path = state_dir() / name
    fd = _shared_open(path)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------


def _read_ledger(fd: int) -> dict:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 1 << 20).decode("utf-8", "replace").strip()
    except OSError:
        return {}
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        # A torn or hand-edited ledger must not wedge the host. Start over;
        # the next calibration refills it.
        return {}
    return data if isinstance(data, dict) else {}


def _write_ledger(fd: int, data: dict) -> None:
    """Rewrite the ledger IN PLACE. Never os.replace -- see module docstring."""
    payload = json.dumps(data, indent = 2, sort_keys = True).encode()
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, payload)
    os.ftruncate(fd, len(payload))


def _blank_ledger() -> dict:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "last_call_at": 0.0,
        "cooldown_until": 0.0,
        "calls": {},  # token_var -> count since the last calibration
        "resources": {},  # token_var -> {resource: [remaining, limit, reset]}
        "calibrated_at": {},  # token_var -> epoch
    }


def _ensure(data: dict) -> dict:
    base = _blank_ledger()
    if data.get("schema_version") != LEDGER_SCHEMA_VERSION:
        return base
    base.update(data)
    return base


def read_state() -> dict:
    # Shared lock: many readers, and _mutate's exclusive lock excludes them all
    # for the brief window it rewrites the file. Locking the ledger file itself
    # (rather than a sidecar) is what makes the read/modify/write in _mutate
    # atomic against other hosts' users.
    with _locked("ledger.json", exclusive = False) as fd:
        return _ensure(_read_ledger(fd))


def _mutate(fn) -> dict:
    """Apply fn(ledger) -> ledger under the exclusive ledger lock."""
    with _locked("ledger.json") as fd:
        data = _ensure(_read_ledger(fd))
        data = fn(data)
        _write_ledger(fd, data)
        return data


# --------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------

_HEADER_RE = re.compile(
    r"^x-ratelimit-(remaining|limit|reset|used|resource):\s*(\S+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_rate_headers(text: str) -> dict | None:
    """Pull the x-ratelimit block out of `gh api -i` output.

    Returns {"resource": str, "remaining": int, "limit": int, "reset": int}
    or None when the headers are absent (e.g. the call failed before a
    response).
    """
    found = {m.group(1).lower(): m.group(2) for m in _HEADER_RE.finditer(text)}
    if "remaining" not in found or "limit" not in found:
        return None
    try:
        return {
            "resource": found.get("resource", "core"),
            "remaining": int(found["remaining"]),
            "limit": int(found["limit"]),
            "reset": int(found.get("reset", 0)),
        }
    except ValueError:
        return None


def record_headers(token_var: str | None, parsed: dict) -> None:
    """Fold a real header reading into the shared ledger."""
    key = token_var or "_ambient"
    now = time.time()

    def _apply(d: dict) -> dict:
        res = d.setdefault("resources", {}).setdefault(key, {})
        res[parsed["resource"]] = [parsed["remaining"], parsed["limit"], parsed["reset"]]
        # A fresh reading supersedes the estimate we were carrying.
        d.setdefault("calls", {})[key] = 0
        d.setdefault("calibrated_at", {})[key] = now
        return d

    _mutate(_apply)


def note_call(token_var: str | None, n: int = 1) -> None:
    """Count a request we made but could not read headers for.

    Between calibrations this is what keeps the estimate honest: subtract the
    locally counted calls from the last true remaining.

    Deliberately does NOT touch last_call_at. That field is a RESERVATION owned
    by _claim_departure(), and stamping it with time.time() here would move it
    backwards past a departure another process has already booked, letting two
    calls leave inside one spacing interval.
    """
    key = token_var or "_ambient"

    def _apply(d: dict) -> dict:
        d.setdefault("calls", {})[key] = int(d.get("calls", {}).get(key, 0)) + n
        return d

    _mutate(_apply)


def _claim_departure() -> float:
    """Book the next departure slot; returns the epoch to leave at.

    The read-modify-write happens under the ledger's exclusive lock, so N
    concurrent callers get N distinct times MIN_SPACING_S apart even though
    they then wait in parallel. Without this the floor is unenforceable for
    MAX_INFLIGHT > 1.
    """
    with _locked("ledger.json") as fd:
        data = _ensure(_read_ledger(fd))
        depart = max(time.time(), float(data.get("last_call_at") or 0.0) + MIN_SPACING_S)
        data["last_call_at"] = depart
        _write_ledger(fd, data)
        return depart


def note_secondary_limit(token_var: str | None = None) -> float:
    """Park every process on the host. Returns the epoch it expires."""
    until = time.time() + COOLDOWN_S

    def _apply(d: dict) -> dict:
        # Extend, never shorten: a second rejection during a cooldown means
        # the first backoff was not enough.
        d["cooldown_until"] = max(float(d.get("cooldown_until") or 0.0), until)
        return d

    d = _mutate(_apply)
    _log(
        f"secondary rate limit hit by {token_var or 'ambient'}; "
        f"host-wide cooldown for {COOLDOWN_S:.0f}s"
    )
    return float(d["cooldown_until"])


def headroom(token_var: str | None = None, state: dict | None = None) -> float:
    """Fraction of the scarcest resource still available, 1.0 if unknown.

    Unknown means "we have never taken a reading", which is different from
    "empty". Returning 1.0 there is deliberate: a brand-new ledger must not
    deadlock every process on the host. The first calibration corrects it.
    """
    key = token_var or "_ambient"
    d = state if state is not None else read_state()
    res = (d.get("resources") or {}).get(key) or {}
    if not res:
        return 1.0
    spent = int((d.get("calls") or {}).get(key, 0))
    now = time.time()
    best = 1.0
    for _name, triple in res.items():
        try:
            remaining, limit, reset = int(triple[0]), int(triple[1]), int(triple[2])
        except (TypeError, ValueError, IndexError):
            continue
        if limit <= 0:
            continue
        if reset and now >= reset:
            # The window rolled over; the old reading says nothing.
            return 1.0
        best = min(best, max(0, remaining - spent) / limit)
    return best


def earliest_reset(token_var: str | None = None, state: dict | None = None) -> int:
    key = token_var or "_ambient"
    d = state if state is not None else read_state()
    res = (d.get("resources") or {}).get(key) or {}
    resets = [int(t[2]) for t in res.values() if len(t) > 2 and int(t[2] or 0)]
    return min(resets) if resets else int(time.time()) + 60


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def calibrate(
    token_var: str | None = None,
    token: str | None = None,
    force: bool = False,
) -> dict | None:
    """Spend one real request to learn the true remaining count.

    ~12 calls/hour at the default interval, against a 5000/hour budget. That is
    the price of knowing; the alternative is the dead sensor this module exists
    to replace.
    """
    key = token_var or "_ambient"
    d = read_state()
    last = float((d.get("calibrated_at") or {}).get(key, 0))
    if not force and (time.time() - last) < CALIBRATE_INTERVAL_S:
        return None

    env = dict(os.environ)
    if token:
        env["GH_TOKEN"] = token
        env.pop("GITHUB_TOKEN", None)
    try:
        r = subprocess.run(
            ["gh", "api", "-i", CALIBRATE_PATH],
            capture_output = True,
            text = True,
            timeout = 60,
            env = env,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        _log(f"calibration failed: {e}")
        return None
    if looks_rate_limited(r.stderr):
        note_secondary_limit(token_var)
        return None
    parsed = parse_rate_headers(r.stdout or "")
    if parsed is None:
        return None
    record_headers(token_var, parsed)
    return parsed


# --------------------------------------------------------------------------
# Admission control
# --------------------------------------------------------------------------


@contextmanager
def slot(
    token_var: str | None = None,
    block: bool = True,
    label: str = "",
) -> "object":
    """Hold the host-wide in-flight slot for the duration of one gh call.

    Order of business, all of it shared across users:
      1. wait out any cooldown another process may have tripped,
      2. refuse to proceed below the reserve floor (block until the window
         rolls over),
      3. take the slot,
      4. enforce the spacing floor since the last call by ANY process.
    """
    deadline = time.time() + MAX_BLOCK_S

    while True:
        st = read_state()
        cooldown = float(st.get("cooldown_until") or 0.0)
        now = time.time()
        if cooldown > now:
            wait = min(cooldown - now, max(1.0, deadline - now))
            if not block or now >= deadline:
                break
            _log(
                f"host-wide cooldown active; sleeping {wait:.0f}s"
                + (f" ({label})" if label else "")
            )
            time.sleep(wait + random.uniform(0, 1.0))
            continue

        if headroom(token_var, st) > RESERVE_FRACTION:
            break

        reset = earliest_reset(token_var, st)
        now = time.time()
        if not block or now >= deadline:
            _log("below the reserve floor and not blocking; proceeding anyway")
            break
        wait = min(max(5.0, reset - now), max(1.0, deadline - now))
        _log(
            f"below the {RESERVE_FRACTION:.0%} reserve floor; "
            f"sleeping {wait:.0f}s for the window to reset"
        )
        time.sleep(wait + random.uniform(0, 5.0))

    # MAX_INFLIGHT distinct lock files; try each until one is free. With the
    # default of 1 this is a plain mutex.
    held = None
    try:
        while held is None:
            for i in range(MAX_INFLIGHT):
                path = state_dir() / f"slot-{i}.lock"
                fd = _shared_open(path)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as e:
                    os.close(fd)
                    if e.errno not in (errno.EAGAIN, errno.EACCES):
                        raise
                    continue
                held = fd
                break
            if held is None:
                if time.time() >= deadline:
                    # Never deadlock a review because a slot leaked.
                    _log("timed out waiting for an in-flight slot; proceeding")
                    break
                time.sleep(0.05 + random.uniform(0, 0.05))

        # Spacing floor, host-wide.
        #
        # This RESERVES a departure time atomically instead of reading
        # last_call_at and sleeping. With MAX_INFLIGHT > 1 the read-then-sleep
        # version is a race: three holders read the same last_call_at, compute
        # the same delay, sleep in parallel and depart together -- which is a
        # burst, precisely what the floor exists to prevent. It measured as 30
        # violations the moment in-flight went from 1 to 3.
        #
        # Claiming the slot under the ledger's exclusive lock serialises the
        # decision even though the waiting itself happens in parallel.
        depart = _claim_departure()
        delay = depart - time.time()
        if delay > 0:
            time.sleep(delay)

        yield
    finally:
        if held is not None:
            try:
                fcntl.flock(held, fcntl.LOCK_UN)
            finally:
                os.close(held)


def run_gh(
    cmd: list[str],
    *,
    token_var: str | None = None,
    token: str | None = None,
    timeout: int = 60,
    attempts: int = 4,
    label: str = "",
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run one read-only `gh` command under the host-wide budget.

    Retries secondary-limit rejections with backoff, tripping the shared
    cooldown each time so the other 39 processes back off too rather than
    each discovering the limit for themselves.
    """
    base_env = dict(env or os.environ)
    if token:
        base_env["GH_TOKEN"] = token
        base_env.pop("GITHUB_TOKEN", None)

    last: subprocess.CompletedProcess | None = None
    for attempt in range(1, attempts + 1):
        calibrate(token_var, token)
        with slot(token_var, label = label):
            try:
                last = subprocess.run(
                    cmd,
                    capture_output = True,
                    text = True,
                    timeout = timeout,
                    env = base_env,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                last = subprocess.CompletedProcess(cmd, 1, "", str(e))
            note_call(token_var)

        if last.returncode == 0:
            return last
        if not looks_rate_limited(last.stderr):
            return last

        note_secondary_limit(token_var)
        if attempt < attempts:
            delay = min(60.0, 2.0**attempt) + random.uniform(0, 3)
            _log(
                f"{label or ' '.join(cmd[:3])} rate-limited "
                f"(attempt {attempt}/{attempts}); retrying in {delay:.0f}s"
            )
            time.sleep(delay)
    return last  # type: ignore[return-value]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _cmd_status(_args: argparse.Namespace) -> int:
    d = read_state()
    print(f"state dir: {state_dir()}")
    print(
        f"inflight slots: {MAX_INFLIGHT}   spacing: {MIN_SPACING_S}s   "
        f"reserve: {RESERVE_FRACTION:.0%}"
    )
    cooldown = float(d.get("cooldown_until") or 0.0)
    if cooldown > time.time():
        print(f"COOLDOWN active for another {cooldown - time.time():.0f}s")
    else:
        print("cooldown: clear")
    for key, res in sorted((d.get("resources") or {}).items()):
        spent = (d.get("calls") or {}).get(key, 0)
        age = time.time() - float((d.get("calibrated_at") or {}).get(key, 0))
        print(
            f"  {key}: headroom={headroom(key, d):.0%} "
            f"uncounted_calls={spent} calibrated {age:.0f}s ago"
        )
        for name, triple in sorted(res.items()):
            print(f"      {name}: remaining={triple[0]}/{triple[1]}")
    if not d.get("resources"):
        print("  (no readings yet -- run `gh_budget.py calibrate`)")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    var = args.token_var
    tok = os.environ.get(var) if var else None
    parsed = calibrate(var, tok, force = True)
    if parsed is None:
        print("calibration produced no reading", file = sys.stderr)
        return 1
    print(json.dumps(parsed, indent = 2, sort_keys = True))
    return 0


def _cmd_exec(args: argparse.Namespace) -> int:
    """Run one `gh` command under the host-wide budget and pass it through.

    This is the entry point for the shell scripts (describe_pr.sh,
    describe_issue.sh). They cannot import the Python module, and they were the
    single largest source of unthrottled traffic: pr_summarizer.py invoked them
    with env=gh_read_env(), which picks a token but skips the concurrency cap,
    the spacing floor and the retry/cooldown entirely.

    stdout and stderr are forwarded verbatim and the child's exit status is
    preserved, so callers that parse JSON are unaffected.
    """
    cmd = list(args.argv or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("usage: gh_budget.py exec -- gh api ...", file = sys.stderr)
        return 2

    var = os.environ.get("GH_BUDGET_TOKEN_VAR") or None
    tok = os.environ.get(var) if var else None
    r = run_gh(cmd, token_var = var, token = tok, timeout = args.timeout, label = " ".join(cmd[:3]))
    if r.stdout:
        sys.stdout.write(r.stdout)
    if r.stderr:
        sys.stderr.write(r.stderr)
    return r.returncode


def _cmd_reset_cooldown(_args: argparse.Namespace) -> int:
    _mutate(lambda d: {**d, "cooldown_until": 0.0})
    print("cooldown cleared")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description = "Host-wide gh read budget.")
    sub = p.add_subparsers(dest = "cmd", required = True)
    sub.add_parser("status").set_defaults(fn = _cmd_status)
    c = sub.add_parser("calibrate")
    c.add_argument("--token-var", default = None, help = "env var holding the token to calibrate")
    c.set_defaults(fn = _cmd_calibrate)
    e = sub.add_parser("exec", help = "run a gh command under the budget")
    e.add_argument("--timeout", type = int, default = 120)
    e.add_argument("argv", nargs = argparse.REMAINDER)
    e.set_defaults(fn = _cmd_exec)
    sub.add_parser("reset-cooldown").set_defaults(fn = _cmd_reset_cooldown)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
