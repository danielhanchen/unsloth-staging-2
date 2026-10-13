"""Shared plumbing for PR before/after Studio scenes.

Everything here exists because it bit me during the PR 8222 pilot. Each helper
guards one failure mode that produced a screenshot pair which looked perfectly
fine and proved nothing:

  pick_free_ports      a port already held by an unrelated Studio -> launch_studio
                       silently fails to bind, /healthz answers from the WRONG
                       install, and both sides photograph someone else's build
  studio_session       the bootstrap password is written to auth/.bootstrap_password,
                       NOT inlined in the log (so lifecycle's regex finds nothing),
                       and every API 403s with "Password change required" until it
                       is rotated
  assert_showing       a detail panel keeps its previous selection while the click
                       lands on the search box, so the shot is of another model
  open_list            the toolbar format filter also reads "GGUF"; clicking it opens
                       a menu identical on both sides, which photographs as NO CHANGE

The last one is the dangerous class: a wrong selector that still yields a clean
image on both sides is indistinguishable from "the PR changed nothing" unless you
look at the picture. Always look at the picture.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Port bands. The old single base of 8990 was shared by every workspace on the
# box, which is what made two of them pick the same "free" port.
_BASE = 8990
_BAND_SIZE = 200
_BANDS = 20                      # 8990..12989, clear of the ephemeral range


def _workspace_port_base() -> int:
    """A scan base this workspace gets to itself, derived from its own path.

    A bind test proves a port free at the instant it runs and RESERVES NOTHING,
    so two callers scanning the same range from the same base within the same
    second pick the SAME port and the loser's Studio never binds. That is not
    hypothetical across workspaces: a sibling took :9001 in the seconds between
    our pick and our launch, and the AFTER side would have photographed someone
    else's build had the login identity check not caught it.

    Every workspace on this box runs the same code, so a shared constant base is
    the whole problem. Hashing the workspace root spreads them deterministically:
    the same workspace keeps the same band run to run (so a leftover Studio is
    still found where it was left), and two different ones have to collide by
    hash rather than by construction.
    """
    root = os.environ.get("WORKSPACE") or os.environ.get("UNSLOTH_WORKSPACE") or str(Path.cwd())
    band = int(hashlib.sha256(root.encode()).hexdigest()[:8], 16) % _BANDS
    return _BASE + band * _BAND_SIZE


def pick_free_ports(count: int, start: Optional[int] = None,
                    stop: Optional[int] = None, seed: Optional[str] = None) -> list[int]:
    """`count` ports nothing is listening on, proven by actually binding them.

    Not a guess from a range: this workspace routinely has a dozen Studios alive
    from earlier sessions, and a collision does not raise -- it serves someone
    else's UI on the port you asked for.

    With no `start`, scan this workspace's own band (see `_workspace_port_base`)
    rather than a base every workspace shares. `UIDIFF_PORT_START` overrides it
    for a run that needs a specific range; an explicit `start=` argument wins
    over both, since a caller naming a port range meant that range.
    """
    if start is None:
        env_start = os.environ.get("UIDIFF_PORT_START")
        if env_start:
            try:
                start = int(env_start)
            except ValueError:
                raise ValueError(
                    f"UIDIFF_PORT_START={env_start!r} is not a port number"
                ) from None
        else:
            start = _workspace_port_base()
    # Checked rather than left to the bind: port 0 BINDS FINE (the kernel hands
    # out an ephemeral port) and would be returned as a free port, launching
    # Studio on `-p 0`; and a port above 65535 raises OverflowError, which the
    # loop's `except OSError` below does not catch.
    if not 1024 <= start <= 65_535 - _BAND_SIZE:
        raise ValueError(f"port scan base {start} is out of range")
    if stop is None:
        stop = start + _BAND_SIZE
    free: list[int] = []
    # Scan from a point spread over the range (wrapping): the bind probe is check-then-use, and two
    # processes starting in the same second would otherwise both take the first free port. A
    # `seed` makes the point stable, so callers that start the same UI twice in sequence (the
    # two sides of one before/after run, where the port shows in the page) get the same port.
    span = list(range(start, stop))
    if not span:
        k = 0
    elif seed is not None:
        k = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % len(span)
    else:
        k = random.randrange(len(span))
    for port in span[k:] + span[:k]:
        s = socket.socket()
        # Probe the way the server will bind: uvicorn sets SO_REUSEADDR, so a port whose last
        # connections sit in TIME_WAIT is free to it. Without this the same seeded scan skips
        # a port the previous side just released and the two sides of a run differ.
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            free.append(port)
        except OSError:
            pass
        finally:
            s.close()
        if len(free) == count:
            return free
    raise RuntimeError(f"could not find {count} free ports in [{start}, {stop})")


@dataclass
class Session:
    """An authenticated Studio, proven to be the install we meant."""

    base_url: str
    home: Path
    access_token: str
    refresh_token: str
    password: str


def _post(url: str, payload: dict, token: Optional[str] = None, timeout: int = 120) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def studio_session(base_url: str, home: Path, new_password: str) -> Session:
    """Log in to `base_url` with THIS home's own credential, rotating if needed.

    Logging in with the home's own credential is what proves the server answering
    is the install we built. A stale Studio on the same port has a different
    password and fails here instead of quietly serving the wrong screenshots.

    Two credential states, because a home is reused across runs:
      first run   auth/.bootstrap_password exists; log in with it and rotate. The
                  rotation is not optional -- until it happens every authenticated
                  route answers 403 "Password change required".
      later runs  Studio DELETES the bootstrap file once rotated, so the only
                  credential left is the one we set. Use it directly.
    """
    home = Path(home)
    boot_file = home / "auth" / ".bootstrap_password"
    if boot_file.exists():
        bootstrap = boot_file.read_text().strip()
        rotate = True
    else:
        bootstrap = new_password
        rotate = False
    try:
        tok = _post(f"{base_url}/api/auth/login",
                    {"username": "unsloth", "password": bootstrap})
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()[:300]
        hint = ("ANOTHER Studio may hold this port and the one we launched never bound"
                if rotate else
                "this home was rotated under a DIFFERENT password; pass it as "
                "--password, or delete the home and reinstall")
        raise RuntimeError(
            f"login to {base_url} with {home}'s own credential failed "
            f"({exc.code}): {body}\n{hint}"
        ) from None
    if rotate:
        tok = _post(f"{base_url}/api/auth/change-password",
                    {"current_password": bootstrap, "new_password": new_password},
                    token=tok["access_token"])
    return Session(base_url=base_url, home=home, password=new_password,
                   access_token=tok["access_token"],
                   refresh_token=tok.get("refresh_token", ""))


def api_get(session: Session, path: str, timeout: int = 600) -> dict:
    """Authenticated GET, for the numeric half of the evidence.

    A screenshot shows a change; an API reading proves what the number IS. Scenes
    should report both, because a picture of a list cannot be diffed and a reviewer
    cannot count 63 rows by eye.
    """
    req = urllib.request.Request(
        f"{session.base_url}{path}",
        headers={"Authorization": f"Bearer {session.access_token}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def api_post(session: Session, path: str, payload: dict, timeout: int = 600) -> dict:
    """Authenticated POST, for endpoints that answer a query rather than mutate.

    Same purpose as :func:`api_get`: read the number the photographed server would put
    on screen. Deliberately NOT wrapped in a try -- a scene decides for itself whether a
    404 is the finding (an endpoint the PR adds) or a failure.
    """
    return _post(f"{session.base_url}{path}", payload, token=session.access_token,
                 timeout=timeout)


def api_delete(session: Session, path: str, payload: Optional[dict] = None,
               timeout: int = 120) -> Optional[dict]:
    """Authenticated DELETE, with an optional JSON body.

    Studio's bulk routes take the body form (`DELETE /api/chat/threads` with
    `{"ids": [...]}`), so the body is not optional decoration. Returns the parsed
    response, or None for the 204s that carry no body.
    """
    headers = {"Authorization": f"Bearer {session.access_token}"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"{session.base_url}{path}", data=data, headers=headers, method="DELETE"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else None


async def hit_test(page, selector: str, samples: int = 19) -> dict:
    """How much of an element actually receives its own clicks.

    Playwright reports an occluded control as a 30 second timeout on the click,
    whose message names the intercepting element but reads like a bad selector.
    That is a poor way to learn that a PR's only entry point is covered: it looks
    like a scene bug, and the honest finding ("this control cannot be clicked")
    needs a MEASUREMENT rather than a stack trace.

    Samples across the element's width at mid height and reports how many points
    `elementFromPoint` resolves back to it, plus what wins where it does not.
    `reachable == 0` means no user can click it at this viewport.
    """
    return await page.evaluate(
        """(args) => {
          const el = document.querySelector(args.selector);
          if (!el) return {found: false};
          const r = el.getBoundingClientRect();
          let reachable = 0; const blockers = {};
          for (let i = 0; i < args.samples; i++) {
            const x = r.x + r.width * ((i + 0.5) / args.samples);
            const hit = document.elementFromPoint(x, r.y + r.height / 2);
            if (el.contains(hit)) { reachable++; continue; }
            let owner = hit;
            while (owner && owner.tagName !== 'BUTTON') owner = owner.parentElement;
            const key = (owner && owner.getAttribute('aria-label'))
                        || (hit && hit.tagName) || 'nothing';
            blockers[key] = (blockers[key] || 0) + 1;
          }
          return {found: true, reachable, samples: args.samples, blockers,
                  rect: {x: Math.round(r.x), y: Math.round(r.y),
                         width: Math.round(r.width), height: Math.round(r.height)}};
        }""",
        {"selector": selector, "samples": samples},
    )


async def assert_clickable(page, selector: str, name: str) -> dict:
    """Fail with a measurement when `selector` cannot receive a click."""
    result = await hit_test(page, selector)
    if not result.get("found"):
        raise RuntimeError(f"{name} ({selector}) is not in the DOM at all")
    if not result["reachable"]:
        raise RuntimeError(
            f"{name} is in the DOM at {result['rect']} but NONE of "
            f"{result['samples']} points across it receive its own clicks; they land "
            f"on {result['blockers']}. It is covered, so no user can click it. This is "
            f"a finding about the build, not a scene bug."
        )
    return result


async def assert_showing(page, name: str, timeout_ms: int = 60_000) -> None:
    """Fail unless the page is actually displaying `name`.

    Guards the quietest scene bug there is: the click misses, the panel keeps its
    previous selection, and the screenshot is of a different model entirely. It
    looks like a valid screenshot, so nothing downstream catches it.
    """
    await page.get_by_role("heading", name=re.compile(re.escape(name))).first.wait_for(
        state="visible", timeout=timeout_ms
    )


async def open_menu(page, trigger, item, attempts: int = 6,
                    item_timeout_ms: int = 4_000) -> None:
    """Click `trigger` until `item` is actually on screen.

    Radix menus lose races with Studio's background refresh: the inventory reloads on a
    timer, and a reload landing between the click and the menu paint remounts the row and
    takes the open dropdown with it. A single attempt failed roughly half the time during
    the 8223 run.

    Worth a helper because of how the failure PRESENTS: a bare `click` timeout on the menu
    ITEM, which reads exactly like a bad selector and sends you rewriting a locator that
    was correct all along.
    """
    for attempt in range(attempts):
        await trigger.scroll_into_view_if_needed()
        await trigger.hover()
        await page.wait_for_timeout(400)
        await trigger.click()
        try:
            await item.wait_for(state="visible", timeout=item_timeout_ms)
            return
        except Exception:  # noqa: BLE001 -- a lost race, not a bad selector
            if attempt == attempts - 1:
                raise
            # Escape first: a half-open menu swallows the next click on the trigger.
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(1_000)


async def open_list(page, item_pattern: str, timeout_ms: int = 15_000) -> None:
    """Open the collapsed control whose CURRENT value matches `item_pattern`.

    Match on a value the list contains (a quant token, a resolution, a precision),
    never on a category word like "GGUF": category words also appear on toolbar
    filters, and opening the wrong menu yields a dropdown with identical contents
    on both sides -- evidence that the PR did nothing.
    """
    combo = page.locator("button").filter(has_text=re.compile(item_pattern)).first
    try:
        await combo.click(timeout=timeout_ms)
    except Exception:  # noqa: BLE001 -- already-open is not a failure
        pass
