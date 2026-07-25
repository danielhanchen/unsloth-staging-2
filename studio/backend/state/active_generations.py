# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Registry of in-flight chat generations, keyed by conversation.

Studio's Chat tab runs several conversations at once: starting a New Chat leaves
the previous conversation streaming in the background. Two things need to know
which conversations are currently generating:

  * ``GET /inference/active-generations`` -- so the UI can label a background
    reload attempt with the chats it would interrupt.
  * ``/inference/load`` and ``/inference/unload`` -- reloading the model tears
    down llama-server underneath every running generation, so the route refuses
    with 409 unless the caller opts in to cancelling them.

A frontend-only guard is not enough: a second browser tab, the desktop app, or a
direct REST call would all walk straight past it. This registry is the gate.

Entries hold the same ``threading.Event`` the per-run cancel registry in
``routes/inference.py`` already owns, so ``cancel_all()`` reuses the existing,
per-request cancellation path -- it closes each generation's own upstream HTTP
stream and never signals llama-server itself.

Everything here is a plain dict plus a ``threading.Lock``: no signals, no process
groups, no event loop affinity, so it behaves identically on Linux, macOS,
Windows and WSL.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Optional

# handle id -> entry. Keyed by an opaque handle rather than thread_id because a
# conversation can legitimately have two overlapping entries for a moment (a tool
# continuation registering before the previous leg unregisters), and a
# thread_id-keyed dict would drop one of them.
_ACTIVE: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


class ActiveGeneration:
    """Context manager registering one in-flight generation for its duration.

    Re-entrant-safe by construction: each ``__enter__`` mints its own handle, so
    nesting or overlapping uses never clobber one another.
    """

    __slots__ = ("thread_id", "cancel_event", "model", "kind", "_handle")

    def __init__(
        self,
        cancel_event: threading.Event,
        *,
        thread_id: Optional[str] = None,
        model: Optional[str] = None,
        kind: str = "chat",
    ):
        self.thread_id = thread_id or None
        self.cancel_event = cancel_event
        self.model = model or None
        self.kind = kind
        self._handle: Optional[str] = None

    def __enter__(self) -> "ActiveGeneration":
        self._handle = uuid.uuid4().hex
        with _LOCK:
            _ACTIVE[self._handle] = {
                "handle": self._handle,
                "thread_id": self.thread_id,
                "model": self.model,
                "kind": self.kind,
                "started_at": time.time(),
                "event": self.cancel_event,
            }
        return self

    def __exit__(self, *exc) -> bool:
        handle, self._handle = self._handle, None
        if handle is not None:
            with _LOCK:
                _ACTIVE.pop(handle, None)
        return False


def snapshot() -> list[dict[str, Any]]:
    """Public view of the in-flight generations, newest last.

    The ``threading.Event`` is deliberately dropped: this feeds an HTTP response.
    """
    with _LOCK:
        entries = list(_ACTIVE.values())
    entries.sort(key = lambda e: e["started_at"])
    return [
        {
            "handle": e["handle"],
            "thread_id": e["thread_id"],
            "model": e["model"],
            "kind": e["kind"],
            "started_at": e["started_at"],
        }
        for e in entries
    ]


def active_thread_ids() -> list[str]:
    """Distinct conversation ids with a generation in flight, in start order.

    Generations started before the frontend knew its thread id (a brand-new chat
    whose first turn races persistence) carry ``thread_id is None`` and are
    counted by ``count()`` but cannot be named here.
    """
    seen: list[str] = []
    for e in snapshot():
        tid = e["thread_id"]
        if tid and tid not in seen:
            seen.append(tid)
    return seen


def count() -> int:
    """Number of generations currently in flight."""
    with _LOCK:
        return len(_ACTIVE)


def cancel_all() -> int:
    """Signal every in-flight generation to stop. Returns how many were signalled.

    Only sets the per-run cancel events; each generation's own stream teardown
    does the rest. Entries are removed by their own ``__exit__``, not here, so a
    generation that is mid-cleanup is not double-counted or lost.
    """
    with _LOCK:
        events = [e["event"] for e in _ACTIVE.values()]
    for ev in events:
        try:
            ev.set()
        except Exception:
            pass
    return len(events)


def cancel_thread(thread_id: str) -> int:
    """Signal only the generations belonging to ``thread_id``."""
    if not thread_id:
        return 0
    with _LOCK:
        events = [e["event"] for e in _ACTIVE.values() if e["thread_id"] == thread_id]
    for ev in events:
        try:
            ev.set()
        except Exception:
            pass
    return len(events)


def reset_for_tests() -> None:
    """Drop every entry. Test-only; never called from request paths."""
    with _LOCK:
        _ACTIVE.clear()
