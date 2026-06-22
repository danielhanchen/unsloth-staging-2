# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Self-contained proof of the #6489 / PR #6548 bug class on the real OS.

This file deliberately does NOT import the Studio backend, so it runs with only
the standard library. It replicates the two I/O sites that ``_python_exec`` used
before PR #6548 -- writing the temp script and reading the child's stdout with
the OS default text codec -- and shows:

  * the OLD code path crashes (or garbles) on a cp1252-class host, and
  * the NEW (explicit utf-8) path round-trips the same payload exactly.

On real Windows the OS default codec is cp1252, so the OLD path fails natively
(no simulation). On a utf-8 host we force cp1252 to emulate the exact codec
Windows would have selected, so the same assertions hold on CI Linux too.
"""

import locale
import os
import subprocess
import sys
import tempfile

import pytest

# Arrow, em-dash, accent, CJK, check, latin-a-macron, astral-plane emoji. ``ā``
# (U+0101) is included on purpose: its utf-8 bytes (0xC4 0x81) contain 0x81,
# which is UNDEFINED in cp1252, so decoding those bytes as cp1252 is a hard
# UnicodeDecodeError rather than silent mojibake.
UNICODE = "café — 数字 → ✓ ā \U0001F600"

_IS_WIN = sys.platform == "win32"
# Native OS default on Windows (cp1252 on en-US runners); forced cp1252 sim
# elsewhere so the same logic demonstrates the class on Linux/macOS CI too.
_OLD_CODEC = locale.getpreferredencoding(False) if _IS_WIN else "cp1252"


def _cp1252_class(codec: str) -> bool:
    try:
        "→".encode(codec)
    except UnicodeEncodeError:
        return True
    return False


# If a Windows runner somehow defaults to utf-8 (e.g. UTF-8 mode on), the old
# path would not fail and there is nothing to prove -- skip rather than lie.
_SKIP_OLD = not _cp1252_class(_OLD_CODEC)
_skip_reason = f"OS default codec {_OLD_CODEC!r} can encode non-ASCII; no bug to reproduce"


@pytest.mark.skipif(_SKIP_OLD, reason=_skip_reason)
def test_old_write_path_crashes():
    """OLD temp-script write (os.fdopen 'w', OS default codec) crashes on cp1252."""
    code = f"print({UNICODE!r})"
    fd, path = tempfile.mkstemp(suffix=".py", prefix="old_", dir=tempfile.mkdtemp())
    try:
        with pytest.raises(UnicodeEncodeError):
            with os.fdopen(fd, "w", encoding=_OLD_CODEC) as f:
                f.write(code)
    finally:
        if os.path.exists(path):
            os.unlink(path)


@pytest.mark.skipif(_SKIP_OLD, reason=_skip_reason)
def test_old_read_path_crashes_or_garbles():
    """OLD stdout read (OS default codec) crashes/garbles; NEW (utf-8) round-trips.

    The source is written as utf-8 so the WRITE succeeds and we isolate the READ
    failure. The child gets PYTHONIOENCODING=utf-8 (the sandbox path always set
    it), so it emits valid utf-8 bytes. We capture those bytes once and decode
    them two ways. Reading raw bytes and decoding in the main thread is the
    deterministic equivalent of the OLD ``Popen(text=True)`` path: on Windows the
    text-mode decode happens in a background reader thread, so the crash would
    surface there and ``communicate()`` would just return ``None`` -- the data is
    still lost, but the failure is not catchable around ``communicate()``. The
    byte-level decode below exercises the identical codec mismatch on every OS.
    """
    fd, path = tempfile.mkstemp(suffix=".py", prefix="src_", dir=tempfile.mkdtemp())
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"print({UNICODE!r})")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.Popen(
            [sys.executable, path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,  # binary mode
        )
        raw, _ = proc.communicate(timeout=30)
    finally:
        if os.path.exists(path):
            os.unlink(path)

    # OLD: decode child bytes with the OS default codec -> hard crash or mojibake.
    old_failed = False
    try:
        decoded_old = raw.decode(_OLD_CODEC)
        old_failed = UNICODE not in decoded_old   # decoded without raising == mojibake
    except UnicodeDecodeError:
        old_failed = True
    assert old_failed, f"old codec {_OLD_CODEC!r} unexpectedly round-tripped"

    # NEW: decode the same bytes with the PR's explicit utf-8 -> exact round-trip.
    assert UNICODE in raw.decode("utf-8", errors="replace")


@pytest.mark.parametrize("disable_sandbox_like", [False, True])
def test_new_path_round_trips(disable_sandbox_like):
    """NEW explicit-utf-8 write + read round-trips the payload exactly on any OS."""
    fd, path = tempfile.mkstemp(suffix=".py", prefix="new_", dir=tempfile.mkdtemp())
    with os.fdopen(fd, "w", encoding="utf-8") as f:   # PR fix: encoding on write
        f.write(f"print({UNICODE!r})")
    # bypass-like env always sets PYTHONIOENCODING after the PR; sandbox-like
    # already did. Either way the child emits utf-8.
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.Popen(
            [sys.executable, path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",  # PR fix: encoding on read
            env=env,
        )
        out, _ = proc.communicate(timeout=30)
    finally:
        if os.path.exists(path):
            os.unlink(path)
    assert UNICODE in (out or ""), repr(out)
