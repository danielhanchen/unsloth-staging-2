#!/usr/bin/env python3
"""Standalone reproduction of the #6489 / PR #6548 UTF-8 bug class.

Demonstrates the two OS-default-codec failure sites in the OLD ``_python_exec``
I/O, and that the PR's explicit-UTF-8 I/O fixes both. Works on any OS:

- On real Windows the OS default text codec IS cp1252, so the OLD code paths
  fail natively (pass --native to skip the forced codec).
- On a UTF-8 host (Linux/macOS CI) we force ``cp1252`` explicitly to emulate the
  exact codec Windows would have selected.

No Studio deps required: it replicates only the temp-file write + subprocess
read that ``_python_exec`` performs, old vs new.
"""

import argparse
import os
import subprocess
import sys
import tempfile

# This harness PRINTS the unicode payload as part of its diagnostics. On a native
# cp1252 Windows console that print() would itself raise the very UnicodeEncodeError
# we are demonstrating, killing the script before it can report results. Force the
# harness's own stdout to utf-8 so only the OLD code paths under test exhibit the
# bug, not the reporting around them.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Arrow, em-dash, accent, CJK, check, astral-plane emoji. ``ā`` (U+0101) is
# included because its UTF-8 bytes (0xC4 0x81) contain 0x81, which is UNDEFINED
# in cp1252 -- so decoding those bytes as cp1252 raises UnicodeDecodeError
# rather than merely producing mojibake. That makes the read-site failure a hard
# crash, not a silent corruption.
UNICODE = "café — 数字 → ✓ ā \U0001F600"


def _os_default_codec(force_cp1252: bool) -> str:
    if force_cp1252:
        return "cp1252"
    import locale
    return locale.getpreferredencoding(False)


def old_write(code: str, codec: str, workdir: str):
    """OLD: os.fdopen(fd, 'w') -> OS default codec on the temp script write."""
    fd, path = tempfile.mkstemp(suffix=".py", prefix="old_", dir=workdir)
    with os.fdopen(fd, "w", encoding=codec) as f:   # codec == OS default on the real OS
        f.write(code)
    return path


def new_write(code: str, workdir: str):
    """NEW: explicit utf-8 on the temp script write."""
    fd, path = tempfile.mkstemp(suffix=".py", prefix="new_", dir=workdir)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    return path


def _child_bytes(path: str) -> bytes:
    """Run the child (PYTHONIOENCODING=utf-8 -> emits utf-8) and capture raw bytes.

    Reading bytes and decoding in the main thread is deterministic on every OS.
    OLD ``Popen(text=True)`` decoded in a background reader thread on Windows, so
    a decode crash there would just lose the data and leave ``communicate()``
    returning None -- the byte-level decode below shows the identical mismatch.
    """
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(
        [sys.executable, path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,  # binary mode
    )
    raw, _ = proc.communicate(timeout=30)
    return raw or b""


def old_read(path: str, codec: str) -> str:
    """OLD: decode the child's utf-8 bytes with the OS default codec (may raise)."""
    return _child_bytes(path).decode(codec)


def new_read(path: str) -> str:
    """NEW: explicit utf-8 + errors=replace on the parent decode."""
    return _child_bytes(path).decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", action="store_true",
                    help="use the real OS default codec (use on actual Windows)")
    args = ap.parse_args()
    codec = _os_default_codec(force_cp1252=not args.native)

    print(f"python={sys.version.split()[0]} platform={sys.platform} "
          f"os-default-codec={codec!r}")
    print(f"payload={UNICODE!r}\n")

    code = f"print({UNICODE!r})"
    results = {}
    workdir = tempfile.mkdtemp(prefix="repro_")

    # --- Write site -------------------------------------------------------
    try:
        old_write(code, codec, workdir)
        results["old_write"] = "OK (host codec encodes payload; no bug here)"
    except UnicodeEncodeError as e:
        results["old_write"] = f"FAILED as expected: UnicodeEncodeError ({e})"

    new_path = new_write(code, workdir)
    results["new_write"] = "OK"

    # --- Read site --------------------------------------------------------
    # Use a payload the host codec CAN encode in source so the write succeeds and
    # we isolate the read failure. We write the source as utf-8 (new_write) so the
    # child source is intact, then read its stdout the OLD way (host codec).
    try:
        out_old = old_read(new_path, codec)
        if UNICODE in out_old:
            results["old_read"] = "OK (round-tripped; no bug here)"
        else:
            results["old_read"] = f"GARBLED as expected: {out_old!r}"
    except UnicodeDecodeError as e:
        results["old_read"] = f"FAILED as expected: UnicodeDecodeError ({e})"

    out_new = new_read(new_path)
    if UNICODE in out_new:
        results["new_read"] = "OK round-tripped exactly"
    else:
        results["new_read"] = f"UNEXPECTED garble: {out_new!r}"

    print("Results:")
    for k in ("old_write", "new_write", "old_read", "new_read"):
        print(f"  {k:10s}: {results[k]}")

    # Acceptance: new path must always round-trip; on a cp1252-class codec the
    # old path must demonstrate at least one failure (write crash or read crash/garble).
    ok_new = results["new_read"].startswith("OK") and results["new_write"] == "OK"
    old_broke = ("FAILED" in results["old_write"]) or ("FAILED" in results["old_read"]) \
        or ("GARBLED" in results["old_read"])
    print(f"\nnew_path_round_trips={ok_new}  old_path_demonstrates_bug={old_broke}")
    sys.exit(0 if (ok_new and old_broke) else 1)


if __name__ == "__main__":
    main()
