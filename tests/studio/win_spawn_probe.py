# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which part of the desktop app's Windows spawn makes the backend hang? (D10)

The backend the app spawns never becomes reachable, and its own *file* session log
stops at "loading PyTorch, Unsloth and Transformers..." -- it never reaches "Imported
FastAPI app". Run by hand on the same machine the identical command imports in 7-11 s
and serves /api/health. So the machine, the venv, torch and the backend are all fine,
and something about *how the app launches it* is not.

A previous round ruled out the stdout pipe: with stdout to a file, to an undrained
pipe, and to a drained pipe, all three came up (14.7 s / 8.8 s / 8.8 s). It also
explains why -- the child writes only a few hundred bytes before binding, nowhere near
enough to fill a pipe buffer.

What that round did NOT control for is the console. process.rs:641 spawns with

    cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);

and CREATE_NO_WINDOW gives the child no console at all, while every arm so far
inherited the runner's console. So vary exactly that, one bit at a time, using the same
binary and arguments the app uses:

    python tests/studio/win_spawn_probe.py --exe <unsloth.exe> --first-port 8920

Each arm gets its own port so a hung one cannot block the next.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Values from the Windows SDK; subprocess only exposes some of these by name.
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008

# What the app actually passes (process.rs:641). Bisecting this is the point.
ARMS = [
    ("baseline", 0, "pipes, console inherited -- the arm already known to come up"),
    ("new_process_group", CREATE_NEW_PROCESS_GROUP, "the app's first flag, alone"),
    ("no_window", CREATE_NO_WINDOW, "the app's second flag, alone -- no console"),
    (
        "app_exact",
        CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
        "both, exactly as process.rs spawns it",
    ),
    ("detached", DETACHED_PROCESS, "no console by a different route, as a cross-check"),
]


def health(port: int, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/health", timeout=timeout
        ) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def run_arm(exe: Path, label: str, flags: int, port: int, wait: float, log_dir: Path) -> dict:
    """Spawn the backend one way and report whether it ever answers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    args = [str(exe), "studio", "--api-only", "-H", "127.0.0.1", "-p", str(port)]
    print(f"\n=== arm {label} (flags=0x{flags:08X}) on port {port} ===", flush=True)

    started = time.monotonic()
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
    )
    # Drain in threads so this probe is never itself the reason a child blocks; the
    # pipe was already exonerated, and leaving it undrained here would reintroduce a
    # variable we have ruled out.
    captured: list[bytes] = []
    import threading

    def drain(stream) -> None:
        try:
            for chunk in iter(lambda: stream.read(4096), b""):
                captured.append(chunk)
        except Exception:
            pass

    for stream in (proc.stdout, proc.stderr):
        threading.Thread(target=drain, args=(stream,), daemon=True).start()

    up = False
    exited = None
    deadline = started + wait
    while time.monotonic() < deadline:
        if health(port):
            up = True
            break
        if proc.poll() is not None:
            exited = proc.returncode
            break
        time.sleep(2)
    elapsed = round(time.monotonic() - started, 1)

    if up:
        verdict = f"UP after {elapsed}s"
    elif exited is not None:
        verdict = f"EXITED with {exited} after {elapsed}s"
    else:
        verdict = f"HUNG -- alive but never answered in {elapsed}s"
    print(f"RESULT arm {label}: {verdict}", flush=True)

    output = b"".join(captured).decode("utf-8", errors="replace")
    (log_dir / f"spawn-{label}.log").write_text(output, encoding="utf-8")
    tail = output.strip().splitlines()[-3:]
    for line in tail:
        print(f"    | {line[:160]}", flush=True)

    try:
        proc.terminate()
        proc.wait(timeout=30)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    time.sleep(3)
    return {
        "arm": label,
        "flags": f"0x{flags:08X}",
        "port": port,
        "up": up,
        "exited": exited,
        "seconds": elapsed,
        "bytes_captured": len(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path, help="the studio venv's unsloth.exe")
    parser.add_argument("--first-port", type=int, default=8920)
    parser.add_argument("--wait", type=float, default=180.0, help="seconds per arm")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/spawn-probe"))
    parser.add_argument("--report", type=Path, help="write JSON results here")
    args = parser.parse_args()

    if not args.exe.is_file():
        print(f"::error::no unsloth.exe at {args.exe}")
        return 1

    results = [
        run_arm(args.exe, label, flags, args.first_port + index, args.wait, args.log_dir)
        for index, (label, flags, _why) in enumerate(ARMS)
    ]

    print("\n=== summary ===", flush=True)
    for arm, (_label, _flags, why) in zip(results, ARMS):
        state = "UP  " if arm["up"] else "HUNG"
        print(f"  {state}  {arm['arm']:<18} {arm['seconds']:>6}s   {why}", flush=True)

    by_name = {arm["arm"]: arm for arm in results}
    baseline_up = by_name["baseline"]["up"]
    app_exact_up = by_name["app_exact"]["up"]
    no_window_up = by_name["no_window"]["up"]
    group_up = by_name["new_process_group"]["up"]

    print("\n=== D10 verdict ===", flush=True)
    if baseline_up and not app_exact_up:
        print(
            "::warning::REPRODUCED -- the app's exact creation flags hang the backend "
            "while the same command with default flags comes up.",
            flush=True,
        )
        if not no_window_up and group_up:
            print(
                "::warning::Bisected to CREATE_NO_WINDOW: the backend hangs when it has "
                "no console. That is process.rs:641.",
                flush=True,
            )
        elif not group_up and no_window_up:
            print("::warning::Bisected to CREATE_NEW_PROCESS_GROUP.", flush=True)
        else:
            print(
                "::warning::Neither flag alone reproduces it; it needs the combination.",
                flush=True,
            )
    elif baseline_up and app_exact_up:
        print(
            "::warning::NOT REPRODUCED -- even the app's exact creation flags come up "
            "here, so the remaining difference is the environment the app passes, its "
            "job object, or its working directory rather than how it creates the process.",
            flush=True,
        )
    else:
        print(f"::warning::inconclusive: {json.dumps(by_name, default=str)[:400]}", flush=True)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
