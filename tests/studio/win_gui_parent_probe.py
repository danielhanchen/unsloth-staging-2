# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does the backend hang when its parent is a GUI process with no console? (D10)

Ruled out so far, each by experiment rather than by argument:

  the stdout pipe      file / undrained pipe / drained pipe all came up (14.7/8.8/8.8 s)
  the creation flags   baseline, CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, both
                       together as process.rs spawns them, and DETACHED_PROCESS: all
                       came up in 8.3-8.4 s
  cold-start cost      a backend started by hand immediately after install, no app
                       involved, came up in 22.6 s against a 900 s budget

What is left is the parent. `process.rs:577-579` sets stdout and stderr to pipes and
**says nothing about stdin**, so the backend inherits the stdin of the Tauri app -- a
GUI-subsystem process that has no console at all. Every probe so far either passed
`stdin=DEVNULL` or inherited a console from pwsh/python.exe. A handle that is neither a
console nor closed, that never delivers a byte and never signals EOF, is a classic
Windows hang, and it would hang only under the app.

So reproduce the whole shape: a console-less GUI parent (`pythonw.exe`, launched with
CREATE_NO_WINDOW), spawning the backend with piped stdout/stderr and **inherited**
stdin, using the app's exact creation flags.

    python tests/studio/win_gui_parent_probe.py --exe <unsloth.exe> --port 8930

The parent cannot print anything (it has no console), so it writes its verdict to
--result and this process reports it.
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

CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
APP_FLAGS = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW


def health(port: int, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/health", timeout=timeout
        ) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def run_child(exe: Path, port: int, wait: float, result: Path, log: Path) -> int:
    """Runs INSIDE the console-less GUI parent. Spawns the backend the way the app does
    and records what happened. Must never raise into a process with nowhere to print."""
    record: dict = {"port": port, "flags": f"0x{APP_FLAGS:08X}"}
    try:
        started = time.monotonic()
        proc = subprocess.Popen(
            [str(exe), "studio", "--api-only", "-H", "127.0.0.1", "-p", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # The whole point: NOT DEVNULL. Inherit this GUI process's stdin, exactly as
            # process.rs does by saying nothing about it.
            creationflags=APP_FLAGS,
        )
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

        record.update(
            up=up,
            exited=exited,
            seconds=round(time.monotonic() - started, 1),
        )
        text = b"".join(captured).decode("utf-8", errors="replace")
        record["bytes_captured"] = len(text)
        record["tail"] = text.strip().splitlines()[-6:]
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(text, encoding="utf-8")
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    except Exception as error:  # nowhere to print; capture it instead
        record["error"] = f"{type(error).__name__}: {error}"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8930)
    parser.add_argument("--wait", type=float, default=300.0)
    parser.add_argument("--result", type=Path, default=Path("artifacts/gui-parent-probe.json"))
    parser.add_argument("--log", type=Path, default=Path("logs/gui-parent-backend.log"))
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        return run_child(args.exe, args.port, args.wait, args.result, args.log)

    if not args.exe.is_file():
        print(f"::error::no unsloth.exe at {args.exe}")
        return 1

    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.is_file():
        print(f"::error::no pythonw.exe next to {sys.executable}")
        return 1

    if args.result.exists():
        args.result.unlink()

    print(f"=== launching a console-less GUI parent: {pythonw} ===", flush=True)
    parent = subprocess.Popen(
        [
            str(pythonw), __file__, "--child",
            "--exe", str(args.exe),
            "--port", str(args.port),
            "--wait", str(args.wait),
            "--result", str(args.result),
            "--log", str(args.log),
        ],
        creationflags=APP_FLAGS,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Give the parent its own budget plus room to write the result.
    try:
        parent.wait(timeout=args.wait + 180)
    except subprocess.TimeoutExpired:
        print("::warning::the GUI parent itself never finished; killing it", flush=True)
        parent.kill()

    if not args.result.exists():
        print("::warning::the GUI parent wrote no result at all -- inconclusive", flush=True)
        return 0

    record = json.loads(args.result.read_text(encoding="utf-8"))
    print(json.dumps(record, indent=2), flush=True)

    print("\n=== D10 verdict ===", flush=True)
    if record.get("error"):
        print(f"::warning::the probe itself failed: {record['error']}", flush=True)
    elif record.get("up"):
        print(
            f"::warning::NOT REPRODUCED -- with a console-less GUI parent, inherited "
            f"stdin and the app's exact flags, the backend still came up in "
            f"{record['seconds']}s. Every mechanical difference between the app's spawn "
            f"and a plain one is now ruled out; what is left is the app's own "
            f"environment or its job object.",
            flush=True,
        )
    else:
        print(
            f"::warning::REPRODUCED -- a console-less GUI parent with INHERITED stdin "
            f"hangs the backend ({record['seconds']}s, {record.get('bytes_captured')} "
            f"bytes captured), while the same flags from a console parent came up in "
            f"8.3s. D10 is the inherited stdin handle: process.rs pipes stdout and "
            f"stderr but leaves stdin alone. Fix: Stdio::null() for stdin.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
