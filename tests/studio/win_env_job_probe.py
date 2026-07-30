# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The last two differences between the app's backend spawn and a plain one. (D10)

Eliminated so far, each by experiment:

  stdout pipe          file / undrained / drained all came up (14.7 / 8.8 / 8.8 s)
  creation flags       baseline, CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, both as
                       process.rs spawns them, DETACHED_PROCESS: all 8.3-8.4 s
  cold-start cost      backend by hand seconds after install, no app: 22.6 s
  console-less parent  pythonw.exe GUI parent, app flags, inherited stdin: 8.5 s

So how the process is created is not the difference. Two things remain:

  env    process.rs:582-614 injects UNSLOTH_STUDIO_DESKTOP_OWNER_TOKEN / _KIND and
         UNSLOTH_STUDIO_NATIVE_PATH_LEASE_SECRET. main.py:377 reads the first two at
         import time, and a "tauri" owner changes startup behaviour (it defaults
         UNSLOTH_STUDIO_ALLOW_STDIO_MCP=1). That is a real behavioural fork on exactly
         the code path that hangs.
  job    main.rs assigns the app to a KILL_ON_JOB_CLOSE job object, which children
         inherit. Job objects can carry limits; a constrained one can stall a child
         that a free one runs fine.

    python tests/studio/win_env_job_probe.py --exe <unsloth.exe> --first-port 8940
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from ctypes import wintypes
from pathlib import Path

CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
APP_FLAGS = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in
                ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                 "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.POINTER(wintypes.ULONG)),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def make_kill_on_close_job() -> int | None:
    """Create a KILL_ON_JOB_CLOSE job and put THIS process in it, so the backend we
    spawn next inherits it exactly as the app's children do."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        print(f"::warning::CreateJobObjectW failed: {ctypes.get_last_error()}", flush=True)
        return None
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job, JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
        ctypes.byref(info), ctypes.sizeof(info),
    ):
        print(f"::warning::SetInformationJobObject failed: {ctypes.get_last_error()}", flush=True)
        return None
    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        # Already in a job that disallows nesting -- common under CI agents.
        print(
            f"::warning::AssignProcessToJobObject failed: {ctypes.get_last_error()} "
            f"(this runner may already place us in a job that forbids nesting)",
            flush=True,
        )
        return None
    return job


def health(port: int, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/health", timeout=timeout
        ) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def run_arm(exe: Path, label: str, port: int, wait: float, env: dict | None, log_dir: Path) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== arm {label} on port {port} ===", flush=True)
    if env:
        for key in sorted(env):
            shown = env[key] if not key.endswith(("TOKEN", "SECRET")) else "<redacted>"
            print(f"    {key}={shown}", flush=True)

    child_env = dict(os.environ)
    child_env.update(env or {})

    started = time.monotonic()
    proc = subprocess.Popen(
        [str(exe), "studio", "--api-only", "-H", "127.0.0.1", "-p", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
        creationflags=APP_FLAGS, env=child_env,
    )
    captured: list[bytes] = []

    def drain(stream) -> None:
        try:
            for chunk in iter(lambda: stream.read(4096), b""):
                captured.append(chunk)
        except Exception:
            pass

    for stream in (proc.stdout, proc.stderr):
        threading.Thread(target=drain, args=(stream,), daemon=True).start()

    up, exited = False, None
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

    verdict = (f"UP after {elapsed}s" if up
               else f"EXITED with {exited} after {elapsed}s" if exited is not None
               else f"HUNG -- alive but silent for {elapsed}s")
    print(f"RESULT arm {label}: {verdict}", flush=True)
    text = b"".join(captured).decode("utf-8", errors="replace")
    (log_dir / f"env-job-{label}.log").write_text(text, encoding="utf-8")
    for line in text.strip().splitlines()[-3:]:
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
    return {"arm": label, "port": port, "up": up, "exited": exited, "seconds": elapsed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--first-port", type=int, default=8940)
    parser.add_argument("--wait", type=float, default=240.0)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/env-job-probe"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if not args.exe.is_file():
        print(f"::error::no unsloth.exe at {args.exe}")
        return 1

    # Shaped like the real ones (process.rs hands over opaque secrets); the backend only
    # hashes the owner token, so any high-entropy value exercises the same path.
    tauri_env = {
        "UNSLOTH_STUDIO_DESKTOP_OWNER_KIND": "tauri",
        "UNSLOTH_STUDIO_DESKTOP_OWNER_TOKEN": secrets.token_hex(32),
        "UNSLOTH_STUDIO_NATIVE_PATH_LEASE_SECRET": secrets.token_hex(32),
    }

    results = [
        run_arm(args.exe, "plain", args.first_port, args.wait, None, args.log_dir),
        run_arm(args.exe, "tauri_env", args.first_port + 1, args.wait, tauri_env, args.log_dir),
    ]

    print("\n=== now inside a KILL_ON_JOB_CLOSE job object ===", flush=True)
    job = make_kill_on_close_job()
    if job:
        results.append(
            run_arm(args.exe, "job_object", args.first_port + 2, args.wait, None, args.log_dir)
        )
        results.append(
            run_arm(args.exe, "job_and_env", args.first_port + 3, args.wait, tauri_env, args.log_dir)
        )
    else:
        print("::warning::could not create/join a job object here; those arms are skipped", flush=True)

    print("\n=== summary ===", flush=True)
    for arm in results:
        print(f"  {'UP  ' if arm['up'] else 'HUNG'}  {arm['arm']:<12} {arm['seconds']:>6}s", flush=True)

    by_name = {arm["arm"]: arm for arm in results}
    hung = [name for name, arm in by_name.items() if not arm["up"]]
    print("\n=== D10 verdict ===", flush=True)
    if not hung:
        print(
            "::warning::NOT REPRODUCED by any of these either. Every difference I can "
            "reproduce from outside the app -- pipes, creation flags, cold start, a "
            "console-less GUI parent, inherited stdin, the injected Tauri env, and a "
            "KILL_ON_JOB_CLOSE job object -- leaves the backend starting normally. "
            "The next step needs instrumentation inside the app rather than another "
            "guess from outside: log the child PID and take a stack dump of it while "
            "it is hung.",
            flush=True,
        )
    else:
        print(f"::warning::REPRODUCED by: {', '.join(hung)}", flush=True)
        if "tauri_env" in hung and "plain" not in hung:
            print(
                "::warning::The injected Tauri owner environment alone hangs it. "
                "main.py:377 reads UNSLOTH_STUDIO_DESKTOP_OWNER_* at import time and a "
                "'tauri' owner forks startup behaviour; that fork is the bug.",
                flush=True,
            )
        if "job_object" in hung and "plain" not in hung:
            print(
                "::warning::The KILL_ON_JOB_CLOSE job object alone hangs it -- that is "
                "main.rs assigning the app to a job its children inherit.",
                flush=True,
            )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
