#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Exec a command with signals 32 and 33 set to SIG_IGN, the way the app leaves them.

D8 background. On ubuntu-22.04 the desktop app's spawned backend exits 1 in under a
second having written zero bytes to either stream, on roughly half of runs. On
ubuntu-24.04 it never does. The child's environment, rlimits and fds are byte-identical
between the two, and run from a normal shell on the same failing machine the same command
succeeded 12 times out of 12 -- so the app's spawn context is required to reproduce it.

The one measured difference is the inherited signal disposition:

    ubuntu-22.04   SigIgn: 0000000180000007      (all legs, passing and failing)
    ubuntu-24.04   SigIgn: 0000000000000007

0x7 is SIGHUP/SIGINT/SIGQUIT, which both have and which `nohup` explains. The extra bits
are 31 and 32, meaning signals **32 and 33** -- glibc's internal NPTL signals, SIGCANCEL
and SIGSETXID. Ignored dispositions survive `execve`, so the backend inherits them, and
nothing about that shows up in the environment or the fd table.

That correlation is with the OS, not with the failure: the 22.04 leg that passed had the
same mask. So this is at best a precondition rather than the trigger, which is exactly
what an intermittent OS-specific defect looks like -- a platform-specific setup plus a
race. This script exists to test the precondition directly instead of arguing about it:
put a normal shell-spawned backend into the same signal state, and see whether the
failure follows it out of the app.

Usage:
    spawn_with_ignored_rt_signals.py [--signals 32,33] -- <command> [args...]
"""

import argparse
import ctypes
import os
import sys


SIG_IGN = 1
SYS_rt_sigaction = 13  # x86_64


class _KernelSigaction(ctypes.Structure):
    """The kernel's struct sigaction, which is not glibc's."""

    _fields_ = [
        ("handler", ctypes.c_void_p),
        ("flags", ctypes.c_ulong),
        ("restorer", ctypes.c_void_p),
        ("mask", ctypes.c_ulong),
    ]


def ignore_signals(signums: "list[int]") -> "list[int]":
    """Set each signal to SIG_IGN, returning the ones that took.

    Goes through the raw rt_sigaction syscall rather than signal()/sigaction(), because
    glibc owns signals 32 and 33 for its own threading internals and refuses to touch
    them: both wrappers return EINVAL, verified. The kernel itself has no such objection.
    That glibc guards these at all is a fair hint that leaving them ignored across an exec
    is not a healthy state for a child to start in.
    """
    libc = ctypes.CDLL("libc.so.6", use_errno = True)
    libc.syscall.restype = ctypes.c_long

    applied = []
    for signum in signums:
        action = _KernelSigaction(
            handler = ctypes.c_void_p(SIG_IGN), flags = 0, restorer = None, mask = 0,
        )
        ctypes.set_errno(0)
        result = libc.syscall(
            ctypes.c_long(SYS_rt_sigaction),
            ctypes.c_long(signum),
            ctypes.byref(action),
            None,
            ctypes.c_size_t(8),   # sizeof(sigset_t) as the kernel counts it
        )
        if result != 0:
            print(
                f"warning: could not ignore signal {signum}: errno {ctypes.get_errno()}",
                file = sys.stderr, flush = True,
            )
            continue
        applied.append(signum)
    return applied


def current_sigign() -> str:
    try:
        with open("/proc/self/status", encoding = "utf-8") as handle:
            for line in handle:
                if line.startswith("SigIgn:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--signals", default = "32,33",
        help = "comma-separated signal numbers to ignore before exec (default 32,33)",
    )
    parser.add_argument("command", nargs = argparse.REMAINDER)
    args = parser.parse_args()

    command = [a for a in args.command if a != "--"]
    if not command:
        print("no command given", file = sys.stderr)
        return 2

    signums = [int(s) for s in args.signals.split(",") if s.strip()]
    applied = ignore_signals(signums)
    print(
        f"ignoring signals {applied}; SigIgn is now {current_sigign()}",
        file = sys.stderr, flush = True,
    )

    # exec, not spawn: ignored dispositions survive execve, and this way the command
    # under test has no extra parent between it and the shell.
    try:
        os.execv(command[0], command)
    except OSError as error:
        print(f"exec failed: {error}", file = sys.stderr, flush = True)
        return 127
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
