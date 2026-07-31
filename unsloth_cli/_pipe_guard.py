# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Keep a closed stdout from killing the process.

Deliberately dependency-free -- it must be importable on its own, both because it runs
before typer at CLI startup and so that it can be exercised in isolation.
"""

import sys as _sys


EPIPE = 32


class _PipeTolerantStream:
    """A stdio stream that goes quiet instead of dying when its reader disappears.

    The desktop app spawns the backend with its stdout and stderr on pipes and reads
    them for the log view. If those pipes lose their reader, the next write raises
    BrokenPipeError; the traceback then cannot be written to a stderr that is gone
    either, and the process exits 1 having printed nothing at all.

    That is not hypothetical. It is the measured signature of a real failure -- the
    backend dying in under a second, silently, exit 1 -- and it was reproduced exactly
    and *only* by giving the same command a reader-less pipe: eight attempts out of
    eight, on both ubuntu-22.04 and ubuntu-24.04, where the identical command with a
    file or with a live reader started 8/8.

    Losing the log pipe is not a reason for a server to stop serving. It has a session
    log on disk, and a caller waiting on its port. So the first EPIPE latches this
    stream closed and everything after it is discarded.
    """

    __slots__ = ("_stream", "_broken")

    def __init__(self, stream):
        object.__setattr__(self, "_stream", stream)
        object.__setattr__(self, "_broken", False)

    def _give_up(self):
        object.__setattr__(self, "_broken", True)

    def write(self, data):
        if object.__getattribute__(self, "_broken"):
            return len(data) if isinstance(data, (str, bytes)) else 0
        try:
            return object.__getattribute__(self, "_stream").write(data)
        except BrokenPipeError:
            self._give_up()
        except ValueError:
            # "I/O operation on closed file" -- same situation, different exception.
            self._give_up()
        except OSError as _error:
            # EPIPE can arrive as a plain OSError depending on the layer that raises.
            if getattr(_error, "errno", None) != 32:
                raise
            self._give_up()
        return len(data) if isinstance(data, (str, bytes)) else 0

    def writelines(self, lines):
        for _line in lines:
            self.write(_line)

    def flush(self):
        if object.__getattribute__(self, "_broken"):
            return
        try:
            object.__getattribute__(self, "_stream").flush()
        except (BrokenPipeError, ValueError):
            self._give_up()
        except OSError as _error:
            if getattr(_error, "errno", None) != 32:
                raise
            self._give_up()

    # Everything else -- fileno, encoding, isatty, buffer, reconfigure -- has to keep
    # working, because rich, typer and uvicorn all interrogate the real stream.
    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_stream"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_stream"), name, value)


def install(streams = ("stdout", "stderr"), module = None):
    """Wrap sys.stdout / sys.stderr so an EPIPE cannot abort the process.

    Idempotent, and never raises: a stream guard that breaks startup would be worse
    than the failure it prevents.
    """
    target = module if module is not None else _sys
    wrapped = []
    for name in streams:
        stream = getattr(target, name, None)
        if stream is None or isinstance(stream, _PipeTolerantStream):
            continue
        try:
            setattr(target, name, _PipeTolerantStream(stream))
            wrapped.append(name)
        except Exception:
            pass
    return wrapped


def shield_stdio_from_epipe(fds = (1, 2)):
    """Guarantee that writing to fd 1 and fd 2 can never raise EPIPE or raise SIGPIPE.

    Why this exists, measured rather than guessed. The desktop app spawns the backend
    with its stdout on a pipe and reads it for the log view. When that reader goes away
    mid-startup, the next write fails and the server dies. From strace on a failing CI
    run, with the successful writes immediately before it:

        write(1, "Starting Unsloth Studio on http:"..., 49) = 49
        write(1, "2026-07-31 20:19:00 [info     ] "..., 95) = 95
        write(1, "Session log: /home/runner/.unslo"..., 88) = -1 EPIPE (Broken pipe)
        --- SIGPIPE ---
        write(4, "Session log: /home/runner/.unslo"..., 88) = 88
        exit_group(1)

    Note the second-to-last line: the same message reaches the session log on disk. The
    server had everything it needed to keep running and stopped only because a log
    consumer went away.

    Wrapping sys.stdout is not enough, and was tried: attribute access delegates
    `.buffer`, so click and typer's byte writes go straight past it, and a subprocess
    inherits the raw descriptor regardless. The guarantee has to be at the descriptor
    level, so it is: the real descriptor is duplicated aside, a fresh pipe is dup2'd over
    the original number, and a daemon thread drains that pipe and forwards to the real
    one, discarding quietly once the far end is gone. After this, fd 1 and fd 2 always
    have a reader -- this process's own thread -- so nothing written by Python, by a C
    extension, or by a child process can ever fail on them.

    Only pipes are shielded. A file or a terminal cannot EPIPE, and interposing on those
    would add a thread and a copy for nothing.
    """
    import os
    import stat
    import threading

    shielded = []
    for fd in fds:
        try:
            if not stat.S_ISFIFO(os.fstat(fd).st_mode):
                continue
        except OSError:
            continue

        # Flush first: anything sitting in Python's buffers belongs to the real
        # descriptor, and must not be re-routed halfway through a line.
        for name in ("stdout", "stderr"):
            stream = getattr(_sys, name, None)
            try:
                if stream is not None and stream.fileno() == fd:
                    stream.flush()
            except Exception:
                pass

        try:
            real_fd = os.dup(fd)
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, fd)
            os.close(write_fd)
        except OSError:
            continue

        def _relay(read_fd = read_fd, real_fd = real_fd):
            broken = False
            try:
                while True:
                    try:
                        chunk = os.read(read_fd, 65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    if broken:
                        continue      # keep draining, or writers block on a full pipe
                    try:
                        os.write(real_fd, chunk)
                    except OSError:
                        broken = True
            finally:
                for victim in (read_fd, real_fd):
                    try:
                        os.close(victim)
                    except OSError:
                        pass

        thread = threading.Thread(
            target = _relay, name = f"stdio-shield-{fd}", daemon = True,
        )
        thread.start()
        shielded.append(fd)
    return shielded
