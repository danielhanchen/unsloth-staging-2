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
