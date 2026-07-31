# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""A closed stdout must not be able to kill the CLI.

D8: the desktop app gives its backend pipes for stdout and stderr. When those pipes lose
their reader, an unguarded write raises BrokenPipeError, the traceback cannot be written
to a stderr that is also gone, and the process exits 1 having printed nothing. Measured:
that shape reproduced the defect 8 times out of 8 on both ubuntu-22.04 and ubuntu-24.04,
while the same command with a file or a live reader started 8/8.
"""

import errno
import os
import subprocess
import sys
import textwrap

import pytest

from unsloth_cli import _PipeTolerantStream


class _Exploding:
    """A stream whose write/flush raise, and which records what got through."""

    def __init__(self, error):
        self.error = error
        self.written = []
        self.encoding = "utf-8"

    def write(self, data):
        if self.error is not None:
            raise self.error
        self.written.append(data)
        return len(data)

    def flush(self):
        if self.error is not None:
            raise self.error

    def fileno(self):
        return 4321

    def isatty(self):
        return False


@pytest.mark.parametrize(
    "error",
    [
        BrokenPipeError(errno.EPIPE, "Broken pipe"),
        OSError(errno.EPIPE, "Broken pipe"),
        ValueError("I/O operation on closed file"),
    ],
    ids = ["brokenpipe", "oserror-epipe", "closed-file"],
)
def test_a_dead_reader_does_not_raise(error):
    stream = _PipeTolerantStream(_Exploding(error))
    # Each of these would abort the process today.
    stream.write("first line\n")
    stream.write("second line\n")
    stream.writelines(["third\n", "fourth\n"])
    stream.flush()


def test_it_reports_the_length_it_was_given():
    """print() checks the return value; returning None there raises TypeError."""
    stream = _PipeTolerantStream(_Exploding(BrokenPipeError(errno.EPIPE, "Broken pipe")))
    assert stream.write("hello") == len("hello")


def test_an_unrelated_oserror_still_propagates():
    """Only EPIPE is swallowed. ENOSPC is a real problem and must not be hidden."""
    stream = _PipeTolerantStream(_Exploding(OSError(errno.ENOSPC, "No space left")))
    with pytest.raises(OSError) as caught:
        stream.write("x")
    assert caught.value.errno == errno.ENOSPC


def test_a_healthy_stream_is_untouched():
    underlying = _Exploding(None)
    stream = _PipeTolerantStream(underlying)
    stream.write("kept\n")
    stream.flush()
    assert underlying.written == ["kept\n"]


def test_attributes_still_delegate():
    """rich, typer and uvicorn all interrogate the real stream."""
    stream = _PipeTolerantStream(_Exploding(None))
    assert stream.fileno() == 4321
    assert stream.isatty() is False
    assert stream.encoding == "utf-8"


def test_it_latches_and_stops_calling_through():
    underlying = _Exploding(BrokenPipeError(errno.EPIPE, "Broken pipe"))
    stream = _PipeTolerantStream(underlying)
    stream.write("dies here\n")
    underlying.error = None          # the pipe "comes back"
    stream.write("after the break\n")
    # Deliberate: once the reader is gone it is gone, and re-probing a dead pipe on
    # every log line is worse than staying quiet.
    assert underlying.written == []


def test_end_to_end_a_readerless_pipe_no_longer_kills_the_process():
    """The real thing: write into a pipe whose read end is closed.

    Without the guard this exits 1 with an empty stderr, which is exactly what the
    desktop backend was doing. `head -0` is what reproduced it in CI; closing the read
    end directly is the same condition without the extra process.
    """
    program = textwrap.dedent(
        """
        import os, sys
        sys.argv[0] = "unsloth"          # the guard is entry-point behaviour
        import unsloth_cli
        assert isinstance(sys.stdout, unsloth_cli._PipeTolerantStream), "guard not installed"
        for _ in range(200):
            print("x" * 512)
            sys.stdout.flush()
        sys.stderr.write("SURVIVED\\n")
        sys.stderr.flush()
        """
    )
    read_fd, write_fd = os.pipe()
    os.close(read_fd)                    # no reader, ever
    try:
        completed = subprocess.run(
            [sys.executable, "-c", program],
            stdout = write_fd,
            stderr = subprocess.PIPE,
            timeout = 120,
        )
    finally:
        os.close(write_fd)
    assert b"SURVIVED" in completed.stderr, completed.stderr[-2000:]
    assert completed.returncode == 0, (
        f"exited {completed.returncode} writing to a reader-less pipe; "
        f"stderr: {completed.stderr[-2000:]!r}"
    )
