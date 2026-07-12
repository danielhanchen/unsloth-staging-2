"""Stateful FastMCP stdio counter for the PR #7080 Studio before/after demo.

Each server process keeps its own in-memory counter. The returned PID reveals
whether Studio reused the same subprocess across tool calls (persistent, PR head)
or spawned a fresh one each call (one-shot, merge-base). Two calls in one
conversation therefore read COUNTER=1 then COUNTER=2 with the same PID AFTER the
fix, and COUNTER=1 then COUNTER=1 with different PIDs BEFORE it.
"""

import os

from fastmcp import FastMCP

mcp = FastMCP("Stateful Counter")
_count = 0


@mcp.tool
def increment() -> str:
    """Increment this server process's in-memory counter by exactly one and
    return the new value together with the server process id."""
    global _count
    _count += 1
    return f"COUNTER={_count}; PID={os.getpid()}"


if __name__ == "__main__":
    mcp.run()
