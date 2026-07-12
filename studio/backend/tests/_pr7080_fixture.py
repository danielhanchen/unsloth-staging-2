"""Minimal real FastMCP stdio server for the PR #7080 cross-OS test."""
import os
from fastmcp import FastMCP

mcp = FastMCP("XPlatFixture")
_n = 0


@mcp.tool
def counter() -> dict:
    global _n
    _n += 1
    return {"counter": _n, "pid": os.getpid()}


@mcp.tool
def crash_mid() -> dict:
    os._exit(24)


if __name__ == "__main__":
    mcp.run()
