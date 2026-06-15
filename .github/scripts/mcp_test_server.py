#!/usr/bin/env python3
"""Tiny MCP server for validating Unsloth Studio's MCP integration.

Exposes a single deterministic tool `add_numbers(a, b) -> {"sum": a+b}` over the
streamable-HTTP transport (the same transport Studio's fastmcp client uses when
the registered URL does not end in /sse). Run it with a Python that has
`fastmcp` installed (the Studio venv does):

    python mcp_test_server.py [PORT]      # default 9100, serves http://127.0.0.1:PORT/mcp

In the Docker lane this is launched INSIDE the container (so Studio reaches it on
container-localhost); in the native lane it runs on the host next to Studio.
"""
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9100

try:
    from fastmcp import FastMCP
except Exception as e:  # pragma: no cover
    print(f"fastmcp import failed: {e!r}", file=sys.stderr)
    sys.exit(3)

mcp = FastMCP("unsloth-test-mcp")


@mcp.tool()
def add_numbers(a: int, b: int) -> dict:
    """Add two integers and return their sum as {"sum": a + b}."""
    return {"sum": int(a) + int(b)}


if __name__ == "__main__":
    # streamable-http serves at http://<host>:<port>/mcp by default.
    mcp.run(transport="streamable-http", host="127.0.0.1", port=PORT)
