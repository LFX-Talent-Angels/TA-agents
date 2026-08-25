"""MCP edge: the taxonomy suite contract exposed to an external MCP client.

Thin, like `api/` — no reasoning here. The client owns the turn and writes the
answer; this package only asserts typed graph facts (ARCHITECTURE.md).
"""

from talent_angels.mcp.server import SERVER_NAME, SERVER_VERSION, build_server, main

__all__ = ["SERVER_NAME", "SERVER_VERSION", "build_server", "main"]
