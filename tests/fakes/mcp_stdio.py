"""Run the MCP edge over the fake suite, on stdio, as a child process.

The tool bodies only exist inside the server, so the honest way to test them
is to speak the protocol to a real one. This module is that server: same
`build_server` the console script uses, fake registry instead of Neo4j.
"""

from __future__ import annotations

from talent_angels.mcp import build_server
from tests.fakes.suite import fake_registry


def main() -> None:
    build_server(registry=fake_registry()).run(transport="stdio")


if __name__ == "__main__":
    main()
