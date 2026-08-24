# `ta-mcp` — the taxonomy graph as an MCP server

Exposes the taxonomy **suite contract** — `search_nodes`, `get_neighbors`,
`enumerate_paths`, `score_paths` — over stdio, so an MCP client (Claude Code)
can walk the graph itself, on the user's own subscription. No LLM call is made
by this process; it spends no project tokens.

The client plays the part the main assistant plays inside this repo: it owns
the goal and writes the answer. This edge only asserts typed graph facts —
IDs, labels, relationship types, confidence, warnings — and never prose, not
even on the failure paths. See `ARCHITECTURE.md` (rules #2 and #8).

## What the tools return

Every payload is structured JSON, validated by the Pydantic models in
`models.py`. Two things a client must not misread, stated in the server
instructions and again in each tool description:

- **IDs are suite-scoped.** `esco:occupation:<uuid>` is meaningful only inside
  ESCO. Nothing here equates IDs across suites, and neither should the client.
- **`confidence` says *how* the match was made**, not how likely it is to be
  right. It is one fixed value per method (ESCO: `0.95` exact preferred label,
  `0.90` exact alternative label, `0.85` case-insensitive unique, `0.80`
  case-insensitive ambiguous, `0.70` substring only). `0.70` does not mean
  "70% sure"; it means only a substring matched.

`warnings` is machine-readable and load-bearing: `not_found`, `ambiguous`,
`no_neighbors`, `node_not_found`, `no_path`, `endpoint_not_found`,
`unknown_suite:<name>`, `suite_unavailable:<ErrorType>`,
`capability_unavailable:<tool>:<suite>`.

## Requirements

- The graph must be loaded and reachable — `NEO4J_URI`, `NEO4J_USER`,
  `NEO4J_PASSWORD`, read from `.env` at the repo root (see `.env.example`) or
  from the environment. The server is **read-only**.
- The concrete `ta-taxonomies` package installed in the same environment.
- Suites are opened on the first tool call, not at startup: a database that is
  down yields `suite_unavailable:<ErrorType>` on one call and can recover on
  the next, instead of a server that refuses to launch.

## Register it with Claude Code

Use the absolute path to the console script in this repo's virtualenv.

```json
{
  "mcpServers": {
    "ta-taxonomy": {
      "command": "/absolute/path/to/TA-agents/.venv/bin/ta-mcp",
      "args": [],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "<your local password>"
      }
    }
  }
}
```

Put that in `.mcp.json` at the repo root to share it with the project, or in
`~/.claude.json` under the project entry to keep it to yourself. Drop the
`env` block if the repo-root `.env` already carries those values — the server
loads it, exactly as the CLI does.

Equivalent one-liner:

```bash
claude mcp add ta-taxonomy -s local -- /absolute/path/to/TA-agents/.venv/bin/ta-mcp
```

Check it with `/mcp` inside Claude Code.

## Layout

| File | Holds |
| ---- | ----- |
| `server.py` | The four tools, their descriptions, suite lifespan |
| `models.py` | Typed payloads returned to the client |
| `mapping.py` | Suite `ToolResult` → those payloads; nothing interprets |
| `protocol.py` | The contract surface, structurally — no concrete import |
| `session.py` | Lazy suite opening, one driver per suite per process |
