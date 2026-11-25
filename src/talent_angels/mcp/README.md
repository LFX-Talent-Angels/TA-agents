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

`map_neighbors` is the one tool that crosses suites. With no `suite` it maps the
name in every taxonomy (`suites` picks a subset; `route_from`/`route_to` map the
top-ranked route in each). Every node keeps its suite-scoped ID and carries
`suite` and `cite`; a `CROSSWALK` edge and `confirmed_by` appear only where an
official code links two nodes (BLS and O*NET by SOC code, JobTech and ESCO by
their ESCO URIs). Names are never matched across suites.

`discover_map` takes a list of up to 100 items (and optional `related_to` pairs, the
user's guesses) and returns the `discover-v1` map: how the items link directly, the
connectors they share, routes, orphans, clusters, a verdict per guess and the gaps
(unclear or not found, per taxonomy). Same rules as `map_neighbors` (a clear label match
only, never a guess), no model call, same bytes for the same matches. Pass its `resolved`
block back as `nodes` to skip resolution. See `docs/decisions/0012-discovery-map.md`.

`roadmap` (`start`, `target`, optional `suites`, `per_step`) and `successors` (`start`, optional
`suites`, `depth`, `breadth`, `max_nodes`) return the `learn-order-v1` JSON: what to learn from one
occupation to another, in four steps, and the tree of what comes after a skill. Each item or edge
says whether its place is `stated` by a taxonomy (ESCO essential-for, the next SFIA level) or comes
from the policy; the result prints that meaning. Names resolve as in `map_neighbors` (a clear match
only; else `ambiguous` with `candidates`), a taxonomy without order data is `no_order_data`, and bad
input is an `invalid_input:` warning. No model call, same bytes for the same data. See
`docs/decisions/0013-learning-order-policy.md`.

`pathfind` (`start`, `target`, optional `suites`) returns, per taxonomy, the skill gap
(`essential`, `optional`, `software`), up to several ranked bridge `routes` (`route-rank-v1`
`score`) and the compact `learning_order`: the data of the chat path card, without phrasing. An
unclear end is `ambiguous` with a `side` and `candidates`, never guessed. No model call, same bytes
for the same data. See `docs/interfaces.md`.

The engine front door adds five tools (built from the capability registry, `talent_angels.engine`):
`capabilities` (the manifest: name, question, input and output JSON Schema, cost, determinism,
`uses_model`, examples; pass `name` for one entry), `run` (`capability`, `input`: any capability by
name or by its tool name, e.g. `map_neighbors`; returns a `Result` with `status`, `output`,
`warnings`, `candidates`, the resolved `call`; only registered names run, unknown fields are
refused, no model call), and `start_job` / `job_status` / `cancel_job` for `tag` (batch document
tagging), which runs for minutes. A job reads `path` under the server's tag root (`TA_TAG_ROOT`,
default `<TA home>/tag`; set it in the client's MCP config to the projects folder), searches without a
model unless the server sets `TA_JOBS_ALLOW_MODEL=1`, and keeps its state in
`<TA home>/jobs/<id>.json`. One job runs at a time per server process.

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
      "args": []
    }
  }
}
```

Keep credentials in the gitignored repo-root `.env` or the calling process
environment; the server loads them exactly as the CLI does. `.mcp.json` is
gitignored too, so it is safe for a per-user local command configuration, but
do not put passwords in it. To share a setup, share this password-free example
and let each developer configure their own credentials.

Equivalent one-liner:

```bash
claude mcp add ta-taxonomy -s local -- /absolute/path/to/TA-agents/.venv/bin/ta-mcp
```

Check it with `/mcp` inside Claude Code.

## Layout

| File | Holds |
| ---- | ----- |
| `server.py` | The four contract tools plus `map_neighbors`, `discover_map`, `roadmap`, `successors` and `pathfind`, their descriptions, suite lifespan |
| `engine_tools.py` | `capabilities`, `run`, `start_job`, `job_status`, `cancel_job` (the engine front door) |
| `models.py` | Typed payloads returned to the client |
| `mapping.py` | Suite `ToolResult` → those payloads; nothing interprets |
| `protocol.py` | The contract surface, structurally — no concrete import |
| `session.py` | Lazy suite opening, one driver per suite per process |
