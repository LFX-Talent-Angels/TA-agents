# Interfaces: every capability, every way in

Everything Talent Angels does is reachable from the command line, the chat, MCP and HTTP, so
you can build on it. All four call the same core functions; there is one implementation per
capability. No path below calls a model, except `tag` without `--no-llm`. The same input and the
same graph data give the same JSON, byte for byte (no timestamps).

| Capability | CLI | Chat (slash) | MCP tool | HTTP API |
| --- | --- | --- | --- | --- |
| **map**: the neighbourhood of a skill or job, or around a route, in every taxonomy | `ta-agent map "Python" --depth 2` | `/map Python --depth 2` | `map_neighbors` | `POST /v1/capabilities/map` |
| **discover**: how a list of roles and skills connects | `ta-agent discover items.csv` | `/discover items.csv` | `discover_map` | `POST /v1/capabilities/discover` |
| **roadmap**: what to learn from job A to job B, in order (`learn-order-v1`) | `ta-agent roadmap "data analyst" "data scientist"` | `/roadmap data analyst \| data scientist` | `roadmap` | `POST /v1/capabilities/roadmap` |
| **successors**: what comes after a skill (`learn-order-v1`) | `ta-agent successors "Python (computer programming)"` | `/successors Python (computer programming)` | `successors` | `POST /v1/capabilities/successors` |
| **pathfind**: skill gap and ranked routes between two jobs | chat or API; `ta-agent map --route A B` maps around the best route | "path from nurse to data analyst" (the path card) | `pathfind` | `POST /v1/capabilities/pathfind` |
| **tag**: batch-tag documents against the taxonomies | `ta-agent tag project/ --no-llm` | `/tag project/ --no-llm` | none, by design | none, by design |

`tag` is not on MCP or HTTP: it is a batch over files that runs for minutes and calls a model.
Run it from the CLI or the chat.

## Chat commands

Type them in `ta-agent`. Each prints a few lines (counts, the top items) and **writes the full
result to files**, printing their paths:

```
<TA home>/outputs/<command>-<slug>-<hash>/
```

`<TA home>` is `TA_AGENTS_HOME`, else `<repo>/.ta-agents` in a checkout (git-ignored), else
`~/.ta-agents`. The folder name comes from the command line, so the same command writes the same
folder and the same bytes; a rerun overwrites only its own files. `map`, `roadmap` and
`successors` write `<command>.json`; `discover` writes `discovery.json`, `.xlsx` and `.html`;
`tag` writes its workbook where the CLI does (inside the project folder).

These commands are not chat turns: they run without the 45 s turn clock (a `/tag` or `/discover`
can take minutes), make no model call and add nothing to the run-log. Bad input gives one
line, never a traceback.

| Command | Syntax |
| --- | --- |
| `/map` | `/map NAME [--depth 1..4] [--breadth 1..10] [--suites a,b] [--kinds k1,k2] [--max-nodes N]`<br>`/map --route FROM \| TO [same flags]` |
| `/discover` | `/discover FILE [--depth --breadth --suites --kinds --max-nodes --nodes discovery.json]` (`.txt`, `.csv`, `.xlsx`, `.json`; at most 100 items) |
| `/roadmap` | `/roadmap FROM \| TO [--suites a,b] [--per-step 1..50]` |
| `/successors` | `/successors SKILL [--suites a,b] [--depth --breadth --max-nodes]` |
| `/tag` | `/tag PATH [--no-llm] [--limit N] [--rows 66-91] [--suites a,b] [--depth leaves\|headings\|all] [--profile P] [--refresh \| --fresh] [--workers N]` |

**Two names are separated by `|`** (a pipe), always with the route flag or command that needs two:
`/roadmap data analyst | data scientist`, `/map --route web developer | software developer`.
A pipe is the one character that cannot be part of a job title. Names can be plain text (no
quotes needed) or node IDs; wrap a path with spaces in quotes.

## HTTP API

```bash
uvicorn talent_angels.api.app:app          # then http://127.0.0.1:8000/docs
```

Request bodies are validated (`422` on a bad or missing field, an unknown field, or a limit
out of range); an unknown suite name is `404`, as on `/v1/query`. A taxonomy that is down or
cannot answer does not fail the call: its entry in `suites` carries `status` (`error:<Type>`,
`ambiguous`, `not_found`, ...) and the others still answer. Responses are the CLI's JSON.

```bash
# pathfind: skill gap and ranked routes, per taxonomy
curl -s localhost:8000/v1/capabilities/pathfind -H 'content-type: application/json' \
  -d '{"from": "nurse", "to": "data analyst", "suites": ["esco"]}'

# map: one start (or route_from + route_to); depth 1..4, breadth 1..10, kinds, max_nodes
curl -s localhost:8000/v1/capabilities/map -H 'content-type: application/json' \
  -d '{"start": "Python", "depth": 2, "breadth": 4, "suites": ["esco", "onet"]}'
curl -s localhost:8000/v1/capabilities/map -H 'content-type: application/json' \
  -d '{"route_from": "web developer", "route_to": "software developer"}'

# discover: items (kind and related_to optional), or "nodes": a previous result's "resolved"
curl -s localhost:8000/v1/capabilities/discover -H 'content-type: application/json' \
  -d '{"items": [{"item": "web developer", "kind": "role", "related_to": ["JavaScript"]},
                 {"item": "data analyst"}], "depth": 2}'

# roadmap: from, to, suites, per_step (1..50)
curl -s localhost:8000/v1/capabilities/roadmap -H 'content-type: application/json' \
  -d '{"from": "data analyst", "to": "data scientist", "per_step": 5}'

# successors: skill, suites, depth, breadth, max_nodes
curl -s localhost:8000/v1/capabilities/successors -H 'content-type: application/json' \
  -d '{"skill": "Python (computer programming)", "depth": 2, "breadth": 4}'
```

`pathfind` returns, per taxonomy: `status`, `start`, `target`, `confidence`, `gap`
(`essential`, `optional`, `software`: `{id, label}` each), `gap_too_large`, `routes` (`node_ids`,
`labels`, `score` under `route-rank-v1`) and `learning_order`. It is the data of the chat path
card without the phrasing. The conversational endpoints (`/v1/query`, `/v1/capabilities/locate`,
`/v1/capabilities/connect`, sessions, `/v1/health`) are described in the [README](../README.md).

## MCP

`ta-mcp` (see [`src/talent_angels/mcp/README.md`](../src/talent_angels/mcp/README.md)) serves the
four suite-contract tools plus `map_neighbors`, `discover_map`, `roadmap`, `successors` and
`pathfind`. Example call (`pathfind`):

```json
{"start": "nurse", "target": "data analyst", "suites": ["esco", "onet"]}
```

Names are resolved per taxonomy with a clear match only. An unclear name makes that taxonomy
`ambiguous` with a `side` and `candidates`; ask the user and call again with the exact title.
