# Interfaces: every capability, every way in

Everything Talent Angels does is reachable from the command line, the chat, MCP and HTTP, so
you can build on it. All four call the same core functions; there is one implementation per
capability, declared once in the engine registry. No path below calls a model, except `tag`
without `--no-llm`. The same input and the
same graph data give the same JSON, byte for byte (no timestamps).

| Capability | CLI | Chat (slash) | MCP tool | HTTP API |
| --- | --- | --- | --- | --- |
| **map**: the neighbourhood of a skill or job, or around a route, in every taxonomy | `ta-agent map "Python" --depth 2` | `/map Python --depth 2` | `map_neighbors` | `POST /v1/capabilities/map` |
| **discover**: how a list of roles and skills connects | `ta-agent discover items.csv` | `/discover items.csv` | `discover_map` | `POST /v1/capabilities/discover` |
| **roadmap**: what to learn from job A to job B, in order (`learn-order-v1`) | `ta-agent roadmap "data analyst" "data scientist"` | `/roadmap data analyst \| data scientist` | `roadmap` | `POST /v1/capabilities/roadmap` |
| **successors**: what comes after a skill (`learn-order-v1`) | `ta-agent successors "Python (computer programming)"` | `/successors Python (computer programming)` | `successors` | `POST /v1/capabilities/successors` |
| **pathfind**: skill gap and ranked routes between two jobs | chat or API; `ta-agent map --route A B` maps around the best route | "path from nurse to data analyst" (the path card) | `pathfind` | `POST /v1/capabilities/pathfind` |
| **tag**: batch-tag documents against the taxonomies | `ta-agent tag project/ --no-llm` | `/tag project/ --no-llm` | `start_job`, `job_status`, `cancel_job` | `POST /v1/jobs` (needs `TA_JOBS_ENABLED=1`; see [Jobs](#jobs-tag)) |
| **locate**, **connect**: resolve text to nodes, one hop around a node | `ta-agent locate`, `connect` (chat turns) | plain questions | `search_nodes`, `get_neighbors` | `POST /v1/run` (the suite calls) |

`tag` is a job: it runs for minutes and may call a model, so over HTTP and MCP it starts in the
background and you poll it.

## The engine front door

One registry (`talent_angels.engine`) declares the eight capabilities: their input and output
models, whether they are deterministic, whether they use a model, what they cost, and their name
on each interface. The generic surfaces are built from it; the per-capability ones above keep
working unchanged (a test asserts that every alias in the registry exists on every interface).

| What | CLI | MCP | HTTP |
| --- | --- | --- | --- |
| The manifest: what can I do? | `ta-agent capabilities [--json]` | `capabilities` | `GET /v1/capabilities` |
| Run any capability by name | `ta-agent run NAME '{...}'` (or `@file.json`) | `run` | `POST /v1/run` |
| Long runs (`tag`) | `ta-agent tag` (foreground) | `start_job`, `job_status`, `cancel_job` | `POST /v1/jobs`, `GET /v1/jobs/{id}`, `POST /v1/jobs/{id}/cancel` |

**Closed action set.** Only a registered capability name (or an MCP tool name such as
`map_neighbors`) runs, and its input must validate against that capability's model: unknown
fields are refused, limits are enforced, and nothing is a free-form tool, a query or a write.
No capability calls a model, except `tag` without `no_llm`. The manifest says so per capability
(`deterministic`, `uses_model`, `cost`, `latency`).

`run` always answers with one envelope, `{status, call, output, warnings, candidates, error, errors}`.
`output` is the capability's own JSON (the same bytes as the dedicated route), `call` is the
resolved call with every default filled (rerun it for the same bytes), and `error` is one line
that never carries a provider message (an unexpected failure shows its type only).

| `status` | Meaning | HTTP | `ta-agent run` exit |
| --- | --- | --- | --- |
| `ok` | at least one taxonomy answered | 200 | 0 |
| `ambiguous` | nobody answered; `candidates` has the pick list per taxonomy | 200 | 2 |
| `no_result` | nobody answered: not found, no order data, no route, same ends | 200 | 2 |
| `unavailable` | every selected taxonomy is down (`output` still says which) | 200 | 2 |
| `invalid_input` | the input failed validation (`errors` lists the fields) | 422 | 2 |
| `unknown_suite` | a suite name is not registered | 404 | 2 |
| `unknown_capability` | not a registered name | 404 | 2 |
| `use_job` | `tag` called synchronously over HTTP/MCP: start a job | 409 | n/a |
| `cancelled` | a job was cancelled | n/a | 1 |
| `error` | unexpected failure; `error` holds the type name only | 500 | 1 |

```bash
curl -s localhost:8000/v1/capabilities                       # the manifest, with JSON Schemas
curl -s localhost:8000/v1/run -H 'content-type: application/json' \
  -d '{"capability": "roadmap", "input": {"from": "data analyst", "to": "data scientist"}}'
ta-agent run roadmap '{"from": "data analyst", "to": "data scientist", "per_step": 5}'
```

The existing `POST /v1/capabilities/{pathfind,map,discover,roadmap,successors}` routes now call
the same engine and return the same JSON as before.

### Jobs (`tag`)

The HTTP API has no auth, so **starting a job over HTTP is off by default**: `POST /v1/jobs`
answers `403` until the operator sets `TA_JOBS_ENABLED=1` on the server. `GET /v1/jobs/{id}` and
`POST /v1/jobs/{id}/cancel` stay open. MCP is a local stdio process and is not gated.

```bash
curl -s localhost:8000/v1/jobs -H 'content-type: application/json' \
  -d '{"capability": "tag", "input": {"path": "uca-pilot", "limit": 20}}'     # 202 + the job state
curl -s localhost:8000/v1/jobs/<id>                                          # poll: status, progress, result
curl -s -X POST localhost:8000/v1/jobs/<id>/cancel
```

- `path` is **relative to the tag root**: `TA_TAG_ROOT`, else `<TA home>/tag`. Absolute paths,
  `..` and symlinks that leave the root are refused (`422`). There is no upload: put the project
  folder on the server's disk under the root. The workbooks are written inside the project
  folder, as `ta-agent tag` does, and listed in `result.output.outputs`.
- `no_llm` is `true` by default here (the CLI's default is the model). A model run
  (`"no_llm": false`) is refused unless the server sets `TA_JOBS_ALLOW_MODEL=1`, because the HTTP
  API has no auth. With the model there is no cost estimate and no budget cap: try `limit` first.
- The root check runs twice: at start, and again inside the worker just before the tag runner
  gets the path. If a link was swapped in between, the job fails with `path_outside_root`.
- State lives in `<TA home>/jobs/<id>.json` (`running`, `done`, `failed`, `cancelled`,
  `interrupted` when the process died, `not_found`). One job runs at a time per server process.
- Cancel is cooperative: the run stops at its next progress tick (an in-flight model call
  finishes first). The ledger is append-only, so a rerun resumes where it stopped.

## Chat commands

Type them in `ta-agent`. Each prints a few lines (counts, the top items) and **writes the full
result to files**, printing their paths:

```
<TA home>/outputs/<command>-<slug>-<hash>/
```

`<TA home>` is `TA_AGENTS_HOME`, else `<repo>/.ta-agents` in a checkout (git-ignored), else
`~/.ta-agents`. The folder name comes from the command line, so the same command writes the same
folder and the same bytes; a rerun overwrites only its own files. `map`, `roadmap` and
`successors` write `<command>.json` and `<command>.html` (a static page to open in a
browser: light and dark, no script); `discover` writes `discovery.json`, `.xlsx` and `.html`;
`tag` writes its workbook where the CLI does (inside the project folder).

These commands are not chat turns: they run without the 45 s turn clock (a `/tag` or `/discover`
can take minutes), make no model call (except `/tag` without `--no-llm`) and add nothing to the
run-log. `/map`, `/roadmap` and `/successors` run through the engine, so a name over 1000
characters is now refused, as over HTTP. Bad input gives one
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

On the command line, `--html PATH` on `map`, `roadmap` and `successors` also writes that page
(the JSON still prints); `discover` always writes `discovery.html`. HTTP and MCP stay JSON-only.
Press Esc during a chat `/discover` or `/tag` to stop it: it prints "Stopped. Nothing was
written." (`/tag` keeps its ledger of finished units but never overwrites the workbook).

## HTTP API

```bash
uvicorn talent_angels.api.app:app          # then http://127.0.0.1:8000/docs
```

Request bodies are validated (`422` on a bad or missing field, an unknown field, or a limit
out of range); an unknown suite name is `404`, as on `/v1/query`. (`POST /v1/run` returns the
status envelope above with the same codes.) A taxonomy that is down or
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
`pathfind`, and the engine tools `capabilities`, `run`, `start_job`, `job_status` and `cancel_job`. Example call (`pathfind`):

```json
{"start": "nurse", "target": "data analyst", "suites": ["esco", "onet"]}
```

Names are resolved per taxonomy with a clear match only. An unclear name makes that taxonomy
`ambiguous` with a `side` and `candidates`; ask the user and call again with the exact title.
