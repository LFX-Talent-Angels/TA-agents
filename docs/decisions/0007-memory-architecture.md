# ADR-0007: Agent memory — stores, cache semantics, and the episode record

- **Status:** Accepted (this repo)
- **Date:** 2026-09-23
- **Deciders:** Aman (driving) + mentee team; mentors review on the shared series
- **Mirrors:** `TA-memory/decisions/0007-memory-architecture.md` (copy on mentor acceptance)

## Context

Memory phase 1 (session state plus `USER.md`/`MEMORY.md`) shipped. The next
phases need durable, queryable, typed records beyond two plain-text files, and
the assistant must not repeat work it has already done.

Three forces are in play. **One home** — memory must live in a single,
predictable place so every process (TUI, API, tests) finds the same data.
**Typed records** — the assistant's internal contract is typed `AgentResult`s;
memory that stores prose or opaque blobs breaks the coupling rule
(ADR-0004) and the "evidence is a pointer" principle. **Freshness vs cost** —
re-querying Neo4j for the same `get_neighbors` on every turn is wasted
round-trips, but a stale cached answer is worse than none.

Three designs competed as the base: the file-based phase-1 stack, a full
SQLite memory implementation on a local branch, and a mem0-backed
`MemoryClient` on another. What matters is the durable standard, not which
branch carried the code.

## Decision

**The agent's memory is a single directory under the user's home, and its
facts are SQLite-backed, typed, and node-ID-only.** Concretely:

**1. `~/.ta-agents/` is the only memory home.** `USER.md` (the human profile,
explicit-confirm), `MEMORY.md` (agent notes), and `memory.db` (cache + the
episode record) all live there. No second default (`TA_MEMORY_DIR`,
`data/local/memory`) exists; only explicit env override for testing points
elsewhere.

**2. `memory.db` (SQLite) is the queryable store.** Two tables of records:

- **`neighbor_cache`** — the result of a `get_neighbors(node, rel_types)`
  call, so Locate→Connect chains reuse prior graph answers. Every row is
  keyed by `(node_id, rel_types)` and **expires after 24 hours**; an expired
  row is a miss, never a stale hit. The cache is a pointer with a TTL, not a
  snapshot contract — it is correct to drop it, never to vend it as fresh.
- **`episodes` / `episode_nodes`** — one typed record per turn: what was
  asked, which suite and plan ran, which node IDs were cited, and whether the
  turn **satisfied** (landed on cited nodes without a `not_found`/warning
  outcome). Records reference nodes **by suite-scoped ID only** — no
  descriptions, no model text, no payload (licensing + ADR-0004).

**3. Every persisted turn writes exactly one episode.** The episode record
fires beside the existing run-log append — one code path, no opt-in hole. A
turn that produced no cited nodes is still an episode; its `satisfied` flag
is `false`, and that is honest signal, not an error row.

**4. Explicit-confirm writes stay the rule.** Machine-authored data goes to
`memory.db` (episodes, cache) or `MEMORY.md` (agent notes). The human profile
`USER.md` is only ever written from user-confirmed events. The episode index
derives from what the user *did*, never from what the machine assumes the user
*wants*.

**5. Episodes are the seeds of the crowdsourced graph.** Node-ID-only records
aggregate cleanly later into shared knowledge (per plan session notes §4)
without re-licensing taxonomy content — because they store pointers, not
payloads.

**6. Semantic (vector) memory is deferred, not decided.** After episodes land
green, choose sqlite-vec in the existing `memory.db` vs a standalone vector
DB. Default inclination: sqlite-vec, to keep `memory.db` the single store and
honor decision 1.

## Alternatives considered

- **mem0 `MemoryClient` (add/search protocol).** Pros: battle-tested memory
  product, easy semantic search later. Cons: external dependency / self-hosted
  service, opaque internal representation, contradicts the deterministic,
  typed, node-ID-only record design. **Parked** as a documented future option.
- **File-only (no SQLite).** Pros: nothing new. Cons: cannot answer "which
  turns cited node X", no cache TTL discipline, files grow unbounded.
  **Rejected** — cache and episodes require queryable storage.
- **Rewrite the SQLite layer from scratch.** Pros: clean history. Cons:
  duplicates finished, notebook-backed work. **Rejected** — adopt, review, and
  reconcile instead.
- **`data/local/memory` as default.** Cons: memory follows the repo, not the
  user; breaks when the repo moves or is cloned. **Rejected** —
  `~/.ta-agents/` matches phase-1 files and survives clones (hosting plan
  ADR-0005).

## Consequences

**Easier.** One store with a single path; cache round-trips drop; "do we know
this node already?" becomes a SQL query; satisfied-detection gives the team a
real (not bookkeeping) success signal; node-ID-only records aggregate into the
crowdsourced-graph idea without licensing pain.

**Harder.** Episodes only reflect what a turn cited — recall quality is bounded
by graph traversal, not by whatever the model happened to say. The cache must
not drift: 24h TTL means the *reuse window* is short, and any tester must
remember the cache masks fresh Neo4j round-trips until it expires.

**Follow-ups.**

- Semantic-layer decision (sqlite-vec vs Chroma) after episodes are green —
  this ADR does not decide it.
- Automated pruning of stale `MEMORY.md` notes (still open, plan session
  notes §2).
- Episodic recall surfaced to the assistant: "we covered this last Thursday"
  is a query over `episodes`, not a new store.
- Mirror to `TA-memory/decisions/` when mentors accept on the shared series.

**Risks.** A cache that outlives its TTL logic would silently vend stale
graph data — the keying must always include node + rel_types, and expiry must
be checked at read. Episode rows accumulate in `memory.db` without pruning by
design; if the file grows unbounded it becomes an index we have to bound —
acceptable until the sniffed pilot shows otherwise.