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

**1. The memory home is project-local, with an env override.** `.ta-agents/`
at the repository root is the default home; `USER.md` (the human profile,
explicit-confirm), `MEMORY.md` (agent notes), and `memory.db` (cache + the
episode record) all live there, gitignored alongside sessions and the run-log.
`TA_AGENTS_HOME` re-points the home (dedicated volume, shared host) — it is
honoured first and is the only supported way to leave the project. *(Amended
2026-09-30: the previous default was `~/.ta-agents/`; see the alternatives
entry below for why.)*

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

**6. Semantic (vector) memory is built and measured; `vector` alone is not
recommended, but `hybrid` is.** ~~After
episodes land green, choose sqlite-vec in the existing `memory.db` vs a
standalone vector DB.~~ **Done, and the answer is sqlite-vec in
`memory.db`** — decision 1 held, so there is one file and one erasure story.
Implemented behind `TA_RECALL=vector` with `memory/vector_index.py`.

The measurement rejected it as a default. Over the 30-turn / 43-query corpus in
`evals/recall.py` (`--vector`): p@1 **0.889 against lexical's 0.917**, and 210
irrelevant hits against lexical's 0. Vector's one structural advantage is real
— it never returns nothing, so it recovered the 2 of 36 queries bm25 could not
serve — but it pays for that by always returning its `k` nearest turns, so on
an off-topic question it supplies 30 past turns, none of them relevant. It also
loses two queries lexical gets right, and costs a network call per turn.

**Superseded in part.** The 210 was diagnosed as a *missing guardrail*, not a
weak model: without a relevance floor a nearest-neighbour search has no way to
express "nothing here is relevant". Two changes followed, each measured on the
same corpus:

* `MIN_RELEVANCE_SCORE` in `memory/vector_retriever.py` — calibrated, not
  guessed. The 7 questions sharing nothing with any turn scored in
  [-1.303, -1.197]; the 36 answerable ones in [-1.018, -0.234]. A floor of
  **-1.10** sits in the empty band between them. The margin is thin and the
  unrelated class is only 7 queries, so this is a value to re-calibrate
  against real usage.
* `FallbackEpisodeRetriever` in `memory/fallback_retriever.py` — lexical first,
  vector only on an empty result, behind `TA_RECALL=hybrid`.

Result: **0/36 unanswered, p@1 0.917, 0 irrelevant hits**, with zero embedding
calls on the common path. Hybrid now beats lexical on every axis instead of
trading against it. A merged ranking (RRF, then a reranker) is the better
general answer and is the next thing to measure — deliberately not built here,
because the failure it fixes is ~5% and merging re-orders *every* query whereas
the ladder changes nothing for the 34 lexical already answers.

Kept, not deleted, and that is the point: the alternative has to be *measured*
rather than assumed, and the measurement is the deliverable. Revisit if the
corpus grows past what bm25 can serve, or if a hybrid is built that applies the
recall floor only where lexical returned nothing — which is the one shape this
data actually points at.

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
- **Project-local default (`.ta-agents/`, amended 2026-09-30).** The earlier
  rationale rejected a repo-following default because the home would break
  when the repo moves or is cloned. Operating practice said otherwise: a
  memory buried in `~` was undiscoverable for the developer and unmanaged by
  the repo, so the default became `.ta-agents/` inside the repo, and the
  "repo moves / clone / host" escape hatch is now the `TA_AGENTS_HOME` override.

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
  this ADR does not decide it. **Still open, and now measured first:** recall
  ships lexical-only (`memory/fts_retriever.py`), and `evals/recall.py` is the
  harness that produces the number a semantic layer would have to beat.
- Automated pruning of stale `MEMORY.md` notes (still open, plan session
  notes §2).
- ~~Episodic recall surfaced to the assistant: "we covered this last
  Thursday" is a query over `episodes`, not a new store.~~ **Done** — as a
  query over `episodes`, with no new store, behind the seam in
  `memory/retrieval.py` and switched by `TA_RECALL=lexical` (default `off`).
  "Last Thursday" specifically is *not* delivered: the lexical retriever ranks
  by bm25 over content words, so a time-based question is the case the deferred
  vector path would serve better — and, per decision 6, the vector path was
  measured and did not solve it any better. A time-aware retriever is the real
  answer here, not a denser one.
- Mirror to `TA-memory/decisions/` when mentors accept on the shared series.

**Risks.** A cache that outlives its TTL logic would silently vend stale
graph data — the keying must always include node + rel_types, and expiry must
be checked at read. Episode rows accumulate in `memory.db` without pruning by
design; if the file grows unbounded it becomes an index we have to bound —
acceptable until the sniffed pilot shows otherwise.
**7. Recall defaults to `hybrid`, not `off` (reverses an earlier decision).**
The original choice was `off`, on the sound reasoning that a feature which
changes what the app says back to you should not surprise anyone who did not ask
for it. It is now `hybrid`, because the measurement ended the question the
decision was waiting on: the feature is measurably better than the alternative
in every column, and shipped dormant it is a feature nobody discovers.

What makes the default defensible rather than reckless is that `hybrid` is
chosen over `vector` for it. `hybrid`'s common path is byte-identical to
`lexical` and costs nothing — no network, no embedding, no index — so the
provider is only reached after keyword search has already returned nothing. On
top of that, `episode_retriever` degrades to keyword-only when there is no
embedding credential or no built index, warning once and naming the fix. So an
install that cannot use the meaning index gets lexical, not a failed request per
question.

The invariant that protects a user who never wanted recall is unchanged and now
tested as such: **a question with nothing to recall leaves the system prompt
byte-identical.** That is the promise the old `off` default was really making,
and it survives the flip.
