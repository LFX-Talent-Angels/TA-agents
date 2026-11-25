# ADR-0012: Discovery map — how a university's list of roles, tasks and skills connects

- **Status:** Proposed
- **Date:** 2026-11-25
- **Deciders:** mentee team; mentors to accept
- **Mirrors:** none yet — mirror to `TA-memory/decisions/` on mentor acceptance

## Context

A university sent a list of market roles, tasks and skills (vague, messy, with
typos and some terms no taxonomy has). It wants one map of how its items
connect to each other and what else links them (connections it had not
considered), which items connect to nothing, which are in no taxonomy, and
whether its own guesses ("A goes with B") hold in the graph.

The Graph-Mapper (ADR-0010) maps one start. This runs it over a list, merges
the maps, and classifies the result. The reason to use official taxonomies in a
knowledge graph is a proven, cited answer, so the whole feature is code: no
model call, no inferred link.

## Decision

`ta-agent discover FILE` and the MCP tool `discover_map`, in
`skills/connect/discover*.py`. It stays inside Connect (ADR-0003).

1. **Input.** `.txt` (one item per line), `.csv`/`.xlsx` (column `item`, else the
   first column; optional `kind` role|occupation|task|skill and `related_to`
   with `;`-separated guesses), `.json` (strings or objects). At most 100
   items; blank and duplicate (case-insensitive) items are dropped with a note.
   `kind` is recorded, never used to filter: suites name kinds differently
   (O*NET lists Python as `Software`).
2. **Resolve** with the mapper's own start rule (`resolve_start`, shared with
   `map_from_text`): a clear label match only (confidence >= 0.85, one hit, not
   ambiguous). Anything else is `unclear` (top 3 candidates: id, label,
   confidence, match method) or `not_found`, per suite, never guessed. A hit
   found only by meaning (Locate's `hybrid_rrf`) is never taken.
   `--nodes resolved.json` replaces resolution with a recorded node set (the
   `resolved` block of a previous `discovery.json`, or one edited by hand).
3. **Map** every resolved node with the Graph-Mapper (map-rank-v1, default depth
   2, breadth 4, `--kinds`), add the official crosswalk `same_as` edges, merge.
   Each node keeps `suite` and `cite`. `--max-nodes` is a hard cap on the merged
   map, split evenly across the resolved nodes (default 2000, up to 5000: a
   list of 25 items needs more than the 500 of a one-name map).
4. **Classify**, in code:
   - `links`: a graph edge directly between two different items' nodes (also
     `same_as`, and `same_node` when two items resolve to one node). Checked
     against each item's full neighbour list, not the breadth-cut walk.
   - `connectors`: non-item nodes reached by two or more items, most items
     first, then label, then ID; hub kinds are flagged `broad`.
   - `routes`: for pairs with no link and no shared connector, Pathfind's
     `find_routes` between the two resolved nodes ranked by route-rank-v1
     (ADR-0011), in any suite that resolves both. The user's own guess pairs are
     always routed; other pairs are capped at 30 (items with no link or
     connector yet first) and the number skipped is reported.
   - `orphans`: resolved items with no link, connector or route; each says
     whether every route to it was checked (the cap can leave some unchecked).
   - `clusters`: connected components (two or more items) over links,
     connectors and routes; `Cluster 1..n` by size, then first item.
   - `guesses`: `confirmed` (direct link), `indirect` (shared connector, or a
     route of at most 3 edges), `not_in_graph`, `unresolved` (an end is not
     resolved or not in the list).
   - `gaps`: unclear and not-found items per suite. An item is
     `not_in_any_taxonomy` when nothing resolved and no suite had a label-level
     candidate (nothing, or only meaning guesses).
5. **Outputs** in `--out` (default: the input's folder): `discovery.json`
   (`policy: discover-v1`), `discovery.xlsx` (Items, Connectors, Links, Routes,
   Orphans, Clusters, Guesses, Gaps, worded for non-engineers) and a static
   `discovery.html` (no script, no external file).

## The determinism contract

Same resolved node set + same graph data gives byte-identical `discovery.json`
(and `discovery.html`), whatever the order of the input rows.

- Items are de-duplicated and sorted case-insensitively on the way in; the kept
  spelling is the first in sorted order, not the first row.
- Every list is a total order ending in the node ID (or the item text): links,
  connectors, routes, clusters, guesses, nodes, edges, candidates.
- Nothing set- or dict-ordered reaches the output unsorted; floats are rounded
  to 4 decimals; no timestamp, run id or machine path is recorded.
- The model never writes an edge and never breaks a tie; there is no model call.
- `--nodes` rerun: resolution is the only step that reads search; recording it
  makes the rerun independent of search. Tests assert both (offline fakes and
  the live golden).

## What it is not

- Not a learning order. "Linked in the graph" is the only meaning; prerequisites
  are the university's job once it has the map.
- No inferred links: no label matching across suites, no embeddings, no
  "probably related". A `not_in_graph` verdict says the graph has no such
  connection within the checked depth, not that the guess is wrong.
- Not a recommender: connectors are shared neighbours, ranked by how many of the
  university's items reach them, not by importance.

## Consequences and known limits

- No change to `contracts/` or `TA-taxonomies`. `resolve_start` is the mapper's
  start rule factored out of `map_from_text` (behaviour unchanged, same tests).
- Cost: one resolve per item per suite, one walk per resolved node, one neighbour
  fetch per resolved node, up to 30 + guess route searches. The 25-item sample
  takes about 30 s on the local graph.
- Connectors are same-ID nodes; two same-as nodes in different suites are not
  merged into one connector.
- A route is found in one suite; items resolved in different suites only are not
  routed to each other (an official crosswalk edge is the only cross-suite link).
- Suites expose no data or model version today, so `versions` is empty until
  they do.
