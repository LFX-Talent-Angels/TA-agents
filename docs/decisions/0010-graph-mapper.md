# ADR-0010: Graph-Mapper — a multi-hop Connect

- **Status:** Proposed
- **Date:** 2026-11-24
- **Deciders:** mentee team; mentors to accept
- **Mirrors:** none yet — mirror to `TA-memory/decisions/` on mentor acceptance

## Context

The mentors asked for a Graph-Mapper: "This Graph-Mapper receives three values: a
skill, with depth and breadth of subsequent search. For each level it gets the
specified breadth of next skills that can be reached/learned, and repeats until
it reaches the required depth of search." The use case is Enmanuel Amaya's
university project, which needs a small map of what surrounds a skill.

Connect (ADR-0003) is single-hop today: one call, the direct neighbors of one
node. Repeating it is mechanical, but "which N neighbors?" needs a fixed rule.

## Decision

1. Graph-Mapper is a **multi-hop Connect**. It stays inside ADR-0003's four
   capabilities and lives in `skills/connect/`. One hop is one `get_neighbors` call.
2. Inputs: a start (node ID or name via the Locator), `depth` 1-4, `breadth`
   1-10, optional `kinds` filter, `max_nodes` 500. One suite per call; no
   cross-suite links without a crosswalk.
3. Per expanded node, neighbors are ranked and the top `breadth` not yet mapped
   are kept. Ranking, lowest first: relation (related, narrower, level, parent,
   needs/needed by), then strength (essential, optional, untyped), then O*NET
   importance, hot technology, label, ID. **Relation comes before strength**
   because strength-first filled Python's map with jobs, since every job link is
   essential or optional (live check, 2026-11-24).
4. Hub kinds (skill groups, SFIA levels and categories, ISCO/SSYK groups,
   headlines) are shown but **never expanded**. This stops drift from Python to
   Maltego through a skill group.
5. The output means "linked in the graph; not a learning order". Learning order
   is the Evaluator's job.
6. **No LLM.** Same input, same bytes. Everything cut is reported per node
   (`considered`/`kept`) and `max_nodes` hits are a warning, never silent.
7. Expectations per suite: ESCO best; SFIA gives the skill's own levels (2-6
   for Programming/software development); JobTech works (Swedish labels); O*NET
   weak (the same 35 skills appear everywhere); BLS lists only broad skills on
   its jobs (Adaptability, Detail oriented, ...), so its maps say little about
   what to learn.

## Consequences

- No change to `contracts/` or `TA-taxonomies`; `NodeRef`/`EdgeRef` are reused.
- Edges are exposed as plain-language relations, not raw edge names.
- Deep maps cost one query per expanded node; a single-query version can follow
  if that proves slow.
- A start given as free text that is ambiguous returns candidates, not a guess.
- Not in scope: ranking by "worth learning next", cross-suite links beyond the official
  crosswalks below, chat.

## All-taxonomy maps

`ta-agent map NAME` with no `--suite` (or `map_neighbors` with no `suite`) maps
the name in **every** taxonomy and merges the results; `--suites a,b` picks a
subset, `--suite a` keeps the single-suite output.

- Each suite resolves the text on its own: mapped, `ambiguous` (pick list, never
  a guess), `not_found`, `via_crosswalk` or `error` (one failing suite never
  hides the others). The node cap is a hard cap for the whole map, split evenly
  across the selected suites (`max_nodes // suites`, at least 1 each).
- Every node keeps its suite-scoped ID and gains `suite` and `cite` (its
  `source_id`, the evidence pointer). Edge keys are `from_id`/`to_id` in every
  output.
- **Crosswalk rules (official codes only):** BLS and O*NET by SOC code
  (`bls:occupation:15-1252` with `onet:occupation:15-1252.00`; only the `.00`
  O*NET code is the same occupation). BLS's `onet_soc_codes` is not used: it
  lists the O*NET jobs BLS borrowed skill scores from, not equivalents
  (Engineers, All Other -> Aerospace Engineers). JobTech to ESCO by `esco_exact` and `esco_close` URIs
  (`http://data.europa.eu/esco/<kind>/<uuid>` is `esco:<kind>:<uuid>`). Links
  are `CROSSWALK` edges (`same_as`, strength `exact`/`close`/`soc`) between
  nodes already in the map; both ends get `confirmed_by`. A suite that cannot
  resolve the text may start from a crosswalk of a start another suite resolved.
  A node ID belongs to one taxonomy: the others are not searched with its text
  (`not_searched`) and may only start from a crosswalk. A name is taken only on a
  label match (Locate confidence >= 0.85); a lone substring or meaning hit is
  shown as a candidate, never picked.
- **Islands:** ESCO + JobTech are linked, O*NET + BLS are linked, SFIA stands
  alone. ESCO to O*NET has no official code, so it is not linked. Meaning-based
  links (same label, embeddings) are deliberately not included.
- The crosswalk rules (`skills/connect/crosswalk.py`) are a **stopgap** in
  TA-agents: TA-taxonomies `crosswalks/` is the intended home ("nothing
  cross-suite lives anywhere else") and they move there once it is filled.
- Route mode (`--route FROM TO`) runs Pathfind in each suite, takes the
  top-ranked route (route-rank-v1) and maps around it; the entry records the
  route's node IDs, score and reasons.
