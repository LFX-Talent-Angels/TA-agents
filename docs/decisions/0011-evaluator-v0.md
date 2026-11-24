# ADR-0011: Evaluator v0 — deterministic ranking under named policies

- **Status:** Proposed
- **Date:** 2026-11-24
- **Deciders:** mentee team; mentors to accept
- **Mirrors:** none yet — mirror to `TA-memory/decisions/` on mentor acceptance

## Context

ADR-0003 names the Evaluator as the capability that ranks under an explicit
policy. Until now Pathfind returned routes in `enumerate_paths` order and the
Graph-Mapper broke ties A–Z (a specialist nurse mapped to "accept own
accountability", "adapt leadership styles..."). The suite contract's
`score_paths` is an unfinished stub upstream, so v0 lives in TA-agents
(`skills/evaluate/`) and uses only graph data. No LLM; same input, same bytes.

## Decision

Two policies, each a frozen dataclass with a name. Evaluator reads routes and
candidates; it never changes a route.

1. **route-rank-v1** (Pathfind routes). `score = 0.5*strength + 0.35*specificity
   + 0.15*length`, rounded to 4 decimals.
   - strength: mean over route edges of essential 1.0, optional or transferable
     0.5, untyped 0.25.
   - specificity: mean over the route's inner (bridge) nodes of
     `1/log2(2 + degree)`; 1.0 when there is no bridge.
   - length: `1 / number_of_edges`.
   - Order: score descending, then route labels, then node ids. Pathfind
     reorders its routes and adds the warnings `ranked:route-rank-v1` and
     `route_scores:<s1>,<s2>,...` (the contract has no score field). The chat
     card shows `Route N (score)`.
2. **map-rank-v1** (Graph-Mapper). From the candidates not yet mapped, take a
   pool of `min(5*breadth, 30)`, look up each member's degree, and sort by
   (relation, strength, importance, hot flag, `-specificity`, label, id); keep
   `breadth`. At most 2000 degree lookups per walk (hubs are never looked up);
   after that the rest keep their old order and the map carries `rank_budget:2000`.
   Output carries `"policy": "map-rank-v1"`.

**Why degree is specificity.** A node linked to thousands of items ("adapt to
change") says little about where it sits; one linked to a few says more. Degree
is in the graph, deterministic, and needs no model. It is a proxy, not a truth.

## What it is not

Not a learning order, not personal fit, and not a statement about quality. It
orders what the graph links; the map's `meaning` stays "linked in the graph; not
a learning order". Personal weights belong to a later policy.

## Versions

A change to any weight, formula, pool or budget is a new policy name
(`route-rank-v2`), never a silent edit of v1. Outputs name the policy they used.
Goldens are re-captured live when a policy changes.

## Consequences

Map walks cost more `get_neighbors` calls (pool lookups, reused by expansion):
`"manage budgets" --depth 3 --breadth 10` took about 26 s on a first run while
the neighbour cache filled, and about 1 s after (local Neo4j). The budget bounds it.
O*NET's relatedness `index` on related-occupation links is used as the ranking
weight when there is no `importance`. Where a taxonomy gives no signal at all
(e.g. O*NET software-to-job links), the pool is still the first
`min(5*breadth, 30)` candidates by label, so specificity only re-ranks that slice;
a v2 needs a degree count from the suite in one query.
