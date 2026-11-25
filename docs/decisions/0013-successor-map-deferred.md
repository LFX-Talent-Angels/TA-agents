# ADR-0013: Successor map ("what comes after skill A") is deferred

- **Status:** Proposed
- **Date:** 2026-11-25
- **Deciders:** mentee team; mentors to accept
- **Mirrors:** none yet — mirror to `TA-memory/decisions/` on mentor acceptance

## Context

A mentor asked for a mapper that starts from one skill A, finds every skill
that has A as a **prerequisite** (A1…An), then repeats for each of them, giving
a recursive "what comes next" tree.

That walk needs prerequisite edges: "learn A before B". We checked all five
suites:

- **ESCO, O*NET, BLS, JobTech:** no prerequisite edges. They link skills by
  broader/narrower, "related", and "needed by the same occupation". None of
  these is a learning order.
- **SFIA:** levels within one skill, not ordering across skills.
- **UCA's own curriculum graph** (the example export shared by the university)
  has 8 `ES_PRERREQUISITO_DE` links between subjects. It is UCA's design: the
  output UCA wants to produce, not an input to TA.

The university's stated process is: market roles, tasks and skills go into TA;
TA returns a map of how they connect (the discovery map, ADR-0012); the
university then defines areas, prerequisites and learning outcomes.

## Decision

Do not build a successor walk now.

- Over the taxonomies it would return nothing, or links we made up. Making
  them up breaks the project rule that only graph data is cited as fact, and
  breaks determinism.
- Over UCA's finished curriculum it would only repeat what UCA already wrote.
- In the university's process, prerequisites are decided **after** the map,
  so the discovery map is what feeds that step.

Graph-Mapper output keeps saying what it is: "linked in the graph; not a
learning order".

## Consequences

- The mentor gets a written answer instead of an implied promise.
- If a graph with real prerequisite edges is loaded later (for example a
  university's curriculum as its own suite), the walk is small: breadth-first
  over one named edge type with Graph-Mapper's existing depth, breadth and
  `max_nodes` caps, deterministic order by node ID. About one day.
- Revisit when such a graph exists, or if a taxonomy publishes prerequisite
  links.
