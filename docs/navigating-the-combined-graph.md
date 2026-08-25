# Navigating the combined graph

The MCP server teaches a client how to read each tool honestly. It does not
teach the client how to navigate, and it does not tell the client that a career
question is a graph question at all. This document is that missing half.

It is written for whoever holds the conversation: a Claude Code session, a
Codex session, or our own assistant.

> Measured on a local graph carrying ESCO, O*NET, BLS and SFIA together, plus
> the ESCO-to-O*NET crosswalk. 41,189 nodes, 427,134 edges.

## Why this document exists

We ran the same MCP server against two clients. One had a file like this
sitting next to it and produced a grounded answer. The other had only the
server's own instructions, never reached for the tools until it was told to
twice, and then spent four calls discovering that the tools existed.

The gap was not the model. It was that nothing told the client when to reach
for the graph, or in what order.

## What each suite knows, and what it cannot answer

| suite | id prefix | what it knows |
|---|---|---|
| ESCO | `esco:` | which occupations and skills exist, in European vocabulary |
| O*NET | `onet:` | **how much** each generic skill matters, surveyed, with margins |
| BLS | `bls:` | **employment, ten-year projection, wages** |
| SFIA | `sfia:` | **how senior** an IT skill is, levels 1 to 7 |

None of them answers another's question. Reaching for the wrong one produces a
confident answer to a question nobody asked.

## The order

**1. Locate both ends, and say what you found.**

`search_nodes` is the entry point; every other tool takes IDs. Report the
confidence and what it means: `0.95` is the official name, `0.90` an official
synonym, `0.85` case-insensitive, `0.70` a substring.

The title the person asks for often does not exist. There is no
"bioinformatics engineer" node in any suite. Say so, name what you resolved to
instead, and let them correct you. Their own words are the thing being
approximated.

Confidence is honest and can still be wrong. `teacher` resolves to
`politics lecturer` at `0.90`, because "teacher" is a listed alternative label
for it in ESCO. Confident and wrong is harder to catch than unsure.

**2. Compare the two ends, and keep the direction.**

Counting shared skills throws away the most useful information. ESCO marks
every `HAS_SKILL` edge `essential` or `optional`, and that is source data.
Split by the relation on both sides:

| from | to | what it means to the person |
|---|---|---|
| optional | essential | this is the work: upgrade it |
| essential | essential | transfers as is |
| essential | optional | you are ahead here |

Measured for biomedical engineer to bioinformatics scientist: 22, 12, and 3.
The 3 are `genetics`, `mathematics` and `scientific research methodology`, and
no shared-skill count would ever surface them.

**The primitive is not overlap. It is overlap with a direction of change.**

**3. When ESCO gives zero, cross to O*NET.**

ESCO skills are written per occupation, so distant occupations correctly share
none:

```
biomedical engineer -> bioinformatics scientist    41 shared    37% overlap
data analyst        -> data scientist              51 shared    45% overlap
specialist nurse    -> data analyst                 0 shared     0% overlap
```

Zero is not a data quality problem, and it is exactly the transition a person
would most want help with. O*NET's elements are generic and every occupation is
rated on all of them, so the same question answers as a distance:

```
specialist nurse -> data analyst, via O*NET

  Programming                  1.44 -> 3.20   (+1.77)
  Computers and Electronics    2.89 -> 4.30   (+1.41)
  Mathematical Reasoning       2.75 -> 3.50   (+0.75)
```

That is what the crosswalk is for. Not more data: answerable hard questions.

## Crossing suites

All four suites are reachable through the tools: pass `suite` to
`search_nodes` and `get_neighbors`. It defaults to `esco`. O*NET neighbours
carry `importance`, `level`, `importance_n` and `importance_lower_ci` on the
edge, so the numbers arrive without writing a query.

**Crossing between suites still needs Cypher.** The tools navigate inside one
suite at a time; `CORRESPONDS_TO` and the SOC code join are not exposed as
tools yet.

ESCO to O*NET weights:

```cypher
MATCH (e:Occupation)-[:CORRESPONDS_TO]-(o:OnetOccupation)-[r:HAS_SKILL]->(s)
WHERE e.id = $id AND r.recommend_suppress = false
RETURN o.pref_label, s.pref_label, r.importance, r.level, r.importance_n
```

BLS joins on the SOC code **inside the identifier**, not on an edge:
`onet:occupation:29-1141.01` to `bls:occupation:29-1141`. That is string
identity, not a semantic correspondence, and saying so is part of using it.

## What cannot be asserted

**One ESCO occupation crosses to several O*NET ones.** Median 2, up to 21, and
only 40.9% are one-to-one. Always say how many it reached and which one each
number came from. The set usually agrees on *which* skills and disagrees on
*how much*: show the range, not the mean. `Biology` for a specialist nurse runs
1.88 to 4.63 while `Problem Sensitivity` runs 3.88 to 4.75. One depends heavily
on the specialisation and the other does not, and an average erases exactly
that.

**The crosswalk is an institutional opinion.** Published jointly by the European
Commission and the US Department of Labor, built with a model and validated by
people. It carries authority; it is not a mechanical derivation. `Provenance`
says who stands behind it, `MappingMethod` says how it was made, and those are
different claims.

**There is no skill mapping between ESCO and O*NET.** 13,939 against 35.
Crossing skills would be inventing.

**Scales are not comparable across suites.** ESCO's 0.95 confidence and O*NET's
importance do not mean the same thing, and SFIA level 7 is not a weight: it does
not mean "matters more", it means "defined for someone who sets strategy".
Never add or average across suites.

**Nothing here knows how people actually move.** No occupation transition data,
no training data, no time or effort. Every route is a skill gap, not an observed
path, and has to be labelled that way. A ranked list of gaps is not a plan.

## The failure to watch for

`score_paths` returns `score_paths_not_implemented` for ESCO, because its edges
are binary and there is nothing to rank on. That refusal is correct and it held
up when a third-party client invented a policy name and asked for it anyway.

The dangerous failure is the opposite one. When a tool returns nothing, or is
blocked, or carries no warning, a client fills the gap in prose. We have seen
it produce "no matching nodes returned" for a permission error, and a
fabricated disclaimer that essential and optional were not marked when the
result carried 25 edges all marked `essential` and `warnings: []`.

**`warnings` is the only place the system says no.** If it is empty, the graph
answered. Do not add a limitation it did not report, and do not describe a
tool failure as an empty result.
