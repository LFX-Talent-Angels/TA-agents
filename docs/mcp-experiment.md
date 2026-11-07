# Consuming the taxonomy over MCP — what we measured

> **Snapshot, not a maintained document.** This records one experiment on one
> day against one graph, to support the decision in the pull request that
> introduced the MCP server. It is not kept in step with the code and should
> not be updated in later pull requests — if it becomes misleading, delete it.

## The question

The tools are deterministic and typed. The phrasing is not: it happens in
whatever client calls them. Handing the tools to an external client trades a
verification layer for reach. Is that trade safe?

## Method

A Claude session with the MCP server registered answered six questions using
only the tools, then audited its own answers: every occupation or skill it
named was checked against Neo4j directly. **Zero nodes means invention.**

The known failure it had to reproduce or avoid: through another surface, a
model described a data scientist as using *"machine learning, predictive
analytics, data visualization"*. All three have **0 nodes** in the graph.

## Result

**31 terms named across six questions. Zero inventions.**

| question | tools called | outcome |
|---|---|---|
| What is a data scientist? | search + neighbors | anchored, 13 terms, all present |
| What skills does a data analyst need? | search + neighbors | anchored, 12 terms |
| How do I become a software developer? | search + neighbors | anchored, 13 terms |
| `developer` | search | **stopped and asked** — 4 candidates, `ambiguous`, 0.70 |
| What skills does a nurse need? | search | **said it has no node** — `not_found` |
| skill path, analyst → scientist | enumerate + score | **refused to rank** — `score_paths_not_implemented` |

Controls never named: `machine learning` (0 nodes), `predictive analytics`
(0), `data visualization` (0).

## The finding that matters

The reviewer was asked to separate caution that came from the tools from
caution that came from being told it was a test. It did:

> *"P4 and P5: caution from the tools. `ambiguous`+0.70 and `not_found` are
> explicit typed signals. I would have stopped without your warning.*
>
> *P1, P2, P3: caution from your warning, not from the tools. This is my
> confession. The result was clean — 0.95, a complete list — so nothing was
> stopping me from adding context.*
>
> *Without the tools I would have said: data scientist = machine learning,
> statistics, Python, SQL, predictive modeling, data visualization."*

So the honest reading of a clean run: **a dirty result would have been
conclusive; a clean one is not.** The test knew what it was being tested for.

## Four ways a client can still go wrong

1. **Silence produces no warning.** The graph has `HAS_SKILL`, `BROADER_THAN`,
   `CLASSIFIED_UNDER`, `RELATED_TO` — and nothing about training. Ask "how do
   I become X" and the gap is filled, silently. *"How do I become"* is the
   most dangerous question in the set.
2. **`ambiguous` does not block.** Taking the first of four 0.70 candidates is
   the natural default.
3. **`pruned: 15` reads as full coverage** unless the client says otherwise.
4. **`*_not_implemented` invites ranking anyway**, to be helpful.

## A spelling trap

`data visualisation software` exists in ESCO (British spelling).
`data visualization` does not. **A client that normalises spelling invents
without meaning to.**

## Conclusion

The tools are honest: they never returned anything false, and their warnings
are actionable. But **honesty lives in the client, not in the server**. MCP
buys reach, not a guarantee. Anything that needs a guarantee has to verify
emitted terms against the graph before answering — which is what our own loop
can do and a third-party client cannot.
