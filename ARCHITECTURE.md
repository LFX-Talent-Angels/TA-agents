# TA-agents — Runtime Architecture

> Internals of the assistant runtime. The cross-repo picture (repo topology,
> suite contract, stack) lives in `TA-workspace/docs/architecture/SYSTEM.md`;
> the decisions behind this design are ADR-0003/0004/0005. This document is the
> team's Sprint 2 architecture, made concrete.

## One assistant, kinds of map work

There is **one main assistant**. It owns the user's goal across turns, decides
which map work each turn needs, and writes the final answer. The map work —
Locate, Connect, Pathfind, Evaluate — is implemented as **skills + tools**, not
as peer agents.

```
User ── chat ─▶ FastAPI edge (thin) ─▶ Main assistant (LangGraph loop)
                                          │  intent → plan → dispatch → merge → answer
                              ┌───────────┼───────────┬────────────┐
                              ▼           ▼           ▼            ▼
                           Locate      Connect     Pathfind     Evaluate
                          (Resolve)    (Reveal)    (Compose)     (Rank)
                              │           │           │            │
                              └───────────┴─────┬─────┴────────────┘
                                                ▼
                                   suite contract (ta-taxonomies)
                                   search_nodes · get_neighbors ·
                                   enumerate_paths · score_paths
```

### The loop (LangGraph)

1. **Intent** — read the user's turn against session state.
2. **Plan** — which capabilities, in what order (L / L+C / L+C+P / …+E), and
   **which suites** (no fan-out to all five by default; the plan picks suites
   by what the question needs — economic signal → BLS, competency levels →
   SFIA, weighted skills → O*NET, multilingual → ESCO).
3. **Dispatch** — run the plan's skills. Inline by default; a scoped subagent
   only where isolation measurably pays (see "Subagent rule").
4. **Merge** — combine typed results across suites. Suite-scoped IDs are never
   equated across suites; disagreements are surfaced, not averaged.
5. **Answer** — one coherent reply: citations to graph evidence, confidence
   carried from Locate, model inference explicitly labeled.
6. **Log** — one structured run-log record per turn (plan, tool I/O,
   assumptions). This is the seam the future Evaluator quality loop attaches to.

### Rules that do not bend

1. Only the main assistant finalizes the user-facing answer or changes the plan.
2. Only graph data is cited as taxonomy fact; model inference is labeled.
3. Skills do not own the user's goal; tools know nothing about agents.
4. Confidence from Locate crosses every later step; low-confidence, high-stakes
   traversal asks the user to confirm.
5. "Better path" is an explicit, named policy — never silent shortest-path.
6. No invented cross-taxonomy identity; only explicit crosswalks, else "no link".
7. Every turn writes a run-log record.
8. Prose never crosses component boundaries — results are typed.

## Skills

A skill is a **passive written procedure** (how to do one kind of map work,
per suite quirks) plus the **tool calls** it drives. Skills are loaded
on demand — only what the plan needs enters context.

| Skill | Does | Mostly |
| ----- | ---- | ------ |
| `locate` | free text → candidates + confidence (exact → alias → vector fallback) | deterministic + judgment on ambiguity |
| `connect` | neighbors/hierarchy/gaps of a resolved node | deterministic (single Cypher hop) |
| `pathfind` | routes between two resolved IDs; chains neighbor traversal — composition lives in code, not in agent-to-agent calls | deterministic (depth-capped, cycle-free) |
| `evaluate` | rank routes under a named scoring policy | deterministic scoring; policy is declared data |

**Determinism is pushed down.** Traversal, cycle avoidance, depth caps, top-K
cuts, scoring — all live in tools/code. The model decides *what to ask* and
*how to answer*, never how to walk the graph.

**Evaluate advanced.** Originally a future fourth agent; as a skill it is
small. O*NET provides weighted edges to score against today. Where a source has
no weights (ESCO is structurally binary), the scoring policy is **our modeling
decision — named, versioned, and cited in the answer**, never presented as
source data.

### Subagent rule

Delegation pays only when a step generates many intermediate tokens and
returns few. Locate and Connect have compression ≈ 1 → always inline.
Pathfind on large graphs is the one candidate; before reaching for a subagent,
prefer pruning in code (top-K by edge weight at the tool edge, returning
counts of what was cut — not the cut rows). Graduation to a subagent requires
the measurement below, not a hunch.

## Token economy (working defaults)

- One assistant = one large stable prompt prefix (system + tool schemas) —
  maximize prompt-cache hits; don't fragment the prefix across processes.
- Typed results between steps (compact), never prose (lossy + expensive).
- Truncate at the tool edge with principled ranking (e.g. O*NET importance),
  and report what was dropped as a count.
- Load skills progressively; pick suites per plan, don't fan out by default.

## Contracts

```python
class AgentResult(BaseModel):  # returned by every skill dispatch
    capability: str  # locate | connect | pathfind | evaluate
    suite: str  # esco | onet | sfia | bls | ...
    nodes: list[NodeRef]  # suite-scoped IDs + source + source_id
    edges: list[EdgeRef]
    evidence: list[EvidencePointer]  # pointer, not payload
    confidence: float | None
    warnings: list[str]  # incl. "no link", truncation counts
```

The suite contract itself (`search_nodes`, `get_neighbors`, `enumerate_paths`,
`score_paths`) is implemented in **`TA-taxonomies`** and consumed here as a
versioned library. This repo may import **only** that surface.

## Layout

```
src/talent_angels/
├── assistant/     # the LangGraph loop: intent, plan, dispatch, merge, answer
├── skills/
│   ├── locate/    connect/    pathfind/    evaluate/
├── contracts/     # AgentResult + typed refs (Pydantic v2)
├── runlog/        # structured per-turn record
└── api/           # FastAPI edge (thin; no reasoning here)
tests/             # pytest; golden evals in tests/evals/
```

## Evaluation

- **Golden evals**: question → expected node(s)/answer pairs per suite,
  runnable via pytest; baselines recorded in `TA-memory`.
- **Falsifiers (measured from Sprint 3)**: tokens per resolved turn, Locate
  accuracy — inline vs. subagent variants. If a single context measurably
  fails somewhere, that capability graduates to a subagent.
- **Not** success metrics: number of agents invoked, path length.

## Open items

Run-log persistence tech · vector-index benchmark (pgvector vs. Neo4j native,
with the embedding model chosen alongside) · session/checkpoint store ·
crosswalk design (SOC/ISCO hub; SFIA bridges skill-to-skill).
