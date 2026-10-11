# ADR-0008: Pathfind — start/target shape, output, and the honest-refusal rules

- **Status:** Accepted (this repo)
- **Date:** 2026-10-11
- **Deciders:** mentee team, finalizing the proposal in `HANDOVER-2026-10-01.md` Part 6.2/6.4
- **Mirrors:** none yet — mirror to `TA-memory/decisions/` on mentor acceptance

## Context

Pathfind ("how do I get from *nurse* to *data analyst*?") is the one skill the
assistant still refuses outright: `capability_not_implemented:pathfind` on
every path. The suite contract already has what Pathfind needs —
`enumerate_paths(from_id, to_id, max_depth, max_paths)` and `score_paths` are
real methods on `Suite`, already passed through by `MeasuredSuite` and
`CachedSuite`, already exposed by the MCP edge — but nothing in TA-agents
calls them. `score_paths` is a stub upstream; treating its output as a real
ranking would violate "deterministic work lives in code, never invented" the
same way an un-cited skill list would.

The handover document proposes starting Pathfind from **a set of skills**,
not only one occupation, so the same tool later serves a business's
certification criteria (Phase 2) without a redesign. That proposal needs a
concrete decision before any code: what exactly "the start" and "the output"
are, and what happens when no reasonable route exists — because "no route"
must read as a finding, not a missing feature.

## Decision

**Pathfind takes a start that is always a skill-ID set, a target that is
always one occupation node, and returns two things: a skill gap always, a
set of bridge routes only when the start also names one occupation.**

**1. Start: a skill-ID set, derived two ways.**
- *Occupation start* ("nurse → data analyst"): Locate resolves "nurse" to one
  occupation node, Connect reads its essential skills, that skill-ID set
  becomes the start. The occupation node ID is kept alongside — it is what
  `enumerate_paths` needs for routes.
- *Bare skill-set start* (not wired into the assistant yet — the chat surface
  only ever produces the occupation form; this shape exists so Evaluate and
  the Phase 2 enrichment pipeline can call the same function with a
  certification's skill IDs without a second code path): skill IDs passed
  directly, no occupation node. Routes are skipped (see 3) — there is no
  graph node to pathfind *from*.

**2. Target: one occupation node, in the same suite as the start.**
Resolved through the same Locate call every other skill uses — ambiguous or
not-found endpoints return the same `ambiguous`/`not_found` warnings Locate
already produces, not a Pathfind-specific error. If Locate resolves "nurse"
and "data analyst" into *different* suites (one only in ESCO, one only in
O\*NET), Pathfind is not attempted for that suite pair — each attached suite
is dispatched independently, exactly as Locate/Connect already do for a
multi-suite question, and a suite that cannot resolve both endpoints reports
its own honest miss rather than a crosswalk nobody built (rule 4,
`ARCHITECTURE.md`: no silent cross-suite identity).

**3. Output part one — the skill gap. Always computed, never skipped.**
`target's essential skills ∪ optional skills` minus `start's skill-ID set`,
as plain set difference over suite-scoped IDs — no embedding, no LLM, no
fuzzy matching between skill labels. Essential-missing skills are listed
before optional-missing ones; within essential, O\*NET's `importance` edge
property orders the list where present, ESCO (which has no such property)
keeps graph return order and says so. A target with zero skill edges
(`no_neighbors` from Connect) returns a gap of zero with that warning
attached, not a silent empty list.

**4. Output part two — routes. Only when the start has an occupation node.**
`enumerate_paths(start_occupation_id, target_occupation_id, max_depth=4,
max_paths=20)` — the suite's own defaults, not re-tuned here. Each returned
path becomes a typed `NodeRef`/`EdgeRef` chain through the *existing*
`AgentResult.nodes`/`.edges` lists (no new contract type: a path is just
nodes connected by cited edges, which is what those fields already hold).
**`score_paths` is never called.** Routes are listed in whatever order
`enumerate_paths` returns them — unscored, because the only scorer that
exists is a stub, and showing a stub's number as a real rank is exactly the
invented-fact failure `ARCHITECTURE.md` rule 1 exists to prevent. Ranking
routes under a named policy is Evaluate's job (a separate, later skill).

**5. The disqualifier is a warning, not an error.** When the essential-skill
gap is larger than `MAX_REALISTIC_GAP_SKILLS` (provisional: **8** — picked to
be clearly "most of a career retrain," not tuned against data yet; the
golden pairs set (PR 6 of this stack) is what should move this number, not a
guess revised twice), the result carries a `gap_too_large` warning alongside
the full gap list. The assistant still answers with the real gap — "no
realistic route" is a thing Pathfind *found*, stated plainly, never a request
it couldn't fulfil.

**6. Everything above lives in `skills/pathfind/`, in code.** Gap
computation, the depth/fan-out caps (inherited from `enumerate_paths`'s own
defaults), and the disqualifier threshold are deterministic Python, called
the same way Locate and Connect are called — never phrased or decided by a
prompt. The planner's only job is extracting the two subjects (`subject` /
`secondary_subject` on `PlanDraft`, already specified for pathfind in
`llm_plan.py`) and routing to the `pathfind` capability; it never sees a
skill ID or a path.

## Alternatives considered

- **Start = one occupation only, no skill-set form.** Simpler, matches every
  worked example in the handover ("nurse → data analyst"). **Rejected** —
  it would mean a second Pathfind implementation when Phase 2's
  certification-skills case arrives, instead of the same function called
  with a different start. The skill-set form costs one extra parameter now
  and avoids a rewrite later.
- **Always attempt routes, even from a bare skill set.** `enumerate_paths`
  needs a graph node to start from; a skill set has no single node by
  construction. Synthesizing a fake "virtual occupation" node to pathfind
  from would invent a node that is not in the taxonomy. **Rejected** — gap
  without routes is an honest, smaller answer; a fabricated start node is not.
- **Call `score_paths` and label its number "experimental."** Still shows an
  invented-looking number next to real graph facts, and every other skill in
  this codebase treats a labeled-but-still-shown stub value as a trap users
  stop reading labels around. **Rejected**, matches existing project
  practice (MENTEE-HANDOFF.md, ADR-0007's own "do not invent a ranking"
  framing).
- **Tune `MAX_REALISTIC_GAP_SKILLS` now, against intuition.** No data exists
  yet to tune it against. **Deferred** to the golden pairs set — the number
  here is provisional and explicitly flagged as such, not presented as
  measured.

## Consequences

**Easier.** Pathfind reuses every existing typed boundary — `NodeRef`,
`EdgeRef`, `AgentResult`, the per-suite dispatch loop, the Locate honesty
rules for ambiguous/not-found endpoints. No change to `contracts/` (so no
code-owner review gate on this stack), no new persisted-state type (so no
`checkpoint.py::PERSISTED_TYPES` entry).

**Harder.** The skill-set start form is unused until Evaluate or Phase 2
enrichment calls it — it is tested directly (unit tests calling the skill
function with a bare skill-ID set) but has no chat-reachable path yet, so it
is dead code from the assistant's point of view until one of those lands.
That is an accepted, named cost, not an oversight: the alternative is
rewriting the function signature later under time pressure.

**Risks.** `MAX_REALISTIC_GAP_SKILLS = 8` is a guess. If the golden pairs
set shows it is badly wrong (too many "no realistic route" answers for pairs
a person would actually consider, or too few for pairs that clearly aren't
realistic), the number moves — that is the set's job, not a sign the
disqualifier itself is wrong.

## Amendments

**2026-10-11 — `MAX_REALISTIC_GAP_SKILLS` moved from a guess to a measured,
per-suite threshold.** PR 6's golden pairs set (`tests/evals/golden_pathfind.json`,
10 real occupation pairs against live ESCO v1.2.1 / O\*NET 31.0) measured
essential-missing counts of 10-36 for ESCO pairs and 51-412 for O\*NET pairs
— confirming the provisional guess of 8 flagged *every* pair tested as
`gap_too_large`, including near-duplicate titles ("web developer" →
"software developer": 11 essential-missing). The two suites are not on the
same scale: O\*NET's `HAS_SKILL`/`USES_SOFTWARE` edges include named
software tools ESCO does not model as skills, so its counts run an order
of magnitude higher.

The threshold is now per-suite (`skills/pathfind/gap.py`,
`_MAX_REALISTIC_GAP_SKILLS_BY_SUITE`), derived from the already-passed
`AgentResult.suite` field rather than a new parameter: **ESCO 15** (rests
on 6 pairs with a clean split between same-family title variants, 10-11,
and genuine transitions, 19+) and **O\*NET 60** (rests on only 4 pairs,
all of which happened to have no bridge route within `max_depth=4` — this
number is the weaker of the two and should be revisited first if more
O\*NET golden pairs land). `SuiteSchema` was deliberately not extended for
this — it is canonically owned by the external `ta_taxonomies` package, so
threading the threshold through it would require a change to that other
repository; a small suite-keyed dict in this repo is the smaller, correct
move for what decision 5 asked for ("move this number from measurement").

A second finding, not acted on in this PR: all 4 O\*NET pairs tested came
back `no_path` from `enumerate_paths`, including occupations that are
obviously close in practice ("Software Quality Assurance Analysts and
Testers" → "Software Developers", essential-missing 51, the lowest O\*NET
count measured). Whether this is a genuine graph-structure difference
between ESCO and O\*NET or an artifact of the suite's own default
`max_depth=4` is unknown from 4 data points — flagged here as a finding
for a wider O\*NET golden pairs set to confirm, not a conclusion.
