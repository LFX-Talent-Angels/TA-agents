# ADR-0014: The LLM boundary

- **Status:** Proposed
- **Date:** 2026-11-25
- **Deciders:** mentee team; mentors to accept
- **Mirrors:** none yet — mirror to `TA-memory/decisions/` on mentor acceptance

## Context

Talent Angels is an engine: a declared set of capabilities (ADR-0012, ADR-0013, the Step 2
registry) that answer from suite-scoped taxonomy graphs, deterministically and with citations.
Any interface plugs into it: the chat, HTTP, MCP and the CLI.

The chat now accepts plain language for every capability ("roadmap from data analyst to data
scientist", "what comes after Python", "tag ./jobs"), and users bring their own model from any
provider. A model makes the chat easier to use. It is also the easiest way to break the engine's
promises: it can invent a skill, a pay figure or an order; it can be talked into skipping the
graph; and text inside an uploaded list can try to instruct it. Users without a model must lose
nothing that the engine itself can do.

## Decision

The model is an **interpreter and a narrator, never a source of facts**.

1. **A closed set of actions.** In the chat, the model may only choose one capability from the
   registry (`map`, `discover`, `roadmap`, `successors`, `tag`) and copy its arguments out of
   the user's words. Code accepts that choice only if the capability is routable, every key is
   allowed, every name occurs in the user's text, every number was stated, every path exists,
   and the capability's input model validates it. Otherwise nothing runs: keyword routing
   answers, or TA asks one clarifying question. There is no free-form tool, no query language
   and no write path; the graph is read-only. The only place a model still chooses graph reads
   is the locate/connect tool loop, limited to two read-only suite tools whose node ids must
   come from earlier results.
2. **Results are written by code.** Routed results are an echo line, a code-written summary and
   the saved call. Cards for locate, connect and pathfind are code-written as before.
3. **Every model sentence is checked.** A sentence of model prose is kept only if every number,
   every name written as a name, every listed example, and every ordering or equivalence word
   in it appears in this turn's code-written facts, and it does not open by affirming. Otherwise
   it is dropped, and the code-written text stands alone. This check is a filter, not the
   guarantee: the guarantee is that facts are always shown as code-written output.
4. **Reproducible.** Model calls use temperature 0. Same words give the same plan and the same
   route (cached). Each routed reply saves the resolved call with every default; rerunning it on
   the same graph data gives byte-identical output.
5. **Files and lists are data.** The router never reads a file. Items from a list or file are
   passed to capabilities as values; only the user's own line, quoted as data, reaches the
   routing model. Over HTTP and MCP, paths must sit under the server's tag root.
6. **Out of scope is refused, in code.** Courses, syllabi, learning outcomes, "invent" or "make
   up" requests, and requests to ignore the taxonomy get a fixed refusal. Pay comes only from
   BLS's published figures. Personal career advice is refused as before.
7. **No model needed.** With no model, every capability works through keyword routing and code
   summaries. A broken or slow model degrades to the same path within the turn clock.
8. **Cost and time limits.** Model calls share one turn clock and a per-call timeout. Tagging
   with a model states its cost basis and runs only after the user says yes; a model can never
   switch it on.
9. **Credentials stay out of the conversation.** Keys are read with hidden input, verified
   before they are stored, stored owner-only in the TA home, and never enter the transcript, the
   run log, a saved call or a prompt. A key typed into the chat is refused before it is recorded.

## Consequences

- Some lines get a clarifying question where a model could have guessed.
- Some harmless model phrasing is dropped by the conservative check; the code-written answer is
  still shown.
- The check does not catch an invented lowercase term outside an example list. It does not need
  to for correctness, because no fact depends on model prose; it lowers the chance that prose
  misleads.
- Providers differ in parameter support; `temperature` is dropped where a model rejects it, so
  reproducibility of prose is best-effort, while capability output stays byte-identical.
- A guardrail test set (about 40 adversarial prompts plus grounding cases) runs offline on every
  change with a scripted misbehaving model; the same set runs live on demand.

## Alternatives considered

- **Let the model answer from its own knowledge.** Rejected: not cited, not reproducible.
- **Give the model the capabilities as free tools, or let it write graph queries.** Rejected: it
  bypasses input validation, limits and the closed action set.
- **Require a model for the chat.** Rejected: the engine must work offline and for free.
- **Let the model write one introduction for routed results.** Rejected for now: zero sentences
  is simpler and meets "at most one".
- **A named-entity model for grounding.** Rejected: a new dependency and another model to trust.

## Review

Revisit if the registry, the citation contract or the deployment model changes. Any exception
to these rules needs a new ADR and a guardrail case that proves the engine is still the only
source of taxonomy facts.
