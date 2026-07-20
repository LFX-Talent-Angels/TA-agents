# TA-agents

The **main project** of Talent Angels: the assistant runtime — one main
assistant (LangGraph) dispatching the Locate/Connect/Pathfind/Evaluate skills
over taxonomy graph suites.

This is a **subrepo** of the Talent Angels workspace.

## Read first

1. The workspace policy: `../CLAUDE.md`
   (or https://github.com/LFX-Talent-Angels/TA-workspace → `CLAUDE.md`).
   It is **authoritative** — git rules, DCO, secrets, agent conventions.
2. `../docs/architecture/SYSTEM.md` — cross-repo architecture + suite contract.
3. `ARCHITECTURE.md` in this repo — runtime internals. **Follow it**; the
   architectural rules below are summaries.
4. This file and `AGENTS.md` for code-specific rules.

## What lives here

```
src/talent_angels/
├── assistant/      # intent → plan → dispatch → merge → answer (LangGraph)
├── skills/         # locate/ connect/ pathfind/ evaluate/ — skills + tools
├── contracts/      # AgentResult + typed refs (Pydantic v2)
├── runlog/         # one structured record per turn
└── api/            # FastAPI edge — thin, no reasoning
tests/              # pytest; golden evals in tests/evals/
```

Taxonomy ingestion, graph schemas, and the suite-contract implementation live
in the sibling repo **`TA-taxonomies`** — never here. This repo imports only
the suite-contract surface.

## Architectural rules (short form — full text in ARCHITECTURE.md)

- Only the **main assistant** talks to the user or changes the plan. Skills
  never own the goal; tools know nothing about agents.
- **Typed results cross every boundary** (AgentResult) — never prose.
- Only graph data is cited as taxonomy fact; model inference is labeled.
- Node IDs are **suite-scoped**; no cross-suite identity without an explicit
  crosswalk. Evidence is a **pointer, not a payload** (licensing).
- **Determinism is pushed down**: traversal, depth caps, top-K cuts, scoring
  live in tools/code, not in model calls.
- Skills run **inline by default**; a subagent only with a measurement that
  isolation pays (see ARCHITECTURE.md "Subagent rule").
- Every turn writes a run-log record.

## Conventions

- **Python 3.11+.** Package is `talent_angels`, src-layout (`src/`).
- Formatting/linting: **ruff**; types: **mypy** (be pragmatic early on).
- Tests: **pytest**. New behavior ships with a test; new skills ship with
  golden evals.
- Keep modules small and single-purpose; one capability per skill package.
- Configuration via environment variables — see `.env.example`. **Never** commit
  real keys or `.env` files.

## Common commands

```bash
# from this repo (after bin/setup-workspace.sh, or: python -m venv .venv && pip install -e ".[dev]")
pytest                 # run tests
ruff check .           # lint
ruff format .          # format
mypy src               # type-check
```

## Git

Branch + PR, **`git commit -s`** (DCO). Never push to `main`. Full rules in the
workspace `CLAUDE.md` and `CONTRIBUTING.md`.

## AI agents

Read this file, `ARCHITECTURE.md`, and `AGENTS.md` before changing code. Review
and test agent output; you own what you submit. Record non-obvious decisions in
`TA-memory`.
