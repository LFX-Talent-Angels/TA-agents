# AGENTS.md — TA-agents

The **main project** of Talent Angels: the assistant runtime — one main
assistant (LangGraph) dispatching the Locate/Connect/Pathfind/Evaluate skills
over taxonomy graph suites.

This file is the source of truth for humans and for every AI coding agent
(Claude Code, Codex, Cursor, Antigravity, Gemini, Aider, and any other).
`CLAUDE.md` is a one-line import of this file.

This is a **subrepo** of the Talent Angels workspace.

## Read first

1. The workspace policy: `../AGENTS.md`
   (or https://github.com/LFX-Talent-Angels/TA-workspace → `AGENTS.md`).
   It is **authoritative** — branch flow, DCO, secrets, agent conventions.
2. `../docs/architecture/SYSTEM.md` — cross-repo architecture + suite contract.
3. `ARCHITECTURE.md` in this repo — runtime internals. **Follow it**; the
   architectural rules below are summaries.
4. This file for code-specific rules.

## What lives here

```
src/talent_angels/
├── assistant/      # intent → plan → dispatch → merge → answer (LangGraph)
├── skills/         # locate/ connect/ pathfind/ evaluate/ — skills + tools
├── contracts/      # AgentResult + typed refs (Pydantic v2)
├── runlog/         # one structured record per turn
└── api/            # FastAPI edge — thin, no reasoning
tests/              # pytest; offline by default, live cases in tests/integration/
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
- Formatting/linting: **ruff**; types: **mypy**, and it blocks CI.
- Tests: **pytest**. Offline tests always run. Cases that need a live Neo4j or
  the concrete `ta-taxonomies` package live in `tests/integration/` and skip
  with an explicit reason.
- New behavior ships with a test; new skills ship with golden evals.
- Keep modules small and single-purpose; one capability per skill package.
- Configuration via environment variables — see `.env.example`. **Never** commit
  real keys or `.env` files.

```bash
pytest -q --ignore=tests/integration   # offline suite
pytest -q tests/integration -rs        # live suite, skips with a reason
ruff check . && ruff format --check .  # lint and format
mypy src                               # type-check
```

## Branches and pull requests

| Branch | What it is        | To merge into it                            |
| ------ | ----------------- | ------------------------------------------- |
| `dev`  | Integration trunk | 1 approval from any contributor + green CI  |
| `main` | What we publish   | 1 approval **from a code owner** + green CI |

- **Open every pull request against `dev`.** `main` only receives promotions
  from `dev`.
- Never push directly to `dev` or `main`.
- **Every commit signed off**: `git commit -s` (DCO). Pull requests without it
  are blocked.
- Without write access, fork and open the pull request from your fork. With
  write access, push the branch to this repo directly, which is what makes
  stacked pull requests possible (`gh stack init --trunk dev`).
- `src/talent_angels/contracts/` and `tests/evals/` need a **code owner
  review**: one defines the typed boundary, the other the correctness
  baseline. See `.github/CODEOWNERS`.

Full details in the workspace `AGENTS.md` and `CONTRIBUTING.md`.

## AI agents

Read this file and `ARCHITECTURE.md` before changing code. Review and test agent
output; you own what you submit. Record non-obvious decisions in `TA-memory`.
