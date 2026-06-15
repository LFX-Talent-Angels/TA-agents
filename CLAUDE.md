# TA-agents

The **main project** of Talent Angels: the suite of AI Graph Agents (Locator,
Connector, Pathfinder) and the Graph-RAG layer that reasons over skill, task, and
occupation taxonomies.

This is a **subrepo** of the Talent Angels workspace.

## Read first

1. The workspace policy: `../TA-workspace/CLAUDE.md`
   (or https://github.com/LFX-Talent-Angels/TA-workspace → `CLAUDE.md`).
   It is **authoritative** — git rules, DCO, secrets, agent conventions.
2. `../TA-workspace/docs/architecture/SYSTEM.md` — high-level architecture.
3. This file and `AGENTS.md` for code-specific rules.

## What lives here

```
src/talent_angels/
├── locator/        # pinpoint a node from natural language
├── connector/      # neighbors of a node
├── pathfinder/     # routes between two nodes
├── graph/          # knowledge graph model, ingestion, Graph-RAG retrieval
└── taxonomies/     # load & normalize ESCO, O*NET, SFIA, BLS, Lightcast
tests/              # pytest
```

## Conventions

- **Python 3.11+.** Package is `talent_angels`, src-layout (`src/`).
- Formatting/linting: **ruff**; types: **mypy** (be pragmatic early on).
- Tests: **pytest**. New behavior ships with a test.
- Keep modules small and single-purpose; one agent concern per package.
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

Read this file and `AGENTS.md` before changing code. Review and test agent
output; you own what you submit. Record non-obvious decisions in `TA-memory`.
