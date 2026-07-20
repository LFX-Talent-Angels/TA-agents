# AGENTS.md — TA-agents

Routing for non-Claude agents (Codex, Antigravity, Cursor, Gemini, Aider, …)
working in `TA-agents`, the main code repo of Talent Angels.

## Read first

1. `../CLAUDE.md` — **authoritative** project policy (git, DCO,
   secrets, conventions). On GitHub: `LFX-Talent-Angels/TA-workspace`.
2. `../docs/architecture/SYSTEM.md` — cross-repo architecture + suite contract.
3. `ARCHITECTURE.md` in this repo — runtime internals (one main assistant +
   Locate/Connect/Pathfind/Evaluate as skills; typed results; determinism
   pushed down into tools). **Follow it.**
4. `CLAUDE.md` in this repo — code-specific rules.

Treat `CLAUDE.md` files as authoritative. This file only routes non-Claude
agents; keep both in sync.

## Rules (summary — see CLAUDE.md for the full text)

- Python 3.11+, src-layout package `talent_angels`. Lint with ruff, test with
  pytest, type-check with mypy.
- Taxonomy ingestion/graph schemas live in the sibling repo `TA-taxonomies`;
  this repo imports only the suite-contract surface.
- Branch + PR flow. Every commit DCO signed-off (`git commit -s`). Never push to
  `main`.
- Never commit `.env*` files or secrets (a local, gitignored `.env` is fine). Use `.env.example`.
- New behavior ships with a test.
- Interpret Claude slash-commands (`/review`, `/ship`, `/qa`) as workflow intent;
  use your own equivalents or do the steps manually.
- Record non-obvious decisions/learnings in `TA-memory`.
