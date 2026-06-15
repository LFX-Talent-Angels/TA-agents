# AGENTS.md — TA-agents

Routing for non-Claude agents (Codex, Antigravity, Cursor, Gemini, Aider, …)
working in `TA-agents`, the main code repo of Talent Angels.

## Read first

1. `../TA-workspace/CLAUDE.md` — **authoritative** project policy (git, DCO,
   secrets, conventions). On GitHub: `LFX-Talent-Angels/TA-workspace`.
2. `../TA-workspace/docs/architecture/SYSTEM.md` — architecture.
3. `CLAUDE.md` in this repo — code-specific rules.

Treat `CLAUDE.md` files as authoritative. This file only routes non-Claude
agents; keep both in sync.

## Rules (summary — see CLAUDE.md for the full text)

- Python 3.11+, src-layout package `talent_angels`. Lint with ruff, test with
  pytest, type-check with mypy.
- Branch + PR flow. Every commit DCO signed-off (`git commit -s`). Never push to
  `main`.
- Never create `.env*` files or commit secrets. Use `.env.example`.
- New behavior ships with a test.
- Interpret Claude slash-commands (`/review`, `/ship`, `/qa`) as workflow intent;
  use your own equivalents or do the steps manually.
- Record non-obvious decisions/learnings in `TA-memory`.
