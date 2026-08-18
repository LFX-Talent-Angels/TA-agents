# Contributing to TA-agents

This repo follows the **project-wide contributing guide** in the workspace:
👉 https://github.com/LFX-Talent-Angels/TA-workspace/blob/main/CONTRIBUTING.md

Quick reminders specific to this code repo:

```bash
git switch -c feature/my-change
# ... code + tests ...
ruff check . && ruff format . && pytest      # keep CI green
git commit -s -m "feat: ..."                  # DCO sign-off is required
git push -u origin feature/my-change
gh pr create --fill
```

- Python 3.11+, package `talent_angels` (src-layout).
- New behavior ships with a pytest test.
- Never commit secrets or `.env` files — use `.env.example`.
- At least one mentor approval is required to merge.
- Every commit must be DCO signed (`git commit -s`). Never push to `main`.

## Next work (post-MVP)

Branch from `feature/honesty-loop` (or the current demo tip), not from
official `main`. The TUI / logging / open memory discussion is **only**
in the local mentee handoff
(`data/local/agent-coordination/MENTEE-HANDOFF.md` §1.1) so it can be
shared as one Slack file. Do not add a second spec in `docs/`.

## Tests

The required offline suite does not need Neo4j or a concrete taxonomy suite:

```bash
ruff check .
ruff format --check .
mypy src
pytest -q --ignore=tests/integration
```

Concrete ESCO and Neo4j checks live separately under `tests/integration/`:

```bash
# Official TA-taxonomies main now includes the contract (#3), loader (#4),
# and EscoSuite tools (#5). That pin is enough for contract checks and
# live Locate/Connect. Pathfind still needs an agents skill on top of
# enumerate_paths; score_paths remains a stub.
pip install -e ../TA-taxonomies

# Configure Neo4j through the documented environment variables, then run:
pytest -q tests/integration -rs
```

The CLI and API load a local `.env` automatically (shell exports still win).
To exercise the same turn over HTTP / Swagger:

```bash
uvicorn talent_angels.api.app:app --reload
# http://127.0.0.1:8000/docs
```

When the taxonomy package or Neo4j is unavailable, the integration command
reports an explicit skip reason. Offline assistant and skill tests must never
be skipped for missing infrastructure.
