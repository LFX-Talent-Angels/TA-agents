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
# From the workspace, install the sibling taxonomy package into this venv.
pip install -e ../TA-taxonomies

# Configure Neo4j through the documented environment variables, then run:
pytest -q tests/integration -rs
```

When the taxonomy package or Neo4j is unavailable, the integration command
reports an explicit skip reason. Offline assistant and skill tests must never
be skipped for missing infrastructure.
