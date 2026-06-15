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
