# TA-agents

> Main project of **Talent Angels** — the assistant runtime that reasons over
> skill, task, and occupation taxonomies via Graph-RAG.

Part of the [`LFX-Talent-Angels`](https://github.com/LFX-Talent-Angels) org. For
project-wide docs, onboarding, and rules, see
[`TA-workspace`](https://github.com/LFX-Talent-Angels/TA-workspace).

## Architecture in one paragraph

**One main assistant** owns the user's goal and dispatches four map-work
capabilities implemented as **skills + tools** (the team's Sprint 2
architecture, ratified in ADR-0003):

- **Locator** *(Resolve)* — pinpoints a skill/task/occupation; attaches confidence.
- **Connector** *(Reveal)* — lists the nodes around a resolved location.
- **Pathfinder** *(Compose)* — traces routes between two locations (learning journeys).
- **Evaluator** *(Rank)* — scores routes under an explicit, named policy.

Taxonomy graphs (O*NET · BLS · ESCO · SFIA structure-only · Sweden JobTech, per ADR-0006) live as
**suites** in the sibling repo
[`TA-taxonomies`](https://github.com/LFX-Talent-Angels/TA-taxonomies), consumed
here as a versioned library through the suite contract. Details:
[`ARCHITECTURE.md`](./ARCHITECTURE.md).

## Quick start

```bash
# Option A: as part of the workspace
git clone https://github.com/LFX-Talent-Angels/TA-workspace.git
cd TA-workspace && bash bin/setup-workspace.sh

# Option B: standalone
git clone https://github.com/LFX-Talent-Angels/TA-agents.git
cd TA-agents
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in your own keys
pytest
```

## Layout

```
src/talent_angels/
├── assistant/    # the LangGraph loop: intent → plan → dispatch → merge → answer
├── skills/       # locate/ connect/ pathfind/ evaluate/
├── contracts/    # typed results (Pydantic v2)
├── runlog/       # structured per-turn record
└── api/          # thin FastAPI edge
tests/
```

## Contributing

See [`CONTRIBUTING.md`](./CONTRIBUTING.md). Branch, `git commit -s` (DCO), open a
PR, request a mentor review.

## License

Apache-2.0 — see [`LICENSE`](./LICENSE).
