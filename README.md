# TA-agents

> Main project of **Talent Angels** — a suite of AI Graph Agents that reason over
> skill, task, and occupation taxonomies via Graph-RAG.

Part of the [`LFX-Talent-Angels`](https://github.com/LFX-Talent-Angels) org. For
project-wide docs, onboarding, and rules, see
[`TA-workspace`](https://github.com/LFX-Talent-Angels/TA-workspace).

## Agents

- **Locator** — pinpoints a skill/task/occupation in the taxonomies.
- **Connector** — lists the nodes directly preceding/succeeding a location.
- **Pathfinder** — traces all routes between two locations (learning journeys).
- **Evaluator** *(future)* — ranks paths by relevance, distance, profile fit.

## Taxonomies

ESCO · O*NET · SFIA · BLS · Lightcast.

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
├── locator/      connector/      pathfinder/      # the agents
├── graph/        # knowledge graph + Graph-RAG retrieval
└── taxonomies/   # ESCO, O*NET, SFIA, BLS, Lightcast loaders
tests/
```

## Contributing

See [`CONTRIBUTING.md`](./CONTRIBUTING.md). Branch, `git commit -s` (DCO), open a
PR, request a mentor review.

## License

Apache-2.0 — see [`LICENSE`](./LICENSE).
