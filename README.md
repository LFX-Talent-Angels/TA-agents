# TA-agents

Headless assistant runtime for **Talent Angels**. One main assistant
interprets a natural-language question, calls deterministic ESCO graph tools
(Locate / Connect), and returns a cited JSON answer with tokens and cost.

Pathfind (routes between two occupations) is not implemented yet: those
questions are refused honestly. Evaluate and multi-taxonomy merge come later.

Sibling graph library: [`TA-taxonomies`](https://github.com/LFX-Talent-Angels/TA-taxonomies).
Workspace policy: [`TA-workspace`](https://github.com/LFX-Talent-Angels/TA-workspace).
Internals: [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## What you get

```text
you  →  CLI or FastAPI (/docs)
           →  main assistant (LLM tool loop)
                 →  search_nodes / get_neighbors   (TA-taxonomies)
                       →  Neo4j ESCO graph
```

- CLI prints one JSON object (`answer`, `plan`, `tools`, `tokens`, `cost_usd`).
- API is the same turn. Swagger: `http://127.0.0.1:8000/docs`.
- `.env` is loaded automatically (cwd, then repo root). Shell exports win.
---

## Prerequisites

- Python **3.11+**
- Docker (local Neo4j)
- ESCO English **DATABASE** xlsx (CC BY 4.0) — not in git
- An LLM key for the agentic demo (OpenRouter via LiteLLM)

This README assumes the workspace layout:

```text
TA-workspace/
  TA-agents/        ← you are here
  TA-taxonomies/    ← graph loader + EscoSuite tools
```

Install the sibling [`TA-taxonomies`](https://github.com/LFX-Talent-Angels/TA-taxonomies)
package from official **`dev`** for day-to-day integration (suite contract,
ESCO loader, indexed Locate tools). Use `main` only when you intentionally
want the last promoted publish cut.

---

## 1. Start Neo4j (once)

```bash
cd ../TA-taxonomies
docker compose up -d
docker compose ps          # wait until ta-neo4j is healthy
```

| | |
| --- | --- |
| Browser | http://localhost:7474 |
| Bolt | `bolt://localhost:7687` |
| User / password | `neo4j` / `taxonomies-dev` |

Do **not** run `docker compose down -v` — that deletes the volume.

---

## 2. Install taxonomies and load **full** ESCO (once)

The small **fixture** (61 nodes) is for CI only. `--mode fixture` **wipes**
whatever is in this Neo4j. Do not run it against the demo database.

```bash
cd ../TA-taxonomies
git fetch origin
git switch dev

python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pip install openpyxl

# xlsx must include occupations_en.xlsx
export ESCO_DATA_DIR=data/esco/raw/DATABASE
ls "$ESCO_DATA_DIR/occupations_en.xlsx"

# This replaces the graph (loader default is wipe). Run it ONCE.
python -m ta_taxonomies.suites.esco.load --mode full
```

Already have a full graph from an older taxonomies tip? Do **not** reload
fixture. Apply schema only so the Locate full-text index exists:

```bash
python - <<'PY'
from ta_taxonomies.suites.esco.db import neo4j_driver
from ta_taxonomies.suites.esco.schema import apply_schema
with neo4j_driver() as (driver, database):
    apply_schema(driver, database)
PY
```

Download the English DATABASE package from
[ESCO download](https://esco.ec.europa.eu/en/use-esco/download) if
`data/esco/raw/DATABASE/` is empty (gitignored).

### Validate the load (must look like thousands, not 61)

```bash
python - <<'PY'
from ta_taxonomies.suites.esco.db import neo4j_driver
with neo4j_driver() as (driver, database):
    with driver.session(database=database) as s:
        print("nodes", s.run("MATCH (n) RETURN count(n) AS c").single()["c"])
        print("occupations", s.run("MATCH (n:Occupation) RETURN count(n) AS c").single()["c"])
        print("nurse", s.run(
            "MATCH (n:Occupation) WHERE toLower(n.pref_label) CONTAINS 'nurse' "
            "RETURN count(n) AS c"
        ).single()["c"])
PY
```

| Check | Fixture (wrong for demo) | Full ESCO (good) |
| --- | ---: | ---: |
| Nodes | 61 | ~18,000 |
| Occupations | 6 | ~3,039 |
| `nurse` occupations | 0 | many |

Or in Neo4j Browser:

```cypher
MATCH (n) RETURN count(n) AS nodes;
MATCH (n:Occupation) WHERE toLower(n.pref_label) CONTAINS 'nurse'
RETURN n.pref_label LIMIT 10;
```

**Keep the graph:** never `--mode fixture` and never `pytest tests/suites/esco`
against this Bolt URL (those tests reload the fixture and wipe full data).

---

## 3. Install TA-agents

```bash
cd ../TA-agents
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install -e ../TA-taxonomies
cp .env.example .env
```

Edit `.env` (never commit it):

```dotenv
# Do not leave this as none for a live run
LLM_PROVIDER=litellm
LLM_MODEL=openrouter/poolside/laguna-s-2.1:free
OPENROUTER_API_KEY=sk-or-...
ANSWER_MODE=natural
LLM_REASONING_ENABLED=false

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=taxonomies-dev
```

Any OpenRouter model slug works as `openrouter/<vendor>/<model>`. The CLI
reads `.env` by itself; you do not need `source .env`.

Offline tests still use `LLM_PROVIDER=none` (pytest sets that).

---

## 4. Query

```bash
cd ../TA-agents
source .venv/bin/activate

python -m talent_angels.cli query "Where is nurse in ESCO?"
python -m talent_angels.cli query "What essential skills does a software developer need?"
python -m talent_angels.cli query "developer"
python -m talent_angels.cli query "What is the skill path from data analyst to data scientist?"

# O*NET is a second suite on the same Neo4j, not a merge. Default remains ESCO.
python -m talent_angels.cli query --suite onet "Where is Software Engineer?"
python -m talent_angels.cli connect --suite onet "What skills does a software developer need?"
```

Expect JSON with `plan`, `answer`, `tools`, `tokens`, `cost_usd`.

| Question shape | What should happen |
| --- | --- |
| Where is *X* | Locate (`search_nodes`) |
| Essential skills of *X* | Locate then Connect (`get_neighbors`) |
| Ambiguous *developer* | Locate + clarification, no neighbors |
| Path / gap from A to B | Honest Pathfind refusal, no fake skill list |

### HTTP / Swagger

```bash
uvicorn talent_angels.api.app:app --reload
```

Open http://127.0.0.1:8000/docs and `POST /v1/query` with
`{"question": "What essential skills does a software developer need?"}`
or `{"question": "Software Engineer", "suite": "onet", "kind": "occupation"}`.

---

## 5. Tests (does not reload ESCO)

```bash
# Offline — no Neo4j, no API key
ruff check .
ruff format --check .
mypy src
pytest -q --ignore=tests/integration

# Live — uses whatever is already in Neo4j (full or fixture)
pytest -q tests/integration -rs
```

---

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Node count `61`, nurse `not_found` | Fixture graph. Load `--mode full` once. |
| Full graph disappeared after tests | Taxonomies `pytest tests/suites/esco` wiped it. Reload full; do not run those tests here. |
| `LLM_PROVIDER=none` / `tokens.calls: 0` | `.env` still has stub mode, or a shell export overrides the file. |
| `Provider List:` banner | LiteLLM ad, not a crash. |
| `ModuleNotFoundError: ta_taxonomies` | `pip install -e ../TA-taxonomies` from official `dev`. |
| `cannot import EscoSuite` | Taxonomies checkout is behind org `dev`. `git fetch origin && git switch dev && git pull`. |
| `fulltext_index_missing` / slow alias search | Graph predates indexed Locate. Run `apply_schema()` (no wipe) — see §2. |

---

## Layout

```
src/talent_angels/
├── assistant/    # LangGraph: intent → plan → tool loop → answer
├── skills/       # locate / connect / pathfind (skeleton) / evaluate
├── contracts/    # AgentResult
├── llm/          # LiteLLM + stub
├── runlog/       # JSONL + rate card
├── api/          # thin FastAPI
└── cli.py
```

---

## Contributing

See [`CONTRIBUTING.md`](./CONTRIBUTING.md). Branch, `git commit -s` (DCO),
open a PR, request a mentor review. Never commit `.env` or secrets.

## License

Apache-2.0 — see [`LICENSE`](./LICENSE).
