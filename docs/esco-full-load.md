# Loading the full ESCO graph locally

The committed fixture is an ICT subset (6 occupations, 51 skills). Every
non-computing question fails on missing data, not on a broken pipeline, so
product evaluation needs the real classification. This is how to load it and
what it actually costs.

Everything below runs against a **dedicated** Neo4j on Bolt port **7690**
(`ta-neo4j-full`), so the day-to-day database on 7687 is never touched.

> `python -m ta_taxonomies.suites.esco.load` with no arguments defaults to
> `--mode fixture` **and wipes the ESCO labels first**. Always pass `--mode`
> and always point `NEO4J_URI` at the database you mean.

## 1. Start an isolated Neo4j

```bash
docker run -d --name ta-neo4j-full \
  -p 7690:7687 -p 7477:7474 \
  -e NEO4J_AUTH=neo4j/escofull-dev \
  -e NEO4J_server_memory_heap_initial__size=1G \
  -e NEO4J_server_memory_heap_max__size=2G \
  -e NEO4J_server_memory_pagecache_size=1G \
  -v ta_neo4j_full_data:/data -v ta_neo4j_full_logs:/logs \
  neo4j:5-community
```

The `ta_neo4j_full_data` volume keeps the loaded graph across restarts — load
once, reuse. `docker restart ta-neo4j-full` is enough after a reboot.

## 2. Download the official English classification

https://esco.ec.europa.eu/en/use-esco/download serves the file behind a
confirmation page, but the confirmation link resolves to a stable path:

```bash
mkdir -p data/esco/raw && cd data/esco/raw
curl -sSL -A "Mozilla/5.0" -o esco-v1.2.1-en-csv.zip \
  "https://ec.europa.eu/esco/download/ESCO%20dataset%20-%20v1.2.1%20-%20classification%20-%20en%20-%20csv.zip"
unzip -q esco-v1.2.1-en-csv.zip -d csv_v1.2.1
```

No registration, no login, no cookie. `data/` is gitignored — the dump stays
out of the repository (pointer, not payload).

**The portal no longer publishes an xlsx bundle** — the classification ships as
CSV, RDF or TTL only. The loader's `--mode full` reads `*_en.xlsx`, so convert
the official CSVs into that shape once. Column names are copied verbatim:

```bash
python scripts/esco_csv_to_xlsx.py data/esco/raw/csv_v1.2.1 data/esco/raw/DATABASE
```

## 3. Load and validate

```bash
NEO4J_URI=bolt://localhost:7690 NEO4J_USER=neo4j NEO4J_PASSWORD=escofull-dev \
python -m ta_taxonomies.suites.esco.load --mode full --data-dir data/esco/raw/DATABASE
```

The loader validates its own load (counts, no dangling `HAS_SKILL`, no blank
identities, every occupation linked to at least one skill) and exits non-zero
if any invariant breaks. ESCO v1.2.1 gives:

| Entity              | Count   |
| ------------------- | ------- |
| Occupation          | 3,039   |
| Skill               | 13,939  |
| IscoGroup           | 619     |
| SkillGroup          | 640     |
| `HAS_SKILL`         | 126,051 |
| `BROADER_THAN`      | 24,467  |
| `CLASSIFIED_UNDER`  | 3,039   |
| `RELATED_TO`        | 5,818   |

## 4. Measure

```bash
NEO4J_URI=bolt://localhost:7690 NEO4J_USER=neo4j NEO4J_PASSWORD=escofull-dev \
python scripts/esco_bench.py --phase warm --repeats 5
```

`--phase cold` after a `docker restart ta-neo4j-full` measures the cold-cache
path. The script also reports, over 30 everyday occupation terms, how many
resolve ambiguously and how many saturate the `LIMIT 25` cap in
`search_nodes`.

## 5. Ask the demo questions

```bash
NEO4J_URI=bolt://localhost:7690 NEO4J_USER=neo4j NEO4J_PASSWORD=escofull-dev \
LLM_PROVIDER=none python -m talent_angels.cli query "carpenter"
```

`LLM_PROVIDER=none` uses the deterministic stub: zero tokens, no API key.

## Measured, 2026-08-23 (Apple Silicon, Docker Desktop, neo4j:5-community)

| Metric                                | Measured                          |
| ------------------------------------- | --------------------------------- |
| Download (v1.2.1 en, csv zip)         | 10.0 MB zip / 49 MB unpacked, 25 s |
| CSV → xlsx conversion (one-off)       | 5.7 s → 14 MB                     |
| xlsx parse + normalize                | 6.7 s                             |
| MERGE into Neo4j                      | 19.7 s                            |
| Validation                            | 0.9 s                             |
| **Total ingest**                      | **22 s empty DB / 28 s with wipe** |
| Store on disk (`/data/databases`)     | 45 MB                             |
| Transaction logs (`/data/transactions`) | 515 MB (default retention)      |
| Container RSS after load              | ~1.5 GB (heap 2 G + pagecache 1 G) |
| `search_nodes` warm, 150 calls        | median 11 ms, p95 75 ms           |
| `search_nodes` cold, 30 calls         | median 37 ms, p95 323 ms          |
| Ambiguous terms (of 30)               | 5 (16.7 %)                        |
| Terms saturating `LIMIT 25`           | 3 (nurse, mechanic, pilot)        |

The transaction logs dominate the volume — the graph itself is 45 MB. Prune
them with `docker exec ta-neo4j-full neo4j-admin database ...` or accept the
one-off cost; they do not grow with read traffic.
