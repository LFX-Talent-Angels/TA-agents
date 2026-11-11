"""Measure search_nodes latency and term ambiguity against a loaded ESCO graph.

Reports median / p95 latency and, for the same 30 everyday occupation terms,
how many resolve to more than one candidate and how many saturate the LIMIT 25
cap inside ``EscoSuite.search_nodes``.

    python scripts/esco_bench.py --phase cold --repeats 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from ta_taxonomies.suites.esco.db import neo4j_driver
from ta_taxonomies.suites.esco.tools import EscoSuite

# Everyday words a user would actually type, not ESCO preferred labels.
TERMS = [
    "nurse",
    "carpenter",
    "data scientist",
    "chef",
    "teacher",
    "electrician",
    "plumber",
    "welder",
    "accountant",
    "lawyer",
    "software developer",
    "driver",
    "farmer",
    "mechanic",
    "pilot",
    "architect",
    "journalist",
    "pharmacist",
    "dentist",
    "translator",
    "waiter",
    "baker",
    "barber",
    "midwife",
    "librarian",
    "economist",
    "surveyor",
    "veterinarian",
    "photographer",
    "psychologist",
]

LIMIT = 25  # the LIMIT each search_nodes stage applies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="warm", help="label for this run (cold/warm)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    latencies_ms: list[float] = []
    per_term: list[dict[str, object]] = []

    with neo4j_driver() as (driver, database):
        suite = EscoSuite(driver, database=database)
        for rep in range(args.repeats):
            for term in TERMS:
                started = time.perf_counter()
                result = suite.search_nodes(term)
                elapsed_ms = (time.perf_counter() - started) * 1000
                latencies_ms.append(elapsed_ms)
                if rep == 0:
                    n = len(result.candidates)
                    per_term.append(
                        {
                            "term": term,
                            "hits": n,
                            "ambiguous": n > 1,
                            "saturated": n >= LIMIT,
                            "method": result.candidates[0].method if n else None,
                            "top": result.candidates[0].node.label if n else None,
                            "kind": result.candidates[0].node.kind if n else None,
                            "ms_first_call": round(elapsed_ms, 1),
                        }
                    )

    ordered = sorted(latencies_ms)
    # first pass only = the cold-cache view of each distinct term
    first_pass = latencies_ms[: len(TERMS)]
    summary = {
        "phase": args.phase,
        "calls": len(latencies_ms),
        "median_ms": round(statistics.median(ordered), 1),
        "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 1),
        "max_ms": round(ordered[-1], 1),
        "first_pass_median_ms": round(statistics.median(first_pass), 1),
        "first_pass_p95_ms": round(sorted(first_pass)[int(0.95 * (len(first_pass) - 1))], 1),
        "terms": len(TERMS),
        "ambiguous": sum(1 for t in per_term if t["ambiguous"]),
        "saturated_limit_25": sum(1 for t in per_term if t["saturated"]),
        "not_found": sum(1 for t in per_term if t["hits"] == 0),
    }
    summary["ambiguous_pct"] = round(100 * summary["ambiguous"] / summary["terms"], 1)

    print(json.dumps({"summary": summary, "per_term": per_term}, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "per_term": per_term}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
