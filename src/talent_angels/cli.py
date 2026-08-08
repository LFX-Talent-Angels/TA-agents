"""Headless CLI — same turn logic as the API, no server required.

    python -m talent_angels.cli query "software developer"
    python -m talent_angels.cli locate "software developer" --kind occupation
    python -m talent_angels.cli bench
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ta_taxonomies.suites.esco.db import neo4j_driver
from ta_taxonomies.suites.esco.tools import EscoSuite

from talent_angels.assistant import ResultCache, run_turn
from talent_angels.llm import get_llm_client

GOLDEN_LOCATE_PATH = Path(__file__).resolve().parents[2] / "tests" / "evals" / "golden_locate.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="talent-angels")
    sub = parser.add_subparsers(dest="command", required=True)

    query_parser = sub.add_parser("query", help="Ask the assistant (heuristic intent routing).")
    query_parser.add_argument("question")
    query_parser.add_argument("--kind", default=None)

    locate_parser = sub.add_parser(
        "locate", help="Run Locate directly (bypasses intent routing)."
    )
    locate_parser.add_argument("question")
    locate_parser.add_argument("--kind", default=None)

    sub.add_parser(
        "bench",
        help="Run the golden Locate set (baseline vs. result-cache); report cost/latency/accuracy.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "bench":
        return _run_bench()

    with neo4j_driver() as (driver, database):
        suite = EscoSuite(driver, database=database)
        llm_client = get_llm_client()
        outcome = run_turn(
            suite=suite,
            llm_client=llm_client,
            question=args.question,
            kind=args.kind,
            force_locate=(args.command == "locate"),
        )

    print(
        json.dumps(
            {
                "run_id": outcome.record.run_id,
                "capability": outcome.capability,
                "answer": outcome.answer,
                "confidence": outcome.result.confidence,
                "warnings": outcome.result.warnings,
                "cost_usd": outcome.record.cost_usd.total,
            },
            indent=2,
        )
    )
    return 0


def _run_bench() -> int:
    cases = json.loads(GOLDEN_LOCATE_PATH.read_text(encoding="utf-8"))["cases"]

    with neo4j_driver() as (driver, database):
        suite = EscoSuite(driver, database=database)
        llm_client = get_llm_client()

        baseline = _run_pass(suite, llm_client, cases, cache=None)

        result_cache = ResultCache()
        _run_pass(suite, llm_client, cases, cache=result_cache)  # warm the cache
        optimized = _run_pass(suite, llm_client, cases, cache=result_cache)  # all hits

    mean_cost_per_locate = baseline["total_cost_usd"] / len(cases)
    report = {
        "questions": len(cases),
        "baseline": baseline,
        "optimized_with_result_cache": optimized,
        "graph_latency_saved_ms": baseline["total_graph_ms"] - optimized["total_graph_ms"],
        "scale_up_usd": {
            "per_1k_locate": mean_cost_per_locate * 1_000,
            "per_10k_locate": mean_cost_per_locate * 10_000,
            "per_100k_locate": mean_cost_per_locate * 100_000,
        },
    }
    print(json.dumps(report, indent=2))
    return 0


def _run_pass(suite: Any, llm_client: Any, cases: list[dict], *, cache: ResultCache | None) -> dict:
    hits = 0
    scored = 0
    total_cost = 0.0
    total_ms = 0.0
    for case in cases:
        outcome = run_turn(
            suite=suite,
            llm_client=llm_client,
            question=case["question"],
            kind=case["kind"],
            force_locate=True,
            cache=cache,
        )
        total_cost += outcome.record.cost_usd.total
        total_ms += outcome.record.graph.total_ms
        if case["expected_top_id"] is not None:
            scored += 1
            if outcome.result.nodes and outcome.result.nodes[0].id == case["expected_top_id"]:
                hits += 1

    return {
        "hit_at_1_accuracy": (hits / scored) if scored else None,
        "total_cost_usd": total_cost,
        "total_graph_ms": total_ms,
        "mean_cost_usd_per_locate": total_cost / len(cases),
    }


if __name__ == "__main__":
    sys.exit(main())
