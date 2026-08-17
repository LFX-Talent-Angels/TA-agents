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

from talent_angels.assistant import ResultCache, run_turn
from talent_angels.assistant.intent import CAPABILITY_CONNECT
from talent_angels.env import load_local_dotenv
from talent_angels.evals import LocateMetrics
from talent_angels.llm.factory import get_answer_mode, get_llm_client
from talent_angels.query_details import write_query_details
from talent_angels.suites import SuiteRegistry, default_suite_registry

GOLDEN_LOCATE_PATH = Path(__file__).resolve().parents[2] / "tests" / "evals" / "golden_locate.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="talent-angels")
    sub = parser.add_subparsers(dest="command", required=True)

    query_parser = sub.add_parser("query", help="Ask the assistant (heuristic intent routing).")
    query_parser.add_argument("question")
    query_parser.add_argument("--kind", default=None)

    locate_parser = sub.add_parser("locate", help="Run Locate directly (bypasses intent routing).")
    locate_parser.add_argument("question")
    locate_parser.add_argument("--kind", default=None)

    connect_parser = sub.add_parser(
        "connect", help="Run Locate → Connect directly (bypasses intent routing)."
    )
    connect_parser.add_argument("question")
    connect_parser.add_argument("--kind", default=None)

    sub.add_parser(
        "bench",
        help="Run the golden Locate set (baseline vs. result-cache); report cost/latency/accuracy.",
    )

    return parser


def main(argv: Sequence[str] | None = None, *, registry: SuiteRegistry | None = None) -> int:
    load_local_dotenv()
    args = _build_parser().parse_args(argv)
    selected_registry = registry or default_suite_registry()

    if args.command == "bench":
        return _run_bench(selected_registry)

    with selected_registry.open() as runtime:
        llm_client = get_llm_client()
        outcome = run_turn(
            suite=runtime.suite,
            suite_name=runtime.name,
            llm_client=llm_client,
            question=args.question,
            kind=args.kind,
            answer_mode=get_answer_mode(),
            force_locate=(args.command == "locate"),
            force_capability=(CAPABILITY_CONNECT if args.command == "connect" else None),
        )

    record = outcome.record
    details_path = write_query_details(outcome, question=args.question)
    print(
        json.dumps(
            {
                "answer": outcome.answer,
                "run_id": record.run_id,
                "capability": outcome.capability,
                "suite": outcome.result.suite,
                "plan": record.plan,
                "confidence": outcome.result.confidence,
                "warnings": outcome.result.warnings,
                "node_count": len(outcome.result.nodes),
                "details": str(details_path),
                "tools": [tool.model_dump() for tool in record.tools],
                "tokens": {
                    "input": record.gen_ai.input_tokens,
                    "output": record.gen_ai.output_tokens,
                    "reasoning": record.gen_ai.reasoning_tokens,
                    "calls": record.gen_ai.calls,
                    "stages": [
                        {
                            "stage": stage.stage,
                            "input": stage.input_tokens,
                            "output": stage.output_tokens,
                            "reasoning": stage.reasoning_tokens,
                        }
                        for stage in record.gen_ai.stages
                    ],
                },
                "cost_usd": {
                    "known": record.cost_usd.known,
                    "total": record.cost_usd.total,
                    "rate_card": record.cost_usd.rate_card,
                },
            },
            indent=2,
        )
    )
    return 0


def _run_bench(registry: SuiteRegistry) -> int:
    cases = json.loads(GOLDEN_LOCATE_PATH.read_text(encoding="utf-8"))["cases"]

    with registry.open() as runtime:
        llm_client = get_llm_client()

        baseline = _run_pass(runtime.suite, runtime.name, llm_client, cases, cache=None)

        result_cache = ResultCache()
        _run_pass(
            runtime.suite, runtime.name, llm_client, cases, cache=result_cache
        )  # warm the cache
        optimized = _run_pass(
            runtime.suite, runtime.name, llm_client, cases, cache=result_cache
        )  # all hits

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


def _run_pass(
    suite: Any,
    suite_name: str,
    llm_client: Any,
    cases: list[dict],
    *,
    cache: ResultCache | None,
) -> dict:
    metrics = LocateMetrics()
    total_cost = 0.0
    total_ms = 0.0
    for case in cases:
        outcome = run_turn(
            suite=suite,
            suite_name=suite_name,
            llm_client=llm_client,
            question=case["question"],
            kind=case["kind"],
            answer_mode="structured",
            force_locate=True,
            cache=cache,
        )
        total_cost += outcome.record.cost_usd.total
        total_ms += outcome.record.graph.total_ms
        metrics.observe(
            case.get("metric", "hit_at_1"),
            case["expected_top_id"],
            [node.id for node in outcome.result.nodes],
        )

    return {
        **metrics.as_dict(),
        "total_cost_usd": total_cost,
        "total_graph_ms": total_ms,
        "mean_cost_usd_per_locate": total_cost / len(cases),
    }


if __name__ == "__main__":
    sys.exit(main())
