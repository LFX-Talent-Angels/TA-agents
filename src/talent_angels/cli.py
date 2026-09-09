"""Headless CLI — same turn logic as the API, no server required.

python -m talent_angels.cli query "software developer"
python -m talent_angels.cli locate "software developer" --kind occupation
python -m talent_angels.cli bench
python -m talent_angels.cli report --last 20
python -m talent_angels.cli quality
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from talent_angels.assistant import ResultCache, run_turn
from talent_angels.assistant.intent import CAPABILITY_CONNECT
from talent_angels.env import load_local_dotenv
from talent_angels.evals import LocateMetrics
from talent_angels.evals.quality import (
    ObservedTurn,
    filter_cases,
    load_quality_suite,
    quality_suite_path,
    score_case,
    summarize,
    write_quality_report,
)
from talent_angels.llm.factory import get_answer_mode, get_llm_client
from talent_angels.query_details import write_query_details
from talent_angels.runlog.report import render_recent_report
from talent_angels.suites import SuiteRegistry, default_suite_registry

GOLDEN_LOCATE_PATH = Path(__file__).resolve().parents[2] / "tests" / "evals" / "golden_locate.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="talent-angels")
    sub = parser.add_subparsers(dest="command", required=True)

    query_parser = sub.add_parser("query", help="Ask the assistant (heuristic intent routing).")
    query_parser.add_argument("question")
    query_parser.add_argument("--kind", default=None)
    query_parser.add_argument(
        "--suite",
        default=None,
        help="Taxonomy suite: esco (default) or onet. Not a merge.",
    )

    locate_parser = sub.add_parser("locate", help="Run Locate directly (bypasses intent routing).")
    locate_parser.add_argument("question")
    locate_parser.add_argument("--kind", default=None)
    locate_parser.add_argument(
        "--suite", default=None, help="Taxonomy suite: esco (default) or onet."
    )

    connect_parser = sub.add_parser(
        "connect", help="Run Locate → Connect directly (bypasses intent routing)."
    )
    connect_parser.add_argument("question")
    connect_parser.add_argument("--kind", default=None)
    connect_parser.add_argument(
        "--suite", default=None, help="Taxonomy suite: esco (default) or onet."
    )

    sub.add_parser(
        "bench",
        help="Run the golden Locate set (baseline vs. result-cache); report cost/latency/accuracy.",
    )

    report_parser = sub.add_parser(
        "report",
        help="Summarize recent turns from runlog.jsonl (tokens, cost, search, warnings).",
    )
    report_parser.add_argument("--last", type=int, default=20)
    report_parser.add_argument("--path", default=None, help="Override RUNLOG_PATH")

    quality_parser = sub.add_parser(
        "quality",
        help="Score the fixed question card and write a full local report.",
    )
    quality_parser.add_argument("--suite", default=None, help="Path to quality_suite.json")
    quality_parser.add_argument("--tier", default=None)
    quality_parser.add_argument("--family", default=None)
    quality_parser.add_argument("--ids", default=None, help="Comma-separated case ids")
    quality_parser.add_argument("--limit", type=int, default=None)
    quality_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print per-case progress (final JSON still goes to stdout).",
    )

    return parser


def main(argv: Sequence[str] | None = None, *, registry: SuiteRegistry | None = None) -> int:
    load_local_dotenv()
    args = _build_parser().parse_args(argv)
    selected_registry = registry or default_suite_registry()

    if args.command == "bench":
        return _run_bench(selected_registry)
    if args.command == "report":
        print(render_recent_report(path=Path(args.path) if args.path else None, last=args.last))
        return 0
    if args.command == "quality":
        return _run_quality(args, selected_registry)

    suite_name = args.suite if args.command in {"query", "locate", "connect"} else None
    with selected_registry.open(suite_name) as runtime:
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


def _progress(message: str, *, quiet: bool) -> None:
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def _run_quality(args: argparse.Namespace, registry: SuiteRegistry) -> int:
    suite_path = quality_suite_path(Path(args.suite) if args.suite else None)
    if not suite_path.is_file():
        print(
            f"quality suite not found: {suite_path}\nSet QUALITY_SUITE_PATH or pass --suite.",
            file=sys.stderr,
        )
        return 2
    card = load_quality_suite(suite_path)
    wanted_ids = [item.strip() for item in args.ids.split(",")] if args.ids else None
    cases = filter_cases(card.cases, tier=args.tier, family=args.family, case_ids=wanted_ids)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        print("no quality cases selected", file=sys.stderr)
        return 2

    quiet = bool(getattr(args, "quiet", False))
    total = len(cases)
    provider = ""
    model = ""
    _progress(f"quality: {total} case(s) from {suite_path}", quiet=quiet)
    verdicts = []
    with registry.open() as runtime:
        llm_client = get_llm_client()
        provider = str(getattr(llm_client, "provider", "") or "none")
        model = str(getattr(llm_client, "model", "") or "—")
        _progress(f"quality: provider={provider} model={model}", quiet=quiet)
        for index, case in enumerate(cases, start=1):
            preview = case.question if len(case.question) <= 72 else case.question[:71] + "…"
            _progress(f"[{index}/{total}] running {case.id} — {preview}", quiet=quiet)
            started = time.perf_counter()
            try:
                outcome = run_turn(
                    suite=runtime.suite,
                    suite_name=runtime.name,
                    llm_client=llm_client,
                    question=case.question,
                    answer_mode=get_answer_mode(),
                )
                write_query_details(outcome, question=case.question)
                observed = ObservedTurn.from_outcome(
                    outcome, elapsed_s=time.perf_counter() - started
                )
            except Exception as exc:  # noqa: BLE001 — live card must record the miss
                observed = ObservedTurn(
                    capability="",
                    warnings=(),
                    node_ids=(),
                    node_labels=(),
                    edge_count=0,
                    confidence=None,
                    tool_names=(),
                    tools=(),
                    search_texts=(),
                    answer="",
                    elapsed_s=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
            verdict = score_case(case, observed)
            verdicts.append(verdict)
            mark = "PASS" if verdict.passed else "FAIL"
            elapsed = verdict.elapsed_s if verdict.elapsed_s is not None else 0.0
            reason = f" — {verdict.reasons[0]}" if verdict.reasons else ""
            _progress(
                f"[{index}/{total}] {mark} {case.id}  "
                f"{elapsed:.1f}s  {verdict.llm_calls} call(s)  "
                f"{verdict.input_tokens}/{verdict.output_tokens} tok"
                f"{reason}",
                quiet=quiet,
            )

    summary = summarize(verdicts)
    markdown_path, json_path = write_quality_report(
        provider=provider,
        model=model,
        summary=summary,
        verdicts=verdicts,
    )
    _progress(
        f"quality: done {summary['passed']}/{summary['questions']}  "
        f"failed {summary['failed']}  "
        f"{summary['input_tokens']}/{summary['output_tokens']} tok  "
        f"report {markdown_path}",
        quiet=quiet,
    )
    print(
        json.dumps(
            {
                "score": f"{summary['passed']}/{summary['questions']}",
                "failed": summary["failed"],
                "tokens": {
                    "input": summary["input_tokens"],
                    "output": summary["output_tokens"],
                    "reasoning": summary["reasoning_tokens"],
                    "calls": summary["llm_calls"],
                },
                "cost_usd": summary["cost_usd"],
                "markdown": str(markdown_path),
                "json": str(json_path),
            },
            indent=2,
        )
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
