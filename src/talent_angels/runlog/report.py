"""Summarize recent runlog.jsonl turns without a telemetry backend."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from talent_angels.runlog.models import RunLogRecord
from talent_angels.runlog.view import excerpt, format_usd, percentile, search_texts
from talent_angels.runlog.writer import read_records, runlog_path


def summarize_records(records: Sequence[RunLogRecord]) -> dict[str, Any]:
    rows = list(records)
    elapsed = [record.graph.total_ms for record in rows]
    calls = [record.gen_ai.calls for record in rows]
    by_capability: Counter[str] = Counter()
    for record in rows:
        by_capability[record.plan[-1] if record.plan else "unknown"] += 1
    return {
        "turns": len(rows),
        "llm_calls": sum(record.gen_ai.calls for record in rows),
        "input_tokens": sum(record.gen_ai.input_tokens for record in rows),
        "output_tokens": sum(record.gen_ai.output_tokens for record in rows),
        "reasoning_tokens": sum(record.gen_ai.reasoning_tokens for record in rows),
        "cost_usd": sum(record.cost_usd.total for record in rows),
        "cost_known": all(record.cost_usd.known for record in rows) if rows else True,
        "graph_queries": sum(record.graph.queries for record in rows),
        "graph_ms": sum(record.graph.total_ms for record in rows),
        "mean_llm_calls": (sum(calls) / len(calls)) if calls else None,
        "p50_graph_ms": percentile(elapsed, 0.50),
        "p95_graph_ms": percentile(elapsed, 0.95),
        "by_capability": dict(by_capability),
    }


def format_runlog_report(records: Sequence[RunLogRecord]) -> str:
    summary = summarize_records(records)
    lines = [
        "# TA-agents run log",
        "",
        f"- **turns:** {summary['turns']}",
        f"- **llm calls:** {summary['llm_calls']}"
        + (
            f" (mean {summary['mean_llm_calls']:.2f})"
            if summary["mean_llm_calls"] is not None
            else ""
        ),
        (
            f"- **tokens:** {summary['input_tokens']} in / {summary['output_tokens']} out"
            f" / {summary['reasoning_tokens']} reasoning"
        ),
        f"- **cost:** {format_usd(float(summary['cost_usd']), known=bool(summary['cost_known']))}",
        (
            f"- **graph:** {summary['graph_queries']} queries · {summary['graph_ms']:.0f}ms"
            + (
                f" · p50 {summary['p50_graph_ms']:.0f}ms · p95 {summary['p95_graph_ms']:.0f}ms"
                if summary["p50_graph_ms"] is not None
                else ""
            )
        ),
        "",
        "| ts | cap | calls | in/out | graph ms | cost | search | top | warnings |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for record in records:
        cap = record.plan[-1] if record.plan else "—"
        searches = ", ".join(search_texts(record)) or "—"
        top = record.result.node_labels[0] if record.result.node_labels else "—"
        warnings = ", ".join(record.result.warnings) or "—"
        lines.append(
            "| "
            + " | ".join(
                [
                    record.ts.replace("T", " ")[:19],
                    cap,
                    str(record.gen_ai.calls),
                    f"{record.gen_ai.input_tokens}/{record.gen_ai.output_tokens}",
                    f"{record.graph.total_ms:.0f}",
                    format_usd(record.cost_usd.total, known=record.cost_usd.known),
                    excerpt(searches, limit=40),
                    excerpt(top, limit=32),
                    excerpt(warnings, limit=40),
                ]
            )
            + " |"
        )
    if not records:
        lines.append("| — | — | 0 | 0/0 | 0 | $0 | — | — | — |")
    lines.append("")
    return "\n".join(lines)


def render_recent_report(*, path: Path | None = None, last: int = 20) -> str:
    target = path or runlog_path()
    records = read_records(path=target, last=last)
    header = f"source: `{target}` · last {last}\n\n"
    return header + format_runlog_report(records)
