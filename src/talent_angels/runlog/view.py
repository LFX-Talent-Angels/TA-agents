"""Human-readable views of a run-log record. JSONL stays the source of truth."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from talent_angels.runlog.models import RunLogRecord, StageUsage, ToolCall

ANSWER_EXCERPT = 400
LABEL_LIMIT = 5


def stage_as_dict(stage: StageUsage) -> dict[str, Any]:
    return {
        "stage": stage.stage,
        "input_tokens": stage.input_tokens,
        "output_tokens": stage.output_tokens,
        "reasoning_tokens": stage.reasoning_tokens,
        "cache_read_input_tokens": stage.cache_read_input_tokens,
        "cache_creation_input_tokens": stage.cache_creation_input_tokens,
        "latency_ms": stage.latency_ms,
        "calls": stage.calls,
        "request_model": stage.request_model,
        "response_model": stage.response_model,
        "provider_name": stage.provider_name,
    }


def tool_as_dict(tool: ToolCall) -> dict[str, Any]:
    return tool.model_dump()


def search_texts(record: RunLogRecord) -> list[str]:
    texts: list[str] = []
    for tool in record.tools:
        if tool.name != "search_nodes":
            continue
        text = tool.args.get("text")
        if text:
            texts.append(str(text))
    return texts


def economics(record: RunLogRecord) -> dict[str, Any]:
    return {
        "run_id": record.run_id,
        "ts": record.ts,
        "provider": record.gen_ai.provider_name,
        "model": record.gen_ai.request_model,
        "calls": record.gen_ai.calls,
        "input_tokens": record.gen_ai.input_tokens,
        "output_tokens": record.gen_ai.output_tokens,
        "reasoning_tokens": record.gen_ai.reasoning_tokens,
        "cache_read_input_tokens": record.gen_ai.cache_read_input_tokens,
        "cache_creation_input_tokens": record.gen_ai.cache_creation_input_tokens,
        "cost_usd": record.cost_usd.total,
        "cost_known": record.cost_usd.known,
        "rate_card": record.cost_usd.rate_card,
        "graph_queries": record.graph.queries,
        "graph_ms": record.graph.total_ms,
        "efficiency": record.efficiency.model_dump(),
        "stages": [stage_as_dict(stage) for stage in record.gen_ai.stages],
        "tools": [tool_as_dict(tool) for tool in record.tools],
        "search_texts": search_texts(record),
        "top_labels": record.result.node_labels[:LABEL_LIMIT],
        "top_ids": record.result.node_ids[:LABEL_LIMIT],
    }


def excerpt(text: str, *, limit: int = ANSWER_EXCERPT) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def format_usd(amount: float, *, known: bool) -> str:
    if not known:
        return f"${amount:.4f} (unpriced)"
    return f"${amount:.6f}"


def format_tool_line(tool: ToolCall) -> str:
    bits = [tool.name, f"{tool.ms:.0f}ms"]
    if not tool.ok:
        bits.append("FAIL")
    for key in ("text", "kind", "node_id", "rel_types"):
        if key in tool.args and tool.args[key] is not None:
            bits.append(f"{key}={tool.args[key]!r}")
    return " ".join(bits)


def format_stage_line(stage: StageUsage) -> str:
    return (
        f"{stage.stage}: {stage.input_tokens} in / {stage.output_tokens} out"
        f" / {stage.reasoning_tokens} reasoning · {stage.latency_ms:.0f}ms"
        f" · {stage.calls} call(s)"
    )


def economics_markdown_lines(record: RunLogRecord) -> list[str]:
    gen = record.gen_ai
    cost = record.cost_usd
    lines = [
        "## Economics",
        "",
        f"- **provider / model:** {gen.provider_name or 'none'} / {gen.request_model or '—'}",
        f"- **llm calls:** {gen.calls}",
        (
            f"- **tokens:** {gen.input_tokens} in / {gen.output_tokens} out"
            f" / {gen.reasoning_tokens} reasoning"
        ),
        (
            f"- **cache tokens:** read {gen.cache_read_input_tokens}"
            f" · write {gen.cache_creation_input_tokens}"
        ),
        f"- **cost:** {format_usd(cost.total, known=cost.known)} (`{cost.rate_card or '—'}`)",
        (
            f"- **graph:** {record.graph.queries} quer"
            f"{'y' if record.graph.queries == 1 else 'ies'}"
            f" · {record.graph.total_ms:.0f}ms"
        ),
        (
            f"- **efficiency:** mode={record.efficiency.mode}"
            f" · heuristic_intent={record.efficiency.heuristic_intent}"
            f" · result_cache={record.efficiency.result_cache_hit}"
        ),
        "",
        "### LLM stages",
        "",
    ]
    if not gen.stages:
        lines.append("None.")
    else:
        for stage in gen.stages:
            lines.append(f"- {format_stage_line(stage)}")
    lines.extend(["", "### Tools", ""])
    if not record.tools:
        lines.append("None.")
    else:
        for tool in record.tools:
            lines.append(f"- {format_tool_line(tool)}")
    searches = search_texts(record)
    if searches:
        lines.extend(["", f"- **search text:** {', '.join(repr(item) for item in searches)}"])
    return lines


def percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * p
    low = int(idx)
    high = min(low + 1, len(ordered) - 1)
    frac = idx - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)
