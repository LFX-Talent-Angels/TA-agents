"""Score a fixed TA-agents question card from typed turn results."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from talent_angels.assistant.turn import TurnOutcome
from talent_angels.runlog.view import (
    excerpt,
    format_usd,
    percentile,
    search_texts,
    stage_as_dict,
    tool_as_dict,
)

Family = Literal[
    "locate_unique",
    "locate_alias",
    "locate_ambiguous",
    "locate_not_found",
    "locate_natural",
    "connect_template",
    "connect_natural",
    "pathfind_refuse",
    "skill_locate",
]
Tier = Literal["contract", "quality"]
Metric = Literal["hit_at_1", "candidate_recall"]

FAMILIES: tuple[Family, ...] = (
    "locate_unique",
    "locate_alias",
    "locate_ambiguous",
    "locate_not_found",
    "locate_natural",
    "connect_template",
    "connect_natural",
    "pathfind_refuse",
    "skill_locate",
)
TIERS: tuple[Tier, ...] = ("contract", "quality")
DEFAULT_QUALITY_SUITE = Path("data/local/evals/quality_suite.json")
LABEL_LIMIT = 5


class QualityCase(BaseModel):
    """One scored question. Expectations are product-correct, not 'whatever we do today'."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    family: Family
    tier: Tier = "contract"
    expected_capability: tuple[str, ...]
    expected_warnings: tuple[str, ...] = ()
    forbidden_warnings: tuple[str, ...] = ()
    expected_top_id: str | None = None
    expected_ids: tuple[str, ...] = ()
    require_id_in_candidates: bool = False
    unique_must_match_expected: bool = False
    min_confidence: float | None = None
    min_neighbors: int | None = None
    require_tools: tuple[str, ...] = ()
    forbid_tools: tuple[str, ...] = ()
    metric: Metric | None = None

    @field_validator(
        "expected_capability",
        "expected_warnings",
        "forbidden_warnings",
        "expected_ids",
        "require_tools",
        "forbid_tools",
        mode="before",
    )
    @classmethod
    def _tuple_of_str(cls, value: object) -> object:
        if isinstance(value, str):
            return (value,)
        return value


class QualitySuiteFile(BaseModel):
    suite: str
    note: str = ""
    ids: dict[str, str] = Field(default_factory=dict)
    cases: list[QualityCase]


@dataclass(frozen=True)
class ObservedTurn:
    capability: str
    warnings: tuple[str, ...]
    node_ids: tuple[str, ...]
    node_labels: tuple[str, ...]
    edge_count: int
    confidence: float | None
    tool_names: tuple[str, ...]
    tools: tuple[dict[str, Any], ...]
    search_texts: tuple[str, ...]
    answer: str
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True
    rate_card: str = ""
    graph_ms: float = 0.0
    stages: tuple[dict[str, Any], ...] = ()
    run_id: str | None = None
    elapsed_s: float | None = None
    error: str | None = None

    @classmethod
    def from_outcome(cls, outcome: TurnOutcome, *, elapsed_s: float | None = None) -> ObservedTurn:
        record = outcome.record
        return cls(
            capability=outcome.capability,
            warnings=tuple(outcome.result.warnings),
            node_ids=tuple(node.id for node in outcome.result.nodes),
            node_labels=tuple(node.pref_label for node in outcome.result.nodes),
            edge_count=len(outcome.result.edges),
            confidence=outcome.result.confidence,
            tool_names=tuple(tool.name for tool in record.tools),
            tools=tuple(tool_as_dict(tool) for tool in record.tools),
            search_texts=tuple(search_texts(record)),
            answer=outcome.answer,
            llm_calls=record.gen_ai.calls,
            input_tokens=record.gen_ai.input_tokens,
            output_tokens=record.gen_ai.output_tokens,
            reasoning_tokens=record.gen_ai.reasoning_tokens,
            cost_usd=record.cost_usd.total,
            cost_known=record.cost_usd.known,
            rate_card=record.cost_usd.rate_card,
            graph_ms=record.graph.total_ms,
            stages=tuple(stage_as_dict(stage) for stage in record.gen_ai.stages),
            run_id=record.run_id,
            elapsed_s=elapsed_s,
        )


@dataclass
class CaseVerdict:
    case_id: str
    question: str
    family: Family
    tier: Tier
    passed: bool
    reasons: list[str] = field(default_factory=list)
    capability: str | None = None
    warnings: list[str] = field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0
    top_id: str | None = None
    top_labels: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    tool_details: list[dict[str, Any]] = field(default_factory=list)
    search_texts: list[str] = field(default_factory=list)
    answer: str = ""
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True
    rate_card: str = ""
    graph_ms: float = 0.0
    stages: list[dict[str, Any]] = field(default_factory=list)
    confidence: float | None = None
    run_id: str | None = None
    elapsed_s: float | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "question": self.question,
            "family": self.family,
            "tier": self.tier,
            "passed": self.passed,
            "reasons": self.reasons,
            "capability": self.capability,
            "warnings": self.warnings,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "top_id": self.top_id,
            "top_labels": self.top_labels,
            "tools": self.tools,
            "tool_details": self.tool_details,
            "search_texts": self.search_texts,
            "answer": self.answer,
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": self.cost_usd,
            "cost_known": self.cost_known,
            "rate_card": self.rate_card,
            "graph_ms": self.graph_ms,
            "stages": self.stages,
            "confidence": self.confidence,
            "run_id": self.run_id,
            "elapsed_s": self.elapsed_s,
            "error": self.error,
        }


def quality_suite_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    raw = os.environ.get("QUALITY_SUITE_PATH", "").strip()
    return Path(raw) if raw else DEFAULT_QUALITY_SUITE


def load_quality_suite(path: Path) -> QualitySuiteFile:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return QualitySuiteFile.model_validate(payload)


def filter_cases(
    cases: Sequence[QualityCase],
    *,
    tier: str | None = None,
    family: str | None = None,
    case_ids: Sequence[str] | None = None,
) -> list[QualityCase]:
    selected = list(cases)
    if tier is not None:
        selected = [case for case in selected if case.tier == tier]
    if family is not None:
        selected = [case for case in selected if case.family == family]
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in selected if case.id in wanted]
    return selected


def score_case(case: QualityCase, observed: ObservedTurn) -> CaseVerdict:
    reasons: list[str] = []
    if observed.error:
        reasons.append(observed.error)
    elif observed.capability not in case.expected_capability:
        reasons.append(
            f"capability {observed.capability!r} not in {list(case.expected_capability)}"
        )

    for warning in case.expected_warnings:
        if warning not in observed.warnings:
            reasons.append(f"missing warning {warning!r}")
    for warning in case.forbidden_warnings:
        if warning in observed.warnings:
            reasons.append(f"forbidden warning {warning!r}")

    if case.family == "locate_not_found" and observed.node_ids:
        reasons.append("expected no nodes for not_found")

    if case.family == "pathfind_refuse" and observed.node_ids:
        reasons.append("pathfind refuse should return no nodes")

    default_hit = case.expected_top_id and not case.require_id_in_candidates
    if case.metric == "hit_at_1" or (case.metric is None and default_hit):
        if case.expected_top_id:
            if not observed.node_ids or observed.node_ids[0] != case.expected_top_id:
                actual = observed.node_ids[0] if observed.node_ids else None
                reasons.append(f"hit@1 expected {case.expected_top_id}, got {actual}")

    allowed = set(case.expected_ids)
    if case.expected_top_id:
        allowed.add(case.expected_top_id)
    if case.require_id_in_candidates and allowed:
        if not allowed.intersection(observed.node_ids):
            reasons.append("expected id not present in returned nodes")

    if case.unique_must_match_expected and "ambiguous" not in observed.warnings and allowed:
        if not observed.node_ids or observed.node_ids[0] not in allowed:
            actual = observed.node_ids[0] if observed.node_ids else None
            reasons.append(f"unique answer {actual} is not an expected occupation")

    if case.min_confidence is not None:
        if observed.confidence is None or observed.confidence < case.min_confidence:
            reasons.append(f"confidence {observed.confidence} below {case.min_confidence}")

    if case.min_neighbors is not None and observed.edge_count < case.min_neighbors:
        reasons.append(f"neighbors {observed.edge_count} below {case.min_neighbors}")

    for tool in case.require_tools:
        if tool not in observed.tool_names:
            reasons.append(f"missing tool {tool!r}")
    for tool in case.forbid_tools:
        if tool in observed.tool_names:
            reasons.append(f"forbidden tool {tool!r}")

    return CaseVerdict(
        case_id=case.id,
        question=case.question,
        family=case.family,
        tier=case.tier,
        passed=not reasons,
        reasons=reasons,
        capability=observed.capability,
        warnings=list(observed.warnings),
        node_count=len(observed.node_ids),
        edge_count=observed.edge_count,
        top_id=observed.node_ids[0] if observed.node_ids else None,
        top_labels=list(observed.node_labels[:LABEL_LIMIT]),
        tools=list(observed.tool_names),
        tool_details=list(observed.tools),
        search_texts=list(observed.search_texts),
        answer=observed.answer,
        llm_calls=observed.llm_calls,
        input_tokens=observed.input_tokens,
        output_tokens=observed.output_tokens,
        reasoning_tokens=observed.reasoning_tokens,
        cost_usd=observed.cost_usd,
        cost_known=observed.cost_known,
        rate_card=observed.rate_card,
        graph_ms=observed.graph_ms,
        stages=list(observed.stages),
        confidence=observed.confidence,
        run_id=observed.run_id,
        elapsed_s=observed.elapsed_s,
        error=observed.error,
    )


def summarize(verdicts: Iterable[CaseVerdict]) -> dict[str, Any]:
    rows = list(verdicts)
    passed = sum(1 for row in rows if row.passed)
    by_tier: Counter[str] = Counter()
    pass_by_tier: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
    pass_by_family: Counter[str] = Counter()
    elapsed = [row.elapsed_s for row in rows if row.elapsed_s is not None]
    for row in rows:
        by_tier[row.tier] += 1
        by_family[row.family] += 1
        if row.passed:
            pass_by_tier[row.tier] += 1
            pass_by_family[row.family] += 1
    total_calls = sum(row.llm_calls for row in rows)
    return {
        "questions": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "pass_rate": (passed / len(rows)) if rows else None,
        "llm_calls": total_calls,
        "mean_llm_calls": (total_calls / len(rows)) if rows else None,
        "input_tokens": sum(row.input_tokens for row in rows),
        "output_tokens": sum(row.output_tokens for row in rows),
        "reasoning_tokens": sum(row.reasoning_tokens for row in rows),
        "cost_usd": sum(row.cost_usd for row in rows),
        "cost_known": all(row.cost_known for row in rows) if rows else True,
        "graph_ms": sum(row.graph_ms for row in rows),
        "mean_elapsed_s": (sum(elapsed) / len(elapsed)) if elapsed else None,
        "p50_elapsed_s": percentile(elapsed, 0.50),
        "p95_elapsed_s": percentile(elapsed, 0.95),
        "by_tier": {
            tier: {
                "questions": by_tier[tier],
                "passed": pass_by_tier[tier],
                "pass_rate": (pass_by_tier[tier] / by_tier[tier]) if by_tier[tier] else None,
            }
            for tier in TIERS
            if by_tier[tier]
        },
        "by_family": {
            family: {
                "questions": by_family[family],
                "passed": pass_by_family[family],
                "pass_rate": (
                    pass_by_family[family] / by_family[family] if by_family[family] else None
                ),
            }
            for family in FAMILIES
            if by_family[family]
        },
    }


def _pct(rate: float | None) -> str:
    if rate is None:
        return "—"
    return f"{rate:.0%}"


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}s"


def _case_detail_lines(row: CaseVerdict) -> list[str]:
    reason = "; ".join(row.reasons) or "unspecified"
    searches = ", ".join(repr(item) for item in row.search_texts) or "—"
    labels = ", ".join(row.top_labels) or "—"
    stages = (
        "; ".join(
            f"{stage.get('stage')}: {stage.get('input_tokens')}/{stage.get('output_tokens')}"
            f" ({float(stage.get('latency_ms') or 0):.0f}ms)"
            for stage in row.stages
        )
        or "none"
    )
    return [
        f"- **question:** {row.question}",
        f"- **tier / family:** {row.tier} / {row.family}",
        f"- **capability:** {row.capability or '—'}",
        f"- **warnings:** {', '.join(row.warnings) or 'none'}",
        f"- **nodes / edges:** {row.node_count} / {row.edge_count}",
        f"- **top id:** `{row.top_id}`" if row.top_id else "- **top id:** —",
        f"- **top labels:** {labels}",
        f"- **search text:** {searches}",
        f"- **tools:** {', '.join(row.tools) or 'none'}",
        (
            f"- **tokens:** {row.llm_calls} calls · {row.input_tokens} in /"
            f" {row.output_tokens} out / {row.reasoning_tokens} reasoning"
        ),
        f"- **stages:** {stages}",
        f"- **graph:** {row.graph_ms:.0f}ms · wall {_fmt_seconds(row.elapsed_s)}",
        f"- **cost:** {format_usd(row.cost_usd, known=row.cost_known)}",
        f"- **run_id:** `{row.run_id}`" if row.run_id else "- **run_id:** —",
        f"- **reason:** {reason}",
        f"- **answer:** {excerpt(row.answer) or '—'}",
    ]


def format_quality_report(
    *,
    provider: str,
    model: str,
    summary: dict[str, Any],
    verdicts: Sequence[CaseVerdict],
) -> str:
    """Human-readable scorecard for a quality run."""
    questions = int(summary.get("questions") or 0)
    passed = int(summary.get("passed") or 0)
    failed = int(summary.get("failed") or 0)
    lines = [
        "# TA-agents quality report",
        "",
        f"- **provider:** {provider or 'none'}",
        f"- **model:** {model or '—'}",
        f"- **score:** {passed}/{questions} ({_pct(summary.get('pass_rate'))})",
        f"- **failed:** {failed}",
        (
            f"- **llm calls:** {summary.get('llm_calls', 0)}"
            + (
                f" (mean {summary['mean_llm_calls']:.2f})"
                if summary.get("mean_llm_calls") is not None
                else ""
            )
        ),
        (
            f"- **tokens:** {summary.get('input_tokens', 0)} in /"
            f" {summary.get('output_tokens', 0)} out /"
            f" {summary.get('reasoning_tokens', 0)} reasoning"
        ),
        (
            "- **cost:** "
            + format_usd(
                float(summary.get("cost_usd") or 0),
                known=bool(summary.get("cost_known", True)),
            )
        ),
        (
            f"- **latency:** mean {_fmt_seconds(summary.get('mean_elapsed_s'))}"
            f" · p50 {_fmt_seconds(summary.get('p50_elapsed_s'))}"
            f" · p95 {_fmt_seconds(summary.get('p95_elapsed_s'))}"
        ),
        f"- **graph:** {float(summary.get('graph_ms') or 0):.0f}ms",
        "",
        "## By tier",
        "",
        "| tier | passed | questions | rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    by_tier = summary.get("by_tier") or {}
    for tier, stats in by_tier.items():
        lines.append(
            f"| {tier} | {stats['passed']} | {stats['questions']} | "
            f"{_pct(stats.get('pass_rate'))} |"
        )
    lines.extend(
        [
            "",
            "## By family",
            "",
            "| family | passed | questions | rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    by_family = summary.get("by_family") or {}
    for family, stats in by_family.items():
        lines.append(
            f"| {family} | {stats['passed']} | {stats['questions']} | "
            f"{_pct(stats.get('pass_rate'))} |"
        )

    failures = [row for row in verdicts if not row.passed]
    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("None.")
    else:
        for row in failures:
            lines.append(f"### `{row.case_id}` — FAIL")
            lines.append("")
            lines.extend(_case_detail_lines(row))
            lines.append("")

    lines.extend(["", "## All cases", "", format_markdown_table(verdicts), ""])
    return "\n".join(lines)


def write_quality_report(
    *,
    provider: str,
    model: str,
    summary: dict[str, Any],
    verdicts: Sequence[CaseVerdict],
    target_dir: Path | None = None,
    stamp: str | None = None,
) -> tuple[Path, Path]:
    """Write markdown + JSON under the local details dir (not git)."""
    from talent_angels.query_details import details_dir

    folder = target_dir or details_dir()
    folder.mkdir(parents=True, exist_ok=True)
    slug = stamp or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    markdown_path = folder / f"quality-{slug}.md"
    json_path = folder / f"quality-{slug}.json"
    markdown_path.write_text(
        format_quality_report(provider=provider, model=model, summary=summary, verdicts=verdicts),
        encoding="utf-8",
    )
    payload = {
        "provider": provider,
        "model": model,
        "summary": summary,
        "cases": [row.as_dict() for row in verdicts],
        "markdown": str(markdown_path),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return markdown_path, json_path


def format_markdown_table(verdicts: Sequence[CaseVerdict]) -> str:
    lines = [
        "| result | id | cap | calls | in/out | ms | cost | nodes | search | reason |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in verdicts:
        mark = "PASS" if row.passed else "FAIL"
        reason = "; ".join(row.reasons) if row.reasons else ""
        searches = ", ".join(row.search_texts) or ""
        wall_ms = f"{row.elapsed_s * 1000:.0f}" if row.elapsed_s is not None else "—"
        lines.append(
            "| "
            + " | ".join(
                [
                    mark,
                    row.case_id,
                    row.capability or "",
                    str(row.llm_calls),
                    f"{row.input_tokens}/{row.output_tokens}",
                    wall_ms,
                    format_usd(row.cost_usd, known=row.cost_known),
                    str(row.node_count),
                    excerpt(searches, limit=28),
                    excerpt(reason, limit=48),
                ]
            )
            + " |"
        )
    return "\n".join(lines)
