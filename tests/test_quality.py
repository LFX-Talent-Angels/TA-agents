"""Offline quality-card scoring and report shape."""

from __future__ import annotations

from pathlib import Path

from talent_angels.evals.quality import (
    ObservedTurn,
    QualityCase,
    format_quality_report,
    score_case,
    summarize,
    write_quality_report,
)


def _observed(**overrides: object) -> ObservedTurn:
    payload: dict[str, object] = {
        "capability": "locate",
        "warnings": (),
        "node_ids": ("esco:occupation:1",),
        "node_labels": ("software developer",),
        "edge_count": 0,
        "confidence": 0.9,
        "tool_names": ("search_nodes",),
        "tools": (
            {
                "name": "search_nodes",
                "ms": 12.0,
                "ok": True,
                "args": {"text": "software developer", "kind": "occupation"},
            },
        ),
        "search_texts": ("software developer",),
        "answer": "software developer (occupation, id=esco:occupation:1)",
        "llm_calls": 2,
        "input_tokens": 800,
        "output_tokens": 40,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
        "cost_known": True,
        "rate_card": "openrouter/free",
        "graph_ms": 12.0,
        "stages": (
            {
                "stage": "intent",
                "input_tokens": 500,
                "output_tokens": 20,
                "latency_ms": 200.0,
            },
            {
                "stage": "answer",
                "input_tokens": 300,
                "output_tokens": 20,
                "latency_ms": 180.0,
            },
        ),
        "run_id": "run-1",
        "elapsed_s": 1.5,
    }
    payload.update(overrides)
    return ObservedTurn(**payload)  # type: ignore[arg-type]


def test_score_case_records_tokens_and_search_on_pass() -> None:
    case = QualityCase(
        id="locate-sd",
        question="software developer",
        family="locate_unique",
        expected_capability=("locate",),
        expected_top_id="esco:occupation:1",
    )

    verdict = score_case(case, _observed())

    assert verdict.passed is True
    assert verdict.llm_calls == 2
    assert verdict.input_tokens == 800
    assert verdict.search_texts == ["software developer"]
    assert verdict.top_labels == ["software developer"]
    assert "intent" in {stage["stage"] for stage in verdict.stages}


def test_quality_report_includes_economics_and_failure_detail(tmp_path: Path) -> None:
    passed = score_case(
        QualityCase(
            id="locate-sd",
            question="software developer",
            family="locate_unique",
            expected_capability=("locate",),
            expected_top_id="esco:occupation:1",
        ),
        _observed(),
    )
    failed = score_case(
        QualityCase(
            id="quality-python-programming",
            question="python programming",
            family="skill_locate",
            tier="quality",
            expected_capability=("locate",),
            forbidden_warnings=("not_found",),
            expected_ids=("esco:skill:py",),
            require_id_in_candidates=True,
        ),
        _observed(
            warnings=("not_found",),
            node_ids=(),
            node_labels=(),
            search_texts=("python programming",),
            llm_calls=1,
            input_tokens=400,
            output_tokens=15,
            answer="No match found",
        ),
    )
    summary = summarize([passed, failed])
    markdown = format_quality_report(
        provider="litellm",
        model="openrouter/poolside/laguna-s-2.1:free",
        summary=summary,
        verdicts=[passed, failed],
    )

    assert summary["questions"] == 2
    assert summary["failed"] == 1
    assert summary["input_tokens"] == 1200
    assert summary["llm_calls"] == 3
    assert "**tokens:** 1200 in" in markdown
    assert "**search text:** 'python programming'" in markdown
    assert "quality-python-programming" in markdown
    assert "| PASS | locate-sd |" in markdown

    markdown_path, json_path = write_quality_report(
        provider="litellm",
        model="laguna",
        summary=summary,
        verdicts=[passed, failed],
        target_dir=tmp_path,
        stamp="test",
    )
    payload = json_path.read_text(encoding="utf-8")
    assert '"input_tokens": 400' in payload
    assert "search_texts" in payload
    written = markdown_path.read_text(encoding="utf-8")
    assert "**search text:** 'python programming'" in written
    assert "1200 in" in written
