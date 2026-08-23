"""Run log tests: cost math (cache-aware) and JSONL append."""

from __future__ import annotations

import json

from talent_angels.llm.protocol import LLMUsage
from talent_angels.runlog import (
    RunLogRecord,
    StageUsage,
    append_record,
    estimate_llm_cost_usd,
    estimate_turn_cost_usd,
    load_rate_card,
)


def test_stub_model_costs_nothing() -> None:
    usage = LLMUsage(input_tokens=1000, output_tokens=500)
    cost = estimate_llm_cost_usd(usage, "stub")
    assert cost.total == 0.0
    assert cost.llm == 0.0


def test_unknown_model_marks_dollar_cost_unknown() -> None:
    usage = LLMUsage(input_tokens=1000, output_tokens=500)
    cost = estimate_llm_cost_usd(usage, "not-a-real-model")

    assert cost.total == 0.0
    assert cost.known is False
    assert cost.rate_card == "unpriced:not-a-real-model"


def test_openrouter_free_model_records_tokens_at_zero_cost() -> None:
    usage = LLMUsage(input_tokens=1_000, output_tokens=500, reasoning_tokens=300)

    cost = estimate_llm_cost_usd(usage, "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free")

    assert cost.total == 0.0
    assert cost.known is True
    assert usage.reasoning_tokens == 300


def test_known_model_prices_input_and_output_tokens() -> None:
    card = load_rate_card()
    rates = card["models"]["claude-sonnet-5"]
    usage = LLMUsage(input_tokens=1_000_000, output_tokens=1_000_000)

    cost = estimate_llm_cost_usd(usage, "claude-sonnet-5")

    expected = rates["input_per_million"] + rates["output_per_million"]
    assert cost.total == expected
    assert cost.llm == expected  # no cache activity -> baseline equals actual


def test_cache_read_is_cheaper_than_full_price() -> None:
    usage_full = LLMUsage(input_tokens=1_000_000, output_tokens=0)
    usage_cached = LLMUsage(input_tokens=0, output_tokens=0, cache_read_input_tokens=1_000_000)

    cost_full = estimate_llm_cost_usd(usage_full, "claude-sonnet-5")
    cost_cached = estimate_llm_cost_usd(usage_cached, "claude-sonnet-5")

    assert cost_cached.total < cost_full.total
    # Baseline (llm) treats cached tokens as if billed at full input price,
    # so it should roughly match the uncached case's actual cost.
    assert cost_cached.llm == cost_full.total
    assert cost_cached.llm_with_cache_savings == cost_cached.total


def test_turn_cost_sums_priced_stages() -> None:
    stages = [
        StageUsage(
            stage="intent",
            request_model="priced-test-model",
            input_tokens=2,
            output_tokens=1,
        ),
        StageUsage(
            stage="answer",
            request_model="priced-test-model",
            input_tokens=3,
            output_tokens=1,
        ),
    ]
    rate_card = {
        "version": "test",
        "models": {
            "priced-test-model": {
                "input_per_million": 1_000_000.0,
                "output_per_million": 2_000_000.0,
                "cache_read_per_million": 0.0,
                "cache_write_5m_per_million": 0.0,
            }
        },
    }

    cost = estimate_turn_cost_usd(stages, "priced-test-model", rate_card=rate_card)

    assert cost.known is True
    assert cost.total == 9.0


def test_turn_cost_unknown_if_any_stage_is_unpriced() -> None:
    stages = [
        StageUsage(stage="intent", request_model="priced-test-model", input_tokens=1),
        StageUsage(stage="answer", request_model="not-a-real-model", input_tokens=1),
    ]
    rate_card = {
        "version": "test",
        "models": {
            "priced-test-model": {
                "input_per_million": 1.0,
                "output_per_million": 1.0,
                "cache_read_per_million": 0.0,
                "cache_write_5m_per_million": 0.0,
            }
        },
    }

    cost = estimate_turn_cost_usd(stages, "priced-test-model", rate_card=rate_card)

    assert cost.known is False


def test_append_record_writes_one_json_line_per_call(tmp_path) -> None:
    path = tmp_path / "runlog.jsonl"
    record1 = RunLogRecord(suite="esco", plan=["locate"], question="software developer")
    record2 = RunLogRecord(suite="esco", plan=["locate"], question="data scientist")

    append_record(record1, path=path)
    append_record(record2, path=path)

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["question"] == "software developer"
    assert parsed[1]["question"] == "data scientist"
    assert parsed[0]["schema_version"] == 1
