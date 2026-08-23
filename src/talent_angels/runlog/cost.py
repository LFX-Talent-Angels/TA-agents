"""Cache-aware cost estimator (MVP plan Sec 2.6) — reads runlog/rate_card.yaml.

Cost math only. No network calls, no provider SDK imports here — this
consumes `LLMUsage` (talent_angels.llm.protocol), never a raw provider response.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any

import yaml

from talent_angels.llm.protocol import LLMUsage
from talent_angels.runlog.models import CostBreakdown, StageUsage

_PER_MILLION = 1_000_000


@lru_cache(maxsize=1)
def load_rate_card() -> dict[str, Any]:
    text = resources.files("talent_angels.runlog").joinpath("rate_card.yaml").read_text()
    return yaml.safe_load(text)


def estimate_llm_cost_usd(
    usage: LLMUsage, model: str, *, rate_card: dict[str, Any] | None = None
) -> CostBreakdown:
    """Cost of one LLM call, cache-aware.

    Registered models, including deliberate $0 entries, stay ``known=True``.
    Unknown models return ``known=False`` and zero dollar amounts so an
    unpriced model is never reported as a priced $0 stub.
    """
    card = rate_card or load_rate_card()
    rates = card["models"].get(model)
    if rates is None:
        return CostBreakdown(
            known=False,
            rate_card=f"unpriced:{model}",
        )

    input_cost = usage.input_tokens * rates["input_per_million"] / _PER_MILLION
    output_cost = usage.output_tokens * rates["output_per_million"] / _PER_MILLION
    cache_read_cost = usage.cache_read_input_tokens * rates["cache_read_per_million"] / _PER_MILLION
    # Assume 5-minute TTL cache writes unless the caller tracks TTL explicitly;
    # LLMUsage does not carry TTL, so this is the conservative (cheaper) default.
    cache_write_cost = (
        usage.cache_creation_input_tokens * rates["cache_write_5m_per_million"] / _PER_MILLION
    )

    total = input_cost + output_cost + cache_read_cost + cache_write_cost
    # Baseline: what the same tokens would have cost with no caching at all
    # (every cached token billed at full input price) — for the efficiency
    # A/B report (MVP plan Sec 2.7). Equal to `total` on a turn with no cache
    # activity, which is why the plan's example record shows them matching.
    uncached_baseline = (
        usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens
    ) * rates["input_per_million"] / _PER_MILLION + output_cost

    return CostBreakdown(
        llm=uncached_baseline,
        llm_with_cache_savings=total,
        total=total,
        rate_card=f"rate_card.yaml#{card['version']}",
    )


def usage_from_stage(stage: StageUsage) -> LLMUsage:
    return LLMUsage(
        input_tokens=stage.input_tokens,
        output_tokens=stage.output_tokens,
        reasoning_tokens=stage.reasoning_tokens,
        cache_read_input_tokens=stage.cache_read_input_tokens,
        cache_creation_input_tokens=stage.cache_creation_input_tokens,
    )


def estimate_turn_cost_usd(
    stages: list[StageUsage],
    fallback_model: str,
    *,
    rate_card: dict[str, Any] | None = None,
) -> CostBreakdown:
    """Sum per-stage costs. Any unpriced stage makes the roll-up unknown."""
    if not stages:
        return estimate_llm_cost_usd(LLMUsage(), fallback_model, rate_card=rate_card)

    known = True
    llm = 0.0
    cached = 0.0
    total = 0.0
    cards: list[str] = []
    for stage in stages:
        part = estimate_llm_cost_usd(
            usage_from_stage(stage),
            stage.request_model or fallback_model,
            rate_card=rate_card,
        )
        known = known and part.known
        llm += part.llm
        cached += part.llm_with_cache_savings
        total += part.total
        if part.rate_card:
            cards.append(part.rate_card)
    return CostBreakdown(
        known=known,
        llm=llm,
        llm_with_cache_savings=cached,
        total=total,
        rate_card=",".join(cards),
    )
