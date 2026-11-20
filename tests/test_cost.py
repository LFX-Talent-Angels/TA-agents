"""Model strings carry a LiteLLM route; the price belongs to the base model."""

from __future__ import annotations

from talent_angels.llm import LLMUsage
from talent_angels.runlog import estimate_llm_cost_usd

USAGE = LLMUsage(input_tokens=1_000_000, output_tokens=0)


def test_routed_model_is_priced_as_its_base_model() -> None:
    """Live: every Azure-routed Claude turn was reported as an unpriced $0."""
    routed = estimate_llm_cost_usd(USAGE, "azure_ai/claude-sonnet-4-6")
    direct = estimate_llm_cost_usd(USAGE, "claude-sonnet-4-6")

    assert routed.known is True
    assert routed.total == direct.total > 0
    assert routed.rate_card.endswith(":claude-sonnet-4-6")


def test_unknown_model_stays_unknown() -> None:
    cost = estimate_llm_cost_usd(USAGE, "azure_ai/some-unlisted-model")
    assert cost.known is False
    assert cost.total == 0.0
