"""Time one LLM completion and record it as a run-log stage."""

from __future__ import annotations

from time import perf_counter

from talent_angels.llm import LLMClient, LLMResult, Message
from talent_angels.runlog import StageUsage


def measure_complete(
    client: LLMClient, messages: list[Message], *, stage: str
) -> tuple[LLMResult, StageUsage]:
    started = perf_counter()
    result = client.complete(messages)
    latency_ms = (perf_counter() - started) * 1000
    usage = result.usage
    return result, StageUsage(
        stage=stage,
        provider_name=result.provider,
        request_model=getattr(client, "model", "") or "",
        response_model=result.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        latency_ms=latency_ms,
        calls=1,
    )
