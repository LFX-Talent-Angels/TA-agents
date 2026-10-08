"""Time one LLM completion and record it as a run-log stage.

The single entry point for model calls: every call is metered here, and every
provider failure leaves here as :class:`~talent_angels.llm.LLMError`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

from talent_angels.llm import LLMClient, LLMError, LLMResult, Message
from talent_angels.runlog import StageUsage

#: Stage names of phrasing calls made *after* a turn's record is built (TUI).
PHRASING_STAGES = frozenset({"phrase", "synthesize"})

_SINK: ContextVar[list[StageUsage] | None] = ContextVar("llm_stage_sink", default=None)


@contextmanager
def collect_stages() -> Iterator[list[StageUsage]]:
    """Collect every metered call made in this context (and copies of it).

    Lets an edge account for calls that happen outside ``run_turn`` — the
    TUI's phrasing — so the run-log's token and cost totals are complete.
    """
    sink: list[StageUsage] = []
    token = _SINK.set(sink)
    try:
        yield sink
    finally:
        _SINK.reset(token)


def measure_complete(
    client: LLMClient,
    messages: list[Message],
    *,
    stage: str,
    tools: list[dict[str, object]] | None = None,
) -> tuple[LLMResult, StageUsage]:
    started = perf_counter()
    try:
        if tools is None:
            result = client.complete(messages)
        else:
            result = client.complete(messages, tools=tools)
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001 — every provider failure becomes one type
        raise LLMError(f"{stage}: {type(exc).__name__}: {exc}") from exc
    latency_ms = (perf_counter() - started) * 1000
    usage = result.usage
    stage_usage = StageUsage(
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
    sink = _SINK.get()
    if sink is not None:
        sink.append(stage_usage)
    return result, stage_usage
