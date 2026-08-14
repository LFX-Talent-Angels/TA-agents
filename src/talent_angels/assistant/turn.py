"""One full assistant turn: dispatch -> answer -> cost -> run-log.

Shared by the FastAPI edge and the CLI so neither duplicates telemetry logic
(ARCHITECTURE.md: the FastAPI edge is thin; the CLI is just another headless
client of the same assistant).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from talent_angels.assistant.answer import build_answer
from talent_angels.assistant.cache import ResultCache
from talent_angels.assistant.graph import SearchableSuite, build_graph
from talent_angels.assistant.intent import CAPABILITY_LOCATE, Capability
from talent_angels.assistant.planning import (
    ExecutionPlan,
    build_plan_for_capability,
)
from talent_angels.contracts import AgentResult
from talent_angels.llm import LLMClient, LLMUsage
from talent_angels.runlog import (
    EfficiencyInfo,
    GenAIUsage,
    GraphStats,
    ResultSummary,
    RunLogRecord,
    ToolCall,
    append_record,
    estimate_llm_cost_usd,
)
from talent_angels.skills.locate import ESCO_SUITE_NAME, locate


@dataclass
class TurnOutcome:
    capability: Capability
    plan: ExecutionPlan
    result: AgentResult
    answer: str
    record: RunLogRecord


def run_turn(
    *,
    suite: SearchableSuite,
    suite_name: str = ESCO_SUITE_NAME,
    llm_client: LLMClient,
    question: str,
    kind: str | None = None,
    answer_mode: str = "structured",
    force_locate: bool = False,
    cache: ResultCache | None = None,
) -> TurnOutcome:
    """Run one turn and append its run-log record.

    `force_locate=True` skips intent classification and calls Locate directly
    — used by the capability-level endpoint/CLI command for clean per-
    capability cost measurement (MVP plan Sec 2.3). `cache`, when given, is
    only consulted on the `force_locate` path — the efficiency A/B experiment
    (MVP plan Sec 2.7) targets Locate specifically.
    """
    start = time.perf_counter()
    cache_hit = False
    if force_locate:
        capability = CAPABILITY_LOCATE
        plan = build_plan_for_capability(capability, suites=(suite_name,))
        cached = cache.get(suite_name, capability, question) if cache else None
        if cached is not None:
            result = cached
            cache_hit = True
        else:
            result = locate(suite, suite_name, question, kind=kind)
            if cache is not None:
                cache.set(suite_name, capability, question, result)
        answer, llm_usage = build_answer(result, llm_client=llm_client, mode=answer_mode)
    else:
        graph = build_graph(
            suite=suite,
            suite_name=suite_name,
            llm_client=llm_client,
            answer_mode=answer_mode,
        )
        final_state = graph.invoke({"question": question, "kind": kind})
        capability = final_state["capability"]
        plan = final_state["plan"]
        result = final_state["result"]
        answer = final_state["answer"]
        llm_usage = final_state.get("llm_usage")
    elapsed_ms = (time.perf_counter() - start) * 1000

    usage = llm_usage or LLMUsage()
    model = os.environ.get("LLM_MODEL", "").strip() or "stub"
    cost = estimate_llm_cost_usd(usage, model)
    gen_ai = GenAIUsage(
        provider_name=os.environ.get("LLM_PROVIDER", "none").strip() or "none",
        request_model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        calls=1 if llm_usage is not None else 0,
    )
    tools = [ToolCall(name="search_nodes", ms=elapsed_ms, ok=True)]
    graph_stats = GraphStats(queries=1, total_ms=elapsed_ms)

    record = RunLogRecord(
        suite=result.suite,
        plan=list(plan.capabilities),
        question=question,
        efficiency=EfficiencyInfo(
            mode=(
                "cached_result"
                if cache_hit
                else ("llm_answer" if llm_usage is not None else "tool_only")
            ),
            result_cache_hit=cache_hit,
        ),
        gen_ai=gen_ai,
        tools=tools,
        graph=graph_stats,
        cost_usd=cost,
        result=ResultSummary(
            confidence=result.confidence,
            node_ids=[n.id for n in result.nodes],
            warnings=result.warnings,
        ),
    )
    append_record(record)

    return TurnOutcome(
        capability=capability,
        plan=plan,
        result=result,
        answer=answer,
        record=record,
    )
