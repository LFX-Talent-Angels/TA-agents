"""One full assistant turn: dispatch -> answer -> cost -> run-log.

Shared by the FastAPI edge and the CLI so neither duplicates telemetry logic
(ARCHITECTURE.md: the FastAPI edge is thin; the CLI is just another headless
client of the same assistant).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from talent_angels.assistant.answer import build_answer
from talent_angels.assistant.cache import ResultCache
from talent_angels.assistant.graph import build_graph
from talent_angels.assistant.intent import CAPABILITY_LOCATE, Capability
from talent_angels.assistant.planning import (
    ExecutionPlan,
    build_plan_for_capability,
)
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.llm import LLMClient, LLMUsage
from talent_angels.runlog import (
    EfficiencyInfo,
    GenAIUsage,
    GraphStats,
    ResultSummary,
    RunLogRecord,
    StageUsage,
    ToolCall,
    append_record,
    estimate_turn_cost_usd,
    usage_from_stage,
)
from talent_angels.skills.locate import ESCO_SUITE_NAME, locate
from talent_angels.skills.locate.rank import group_and_sort_locate
from talent_angels.suites.measured import MeasuredSuite
from talent_angels.suites.protocol import SuiteTools


def _usage_from_stages(stages: list[StageUsage]) -> LLMUsage:
    if not stages:
        return LLMUsage()
    merged = LLMUsage()
    for stage in stages:
        part = usage_from_stage(stage)
        merged = LLMUsage(
            input_tokens=merged.input_tokens + part.input_tokens,
            output_tokens=merged.output_tokens + part.output_tokens,
            reasoning_tokens=merged.reasoning_tokens + part.reasoning_tokens,
            cache_read_input_tokens=merged.cache_read_input_tokens + part.cache_read_input_tokens,
            cache_creation_input_tokens=(
                merged.cache_creation_input_tokens + part.cache_creation_input_tokens
            ),
        )
    return merged


@dataclass
class TurnOutcome:
    capability: Capability
    plan: ExecutionPlan
    result: AgentResult
    answer: str
    record: RunLogRecord


def run_turn(
    *,
    suite: SuiteTools,
    suite_name: str = ESCO_SUITE_NAME,
    llm_client: LLMClient,
    question: str,
    kind: str | None = None,
    answer_mode: str = "structured",
    force_locate: bool = False,
    force_capability: Capability | None = None,
    cache: ResultCache | None = None,
    bound_node: NodeRef | None = None,
    persist: bool = True,
) -> TurnOutcome:
    """Run one turn and append its run-log record.

    `force_locate=True` skips intent classification and calls Locate directly
    — used by the capability-level endpoint/CLI command for clean per-
    capability cost measurement (MVP plan Sec 2.3). `cache`, when given, is
    only consulted on the `force_locate` path — the efficiency A/B experiment
    (MVP plan Sec 2.7) targets Locate specifically.
    """
    if force_locate and force_capability not in (None, CAPABILITY_LOCATE):
        raise ValueError("force_locate cannot be combined with another forced capability")

    cache_hit = False
    selected_force = CAPABILITY_LOCATE if force_locate else force_capability
    tools: list[ToolCall]
    if selected_force == CAPABILITY_LOCATE:
        capability = CAPABILITY_LOCATE
        plan = build_plan_for_capability(capability, suites=(suite_name,))
        cached = cache.get(suite_name, capability, question) if cache else None
        if cached is not None:
            result = cached
            cache_hit = True
            tools = []
        else:
            measured = MeasuredSuite(suite)
            result = locate(measured, suite_name, question, kind=kind)
            result = group_and_sort_locate(measured, result, question, suite_name=suite_name)
            tools = measured.tool_calls
            if cache is not None:
                cache.set(suite_name, capability, question, result)
        answer, answer_stage = build_answer(result, llm_client=llm_client, mode=answer_mode)
        stages = [answer_stage] if answer_stage is not None else []
        heuristic_intent = True
    else:
        graph = build_graph(
            suite=suite,
            suite_name=suite_name,
            llm_client=llm_client,
            answer_mode=answer_mode,
            forced_capability=selected_force,
        )
        final_state = graph.invoke({"question": question, "kind": kind, "bound_node": bound_node})
        capability = final_state["capability"]
        plan = final_state["plan"]
        result = final_state["result"]
        answer = final_state["answer"]
        stages = list(final_state.get("llm_stages") or [])
        heuristic_intent = bool(final_state.get("heuristic_intent", True))
        tools = final_state["tool_calls"]
    usage = _usage_from_stages(stages)
    model = os.environ.get("LLM_MODEL", "").strip() or "stub"
    cost = estimate_turn_cost_usd(stages, model)
    gen_ai = GenAIUsage(
        provider_name=os.environ.get("LLM_PROVIDER", "none").strip() or "none",
        request_model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        calls=sum(stage.calls for stage in stages),
        stages=stages,
    )
    graph_stats = GraphStats(
        queries=len(tools),
        total_ms=sum(tool.ms for tool in tools),
    )

    record = RunLogRecord(
        suite=result.suite,
        plan=list(plan.capabilities),
        question=question,
        efficiency=EfficiencyInfo(
            mode=("cached_result" if cache_hit else ("llm_answer" if stages else "tool_only")),
            result_cache_hit=cache_hit,
            heuristic_intent=heuristic_intent,
        ),
        gen_ai=gen_ai,
        tools=tools,
        graph=graph_stats,
        cost_usd=cost,
        result=ResultSummary(
            confidence=result.confidence,
            node_ids=[n.id for n in result.nodes],
            node_labels=[n.pref_label for n in result.nodes],
            warnings=result.warnings,
        ),
    )
    if persist:
        append_record(record)

    return TurnOutcome(
        capability=capability,
        plan=plan,
        result=result,
        answer=answer,
        record=record,
    )
