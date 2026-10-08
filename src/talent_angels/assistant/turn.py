"""One full assistant turn: dispatch -> answer -> cost -> run-log.

Shared by the FastAPI edge and the CLI so neither duplicates telemetry logic
(ARCHITECTURE.md: the FastAPI edge is thin; the CLI is just another headless
client of the same assistant).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from talent_angels.assistant.answer import build_answer
from talent_angels.assistant.cache import ResultCache
from talent_angels.assistant.graph import build_graph, dispatch_plan
from talent_angels.assistant.honesty import honesty_warnings
from talent_angels.assistant.intent import CAPABILITY_LOCATE, Capability
from talent_angels.assistant.llm_plan import interpret_question
from talent_angels.assistant.merge import merge_answers
from talent_angels.assistant.planning import (
    ExecutionPlan,
    build_plan_for_capability,
)
from talent_angels.assistant.suite_select import select_suites
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
from talent_angels.suites.registry import SuiteRegistry, UnknownSuiteError


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
    results: tuple[AgentResult, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.results:
            self.results = (self.result,)


def _bound_for_suite(
    suite_name: str,
    bound_node: NodeRef | None,
    bound_nodes: dict[str, NodeRef] | None,
) -> NodeRef | None:
    if bound_nodes:
        return bound_nodes.get(suite_name)
    if bound_node is not None and bound_node.suite == suite_name:
        return bound_node
    if bound_node is not None and not bound_nodes:
        return bound_node
    return None


def _locate_one(
    suite: SuiteTools,
    suite_name: str,
    question: str,
    *,
    kind: str | None,
    cache: ResultCache | None,
) -> tuple[AgentResult, list[ToolCall], bool]:
    cached = cache.get(suite_name, CAPABILITY_LOCATE, question) if cache else None
    if cached is not None:
        return cached, [], True
    measured = MeasuredSuite(suite)
    result = locate(measured, suite_name, question, kind=kind)
    result = group_and_sort_locate(measured, result, question, suite_name=suite_name)
    if cache is not None:
        cache.set(suite_name, CAPABILITY_LOCATE, question, result)
    return result, measured.tool_calls, False


def _dispatch_opened(
    *,
    suite: SuiteTools,
    suite_name: str,
    question: str,
    kind: str | None,
    bound_node: NodeRef | None,
    llm_client: LLMClient,
    answer_mode: str,
    force_capability: Capability | None,
    cache: ResultCache | None,
    force_locate: bool,
) -> tuple[
    Capability,
    ExecutionPlan,
    AgentResult,
    str,
    list[StageUsage],
    bool,
    list[ToolCall],
    bool,
]:
    cache_hit = False
    if force_locate:
        result, tools, cache_hit = _locate_one(suite, suite_name, question, kind=kind, cache=cache)
        answer, answer_stage = build_answer(result, llm_client=llm_client, mode=answer_mode)
        stages = [answer_stage] if answer_stage is not None else []
        plan = build_plan_for_capability(CAPABILITY_LOCATE, suites=(suite_name,))
        return CAPABILITY_LOCATE, plan, result, answer, stages, True, tools, cache_hit

    graph = build_graph(
        suite=suite,
        suite_name=suite_name,
        llm_client=llm_client,
        answer_mode=answer_mode,
        forced_capability=force_capability,
    )
    final_state = graph.invoke({"question": question, "kind": kind, "bound_node": bound_node})
    return (
        final_state["capability"],
        final_state["plan"],
        final_state["result"],
        final_state["answer"],
        list(final_state.get("llm_stages") or []),
        bool(final_state.get("heuristic_intent", True)),
        final_state["tool_calls"],
        False,
    )


def _record_turn(
    *,
    suite_label: str,
    plan: ExecutionPlan,
    question: str,
    result: AgentResult,
    results: tuple[AgentResult, ...],
    stages: list[StageUsage],
    tools: list[ToolCall],
    cache_hit: bool,
    heuristic_intent: bool,
    persist: bool,
) -> RunLogRecord:
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
    all_nodes = [node for item in results for node in item.nodes]
    all_warnings = [warning for item in results for warning in item.warnings]
    record = RunLogRecord(
        suite=suite_label,
        plan=list(plan.capabilities),
        question=question,
        efficiency=EfficiencyInfo(
            mode=("cached_result" if cache_hit else ("llm_answer" if stages else "tool_only")),
            result_cache_hit=cache_hit,
            heuristic_intent=heuristic_intent,
        ),
        gen_ai=gen_ai,
        tools=tools,
        graph=GraphStats(
            queries=len(tools),
            total_ms=sum(tool.ms for tool in tools),
        ),
        cost_usd=cost,
        result=ResultSummary(
            confidence=result.confidence,
            node_ids=[node.id for node in all_nodes],
            node_labels=[node.pref_label for node in all_nodes],
            warnings=all_warnings,
        ),
    )
    if persist:
        append_record(record)
    return record


def run_turn(
    *,
    llm_client: LLMClient,
    question: str,
    suite: SuiteTools | None = None,
    suite_name: str = ESCO_SUITE_NAME,
    registry: SuiteRegistry | None = None,
    suite_override: str | None = None,
    kind: str | None = None,
    answer_mode: str = "structured",
    force_locate: bool = False,
    force_capability: Capability | None = None,
    cache: ResultCache | None = None,
    bound_node: NodeRef | None = None,
    bound_nodes: dict[str, NodeRef] | None = None,
    persist: bool = True,
) -> TurnOutcome:
    """Run one turn and append its run-log record.

    Pass ``registry`` (CLI/API/TUI) to search every attached suite, or a
    single ``suite`` for unit tests and per-suite benches. ``suite_override``
    forces one attached suite (debug).

    `force_locate=True` skips intent classification and calls Locate directly
    — used by the capability-level endpoint/CLI command for clean per-
    capability cost measurement (MVP plan Sec 2.3). `cache`, when given, is
    only consulted on the `force_locate` path — the efficiency A/B experiment
    (MVP plan Sec 2.7) targets Locate specifically.
    """
    if force_locate and force_capability not in (None, CAPABILITY_LOCATE):
        raise ValueError("force_locate cannot be combined with another forced capability")
    if registry is None:
        if suite is None:
            raise ValueError("run_turn requires registry or suite")
        (
            capability,
            plan,
            result,
            answer,
            stages,
            heuristic_intent,
            tools,
            cache_hit,
        ) = _dispatch_opened(
            suite=suite,
            suite_name=suite_name,
            question=question,
            kind=kind,
            bound_node=_bound_for_suite(suite_name, bound_node, bound_nodes),
            llm_client=llm_client,
            answer_mode=answer_mode,
            force_capability=force_capability,
            cache=cache,
            force_locate=force_locate,
        )
        record = _record_turn(
            suite_label=result.suite,
            plan=plan,
            question=question,
            result=result,
            results=(result,),
            stages=stages,
            tools=tools,
            cache_hit=cache_hit,
            heuristic_intent=heuristic_intent,
            persist=persist,
        )
        return TurnOutcome(
            capability=capability,
            plan=plan,
            result=result,
            answer=answer,
            record=record,
            results=(result,),
        )

    selected = select_suites(
        available=registry.available,
        override=suite_override,
        question=question,
    )
    extra_warnings: list[str] = []
    collected: list[AgentResult] = []
    tools = []
    stages = []
    cache_hit = False
    selected_force = CAPABILITY_LOCATE if force_locate else force_capability

    interpreted = interpret_question(
        question,
        suites=selected,
        llm_client=llm_client,
        forced_capability=selected_force,
    )
    if interpreted.stage is not None:
        stages.append(interpreted.stage)
    plan = interpreted.plan
    capability = plan.intent.target
    seed_state = {
        "question": question,
        "kind": kind,
        "bound_node": bound_node,
        "capability": capability,
        "plan": plan,
        "plan_draft": interpreted.draft,
        "heuristic_intent": interpreted.heuristic,
        "llm_stages": stages,
    }

    for name in selected:
        per_state = {
            **seed_state,
            "bound_node": _bound_for_suite(name, bound_node, bound_nodes),
        }
        try:
            with registry.open(name) as runtime:
                if force_locate:
                    one, one_tools, one_hit = _locate_one(
                        runtime.suite, name, question, kind=kind, cache=cache
                    )
                    cache_hit = cache_hit or one_hit
                    collected.append(one)
                    tools.extend(one_tools)
                    continue
                dispatched = dispatch_plan(
                    per_state,  # type: ignore[arg-type]
                    suite=runtime.suite,
                    suite_name=name,
                )
                collected.append(dispatched["result"])
                tools.extend(dispatched.get("tool_calls") or [])
        except UnknownSuiteError:
            raise
        except Exception:  # noqa: BLE001 — a down suite must not fail the turn
            extra_warnings.append(f"suite_unavailable:{name}")

    result_tuple = tuple(collected)
    extra_warnings.extend(honesty_warnings(result_tuple))
    if not result_tuple:
        empty = AgentResult(
            capability=capability,
            suite=selected[0] if selected else suite_name,
            warnings=extra_warnings or ["not_found"],
        )
        result_tuple = (empty,)
        answer = merge_answers((), extra_warnings=extra_warnings)
        answer_stage = None
    elif len(result_tuple) == 1 and not extra_warnings:
        answer, answer_stage = build_answer(
            result_tuple[0], llm_client=llm_client, mode=answer_mode
        )
    else:
        answer = merge_answers(result_tuple, extra_warnings=extra_warnings)
        answer_stage = None

    if answer_stage is not None:
        stages.append(answer_stage)

    primary = result_tuple[0]
    if extra_warnings:
        primary = primary.model_copy(update={"warnings": [*primary.warnings, *extra_warnings]})
        result_tuple = (primary, *result_tuple[1:])

    record = _record_turn(
        suite_label=",".join(item.suite for item in result_tuple),
        plan=plan,
        question=question,
        result=primary,
        results=result_tuple,
        stages=stages,
        tools=tools,
        cache_hit=cache_hit,
        heuristic_intent=interpreted.heuristic,
        persist=persist,
    )
    return TurnOutcome(
        capability=capability,
        plan=plan,
        result=primary,
        answer=answer,
        record=record,
        results=result_tuple,
    )
