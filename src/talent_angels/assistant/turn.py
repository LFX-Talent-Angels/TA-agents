"""One full assistant turn: dispatch -> answer -> cost -> run-log.

Shared by the FastAPI edge and the CLI so neither duplicates telemetry logic
(ARCHITECTURE.md: the FastAPI edge is thin; the CLI is just another headless
client of the same assistant).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field

from talent_angels.assistant.answer import NO_SUBJECT, build_answer
from talent_angels.assistant.cache import ResultCache
from talent_angels.assistant.graph import build_graph, fresh_state_input
from talent_angels.assistant.intent import CAPABILITY_LOCATE, Capability
from talent_angels.assistant.llm_plan import PlanDraft
from talent_angels.assistant.planning import (
    ExecutionPlan,
    build_plan_for_capability,
)
from talent_angels.assistant.turn_graph import (
    TurnRuntime,
    build_turn_graph,
    fresh_turn_input,
    locate_one,
    thread_config,
)
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.llm import LLMClient, LLMUsage
from talent_angels.memory.episodes import record_episode
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
from talent_angels.skills.locate import ESCO_SUITE_NAME
from talent_angels.suites.protocol import SuiteTools
from talent_angels.suites.registry import SuiteRegistry, UnknownSuiteError

logger = logging.getLogger(__name__)

#: Hard cap on the question the runtime will process. The API rejects longer
#: input; other callers are truncated (with a warning) so a pasted document can
#: never be sent whole to every prompt and to full-text search.
MAX_QUESTION_CHARS = 1000

NO_SUBJECT_ANSWER = NO_SUBJECT


def normalize_question(question: str) -> tuple[str, list[str]]:
    """Strip and bound the question. Returns the text and any warnings."""
    text = " ".join(question.split())
    if len(text) > MAX_QUESTION_CHARS:
        return text[:MAX_QUESTION_CHARS].rstrip(), ["question_truncated"]
    return text, []


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
    plan_draft: PlanDraft | None = None

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
    thread_id: str | None = None,
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
        result, tools, cache_hit = locate_one(suite, suite_name, question, kind=kind, cache=cache)
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
        thread_id=thread_id,
    )
    config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
    final_state = graph.invoke(  # type: ignore[call-overload]
        fresh_state_input(question=question, kind=kind, bound_node=bound_node), config=config
    )
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
    run_id: str | None = None,
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
    if run_id:
        record = record.model_copy(update={"run_id": run_id})
    if persist:
        persist_turn_record(record)
    return record


def persist_turn_record(record: RunLogRecord) -> None:
    """Append the run-log line and the episode. Never fails the turn.

    The answer has already been computed; a locked, corrupt or read-only store
    must degrade to a logged warning, not turn a good answer into a 500.
    """
    try:
        append_record(record)
    except OSError:
        logger.warning("run-log append failed", exc_info=True)
    try:
        record_episode(record)
    except (sqlite3.Error, OSError):
        logger.warning("episode record failed", exc_info=True)


def _early_turn(
    *,
    capability: Capability,
    suite_name: str,
    question: str,
    warnings: list[str],
    answer: str,
    persist: bool,
    stages: list[StageUsage] | None = None,
    plan_draft: PlanDraft | None = None,
    heuristic_intent: bool = True,
) -> TurnOutcome:
    """A turn that stops before any suite is queried — still logged (rule 7)."""
    result = AgentResult(capability=capability, suite=suite_name, warnings=warnings)
    plan = build_plan_for_capability(capability, suites=(suite_name,))
    record = _record_turn(
        suite_label=suite_name,
        plan=plan,
        question=question,
        result=result,
        results=(result,),
        stages=list(stages or []),
        tools=[],
        cache_hit=False,
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
        plan_draft=plan_draft,
    )


def _has_bound(bound_node: NodeRef | None, bound_nodes: dict[str, NodeRef] | None) -> bool:
    return bound_node is not None or bool(bound_nodes)


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
    thread_id: str | None = None,
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

    Robustness contract: every turn returns a typed outcome and writes a
    run-log record. An empty question is answered without any model call; an
    unexpected failure degrades to a ``turn_failed:<Type>`` warning instead of
    raising. Only ``UnknownSuiteError`` (a caller error) propagates.
    """
    if force_locate and force_capability not in (None, CAPABILITY_LOCATE):
        raise ValueError("force_locate cannot be combined with another forced capability")
    if registry is None and suite is None:
        raise ValueError("run_turn requires registry or suite")
    question, input_warnings = normalize_question(question)
    default_suite = registry.default if registry is not None else suite_name
    if not question:
        return _early_turn(
            capability=force_capability or CAPABILITY_LOCATE,
            suite_name=default_suite,
            question=question,
            warnings=["no_subject"],
            answer=NO_SUBJECT_ANSWER,
            persist=persist,
        )
    try:
        outcome = _run_turn_checked(
            llm_client=llm_client,
            question=question,
            suite=suite,
            suite_name=suite_name,
            registry=registry,
            suite_override=suite_override,
            kind=kind,
            answer_mode=answer_mode,
            force_locate=force_locate,
            force_capability=force_capability,
            cache=cache,
            bound_node=bound_node,
            bound_nodes=bound_nodes,
            persist=persist,
            thread_id=thread_id,
            input_warnings=input_warnings,
        )
    except UnknownSuiteError:
        raise
    except Exception as exc:  # noqa: BLE001 — a turn must degrade, never crash the edge
        logger.exception("turn failed")
        return _failed_turn(exc, question, default_suite, force_capability, persist)
    return outcome


def _failed_turn(
    exc: Exception,
    question: str,
    suite_name: str,
    force_capability: Capability | None,
    persist: bool,
) -> TurnOutcome:
    return _early_turn(
        capability=force_capability or CAPABILITY_LOCATE,
        suite_name=suite_name,
        question=question,
        warnings=[f"turn_failed:{type(exc).__name__}"],
        answer=(
            "Something went wrong while answering that, and nothing was looked up. "
            "Please try again; if it keeps happening, the run log has the details."
        ),
        persist=persist,
    )


def _run_turn_checked(
    *,
    llm_client: LLMClient,
    question: str,
    suite: SuiteTools | None,
    suite_name: str,
    registry: SuiteRegistry | None,
    suite_override: str | None,
    kind: str | None,
    answer_mode: str,
    force_locate: bool,
    force_capability: Capability | None,
    cache: ResultCache | None,
    bound_node: NodeRef | None,
    bound_nodes: dict[str, NodeRef] | None,
    persist: bool,
    thread_id: str | None,
    input_warnings: list[str],
) -> TurnOutcome:
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
            thread_id=thread_id,
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

    from talent_angels.assistant.checkpoint import shared_checkpointer

    run_id = str(uuid.uuid4())
    runtime = TurnRuntime(
        registry=registry, llm_client=llm_client, answer_mode=answer_mode, cache=cache
    )
    graph = build_turn_graph(runtime, checkpointer=shared_checkpointer() if thread_id else None)
    final = graph.invoke(  # type: ignore[call-overload]
        fresh_turn_input(
            run_id=run_id,
            question=question,
            kind=kind,
            bound_node=bound_node,
            bound_nodes=bound_nodes,
            force_capability=force_capability,
            force_locate=force_locate,
            suite_override=suite_override,
            input_warnings=input_warnings,
        ),
        config=thread_config(thread_id) if thread_id else {},
    )
    final_plan: ExecutionPlan = final["plan"]
    result_tuple = tuple(final["results"])
    primary = result_tuple[0]
    record = _record_turn(
        suite_label=",".join(item.suite for item in result_tuple),
        plan=final_plan,
        question=question,
        result=primary,
        results=result_tuple,
        stages=list(final.get("stages") or []),
        tools=list(final.get("tools") or []),
        cache_hit=bool(final.get("cache_hit")),
        heuristic_intent=bool(final.get("heuristic_intent", True)),
        persist=persist,
        run_id=run_id,
    )
    return TurnOutcome(
        capability=final_plan.intent.target,
        plan=final_plan,
        result=primary,
        answer=final["answer"],
        record=record,
        results=result_tuple,
        plan_draft=final.get("plan_draft"),
    )
