"""The production turn as a LangGraph: plan → dispatch → answer → remember.

This is the loop ARCHITECTURE.md describes, run for every registry turn (API,
CLI, TUI). With a ``thread_id`` the graph is compiled with the shared durable
checkpointer (``assistant.checkpoint``), and the only state that is *meant* to
carry across turns — the bounded ``history`` of :class:`TurnMemo` — does.

Two rules make that safe:

1. **Every per-turn channel is reset on every invoke.** A LangGraph channel
   without a reducer keeps its last value when the input omits it. The first
   checkpointer shipped with an input of only ``question``/``kind``/``bound``,
   so turn 2 on a thread returned turn 1's answer. :func:`fresh_turn_input`
   builds the input from :data:`PER_TURN_KEYS` — every channel except
   ``history`` — so nothing leaks between turns by omission.
2. **Only data crosses into state; runtime objects do not.** The registry, the
   LLM client and the result cache are closed over by the node functions,
   never written to a checkpointed channel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from talent_angels.assistant.answer import NO_SUBJECT, build_answer
from talent_angels.assistant.cache import ResultCache
from talent_angels.assistant.graph import dispatch_plan
from talent_angels.assistant.honesty import honesty_warnings
from talent_angels.assistant.intent import (
    CAPABILITY_CONNECT,
    CAPABILITY_LOCATE,
    Capability,
)
from talent_angels.assistant.llm_plan import (
    PlanDraft,
    interpret_question,
    is_compare,
    is_trivial_subject,
)
from talent_angels.assistant.memo import TurnMemo, append_bounded, memo_for
from talent_angels.assistant.merge import merge_answers
from talent_angels.assistant.planning import ExecutionPlan
from talent_angels.assistant.state import AssistantState
from talent_angels.assistant.suite_select import named_unattached, select_suites
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.llm import LLMClient
from talent_angels.memory.cache import CachedSuite
from talent_angels.memory.profile import profile_line
from talent_angels.runlog import StageUsage, ToolCall
from talent_angels.skills.connect import ConnectRequest, connect
from talent_angels.skills.connect.compare import compare_result
from talent_angels.skills.locate import locate
from talent_angels.skills.locate.areas import Area, AreaRequest, area_summary, search_area
from talent_angels.skills.locate.explore import explore
from talent_angels.skills.locate.rank import group_and_sort_locate
from talent_angels.suites.measured import MeasuredSuite
from talent_angels.suites.protocol import SuiteTools
from talent_angels.suites.registry import SuiteRegistry, UnknownSuiteError

logger = logging.getLogger(__name__)


class TurnState(TypedDict, total=False):
    # --- inputs (reset every turn) ---
    run_id: str
    question: str
    kind: str | None
    bound_node: NodeRef | None
    bound_nodes: dict[str, NodeRef] | None
    force_capability: Capability | None
    force_locate: bool
    suite_override: str | None
    input_warnings: list[str]
    #: "Which area?" answered: re-run one search inside one occupation group.
    area: AreaRequest | None
    # --- working state (reset every turn) ---
    selected: list[str]
    plan: ExecutionPlan | None
    plan_draft: PlanDraft | None
    heuristic_intent: bool
    stages: list[StageUsage]
    tools: list[ToolCall]
    cache_hit: bool
    results: list[AgentResult]
    #: Per suite, how a broad match set splits into occupation groups.
    areas: dict[str, list[Area]]
    extra_warnings: list[str]
    answer: str
    stopped: bool
    # --- carried across turns on a thread ---
    history: Annotated[list[TurnMemo], append_bounded]


PER_TURN_KEYS: tuple[str, ...] = tuple(k for k in TurnState.__annotations__ if k != "history")

_DEFAULTS: dict[str, Any] = {
    "force_locate": False,
    "input_warnings": [],
    "selected": [],
    "heuristic_intent": True,
    "stages": [],
    "tools": [],
    "cache_hit": False,
    "results": [],
    "areas": {},
    "extra_warnings": [],
    "answer": "",
    "stopped": False,
    "question": "",
    "run_id": "",
}


def fresh_turn_input(**values: Any) -> dict[str, Any]:
    """Every per-turn channel, explicitly set (see rule 1 in the module doc)."""
    unknown = set(values) - set(PER_TURN_KEYS)
    if unknown:
        raise KeyError(f"not per-turn channels: {sorted(unknown)}")
    return {key: values.get(key, _DEFAULTS.get(key)) for key in PER_TURN_KEYS}


@dataclass(frozen=True)
class TurnRuntime:
    registry: SuiteRegistry
    llm_client: LLMClient
    answer_mode: str = "structured"
    cache: ResultCache | None = None


def _bound_for_suite(
    suite_name: str,
    bound_node: NodeRef | None,
    bound_nodes: dict[str, NodeRef] | None,
) -> NodeRef | None:
    """The bound node for this suite only — never another suite's node (rule 6)."""
    if bound_nodes:
        return bound_nodes.get(suite_name)
    if bound_node is not None and bound_node.suite == suite_name:
        return bound_node
    return None


def _served_from_cache(suite: object) -> bool:
    return isinstance(suite, CachedSuite) and suite.hits > 0


def locate_one(
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


def compare_one(
    suite: SuiteTools, suite_name: str, first: str, second: str
) -> tuple[AgentResult, list[ToolCall]]:
    """Locate both titles and Connect each, in code; no tool loop, no model.

    A title with several matches comes back as that locate result, so the user
    picks it first — a compare never guesses one side.
    """
    measured = MeasuredSuite(suite)
    schema = suite.suite_schema
    sides: list[AgentResult] = []
    for subject in (first, second):
        located = locate(measured, suite_name, subject, kind="occupation")
        located = group_and_sort_locate(
            measured,
            located,
            subject,
            suite_name=suite_name,
            group_rel_type=schema.group_rel_type,
            group_node_kinds=schema.group_node_kinds,
        )
        if not located.nodes or "ambiguous" in located.warnings:
            return located, measured.tool_calls
        sides.append(
            connect(
                measured,
                suite_name,
                located.nodes[0],
                request=ConnectRequest(subject=subject, rel_types=schema.skill_rel_types),
                confidence=located.confidence,
                locate_evidence=located.evidence,
                optional_rel_values=schema.optional_rel_values,
            )
        )
    return compare_result(sides[0], sides[1]), measured.tool_calls


def explore_one(
    suite: SuiteTools, suite_name: str, draft: PlanDraft, question: str = ""
) -> tuple[AgentResult, list[Area], list[ToolCall]] | None:
    """A vague subject: the planner's titles, checked against the map, as a pick list.

    None when the subject is already one clear title; the normal dispatch runs.
    """
    assert draft.subject
    measured = MeasuredSuite(suite)
    schema = suite.suite_schema
    explored = explore(
        measured,
        suite_name,
        draft.subject,
        draft.candidates,
        kind=draft.kind or "occupation",
        subject_is_users=draft.subject.casefold() in question.casefold(),
        group_rel_type=schema.group_rel_type,
        group_node_kinds=schema.group_node_kinds,
    )
    if explored is None:
        return None
    result, areas = explored
    return result, areas, measured.tool_calls


def _is_broad(result: AgentResult) -> bool:
    """More matching titles than the pick list shows."""
    return (
        result.capability == CAPABILITY_LOCATE
        and "ambiguous" in result.warnings
        and "truncated" in result.warnings
    )


def _plan(state: TurnState, rt: TurnRuntime) -> dict[str, Any]:
    question = state["question"]
    registry = rt.registry
    selected = select_suites(
        available=registry.available, override=state.get("suite_override"), question=question
    )
    extra = [
        *(state.get("input_warnings") or []),
        *(f"suite_not_attached:{name}" for name in named_unattached(question, registry.available)),
    ]
    forced = CAPABILITY_LOCATE if state.get("force_locate") else state.get("force_capability")
    if state.get("area") is not None:
        forced = CAPABILITY_LOCATE  # the user chose an area: nothing to interpret
    interpreted = interpret_question(
        question,
        suites=selected,
        llm_client=rt.llm_client,
        forced_capability=forced,
        profile=profile_line() if forced is None else None,
    )
    stages = [interpreted.stage] if interpreted.stage is not None else []
    update: dict[str, Any] = {
        "selected": list(selected),
        "plan": interpreted.plan,
        "plan_draft": interpreted.draft,
        "heuristic_intent": interpreted.heuristic,
        "stages": stages,
        "extra_warnings": extra,
    }
    capability = interpreted.plan.intent.target
    draft = interpreted.draft
    trivial = draft is not None and bool(draft.subject) and is_trivial_subject(draft.subject)
    if trivial:
        assert draft is not None
        draft = draft.model_copy(update={"subject": None, "profile_intent": None})
        update["plan_draft"] = draft
    has_bound = state.get("bound_node") is not None or bool(state.get("bound_nodes"))
    if (
        (not interpreted.heuristic or trivial)
        and draft is not None
        and not draft.subject
        and capability in (CAPABILITY_LOCATE, CAPABILITY_CONNECT)
        and forced is None
        and not has_bound
    ):
        # The planner found no occupation or skill to look up (a greeting, an
        # instruction to the assistant, noise). Querying the graph anyway is how
        # a prompt example once became a confident, cited, unrelated answer.
        suite_name = selected[0] if selected else registry.default
        update["results"] = [
            AgentResult(capability=capability, suite=suite_name, warnings=["no_subject", *extra])
        ]
        update["answer"] = NO_SUBJECT
        update["extra_warnings"] = []
        update["stopped"] = True
    return update


def _dispatch(state: TurnState, rt: TurnRuntime) -> dict[str, Any]:
    plan = state["plan"]
    assert plan is not None
    question = state["question"]
    kind = state.get("kind")
    bound_node = state.get("bound_node")
    bound_nodes = state.get("bound_nodes")
    stages = list(state.get("stages") or [])
    extra = list(state.get("extra_warnings") or [])
    tools: list[ToolCall] = []
    collected: list[AgentResult] = []
    areas: dict[str, list[Area]] = {}
    area = state.get("area")
    cache_hit = False
    seed: dict[str, Any] = {
        "question": question,
        "kind": kind,
        "capability": plan.intent.target,
        "plan": plan,
        "plan_draft": state.get("plan_draft"),
        "heuristic_intent": state.get("heuristic_intent", True),
        "llm_stages": stages,
    }
    draft = state.get("plan_draft")
    for name in state.get("selected") or []:
        if area is not None and area.suite != name:
            continue
        prior = len(stages)
        bound = _bound_for_suite(name, bound_node, bound_nodes)
        try:
            with rt.registry.open(name) as runtime:
                if area is not None:
                    measured = MeasuredSuite(runtime.suite)
                    collected.append(search_area(measured, name, area))
                    tools.extend(measured.tool_calls)
                    continue
                if state.get("force_locate"):
                    one, one_tools, hit = locate_one(
                        runtime.suite, name, question, kind=kind, cache=rt.cache
                    )
                    cache_hit = cache_hit or hit
                    collected.append(one)
                    tools.extend(one_tools)
                    continue
                if draft is not None and is_compare(draft) and not state.get("force_capability"):
                    assert draft.subject and draft.secondary_subject
                    pair, pair_tools = compare_one(
                        runtime.suite, name, draft.subject, draft.secondary_subject
                    )
                    collected.append(pair)
                    tools.extend(pair_tools)
                    continue
                if (
                    draft is not None
                    and draft.subject
                    and draft.candidates
                    and bound is None
                    and not state.get("force_capability")
                    and plan.intent.target in (CAPABILITY_LOCATE, CAPABILITY_CONNECT)
                ):
                    explored = explore_one(runtime.suite, name, draft, question)
                    if explored is not None:
                        guided, guided_areas, guided_tools = explored
                        collected.append(guided)
                        tools.extend(guided_tools)
                        if guided_areas:
                            areas[name] = guided_areas
                        continue
                dispatched = dispatch_plan(
                    cast(AssistantState, {**seed, "bound_node": bound}),
                    suite=runtime.suite,
                    suite_name=name,
                    llm_client=rt.llm_client,
                    bound_node=bound,
                    # The merged answer is written in code; the model's own
                    # final phrasing round would be billed and then discarded.
                    need_answer=False,
                )
                cache_hit = cache_hit or _served_from_cache(runtime.suite)
                collected.append(dispatched["result"])
                if draft is not None and draft.subject and bound is None:
                    if _is_broad(dispatched["result"]):
                        measured = MeasuredSuite(runtime.suite)
                        found = area_summary(
                            measured, name, draft.subject, kind=draft.kind or "occupation"
                        )
                        tools.extend(measured.tool_calls)
                        if found:
                            areas[name] = found
                tools.extend(dispatched.get("tool_calls") or [])
                stages.extend((dispatched.get("llm_stages") or [])[prior:])
        except UnknownSuiteError:
            raise
        except Exception:  # noqa: BLE001 — a down suite must not fail the turn
            logger.warning("suite %s unavailable", name, exc_info=True)
            extra.append(f"suite_unavailable:{name}")
    return {
        "results": collected,
        "areas": areas,
        "tools": tools,
        "stages": stages,
        "cache_hit": cache_hit,
        "extra_warnings": extra,
    }


def _answer(state: TurnState, rt: TurnRuntime) -> dict[str, Any]:
    plan = state["plan"]
    assert plan is not None
    capability = plan.intent.target
    results = tuple(state.get("results") or [])
    extra = [*(state.get("extra_warnings") or []), *honesty_warnings(results)]
    stages = list(state.get("stages") or [])
    selected = state.get("selected") or []
    if not results:
        # Warnings go on the placeholder once — they are not appended again below.
        placeholder = AgentResult(
            capability=capability,
            suite=selected[0] if selected else rt.registry.default,
            warnings=extra or ["not_found"],
        )
        return {
            "results": [placeholder],
            "answer": merge_answers((), extra_warnings=extra),
            "extra_warnings": [],
        }
    only = results[0]
    if (
        len(results) == 1
        and not extra
        and rt.answer_mode == "natural"
        and only.nodes
        and not only.warnings
    ):
        # Only a clean single-suite hit is worth a natural-language rephrase;
        # everything else gets the same deterministic answer as a multi-suite turn.
        answer, stage = build_answer(only, llm_client=rt.llm_client, mode=rt.answer_mode)
        if stage is not None:
            stages.append(stage)
    else:
        answer = merge_answers(results, extra_warnings=extra)
    if extra:
        primary = results[0].model_copy(update={"warnings": [*results[0].warnings, *extra]})
        results = (primary, *results[1:])
    return {"results": list(results), "answer": answer, "stages": stages, "extra_warnings": []}


def _remember(state: TurnState) -> dict[str, Any]:
    plan = state.get("plan")
    results = state.get("results") or []
    capability = plan.intent.target if plan is not None else CAPABILITY_LOCATE
    memo = memo_for(
        question=state["question"],
        capability=capability,
        answer=state.get("answer") or "",
        results=results,
        run_id=state.get("run_id") or "",
    )
    return {"history": [memo]}


def build_turn_graph(rt: TurnRuntime, *, checkpointer: Any = None) -> CompiledStateGraph:
    graph = StateGraph(TurnState)
    graph.add_node("plan", lambda s: _plan(s, rt))
    graph.add_node("dispatch", lambda s: _dispatch(s, rt))
    graph.add_node("answer", lambda s: _answer(s, rt))
    graph.add_node("remember", _remember)
    graph.set_entry_point("plan")
    graph.add_conditional_edges(
        "plan", lambda s: "remember" if s.get("stopped") else "dispatch", ["remember", "dispatch"]
    )
    graph.add_edge("dispatch", "answer")
    graph.add_edge("answer", "remember")
    graph.add_edge("remember", END)
    return graph.compile(checkpointer=checkpointer)


def thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def conversation_history(thread_id: str) -> list[TurnMemo]:
    """The persisted turns of one conversation, oldest first. [] if none."""
    from talent_angels.assistant.checkpoint import shared_checkpointer
    from talent_angels.memory.paths import checkpoint_db_path

    if not checkpoint_db_path().exists():
        return []
    graph = StateGraph(TurnState)
    graph.add_node("remember", _remember)
    graph.set_entry_point("remember")
    graph.add_edge("remember", END)
    compiled = graph.compile(checkpointer=shared_checkpointer())
    values = compiled.get_state(thread_config(thread_id)).values
    return list(values.get("history") or [])
