"""The main assistant's intent -> plan -> skill dispatch -> answer loop."""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from talent_angels.assistant.answer import build_answer
from talent_angels.assistant.connect_request import (
    UnsupportedConnectQuery,
    extract_connect_request,
    followup_connect_request,
)
from talent_angels.assistant.intent import (
    CAPABILITY_CONNECT,
    CAPABILITY_LOCATE,
    CAPABILITY_PATHFIND,
    Capability,
    extract_locate_subject,
    extract_pathfind_endpoints,
)
from talent_angels.assistant.llm_plan import (
    connect_request_from_draft,
    interpret_question,
)
from talent_angels.assistant.state import AssistantState
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.llm import LLMClient
from talent_angels.runlog import usage_from_stage
from talent_angels.skills.connect import connect
from talent_angels.skills.locate import ESCO_SUITE_NAME, locate
from talent_angels.skills.locate.rank import group_and_sort_locate
from talent_angels.skills.pathfind import pathfind
from talent_angels.suites.measured import MeasuredSuite
from talent_angels.suites.protocol import SuiteTools


def _interpret_intent(
    state: AssistantState,
    *,
    suite_name: str,
    llm_client: LLMClient,
    forced_capability: Capability | None,
) -> AssistantState:
    interpreted = interpret_question(
        state["question"],
        suites=(suite_name,),
        llm_client=llm_client,
        forced_capability=forced_capability,
    )
    stages = list(state.get("llm_stages") or [])
    if interpreted.stage is not None:
        stages.append(interpreted.stage)
    return {
        "capability": interpreted.plan.intent.target,
        "plan": interpreted.plan,
        "plan_draft": interpreted.draft,
        "heuristic_intent": interpreted.heuristic,
        "llm_stages": stages,
    }


def dispatch_plan(state: AssistantState, *, suite: SuiteTools, suite_name: str) -> AssistantState:
    """Run the planned capability against one opened suite."""
    capability = state["plan"].intent.target
    measured = MeasuredSuite(suite)
    draft = state.get("plan_draft")
    if capability == CAPABILITY_LOCATE:
        locate_text = (
            draft.subject if draft and draft.subject else extract_locate_subject(state["question"])
        )
        locate_kind = state.get("kind") or (draft.kind if draft else None)
        result = locate(
            measured,
            suite_name,
            locate_text,
            kind=locate_kind,
        )
        result = group_and_sort_locate(measured, result, locate_text, suite_name=suite_name)
        return {"result": result, "tool_calls": measured.tool_calls}

    if capability == CAPABILITY_CONNECT:
        bound = state.get("bound_node")
        if isinstance(bound, NodeRef) and bound.suite == suite_name:
            followup = followup_connect_request(state["question"], bound)
            if followup is not None:
                result = connect(
                    measured,
                    suite_name,
                    bound,
                    request=followup,
                    confidence=None,
                    locate_evidence=[],
                )
                return {
                    "result": result,
                    "tool_calls": measured.tool_calls,
                }

        request = connect_request_from_draft(draft) if draft is not None else None
        if request is None:
            try:
                request = extract_connect_request(state["question"])
            except UnsupportedConnectQuery:
                return {
                    "result": AgentResult(
                        capability=capability,
                        suite=suite_name,
                        warnings=["unsupported_connect_query"],
                    ),
                    "tool_calls": [],
                }

        locate_kind = state.get("kind") or (draft.kind if draft else None)
        if locate_kind is None and request.rel_types == ("HAS_SKILL",):
            locate_kind = "occupation"
        located = locate(
            measured,
            suite_name,
            request.subject,
            kind=locate_kind,
        )
        located = group_and_sort_locate(measured, located, request.subject, suite_name=suite_name)
        if not located.nodes or "ambiguous" in located.warnings:
            return {
                "result": located.model_copy(update={"capability": capability}),
                "tool_calls": measured.tool_calls,
            }

        result = connect(
            measured,
            suite_name,
            located.nodes[0],
            request=request,
            confidence=located.confidence,
            locate_evidence=located.evidence,
        )
        return {
            "result": result,
            "tool_calls": measured.tool_calls,
        }

    if capability == CAPABILITY_PATHFIND:
        return _dispatch_pathfind(
            state,
            suite=measured,
            suite_name=suite_name,
            draft=draft,
        )

    return {
        "result": AgentResult(
            capability=capability,
            suite=suite_name,
            warnings=[f"capability_not_implemented:{capability}"],
        ),
        "tool_calls": [],
    }


def _dispatch_pathfind(
    state: AssistantState,
    *,
    suite: MeasuredSuite,
    suite_name: str,
    draft: object,
) -> AssistantState:
    ends: tuple[str, str] | None = None
    subject = getattr(draft, "subject", None) if draft is not None else None
    secondary = getattr(draft, "secondary_subject", None) if draft is not None else None
    if isinstance(subject, str) and isinstance(secondary, str) and subject and secondary:
        ends = (subject, secondary)
    else:
        ends = extract_pathfind_endpoints(state["question"])
    if ends is None:
        return {
            "result": AgentResult(
                capability=CAPABILITY_PATHFIND,
                suite=suite_name,
                warnings=["unsupported_pathfind_query"],
            ),
            "tool_calls": suite.tool_calls,
        }
    from_text, to_text = ends
    located_from = group_and_sort_locate(
        suite,
        locate(suite, suite_name, from_text, kind="occupation"),
        from_text,
        suite_name=suite_name,
    )
    located_to = group_and_sort_locate(
        suite,
        locate(suite, suite_name, to_text, kind="occupation"),
        to_text,
        suite_name=suite_name,
    )
    if not located_from.nodes or "ambiguous" in located_from.warnings:
        return {
            "result": located_from.model_copy(
                update={
                    "capability": CAPABILITY_PATHFIND,
                    "warnings": [*located_from.warnings, "endpoint_not_found"],
                }
            ),
            "tool_calls": suite.tool_calls,
        }
    if not located_to.nodes or "ambiguous" in located_to.warnings:
        return {
            "result": located_to.model_copy(
                update={
                    "capability": CAPABILITY_PATHFIND,
                    "warnings": [*located_to.warnings, "endpoint_not_found"],
                }
            ),
            "tool_calls": suite.tool_calls,
        }
    result = pathfind(suite, suite_name, located_from.nodes[0], located_to.nodes[0])
    return {"result": result, "tool_calls": suite.tool_calls}


def _answer(state: AssistantState, *, llm_client: LLMClient, answer_mode: str) -> AssistantState:
    answer, stage = build_answer(state["result"], llm_client=llm_client, mode=answer_mode)
    stages = list(state.get("llm_stages") or [])
    if stage is not None:
        stages.append(stage)
    return {
        "answer": answer,
        "llm_usage": usage_from_stage(stage) if stage is not None else None,
        "llm_stages": stages,
    }


def build_graph(
    *,
    suite: SuiteTools,
    llm_client: LLMClient,
    suite_name: str = ESCO_SUITE_NAME,
    answer_mode: str = "structured",
    forced_capability: Capability | None = None,
) -> CompiledStateGraph:
    """Intent → plan → dispatch → answer. The model names the goal; code walks the graph."""
    graph = StateGraph(AssistantState)
    graph.add_node(
        "interpret_intent",
        lambda s: _interpret_intent(
            s,
            suite_name=suite_name,
            llm_client=llm_client,
            forced_capability=forced_capability,
        ),
    )
    graph.add_node(
        "dispatch_plan",
        lambda s: dispatch_plan(s, suite=suite, suite_name=suite_name),
    )
    graph.add_node("answer", lambda s: _answer(s, llm_client=llm_client, answer_mode=answer_mode))
    graph.set_entry_point("interpret_intent")
    graph.add_edge("interpret_intent", "dispatch_plan")
    graph.add_edge("dispatch_plan", "answer")
    graph.add_edge("answer", END)
    return graph.compile()
