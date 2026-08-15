"""The main assistant's intent -> plan -> skill dispatch -> answer loop."""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from talent_angels.assistant.agent_loop import run_tool_loop
from talent_angels.assistant.answer import build_answer
from talent_angels.assistant.connect_request import (
    UnsupportedConnectQuery,
    extract_connect_request,
)
from talent_angels.assistant.intent import (
    CAPABILITY_CONNECT,
    CAPABILITY_LOCATE,
    Capability,
    classify_capability,
    extract_locate_subject,
)
from talent_angels.assistant.llm_plan import (
    connect_request_from_draft,
    interpret_question,
    uses_llm_planner,
)
from talent_angels.assistant.state import AssistantState
from talent_angels.contracts import AgentResult
from talent_angels.llm import LLMClient
from talent_angels.runlog import usage_from_stage
from talent_angels.skills.connect import connect
from talent_angels.skills.locate import ESCO_SUITE_NAME, locate
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
        suite_name=suite_name,
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


def _dispatch_plan(state: AssistantState, *, suite: SuiteTools, suite_name: str) -> AssistantState:
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
        return {"result": result, "tool_calls": measured.tool_calls}

    if capability == CAPABILITY_CONNECT:
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

    return {
        "result": AgentResult(
            capability=capability,
            suite=suite_name,
            warnings=[f"capability_not_implemented:{capability}"],
        ),
        "tool_calls": [],
    }


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
    graph = StateGraph(AssistantState)
    if uses_llm_planner(llm_client) and forced_capability is None:

        def _agent(state: AssistantState) -> AssistantState:
            try:
                outcome = run_tool_loop(
                    question=state["question"],
                    suite=suite,
                    suite_name=suite_name,
                    llm_client=llm_client,
                    kind=state.get("kind"),
                )
            except RuntimeError:
                # Provider failed mid-loop. Finish with heuristic plan +
                # structured facts so the CLI does not crash. Never treat a
                # path/gap question as Connect.
                interpreted = _interpret_intent(
                    state,
                    suite_name=suite_name,
                    llm_client=llm_client,
                    forced_capability=classify_capability(state["question"]),
                )
                dispatched = _dispatch_plan(
                    {**state, **interpreted},
                    suite=suite,
                    suite_name=suite_name,
                )
                result = dispatched["result"]
                answer, stage = build_answer(result, llm_client=llm_client, mode="structured")
                return {
                    **interpreted,
                    **dispatched,
                    "answer": answer,
                    "llm_usage": usage_from_stage(stage) if stage is not None else None,
                }
            return {
                "capability": outcome.plan.intent.target,
                "plan": outcome.plan,
                "heuristic_intent": False,
                "result": outcome.result,
                "tool_calls": outcome.tool_calls,
                "answer": outcome.answer,
                "llm_stages": outcome.stages,
                "llm_usage": usage_from_stage(outcome.stages[-1]) if outcome.stages else None,
            }

        graph.add_node("agent", _agent)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)
        return graph.compile()

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
        lambda s: _dispatch_plan(s, suite=suite, suite_name=suite_name),
    )
    graph.add_node("answer", lambda s: _answer(s, llm_client=llm_client, answer_mode=answer_mode))
    graph.set_entry_point("interpret_intent")
    graph.add_edge("interpret_intent", "dispatch_plan")
    graph.add_edge("dispatch_plan", "answer")
    graph.add_edge("answer", END)
    return graph.compile()
