"""The main assistant's intent -> plan -> skill dispatch -> answer loop."""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from talent_angels.assistant.answer import build_answer
from talent_angels.assistant.connect_request import (
    UnsupportedConnectQuery,
    extract_connect_request,
)
from talent_angels.assistant.intent import CAPABILITY_CONNECT, CAPABILITY_LOCATE, Capability
from talent_angels.assistant.planning import build_plan, build_plan_for_capability
from talent_angels.assistant.state import AssistantState
from talent_angels.contracts import AgentResult
from talent_angels.llm import LLMClient
from talent_angels.skills.connect import connect
from talent_angels.skills.locate import ESCO_SUITE_NAME, locate
from talent_angels.suites.measured import MeasuredSuite
from talent_angels.suites.protocol import SuiteTools


def _interpret_intent(
    state: AssistantState,
    *,
    suite_name: str,
    forced_capability: Capability | None,
) -> AssistantState:
    plan = (
        build_plan(state["question"], suites=(suite_name,))
        if forced_capability is None
        else build_plan_for_capability(forced_capability, suites=(suite_name,))
    )
    return {"capability": plan.intent.target, "plan": plan}


def _dispatch_plan(state: AssistantState, *, suite: SuiteTools, suite_name: str) -> AssistantState:
    capability = state["plan"].intent.target
    measured = MeasuredSuite(suite)
    if capability == CAPABILITY_LOCATE:
        result = locate(
            measured,
            suite_name,
            state["question"],
            kind=state.get("kind"),
        )
        return {"result": result, "tool_calls": measured.tool_calls}

    if capability == CAPABILITY_CONNECT:
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

        locate_kind = state.get("kind")
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
    answer, llm_usage = build_answer(state["result"], llm_client=llm_client, mode=answer_mode)
    return {"answer": answer, "llm_usage": llm_usage}


def build_graph(
    *,
    suite: SuiteTools,
    llm_client: LLMClient,
    suite_name: str = ESCO_SUITE_NAME,
    answer_mode: str = "structured",
    forced_capability: Capability | None = None,
) -> CompiledStateGraph:
    graph = StateGraph(AssistantState)
    graph.add_node(
        "interpret_intent",
        lambda s: _interpret_intent(
            s,
            suite_name=suite_name,
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
