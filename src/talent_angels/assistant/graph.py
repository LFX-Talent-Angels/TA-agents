"""The main assistant's LangGraph loop (ARCHITECTURE.md): intent -> plan ->
dispatch -> answer.

Gate A scope: only the `locate` capability is dispatched for real; `connect`/
`pathfind` route through intent classification but resolve to an explicit
"not implemented" `AgentResult` rather than silently falling back to Locate.
Only this module (the assistant) may change the plan or produce the
user-facing answer — skills stay passive procedures (ARCHITECTURE.md rule #3).
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from talent_angels.assistant.answer import build_answer
from talent_angels.assistant.intent import CAPABILITY_LOCATE, classify_capability
from talent_angels.assistant.state import AssistantState
from talent_angels.contracts import AgentResult
from talent_angels.llm import LLMClient
from talent_angels.skills.locate import ESCO_SUITE_NAME, locate
from talent_angels.skills.locate.resolve import SearchableSuite


def _interpret_intent(state: AssistantState) -> AssistantState:
    return {"capability": classify_capability(state["question"])}


def _dispatch_locate(
    state: AssistantState, *, suite: SearchableSuite, suite_name: str
) -> AssistantState:
    capability = state["capability"]
    if capability != CAPABILITY_LOCATE:
        result = AgentResult(
            capability=capability,
            suite=suite_name,
            warnings=[f"capability_not_implemented:{capability}"],
        )
    else:
        result = locate(suite, suite_name, state["question"], kind=state.get("kind"))
    return {"result": result}


def _answer(state: AssistantState, *, llm_client: LLMClient, answer_mode: str) -> AssistantState:
    answer, llm_usage = build_answer(state["result"], llm_client=llm_client, mode=answer_mode)
    return {"answer": answer, "llm_usage": llm_usage}


def build_graph(
    *,
    suite: SearchableSuite,
    llm_client: LLMClient,
    suite_name: str = ESCO_SUITE_NAME,
    answer_mode: str = "structured",
) -> CompiledStateGraph:
    graph = StateGraph(AssistantState)
    graph.add_node("interpret_intent", _interpret_intent)
    graph.add_node(
        "dispatch_locate",
        lambda s: _dispatch_locate(s, suite=suite, suite_name=suite_name),
    )
    graph.add_node("answer", lambda s: _answer(s, llm_client=llm_client, answer_mode=answer_mode))
    graph.set_entry_point("interpret_intent")
    graph.add_edge("interpret_intent", "dispatch_locate")
    graph.add_edge("dispatch_locate", "answer")
    graph.add_edge("answer", END)
    return graph.compile()
