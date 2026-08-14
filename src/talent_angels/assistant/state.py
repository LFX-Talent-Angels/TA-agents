"""Typed state threaded through the LangGraph loop."""

from __future__ import annotations

from typing import TypedDict

from talent_angels.assistant.intent import Capability
from talent_angels.assistant.llm_plan import PlanDraft
from talent_angels.assistant.planning import ExecutionPlan
from talent_angels.contracts import AgentResult
from talent_angels.llm import LLMUsage
from talent_angels.runlog import StageUsage, ToolCall


class AssistantState(TypedDict, total=False):
    question: str
    kind: str | None
    capability: Capability
    plan: ExecutionPlan
    plan_draft: PlanDraft | None
    heuristic_intent: bool
    result: AgentResult
    tool_calls: list[ToolCall]
    answer: str
    llm_usage: LLMUsage | None
    llm_stages: list[StageUsage]
