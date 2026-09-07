"""Typed state threaded through the LangGraph loop."""

from __future__ import annotations

from typing import TypedDict

from talent_angels.contracts import AgentResult
from talent_angels.llm import LLMUsage
from talent_angels.memory import UserMemory


class AssistantState(TypedDict, total=False):
    question: str
    kind: str | None
    user_id: str | None
    user_memories: list[UserMemory]
    capability: str
    result: AgentResult
    answer: str
    llm_usage: LLMUsage | None
