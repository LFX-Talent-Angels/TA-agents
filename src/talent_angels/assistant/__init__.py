"""The main assistant: the only piece that talks to the user.

Owns intent, the plan (L / L+C / L+C+P / ...+E), suite selection, merging
across suites, honesty rules, and the final answer. Implemented as a LangGraph
stateful loop. See ARCHITECTURE.md.
"""

from talent_angels.assistant.cache import ResultCache
from talent_angels.assistant.graph import build_graph
from talent_angels.assistant.intent import (
    CAPABILITY_CONNECT,
    CAPABILITY_LOCATE,
    CAPABILITY_PATHFIND,
    classify_capability,
)
from talent_angels.assistant.state import AssistantState
from talent_angels.assistant.turn import TurnOutcome, run_turn
from talent_angels.memory import MemoryClient, get_memory_client

__all__ = [
    "CAPABILITY_CONNECT",
    "CAPABILITY_LOCATE",
    "CAPABILITY_PATHFIND",
    "AssistantState",
    "MemoryClient",
    "ResultCache",
    "TurnOutcome",
    "build_graph",
    "classify_capability",
    "get_memory_client",
    "run_turn",
]
