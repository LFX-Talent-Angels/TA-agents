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
    extract_locate_subject,
)
from talent_angels.assistant.planning import (
    ExecutionPlan,
    Intent,
    PlanStep,
    build_plan,
    build_plan_for_capability,
)
from talent_angels.assistant.state import AssistantState
from talent_angels.assistant.turn import TurnOutcome, run_turn

__all__ = [
    "CAPABILITY_CONNECT",
    "CAPABILITY_LOCATE",
    "CAPABILITY_PATHFIND",
    "AssistantState",
    "ExecutionPlan",
    "Intent",
    "PlanStep",
    "ResultCache",
    "TurnOutcome",
    "build_graph",
    "build_plan",
    "build_plan_for_capability",
    "classify_capability",
    "extract_locate_subject",
    "run_turn",
]
