"""Request/response models for the FastAPI edge."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from talent_angels.assistant.memo import TurnMemo
from talent_angels.contracts import AgentResult
from talent_angels.engine.inputs import (  # noqa: F401  re-exported for the HTTP edge
    MAX_QUESTION_CHARS,
    DiscoverItem,
    DiscoverRequest,
    MapRequest,
    PathfindRequest,
    RoadmapRequest,
    SuccessorsRequest,
    _Request,
)
from talent_angels.runlog import CostBreakdown, GenAIUsage, GraphStats, ToolCall

SESSION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"


class QueryRequest(BaseModel):
    question: str = Field(max_length=MAX_QUESTION_CHARS)
    suite: str | None = None
    kind: str | None = None
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN)


class UsageInfo(BaseModel):
    gen_ai: GenAIUsage
    cost_usd: CostBreakdown
    graph: GraphStats
    tools: list[ToolCall]


class QueryResponse(BaseModel):
    run_id: str
    session_id: str
    capability: str
    plan: list[str] = []
    suite: str
    suites: list[str] = []
    result: AgentResult
    results: list[AgentResult] = []
    answer: str
    usage: UsageInfo


class SuiteHealth(BaseModel):
    name: str
    reachable: bool


class HealthResponse(BaseModel):
    #: "ok" when every attached suite is reachable, else "degraded".
    status: str
    neo4j_reachable: bool
    suites: list[SuiteHealth] = []
    llm_provider: str = "none"
    llm_model: str = ""
    llm_error: str | None = None


class SessionCreated(BaseModel):
    session_id: str


class SessionHistory(BaseModel):
    session_id: str
    turns: list[TurnMemo]


class SessionErased(BaseModel):
    session_id: str
    session_files_deleted: int
    history_deleted: bool


# --- Deterministic capabilities: same JSON as the CLI, no model call. -----------------
# The request models live in the engine (one declaration per capability) and are re-exported
# above, so the routes and their OpenAPI names stay the same.


class RunRequest(_Request):
    """`POST /v1/run` and `POST /v1/jobs`: one capability by name, with its input."""

    capability: str = Field(min_length=1, max_length=64)
    input: dict[str, Any] = Field(default_factory=dict)
