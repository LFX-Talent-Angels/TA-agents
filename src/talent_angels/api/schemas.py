"""Request/response models for the FastAPI edge."""

from __future__ import annotations

from pydantic import BaseModel, Field

from talent_angels.assistant.memo import TurnMemo
from talent_angels.contracts import AgentResult
from talent_angels.runlog import CostBreakdown, GenAIUsage, GraphStats, ToolCall

#: Longer questions are rejected with 422; the runtime also bounds what it sends.
MAX_QUESTION_CHARS = 2000
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
