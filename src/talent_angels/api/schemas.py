"""Request/response models for the FastAPI edge."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from talent_angels.assistant.memo import TurnMemo
from talent_angels.contracts import AgentResult
from talent_angels.runlog import CostBreakdown, GenAIUsage, GraphStats, ToolCall
from talent_angels.skills.connect.discover import DEFAULT_DISCOVER_MAX_NODES, MAX_DISCOVER_NODES
from talent_angels.skills.connect.discover_io import MAX_ITEMS
from talent_angels.skills.connect.neighbourhood import DEFAULT_MAX_NODES, MAX_BREADTH, MAX_DEPTH
from talent_angels.skills.roadmap import DEFAULT_MAX_NODES as SUCCESSORS_MAX_NODES
from talent_angels.skills.roadmap import MAX_PER_STEP

#: Longer questions are rejected with 422; the runtime also bounds what it sends.
# The same bound the runtime applies (assistant/turn.py), so the API rejects
# what the runtime would otherwise shorten.
MAX_QUESTION_CHARS = 1000
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
Name = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
Suites = Field(default=None, description="Subset of taxonomies; omit for all.")
Depth = Field(default=2, ge=1, le=MAX_DEPTH)
Breadth = Field(default=4, ge=1, le=MAX_BREADTH)


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class PathfindRequest(_Request):
    from_: str = Field(alias="from", min_length=1, max_length=MAX_QUESTION_CHARS)
    to: str = Name
    suites: list[str] | None = Suites


class RoadmapRequest(PathfindRequest):
    per_step: int = Field(default=10, ge=1, le=MAX_PER_STEP)


class SuccessorsRequest(_Request):
    skill: str = Name
    suites: list[str] | None = Suites
    depth: int = Depth
    breadth: int = Breadth
    max_nodes: int = Field(default=SUCCESSORS_MAX_NODES, ge=1, le=DEFAULT_MAX_NODES)


class MapRequest(_Request):
    start: str | None = Field(default=None, max_length=MAX_QUESTION_CHARS)
    route_from: str | None = Field(default=None, max_length=MAX_QUESTION_CHARS)
    route_to: str | None = Field(default=None, max_length=MAX_QUESTION_CHARS)
    suites: list[str] | None = Suites
    depth: int = Field(default=1, ge=1, le=MAX_DEPTH)
    breadth: int | None = Field(
        default=None, ge=1, le=MAX_BREADTH, description="Default 4 (3 on a route)."
    )
    kinds: list[str] | None = None
    max_nodes: int = Field(default=DEFAULT_MAX_NODES, ge=1, le=DEFAULT_MAX_NODES)

    @model_validator(mode="after")
    def _one_mode(self) -> MapRequest:
        route = [self.route_from, self.route_to]
        if bool(self.start) == all(route) or (any(route) and not all(route)):
            raise ValueError("give either start, or both route_from and route_to")
        return self


class DiscoverItem(_Request):
    item: str = Name
    kind: str | None = None
    related_to: list[str] = []


class DiscoverRequest(_Request):
    items: list[DiscoverItem] = Field(min_length=1, max_length=MAX_ITEMS)
    suites: list[str] | None = Suites
    depth: int = Depth
    breadth: int = Breadth
    kinds: list[str] | None = None
    max_nodes: int = Field(default=DEFAULT_DISCOVER_MAX_NODES, ge=1, le=MAX_DISCOVER_NODES)
    nodes: list[dict[str, Any]] | None = Field(
        default=None, description="A previous result's `resolved` block: skips resolution."
    )
