"""Request/response models for the FastAPI edge."""

from __future__ import annotations

from pydantic import BaseModel

from talent_angels.contracts import AgentResult
from talent_angels.runlog import CostBreakdown, GenAIUsage, GraphStats, ToolCall


class QueryRequest(BaseModel):
    question: str
    suite: str = "esco"
    kind: str | None = None
    # Omit to opt out of memory entirely: no retrieval, nothing stored.
    user_id: str | None = None


class UsageInfo(BaseModel):
    gen_ai: GenAIUsage
    cost_usd: CostBreakdown
    graph: GraphStats
    tools: list[ToolCall]


class QueryResponse(BaseModel):
    run_id: str
    capability: str
    suite: str
    result: AgentResult
    answer: str
    usage: UsageInfo


class HealthResponse(BaseModel):
    status: str
    neo4j_reachable: bool
