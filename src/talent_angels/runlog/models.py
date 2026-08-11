"""Run-log record schema (MVP plan Sec 2.6) — one JSON object per turn.

OTel GenAI-aligned field names for the ``gen_ai`` block so the JSONL stays
readable against that convention even though Tier 0 (this file) is the
source of truth, not an OTel exporter.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


class EfficiencyInfo(BaseModel):
    mode: str = "tool_only"  # tool_only | llm_answer | cached_result
    result_cache_hit: bool = False
    prompt_cache_hit: bool = False
    heuristic_intent: bool = True


class GenAIUsage(BaseModel):
    provider_name: str = "none"
    request_model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    calls: int = 0


class ToolCall(BaseModel):
    name: str
    ms: float
    ok: bool


class GraphStats(BaseModel):
    queries: int = 0
    total_ms: float = 0.0


class CostBreakdown(BaseModel):
    llm: float = 0.0
    llm_with_cache_savings: float = 0.0
    total: float = 0.0
    rate_card: str = ""


class ResultSummary(BaseModel):
    confidence: float | None = None
    node_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RunLogRecord(BaseModel):
    schema_version: int = SCHEMA_VERSION
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    suite: str
    plan: list[str]  # e.g. ["locate"], ["locate", "connect"]
    question: str
    efficiency: EfficiencyInfo = Field(default_factory=EfficiencyInfo)
    gen_ai: GenAIUsage = Field(default_factory=GenAIUsage)
    tools: list[ToolCall] = Field(default_factory=list)
    graph: GraphStats = Field(default_factory=GraphStats)
    cost_usd: CostBreakdown = Field(default_factory=CostBreakdown)
    result: ResultSummary = Field(default_factory=ResultSummary)
