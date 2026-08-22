"""Typed session state for the interactive ta-agent TUI."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from talent_angels.contracts import AgentResult, NodeRef


class TranscriptLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system"]
    text: str
    ts: str


class PendingChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int
    node: NodeRef
    group_label: str = ""


class LastBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node: NodeRef


class SessionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    name: str | None = None
    transcript: list[TranscriptLine] = Field(default_factory=list)
    binding: LastBinding | None = None
    pending: list[PendingChoice] = Field(default_factory=list)
    last_result: AgentResult | None = None
