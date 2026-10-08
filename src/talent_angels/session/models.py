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


class AreaChoice(BaseModel):
    """An occupation group offered as "which area?", answered with its letter."""

    model_config = ConfigDict(extra="forbid")

    letter: str
    suite: str
    code: str
    label: str
    count: int
    #: The search the area narrows ("engineer"), re-run inside the group.
    query: str
    kind: str | None = "occupation"


class LastBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node: NodeRef


class SessionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    name: str | None = None
    transcript: list[TranscriptLine] = Field(default_factory=list)
    binding: LastBinding | None = None
    bindings: dict[str, NodeRef] = Field(default_factory=dict)
    pending: list[PendingChoice] = Field(default_factory=list)
    #: A profile statement ("I am a X") waiting on the user's pick from ``pending``.
    pending_profile_intent: str | None = None
    #: Titles resolved in this conversation, as searched, newest last (working set).
    recent: list[str] = Field(default_factory=list)
    #: Occupation groups offered with the pick list ("which area?").
    areas: list[AreaChoice] = Field(default_factory=list)
    #: What the open pick list was searched for ("engineer"); hints keep it.
    list_topic: str = ""
    last_result: AgentResult | None = None
    last_results: list[AgentResult] = Field(default_factory=list)
