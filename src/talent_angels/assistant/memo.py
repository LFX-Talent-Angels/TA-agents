"""The compact, typed record of one turn kept in a conversation's history.

Checkpointed per ``thread_id`` (the session id) so a conversation's history
survives a restart and can be shown back to the user (``GET
/v1/sessions/{id}/history``). Node ids and labels only — pointers, never
taxonomy payloads (descriptions, edge properties). Bounded: at most
:data:`MAX_TOPS` nodes per suite and :data:`MAX_HISTORY` turns per thread.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from talent_angels.contracts import AgentResult

MAX_TOPS = 5
MAX_HISTORY = 50
MAX_ANSWER_CHARS = 2000


class SuiteMemo(BaseModel):
    suite: str
    capability: str
    node_ids: list[str] = Field(default_factory=list)
    node_labels: list[str] = Field(default_factory=list)
    node_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class TurnMemo(BaseModel):
    ts: str
    run_id: str = ""
    question: str
    capability: str
    answer: str
    suites: list[SuiteMemo] = Field(default_factory=list)


def memo_for(
    *,
    question: str,
    capability: str,
    answer: str,
    results: Sequence[AgentResult],
    run_id: str = "",
) -> TurnMemo:
    return TurnMemo(
        ts=datetime.now(UTC).isoformat(),
        run_id=run_id,
        question=question,
        capability=capability,
        answer=answer[:MAX_ANSWER_CHARS],
        suites=[
            SuiteMemo(
                suite=result.suite,
                capability=result.capability,
                node_ids=[node.id for node in result.nodes[:MAX_TOPS]],
                node_labels=[node.pref_label for node in result.nodes[:MAX_TOPS]],
                node_count=len(result.nodes),
                warnings=list(result.warnings),
            )
            for result in results
        ],
    )


def append_bounded(left: list[TurnMemo] | None, right: list[TurnMemo] | None) -> list[TurnMemo]:
    """LangGraph reducer: append this turn, keep the newest ``MAX_HISTORY``."""
    merged = [*(left or []), *(right or [])]
    return merged[-MAX_HISTORY:]
