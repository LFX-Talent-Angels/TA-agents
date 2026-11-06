"""Typed rows, units, and verdicts for batch tagging (PILOT-PLAN.md "Architecture").

Pydantic v2 so every boundary inside ``tagging/`` — reader -> select -> rewrite
-> search -> choose -> verdict -> writer — stays typed, matching the suite
contract's own pointer-not-payload discipline (contracts/models.py).
"""

from __future__ import annotations

from collections import defaultdict
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class SourceRow(BaseModel):
    """One row of the client's structured document, in document order."""

    model_config = {"frozen": True}

    row_id: str
    parent_id: str | None
    label: str
    text: str
    order: int


class Document(BaseModel):
    """A client document: its source rows plus how to look them up."""

    key: str
    path: str
    source_format: str
    rows: list[SourceRow] = Field(default_factory=list)

    def by_id(self) -> dict[str, SourceRow]:
        return {row.row_id: row for row in self.rows}

    def children(self) -> dict[str | None, list[SourceRow]]:
        """Rows grouped by ``parent_id``, each group in document order."""
        groups: dict[str | None, list[SourceRow]] = defaultdict(list)
        for row in self.rows:
            groups[row.parent_id].append(row)
        return dict(groups)


class Verdict(StrEnum):
    """Per unit x suite outcome (tag-v1 policy, ``verdict.py`` — PR3)."""

    ACCEPTED = "accepted"
    REVIEW = "review"
    ADJACENT = "adjacent"
    GAP = "gap"
    SKIPPED = "skipped"
    NOT_APPLICABLE = "n/a"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    HUMAN_ACCEPTED = "human_accepted"
    HUMAN_GAP = "human_gap"


class TagUnit(BaseModel):
    """One taggable row: client text plus the profile-derived kind/context."""

    model_config = {"frozen": True}

    unit_key: str
    doc_key: str
    row_id: str
    kind: str
    role: Literal["leaf", "heading", "row_rule"]
    text: str
    clean_text: str
    #: Ancestor texts from top (excluding the Document root) to the parent,
    #: raw stripped text — not the cleaned text.
    context_path: list[str] = Field(default_factory=list)
    group_key: str
    seed_phrases: list[str] = Field(default_factory=list)


class SkippedRow(BaseModel):
    """A row that ``select_units`` routed out of tagging, and why."""

    model_config = {"frozen": True}

    row_id: str
    reason: Literal["root", "metadata", "heading", "excluded_kind", "filtered"]
    kind: str | None = None


class Versions(BaseModel):
    """Everything a fingerprint pins, so a changed rule invalidates the ledger."""

    model_config = {"frozen": True}

    pipeline: str
    policy: str
    prompts: str
    ta_taxonomies: str
    model: str
