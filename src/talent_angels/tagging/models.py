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
    role: Literal["leaf", "heading", "group", "row_rule"]
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


class CandidateRef(BaseModel):
    """A suite hit worth keeping: a pointer + label, never a description.

    Licensing (ADR-0004's pointer-not-payload rule): the ledger and the
    workbook cite ids/labels/URIs, never the suite's own descriptive text.
    """

    model_config = {"frozen": True}

    suite: str
    id: str
    source_id: str
    kind: str
    label: str
    method: str
    best_rank: int
    hits: int
    queries: list[str] = Field(default_factory=list)
    alias_unconfirmed: bool = False


class QueryLog(BaseModel):
    """One query sent to one suite, and what it came back with."""

    q: str
    kind: str | None
    n: int
    top_method: str | None
    warnings: list[str] = Field(default_factory=list)


class SearchOutcome(BaseModel):
    """Everything `search.search_unit` learned about one unit x suite."""

    suite: str
    status: Literal["ok", "n/a", "unavailable"]
    routed_kind: str | None
    queries: list[QueryLog] = Field(default_factory=list)
    candidates: list[CandidateRef] = Field(default_factory=list)
    error: str | None = None


class VerdictResult(BaseModel):
    """`verdict.decide`'s output: the policy call plus the tags it kept."""

    verdict: Verdict
    reason_code: str
    tags: list[CandidateRef] = Field(default_factory=list)
    alternatives: list[CandidateRef] = Field(default_factory=list)


class TagRecord(BaseModel):
    """One ledger line: unit x suite x attempt, everything the run decided.

    `rewrite`/`choice` stay untyped `dict`s in PR2 — PR3 defines `Rewrite` and
    `Choice` and starts populating them; the ledger schema does not change.
    """

    schema_version: int = 1
    run_id: str
    ts: str
    doc_key: str
    unit_key: str
    row_id: str
    suite: str
    fingerprint: str
    attempt: int = 0
    versions: Versions
    kind: str
    clean_text: str
    context_path: list[str] = Field(default_factory=list)
    hint: str = ""
    queries: list[QueryLog] = Field(default_factory=list)
    candidates: list[CandidateRef] = Field(default_factory=list)
    rewrite: dict[str, object] | None = None
    choice: dict[str, object] | None = None
    verdict: Verdict
    reason_code: str
    tags: list[CandidateRef] = Field(default_factory=list)
    alternatives: list[CandidateRef] = Field(default_factory=list)
    usage: dict[str, float] = Field(default_factory=dict)
    decided_by: Literal["policy", "human"] = "policy"


class RunOptions(BaseModel):
    """CLI/runner knobs for one `ta-agent tag` invocation."""

    no_llm: bool = True
    limit: int | None = None
    rows: set[str] | None = None
    suites: list[str] | None = None
    depth: str | None = None
    refresh: bool = False
    fresh: bool = False
    workers: int = 1


class RunSummary(BaseModel):
    """What `runner.run` reports back to the CLI, printable as-is."""

    project: str
    documents: list[str] = Field(default_factory=list)
    units: int = 0
    counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    reused: int = 0
    searched: int = 0
    outputs: list[str] = Field(default_factory=list)
    seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)
