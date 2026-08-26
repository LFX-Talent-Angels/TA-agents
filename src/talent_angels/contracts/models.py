"""Typed results that cross every component boundary (ARCHITECTURE.md).

Skills return only these models to the assistant — never prose. Node/edge IDs
are suite-scoped (e.g. ``esco:occupation:<uuid>``); nothing here equates IDs
across suites. Evidence is a pointer (a citation string from the suite tool),
never a payload, per ESCO's CC BY 4.0 attribution requirement and the
workspace's pointer-not-payload rule.

These are TA-agents' own contract types. They are populated *from* the suite
contract's ``ToolResult``/``Node``/``Edge``/``Candidate`` (``ta_taxonomies.
contract.models``) inside each skill — kept as separate types so this repo's
public shape doesn't change every time the suite contract does.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class NodeRef(BaseModel):
    """A single resolved or traversed graph node."""

    id: str  # suite-scoped, e.g. "esco:occupation:<uuid>"
    suite: str  # esco | onet | sfia | bls | ...
    source: str  # taxonomy family, e.g. "esco"
    source_id: str  # native identifier in that source, e.g. full ESCO URI
    kind: str  # Occupation | Skill | ISCOGroup | SkillGroup | ...
    pref_label: str
    alt_labels: list[str] = Field(default_factory=list)
    description: str | None = None  # official suite text when the tool returned it


class EdgeRef(BaseModel):
    """A single typed relationship between two suite-scoped nodes."""

    type: str  # HAS_SKILL | CLASSIFIED_UNDER | BROADER_THAN | ...
    suite: str
    source_node_id: str
    target_node_id: str
    properties: dict[str, object] = Field(default_factory=dict)


class EvidencePointer(BaseModel):
    """A pointer to the graph fact backing a result — never the raw payload."""

    suite: str
    pointer: str  # citation string from the suite tool, e.g. "esco:search:exact_pref:..."


class AgentResult(BaseModel):
    """Returned by every skill dispatch (locate/connect/pathfind/evaluate)."""

    capability: str  # locate | connect | pathfind | evaluate
    suite: str  # esco | onet | sfia | bls | ...
    nodes: list[NodeRef] = Field(default_factory=list)
    edges: list[EdgeRef] = Field(default_factory=list)
    evidence: list[EvidencePointer] = Field(default_factory=list)
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)  # e.g. "ambiguous", "not_found"
