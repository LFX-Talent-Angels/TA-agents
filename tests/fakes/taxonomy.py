"""Taxonomy result fakes for testing agent behaviour without TA-taxonomies."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeNode:
    id: str
    kind: str
    label: str
    source: str
    source_id: str
    properties: dict[str, object] = field(default_factory=dict)


@dataclass
class FakeCandidate:
    node: FakeNode
    confidence: float
    method: str


@dataclass
class FakeEdge:
    type: str
    from_id: str
    to_id: str
    properties: dict[str, object] = field(default_factory=dict)


@dataclass
class FakeToolResult:
    candidates: list[FakeCandidate] = field(default_factory=list)
    nodes: list[FakeNode] = field(default_factory=list)
    edges: list[FakeEdge] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
