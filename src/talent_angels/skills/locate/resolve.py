"""Locate skill: free text -> node candidates + confidence (ARCHITECTURE.md).

A skill is a passive procedure, not an agent — it never decides the plan or
talks to the user (rule #3, ARCHITECTURE.md). This module only converts a
suite tool's `search_nodes` result (deterministic; confidence policy owned by
the suite, see `ta_taxonomies.suites.esco.config`) into TA-agents'
`AgentResult`. It never invents or adjusts confidence itself.
"""

from __future__ import annotations

from typing import Protocol

from ta_taxonomies.contract.models import Node, ToolResult

from talent_angels.contracts import AgentResult, EvidencePointer, NodeRef


class SearchableSuite(Protocol):
    """The slice of the suite contract Locate needs — any suite qualifies."""

    def search_nodes(self, text: str, kind: str | None = None) -> ToolResult: ...


def _node_ref(node: Node, suite: str) -> NodeRef:
    return NodeRef(
        id=node.id,
        suite=suite,
        source=node.source,
        source_id=node.source_id,
        kind=node.kind,
        pref_label=node.label,
        alt_labels=list(node.properties.get("alt_labels") or []),
    )


def locate(
    suite: SearchableSuite, suite_name: str, text: str, *, kind: str | None = None
) -> AgentResult:
    """Resolve free text to node candidates via `suite.search_nodes`.

    `suite_name` is passed separately (rather than read off `suite`) so this
    stays agnostic to how each suite implementation names itself.
    """
    result = suite.search_nodes(text, kind=kind)

    nodes = [_node_ref(candidate.node, suite_name) for candidate in result.candidates]
    confidence = result.candidates[0].confidence if result.candidates else None
    evidence = [EvidencePointer(suite=suite_name, pointer=pointer) for pointer in result.evidence]

    return AgentResult(
        capability="locate",
        suite=suite_name,
        nodes=nodes,
        edges=[],
        evidence=evidence,
        confidence=confidence,
        warnings=list(result.warnings),
    )
