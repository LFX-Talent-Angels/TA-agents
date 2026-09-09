"""Locate skill tests.

These unit tests use a fake suite (no Neo4j) to pin down the ToolResult ->
AgentResult mapping. Live ESCO coverage lives under ``tests/integration``.
"""

from __future__ import annotations

from talent_angels.skills.locate import ESCO_SUITE_NAME, locate
from tests.fakes.taxonomy import FakeCandidate, FakeNode, FakeToolResult


class FakeSuite:
    """Duck-types `search_nodes` so the mapping logic can be tested offline."""

    def __init__(self, result: FakeToolResult) -> None:
        self._result = result

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        return self._result


def _occupation_node(node_id: str = "esco:occupation:fixture-1") -> FakeNode:
    return FakeNode(
        id=node_id,
        kind="Occupation",
        label="software developer",
        source="esco",
        source_id="http://data.europa.eu/esco/occupation/fixture-1",
        properties={
            "alt_labels": ["developer", "programmer"],
            "description": "Designs and implements software applications.",
        },
    )


def test_locate_maps_exact_match_candidate_to_agent_result() -> None:
    node = _occupation_node()
    result = FakeToolResult(
        candidates=[FakeCandidate(node=node, confidence=0.95, method="exact_pref")],
        nodes=[node],
        evidence=["esco:search:exact_pref:software developer"],
    )
    suite = FakeSuite(result)

    outcome = locate(suite, ESCO_SUITE_NAME, "software developer")

    assert outcome.capability == "locate"
    assert outcome.suite == "esco"
    assert outcome.confidence == 0.95
    assert outcome.warnings == []
    assert len(outcome.nodes) == 1
    assert outcome.nodes[0].id == node.id
    assert outcome.nodes[0].pref_label == "software developer"
    assert outcome.nodes[0].alt_labels == ["developer", "programmer"]
    assert outcome.nodes[0].description == "Designs and implements software applications."
    assert outcome.evidence[0].pointer == "esco:search:exact_pref:software developer"


def test_locate_not_found_has_no_nodes_and_carries_warning() -> None:
    result = FakeToolResult(warnings=["not_found"], evidence=["esco:search:not_found:xyzzy"])
    suite = FakeSuite(result)

    outcome = locate(suite, ESCO_SUITE_NAME, "xyzzy")

    assert outcome.nodes == []
    assert outcome.confidence is None
    assert outcome.warnings == ["not_found"]


def test_locate_ambiguous_reports_top_candidate_confidence_and_warning() -> None:
    top = _occupation_node("esco:occupation:top")
    other = _occupation_node("esco:occupation:other")
    result = FakeToolResult(
        candidates=[
            FakeCandidate(node=top, confidence=0.70, method="contains"),
            FakeCandidate(node=other, confidence=0.55, method="contains"),
        ],
        nodes=[top, other],
        warnings=["ambiguous"],
        evidence=["esco:search:contains:developer"],
    )
    suite = FakeSuite(result)

    outcome = locate(suite, ESCO_SUITE_NAME, "developer")

    assert outcome.confidence == 0.70
    assert outcome.warnings == ["ambiguous"]
    assert len(outcome.nodes) == 2
