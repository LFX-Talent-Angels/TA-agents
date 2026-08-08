"""Locate skill tests.

Unit tests use a fake suite (no Neo4j) to pin down the ToolResult -> AgentResult
mapping. The integration test exercises the real ESCO fixture over a live
Neo4j and is skipped automatically when one isn't reachable (see
CONTRIBUTING.md: fixture-backed tests should skip, not hard-fail, in CI).
"""

from __future__ import annotations

import pytest
from ta_taxonomies.contract.models import Candidate, Node, ToolResult

from talent_angels.skills.locate import ESCO_SUITE_NAME, locate, open_esco_suite


class FakeSuite:
    """Duck-types `search_nodes` so the mapping logic can be tested offline."""

    def __init__(self, result: ToolResult) -> None:
        self._result = result

    def search_nodes(self, text: str, kind: str | None = None) -> ToolResult:
        return self._result


def _occupation_node(node_id: str = "esco:occupation:fixture-1") -> Node:
    return Node(
        id=node_id,
        kind="Occupation",
        label="software developer",
        source="esco",
        source_id="http://data.europa.eu/esco/occupation/fixture-1",
        properties={"alt_labels": ["developer", "programmer"]},
    )


def test_locate_maps_exact_match_candidate_to_agent_result() -> None:
    node = _occupation_node()
    result = ToolResult(
        candidates=[Candidate(node=node, confidence=0.95, method="exact_pref")],
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
    assert outcome.evidence[0].pointer == "esco:search:exact_pref:software developer"


def test_locate_not_found_has_no_nodes_and_carries_warning() -> None:
    result = ToolResult(warnings=["not_found"], evidence=["esco:search:not_found:xyzzy"])
    suite = FakeSuite(result)

    outcome = locate(suite, ESCO_SUITE_NAME, "xyzzy")

    assert outcome.nodes == []
    assert outcome.confidence is None
    assert outcome.warnings == ["not_found"]


def test_locate_ambiguous_reports_top_candidate_confidence_and_warning() -> None:
    top = _occupation_node("esco:occupation:top")
    other = _occupation_node("esco:occupation:other")
    result = ToolResult(
        candidates=[
            Candidate(node=top, confidence=0.70, method="contains"),
            Candidate(node=other, confidence=0.55, method="contains"),
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


def _neo4j_reachable() -> bool:
    try:
        with open_esco_suite() as suite:
            suite._driver.verify_connectivity()  # noqa: SLF001 - availability probe only
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _neo4j_reachable(), reason="Neo4j not reachable; see TA-taxonomies NOTES.md"
)
def test_locate_against_live_esco_fixture_finds_software_developer() -> None:
    with open_esco_suite() as suite:
        outcome = locate(suite, ESCO_SUITE_NAME, "software developer", kind="occupation")

    assert outcome.confidence == 0.95
    assert outcome.nodes
    assert outcome.nodes[0].kind == "Occupation"
    assert outcome.nodes[0].pref_label.lower() == "software developer"
