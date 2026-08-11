"""Contract model tests — shape, defaults, and validation."""

import pytest
from pydantic import ValidationError

from talent_angels.contracts import AgentResult, EdgeRef, EvidencePointer, NodeRef


def test_agent_result_defaults_are_empty_not_none() -> None:
    result = AgentResult(capability="locate", suite="esco")

    assert result.nodes == []
    assert result.edges == []
    assert result.evidence == []
    assert result.warnings == []
    assert result.confidence is None


def test_agent_result_round_trip_with_populated_fields() -> None:
    node = NodeRef(
        id="esco:occupation:abc",
        suite="esco",
        source="esco",
        source_id="http://data.europa.eu/esco/occupation/abc",
        kind="Occupation",
        pref_label="software developer",
        alt_labels=["developer", "programmer"],
    )
    edge = EdgeRef(
        type="HAS_SKILL",
        suite="esco",
        source_node_id="esco:occupation:abc",
        target_node_id="esco:skill:def",
        properties={"relation_type": "essential"},
    )
    evidence = EvidencePointer(suite="esco", pointer="esco:search:exact_pref:software developer")

    result = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[node],
        edges=[edge],
        evidence=[evidence],
        confidence=0.95,
        warnings=[],
    )

    assert result.nodes[0].id == "esco:occupation:abc"
    assert result.edges[0].properties["relation_type"] == "essential"
    assert result.evidence[0].pointer.startswith("esco:search:")
    assert result.confidence == 0.95


def test_node_ref_requires_core_identity_fields() -> None:
    with pytest.raises(ValidationError):
        NodeRef(suite="esco")  # missing id, source, source_id, kind, pref_label
