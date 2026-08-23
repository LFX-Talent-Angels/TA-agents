"""Connect skill behavior through its public typed seams."""

from __future__ import annotations

import pytest

from talent_angels.assistant.connect_request import (
    UnsupportedConnectQuery,
    extract_connect_request,
)
from talent_angels.contracts import EvidencePointer, NodeRef
from talent_angels.skills.connect import connect
from tests.fakes.taxonomy import FakeEdge, FakeNode, FakeToolResult


class RecordingConnectSuite:
    def __init__(self, result: FakeToolResult) -> None:
        self.result = result
        self.calls: list[tuple[str, list[str] | None]] = []

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        self.calls.append((node_id, rel_types))
        return self.result


def _node(node_id: str, kind: str, label: str) -> FakeNode:
    return FakeNode(
        id=node_id,
        kind=kind,
        label=label,
        source="esco",
        source_id=f"https://example.test/{node_id}",
        properties={},
    )


def _center_ref() -> NodeRef:
    return NodeRef(
        id="esco:occupation:software-developer",
        suite="esco",
        source="esco",
        source_id="https://example.test/software-developer",
        kind="Occupation",
        pref_label="software developer",
    )


@pytest.mark.parametrize(
    ("question", "subject", "rel_types", "relation_kind"),
    [
        (
            "What essential skills does a software developer need?",
            "software developer",
            ("HAS_SKILL",),
            "essential",
        ),
        ("What skills does software developer need?", "software developer", ("HAS_SKILL",), None),
        ("Show skills for an accountant", "accountant", ("HAS_SKILL",), None),
        ("Show neighbors of data scientist", "data scientist", (), None),
    ],
)
def test_extract_connect_request(
    question: str,
    subject: str,
    rel_types: tuple[str, ...],
    relation_kind: str | None,
) -> None:
    request = extract_connect_request(question)

    assert request.subject == subject
    assert request.rel_types == rel_types
    assert request.relation_kind == relation_kind


def test_extract_connect_request_rejects_unsupported_shape() -> None:
    with pytest.raises(UnsupportedConnectQuery, match="could not extract"):
        extract_connect_request("Tell me something connected")


def test_connect_filters_edge_property_and_preserves_graph_facts() -> None:
    center = _node("esco:occupation:software-developer", "Occupation", "software developer")
    essential = _node("esco:skill:programming", "Skill", "computer programming")
    optional = _node("esco:skill:presenting", "Skill", "present information")
    suite = RecordingConnectSuite(
        FakeToolResult(
            nodes=[center, essential, optional],
            edges=[
                FakeEdge(
                    type="HAS_SKILL",
                    from_id=center.id,
                    to_id=essential.id,
                    properties={"relation_type": "essential"},
                ),
                FakeEdge(
                    type="HAS_SKILL",
                    from_id=center.id,
                    to_id=optional.id,
                    properties={"relation_type": "optional"},
                ),
            ],
            evidence=["esco:neighbors:software-developer"],
        )
    )
    request = extract_connect_request("What essential skills does a software developer need?")

    result = connect(
        suite,
        "esco",
        _center_ref(),
        request=request,
        confidence=0.95,
        locate_evidence=[EvidencePointer(suite="esco", pointer="esco:search:exact_pref")],
    )

    assert suite.calls == [(center.id, ["HAS_SKILL"])]
    assert [node.id for node in result.nodes] == [center.id, essential.id]
    assert len(result.edges) == 1
    assert result.edges[0].source_node_id == center.id
    assert result.edges[0].target_node_id == essential.id
    assert result.edges[0].properties == {"relation_type": "essential"}
    assert result.confidence == 0.95
    assert [item.pointer for item in result.evidence] == [
        "esco:search:exact_pref",
        "esco:neighbors:software-developer",
    ]


def test_connect_reports_no_matching_neighbors_without_dropping_center() -> None:
    center = _node("esco:occupation:software-developer", "Occupation", "software developer")
    optional = _node("esco:skill:presenting", "Skill", "present information")
    suite = RecordingConnectSuite(
        FakeToolResult(
            nodes=[center, optional],
            edges=[
                FakeEdge(
                    type="HAS_SKILL",
                    from_id=center.id,
                    to_id=optional.id,
                    properties={"relation_type": "optional"},
                )
            ],
        )
    )

    result = connect(
        suite,
        "esco",
        _center_ref(),
        request=extract_connect_request("What essential skills does a software developer need?"),
        confidence=0.95,
        locate_evidence=[],
    )

    assert [node.id for node in result.nodes] == [center.id]
    assert result.edges == []
    assert "no_matching_neighbors" in result.warnings
