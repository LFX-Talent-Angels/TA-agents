"""The MCP edge asserts typed graph data and nothing else."""

from __future__ import annotations

import pytest

from talent_angels.mcp.mapping import (
    neighbors_payload,
    paths_payload,
    scored_paths_payload,
    search_payload,
    suite_path_arg,
)
from talent_angels.mcp.models import PolicyRef
from tests.fakes.suite import DEV, HAS_PYTHON, PYTHON, FakeSuite
from tests.fakes.taxonomy import FakePath, FakeScoredPath, FakeToolResult


def test_search_keeps_the_match_method_next_to_the_confidence() -> None:
    suite = FakeSuite()

    payload = search_payload(
        suite.search_nodes("programmer"), suite="fake", query="programmer", kind=None
    )

    candidate = payload.candidates[0]
    assert (candidate.confidence, candidate.method) == (0.9, "exact_alt")
    assert candidate.node.id == DEV.id
    assert candidate.node.suite == "fake"
    assert candidate.node.alt_labels == ["programmer"]


def test_a_miss_is_an_empty_result_with_a_warning_not_a_sentence() -> None:
    suite = FakeSuite()

    payload = search_payload(
        suite.search_nodes("stonemason"), suite="fake", query="stonemason", kind=None
    )

    assert payload.candidates == []
    assert payload.warnings == ["not_found"]
    assert [pointer.pointer for pointer in payload.evidence] == ["fake:search:not_found:stonemason"]


def test_neighbors_preserve_the_edge_qualifier_that_changes_the_meaning() -> None:
    suite = FakeSuite()

    payload = neighbors_payload(suite.get_neighbors(DEV.id), suite="fake", center_id=DEV.id)

    edge = payload.edges[0]
    assert (edge.type, edge.source_node_id, edge.target_node_id) == ("HAS_SKILL", DEV.id, PYTHON.id)
    assert edge.properties == {"relation_type": "essential"}
    assert [node.id for node in payload.nodes] == [DEV.id, PYTHON.id]


def test_paths_report_what_was_cut_as_counts() -> None:
    suite = FakeSuite()

    payload = paths_payload(
        suite.enumerate_paths(DEV.id, PYTHON.id), suite="fake", from_id=DEV.id, to_id=PYTHON.id
    )

    assert payload.paths[0].node_ids == [DEV.id, PYTHON.id]
    assert payload.pruning is not None
    assert (payload.pruning.considered, payload.pruning.returned, payload.pruning.pruned) == (
        3,
        1,
        2,
    )


def test_a_route_round_trips_back_into_the_suite_contract_shape() -> None:
    payload = paths_payload(
        FakeSuite().enumerate_paths(DEV.id, PYTHON.id),
        suite="fake",
        from_id=DEV.id,
        to_id=PYTHON.id,
    )

    assert suite_path_arg(payload.paths[0]) == {
        "node_ids": [DEV.id, PYTHON.id],
        "edges": [
            {
                "type": "HAS_SKILL",
                "from_id": DEV.id,
                "to_id": PYTHON.id,
                "properties": {"relation_type": "essential"},
            }
        ],
    }


def test_a_score_carries_the_policy_the_suite_answered_under() -> None:
    asked = PolicyRef(name="essential-first", version="0.1.0")
    answered = PolicyRef(name="essential-first", version="0.0.9")
    result = FakeToolResult(
        scored_paths=[
            FakeScoredPath(
                path=FakePath(node_ids=[DEV.id, PYTHON.id], edges=[HAS_PYTHON]),
                score=1.0,
                policy=answered,
            )
        ]
    )

    payload = scored_paths_payload(result, suite="fake", policy=asked)

    assert payload.policy == asked
    assert payload.scored_paths[0].policy == answered


@pytest.mark.parametrize("text", ["software developer", "programmer"])
def test_ids_are_reported_under_the_suite_that_issued_them(text: str) -> None:
    payload = search_payload(FakeSuite().search_nodes(text), suite="fake", query=text, kind=None)

    assert all(candidate.node.suite == "fake" for candidate in payload.candidates)
    assert all(candidate.node.id.startswith("fake:") for candidate in payload.candidates)
