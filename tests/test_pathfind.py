"""Pathfind skill over enumerate_paths — per suite, no cross-suite walk."""

from __future__ import annotations

from talent_angels.assistant.intent import extract_pathfind_endpoints
from talent_angels.skills.pathfind import pathfind
from tests.fakes.suite import DEV, PYTHON, FakeSuite


def test_extract_pathfind_endpoints() -> None:
    assert extract_pathfind_endpoints("path from data analyst to data scientist") == (
        "data analyst",
        "data scientist",
    )


def test_pathfind_returns_a_route_on_fake_suite() -> None:
    result = pathfind(FakeSuite(), "fake", _ref(DEV), _ref(PYTHON))
    assert result.capability == "pathfind"
    assert result.suite == "fake"
    assert result.edges
    assert "capability_not_implemented:pathfind" not in result.warnings


def test_pathfind_without_enumerate_is_unimplemented() -> None:
    class LocateOnly:
        def search_nodes(self, text: str, kind: str | None = None) -> object:
            raise AssertionError("pathfind must not search")

        def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> object:
            raise AssertionError("pathfind must not hop")

    result = pathfind(LocateOnly(), "test", _ref(DEV), _ref(PYTHON))
    assert result.warnings == ["capability_not_implemented:pathfind"]


def _ref(node) -> object:
    from talent_angels.contracts import NodeRef

    return NodeRef(
        id=node.id,
        suite="fake",
        source=node.source,
        source_id=node.source_id,
        kind=node.kind,
        pref_label=node.label,
    )
