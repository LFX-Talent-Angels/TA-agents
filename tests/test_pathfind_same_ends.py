"""Pathfind with the same start and target says so (BX-20)."""

from __future__ import annotations

from talent_angels.skills.pathfind.dispatch import pathfind_one
from talent_angels.suites.schema import SuiteSchema
from tests.fakes.taxonomy import FakeCandidate, FakeNode, FakeToolResult

DEV = FakeNode(
    "bls:occupation:15-1252", "Occupation", "Software Developers", source="bls", source_id="15-1252"
)


class _Suite:
    suite_schema = SuiteSchema(
        skill_rel_types=("HAS_SKILL",),
        optional_rel_values=frozenset({"optional"}),
        group_rel_type=None,
        group_node_kinds=frozenset(),
    )

    def __init__(self) -> None:
        self.paths_asked = False

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        return FakeToolResult(candidates=[FakeCandidate(DEV, 0.95, "exact_pref")])

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        result = FakeToolResult()
        result.nodes = [DEV]  # type: ignore[attr-defined]
        return result

    def enumerate_paths(self, *a: object, **k: object) -> FakeToolResult:
        self.paths_asked = True
        return FakeToolResult(warnings=["no_path"])


def test_same_start_and_target() -> None:
    suite = _Suite()
    result, _calls = pathfind_one(suite, "bls", "Software Developers", "software developers")
    assert result.warnings == ["same_start_and_target"]
    assert [n.id for n in result.nodes] == ["bls:occupation:15-1252"]
    assert not suite.paths_asked
