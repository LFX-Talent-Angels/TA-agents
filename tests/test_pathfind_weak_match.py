"""A weak single-candidate match is a pick, never a silent guess (PF-3)."""

from __future__ import annotations

from talent_angels.skills.pathfind.dispatch import pathfind_one
from talent_angels.suites.schema import SuiteSchema
from tests.fakes.taxonomy import FakeCandidate, FakeNode, FakeToolResult

_SCHEMA = SuiteSchema(
    skill_rel_types=("HAS_SKILL",),
    optional_rel_values=frozenset({"optional"}),
    group_rel_type=None,
    group_node_kinds=frozenset(),
)

TREE_SURGEON = FakeNode(
    id="esco:occupation:tree-surgeon",
    kind="Occupation",
    label="tree surgeon",
    source="esco",
    source_id="t1",
    properties={"alt_labels": ["surgeon"]},
)
GENERAL_PRACTITIONER = FakeNode(
    id="esco:occupation:gp",
    kind="Occupation",
    label="general practitioner",
    source="esco",
    source_id="g1",
)
DATA_ANALYST = FakeNode(
    id="esco:occupation:data-analyst",
    kind="Occupation",
    label="data analyst",
    source="esco",
    source_id="d1",
)


class _Suite:
    suite_schema = _SCHEMA

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        if text == "surgeon":
            # "surgeon" is an exact alias of tree surgeon, and otherwise only
            # matches general practitioner by meaning — group_and_sort_locate
            # confidently auto-selects tree surgeon, same as live ESCO today.
            return FakeToolResult(
                candidates=[
                    FakeCandidate(TREE_SURGEON, 0.85, "exact_alias"),
                    FakeCandidate(GENERAL_PRACTITIONER, 0.4, "meaning"),
                ],
                nodes=[TREE_SURGEON, GENERAL_PRACTITIONER],
            )
        if text == "data analyst":
            return FakeToolResult(
                candidates=[FakeCandidate(DATA_ANALYST, 0.95, "exact_pref")],
                nodes=[DATA_ANALYST],
            )
        return FakeToolResult(warnings=["not_found"])

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        return FakeToolResult(warnings=["no_neighbors"])


def test_a_weak_alias_match_is_reported_as_ambiguous_not_guessed() -> None:
    result, _calls = pathfind_one(_Suite(), "esco", "surgeon", "data analyst")

    assert "pathfind_side:start" in result.warnings
    assert "ambiguous" in result.warnings
    # Both candidates are shown — the pick list, not just the guessed winner.
    assert {node.pref_label for node in result.nodes} == {"tree surgeon", "general practitioner"}


def test_a_clear_match_is_not_treated_as_weak() -> None:
    result, _calls = pathfind_one(_Suite(), "esco", "data analyst", "data analyst")

    assert result.warnings == ["same_start_and_target"]
