"""Areas of a broad search, narrowing to one, and checking planner titles."""

from __future__ import annotations

from talent_angels.skills.locate.areas import (
    MIN_AREA_TITLES,
    AreaRequest,
    areas_from,
    search_area,
)
from talent_angels.skills.locate.explore import explore
from tests.fakes.taxonomy import FakeCandidate, FakeEdge, FakeNode, FakeToolResult


def _occ(label: str, *aliases: str) -> FakeNode:
    return FakeNode(
        id=f"esco:occ:{label}",
        kind="Occupation",
        label=label,
        source="esco",
        source_id=label,
        properties={"alt_labels": list(aliases)},
    )


def _hits(*nodes: FakeNode, method: str = "contains", **extra: object) -> FakeToolResult:
    warnings = ["ambiguous"] if len(nodes) > 1 else []
    return FakeToolResult(
        candidates=[FakeCandidate(node=n, confidence=0.7, method=method) for n in nodes],
        nodes=list(nodes),
        evidence=[f"esco:search:{method}:x"],
        warnings=[*warnings, *extra.pop("warnings", [])],  # type: ignore[list-item]
        **extra,  # type: ignore[arg-type]
    )


def _groups(*pairs: tuple[str, str, int]) -> dict[str, object]:
    return {"groups": [{"code": c, "label": lbl, "count": n} for c, lbl, n in pairs]}


CIVIL = _occ("civil engineer")
CONSTRUCTION = _occ("construction engineer")
TEST = _occ("test engineer")
GROUP = FakeNode(
    id="esco:isco:2142", kind="ISCOGroup", label="Civil engineers", source="esco", source_id="2142"
)


class _Suite:
    """Answers by exact text; every occupation sits in ISCO 2142."""

    def __init__(
        self, answers: dict[str, FakeToolResult], group_hits: FakeToolResult | None = None
    ):
        self.answers = answers
        self.group_hits = group_hits
        self.searched: list[str] = []
        self.group_calls: list[tuple[str, str]] = []

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        self.searched.append(text)
        return self.answers.get(text, FakeToolResult(warnings=["not_found"]))

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        return FakeToolResult(
            nodes=[GROUP],
            edges=[FakeEdge(type="CLASSIFIED_UNDER", from_id=node_id, to_id=GROUP.id)],
        )

    def search_group(self, text: str, group: str) -> FakeToolResult:
        self.group_calls.append((text, group))
        assert self.group_hits is not None
        return self.group_hits


# -- areas --------------------------------------------------------------------


def test_areas_keep_groups_with_enough_titles() -> None:
    raw = _hits(
        meta=_groups(
            ("2144", "Mechanical engineers", 24),
            ("2142", "Civil engineers", 11),
            ("5321", "Health care assistants", MIN_AREA_TITLES - 1),
        )
    )
    assert [(a.code, a.count) for a in areas_from(raw, "esco")] == [("2144", 24), ("2142", 11)]


def test_one_weighty_area_is_not_a_choice() -> None:
    raw = _hits(meta=_groups(("2221", "Nursing", 9), ("6112", "Farming", 1)))
    assert areas_from(raw, "esco") == []


def test_no_meta_no_areas() -> None:
    assert areas_from(FakeToolResult(), "esco") == []


def test_search_area_runs_inside_the_group_and_never_binds_one_title() -> None:
    suite = _Suite({}, group_hits=_hits(CIVIL))
    result = search_area(suite, "esco", AreaRequest(suite="esco", code="2142", query="engineer"))
    assert suite.group_calls == [("engineer", "2142")]
    assert [n.pref_label for n in result.nodes] == ["civil engineer"]
    assert "ambiguous" in result.warnings


def test_search_area_without_group_search_says_so() -> None:
    class _Plain:
        def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
            raise AssertionError("must not search the whole map")

    result = search_area(_Plain(), "onet", AreaRequest(suite="onet", code="17", query="x"))
    assert result.warnings == ["area_not_supported"]


# -- explore -------------------------------------------------------------------


def test_a_clear_subject_needs_no_exploring() -> None:
    suite = _Suite({"civil engineer": _hits(CIVIL, method="exact_pref")})
    assert explore(suite, "esco", "civil engineer", ["bridge engineer"], kind="occupation") is None


def test_confirmed_titles_come_first_and_guesses_the_map_lacks_are_dropped() -> None:
    suite = _Suite(
        {
            "engineer": _hits(
                TEST,
                CIVIL,
                warnings=["truncated"],
                meta=_groups(("2149", "Other", 36), ("2142", "Civil", 11)),
            ),
            "construction engineer": _hits(CONSTRUCTION, method="exact_pref"),
            "civil engineer": _hits(CIVIL, method="exact_pref"),
            "structural engineer": _hits(_occ("structural engineering technician")),
        }
    )
    explored = explore(
        suite,
        "esco",
        "engineer",
        ["construction engineer", "structural engineer", "civil engineer"],
        kind="occupation",
    )
    assert explored is not None
    result, areas = explored
    assert [n.pref_label for n in result.nodes] == [
        "construction engineer",
        "civil engineer",
        "test engineer",
    ]
    assert {"ambiguous", "guided", "truncated"} <= set(result.warnings)
    assert {e.source_node_id for e in result.edges} >= {CONSTRUCTION.id}
    assert [a.code for a in areas] == ["2149", "2142"]


def test_one_confirmed_title_is_still_offered_not_picked() -> None:
    suite = _Suite({"civil engineers": _hits(CIVIL, method="exact_alt")})
    explored = explore(suite, "esco", "someone who builds", ["civil engineers"], kind="occupation")
    assert explored is not None
    assert [n.pref_label for n in explored[0].nodes] == ["civil engineer"]
    assert "ambiguous" in explored[0].warnings


def test_alias_confirms_a_candidate() -> None:
    dev = _occ("software developer", "programmer")
    suite = _Suite({"programmer": _hits(dev, method="exact_alt")})
    explored = explore(suite, "esco", "computers", ["programmer"], kind="occupation")
    assert explored is not None and explored[0].nodes[0].pref_label == "software developer"


def test_a_title_the_planner_wrote_is_offered_not_bound() -> None:
    suite = _Suite({"civil engineer": _hits(CIVIL, method="exact_pref")})
    explored = explore(
        suite,
        "esco",
        "civil engineer",
        ["construction engineer"],
        kind="occupation",
        subject_is_users=False,
    )
    assert explored is not None
    assert [n.pref_label for n in explored[0].nodes] == ["civil engineer"]
    assert "ambiguous" in explored[0].warnings


def test_subject_matches_that_do_not_name_it_are_dropped_once_titles_are_confirmed() -> None:
    dev = _occ("software developer", "software engineer")
    sweep = _occ("chimney sweep")
    swe_alias = _occ("systems engineer", "SWE lead")
    suite = _Suite(
        {
            "SWE": _hits(sweep, _occ("street sweeper"), swe_alias),
            "software engineer": _hits(dev, method="exact_alt"),
        }
    )
    explored = explore(suite, "esco", "SWE", ["software engineer"], kind="occupation")
    assert explored is not None
    assert [n.pref_label for n in explored[0].nodes] == ["software developer", "systems engineer"]


def test_a_short_word_with_no_confirmed_title_is_an_honest_miss() -> None:
    sweep = _occ("chimney sweep")
    suite = _Suite({"SWE": _hits(sweep, _occ("street sweeper"))})
    explored = explore(suite, "esco", "SWE", ["no such title"], kind="occupation")
    assert explored is not None
    assert explored[0].nodes == []
    assert explored[0].warnings == ["not_found"]


def test_a_longer_subject_with_no_confirmed_title_keeps_its_list() -> None:
    suite = _Suite({"engineer": _hits(_occ("civil engineer"), _occ("test engineer"))})
    explored = explore(suite, "esco", "engineer", ["no such title"], kind="occupation")
    assert explored is not None
    assert {n.pref_label for n in explored[0].nodes} == {"civil engineer", "test engineer"}


def test_a_planner_subject_with_no_confirmed_title_is_still_not_bound() -> None:
    near = _occ("basket maker")
    suite = _Suite({"basket weaver": _hits(near)})
    explored = explore(
        suite,
        "esco",
        "basket weaver",
        ["quantum weaver"],
        kind="occupation",
        subject_is_users=False,
    )
    assert explored is not None
    assert "ambiguous" in explored[0].warnings
