from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.skills.locate.rank import group_and_sort_locate, lexical_rank
from tests.fakes.taxonomy import FakeEdge, FakeNode, FakeToolResult


def _occ(label: str, n: int, *, alts: list[str] | None = None) -> NodeRef:
    return NodeRef(
        id=f"esco:occupation:{n}",
        suite="esco",
        source="esco",
        source_id=f"s{n}",
        kind="Occupation",
        pref_label=label,
        alt_labels=list(alts or []),
    )


def test_lexical_rank_prefers_pref_token_over_short_alt_substring() -> None:
    query = "developer"
    web = _occ("web developer", 2)
    modeller = _occ("3D modeller", 1, alts=["3D developer"])
    assert lexical_rank(query, web) < lexical_rank(query, modeller)


def test_group_and_sort_marks_multi_hit_ambiguous_and_groups() -> None:
    web = _occ("web developer", 2)
    short = _occ("3D modeller", 1, alts=["3D developer"])
    result = AgentResult(capability="locate", suite="esco", nodes=[short, web], confidence=0.7)

    isco = FakeNode(
        id="esco:isco:1",
        kind="ISCOGroup",
        label="Software and applications developers",
        source="esco",
        source_id="isco-1",
    )

    class Suite:
        def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
            assert rel_types == ["CLASSIFIED_UNDER"]
            return FakeToolResult(
                nodes=[
                    FakeNode(
                        id=node_id,
                        kind="Occupation",
                        label="x",
                        source="esco",
                        source_id="n",
                    ),
                    isco,
                ],
                edges=[
                    FakeEdge(
                        type="CLASSIFIED_UNDER",
                        from_id=node_id,
                        to_id=isco.id,
                    )
                ],
            )

    ranked = group_and_sort_locate(Suite(), result, "developer", suite_name="esco")
    labels = [node.pref_label for node in ranked.nodes]
    assert labels.index("web developer") < labels.index("3D modeller")
    assert "ambiguous" in ranked.warnings
    assert any(edge.type == "CLASSIFIED_UNDER" for edge in ranked.edges)


def test_exact_alt_top_hit_is_unique_enough() -> None:
    winner = _occ("Software Developers", 1, alts=["Software Engineer"])
    noise = _occ("Blockchain Engineers", 2)
    result = AgentResult(
        capability="locate",
        suite="onet",
        nodes=[noise, winner],
        confidence=0.7,
    )

    class Quiet:
        def get_neighbors(self, *_a, **_k) -> FakeToolResult:
            return FakeToolResult()

    ranked = group_and_sort_locate(Quiet(), result, "Software Engineer", suite_name="onet")
    assert [node.id for node in ranked.nodes] == [winner.id]
    assert "ambiguous" not in ranked.warnings
    assert "also_matched:1" in ranked.warnings
    assert ranked.confidence == 0.90


def test_singular_query_matches_plural_pref_as_unique() -> None:
    winner = _occ("Software Developers", 1)
    noise = _occ("Blockchain Engineers", 2)
    result = AgentResult(
        capability="locate",
        suite="onet",
        nodes=[noise, winner],
        confidence=0.7,
    )

    class Quiet:
        def get_neighbors(self, *_a, **_k) -> FakeToolResult:
            return FakeToolResult()

    ranked = group_and_sort_locate(Quiet(), result, "software developer", suite_name="onet")
    assert [node.id for node in ranked.nodes] == [winner.id]
    assert "ambiguous" not in ranked.warnings


def test_two_exact_pref_hits_stay_ambiguous() -> None:
    a = _occ("developer", 1)
    b = _occ("developer", 2)
    # Same pref_label, different ids — lexical tier ties.
    result = AgentResult(capability="locate", suite="esco", nodes=[a, b], confidence=0.95)

    class Quiet:
        def get_neighbors(self, *_a, **_k) -> FakeToolResult:
            return FakeToolResult()

    ranked = group_and_sort_locate(Quiet(), result, "developer", suite_name="esco")
    assert len(ranked.nodes) == 2
    assert "ambiguous" in ranked.warnings


def test_unique_locate_is_not_regrouped() -> None:
    only = _occ("software developer", 1)
    result = AgentResult(capability="locate", suite="esco", nodes=[only], confidence=0.95)

    class Boom:
        def get_neighbors(self, *_a, **_k):
            raise AssertionError("unique locate must not hop CLASSIFIED_UNDER")

    assert group_and_sort_locate(Boom(), result, "software developer", suite_name="esco") is result
