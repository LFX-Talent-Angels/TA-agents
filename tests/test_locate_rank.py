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
    # "web developer" has "developer" as a pref_label token (tier 1);
    # "3D modeller" only has it in an alt_label (tier 4). Clear tier gap →
    # auto-select the pref_label winner rather than marking ambiguous.
    assert [n.pref_label for n in ranked.nodes] == ["web developer"]
    assert "ambiguous" not in ranked.warnings
    assert "also_matched:1" in ranked.warnings
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


class _NoGroups:
    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        return FakeToolResult()


def test_alias_substring_is_never_auto_selected() -> None:
    """Live repro: "nurse" resolved to CEO via the alias "senior nurse manager"."""
    ceo = _occ("chief executive officer", 1, alts=["senior nurse manager"])
    other = _occ("hospital porter", 2)
    result = AgentResult(capability="locate", suite="esco", nodes=[ceo, other], confidence=0.7)

    ranked = group_and_sort_locate(
        _NoGroups(), result, "nurse", suite_name="esco", group_rel_type=None
    )

    assert "ambiguous" in ranked.warnings
    assert ranked.confidence == 0.7
    assert len(ranked.nodes) == 2


class _Groups:
    """CLASSIFIED_UNDER parents from a node-id -> group-label map."""

    def __init__(self, groups: dict[str, str]) -> None:
        self.groups = groups

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        label = self.groups.get(node_id)
        if label is None:
            return FakeToolResult()
        group = FakeNode(
            id=f"esco:isco:{label}", kind="ISCOGroup", label=label, source="esco", source_id=label
        )
        return FakeToolResult(
            nodes=[group],
            edges=[FakeEdge(type="CLASSIFIED_UNDER", from_id=node_id, to_id=group.id)],
        )


def test_equally_good_hit_in_another_group_blocks_auto_select() -> None:
    """The runner-up is the best of the rest, not the next node in group order."""
    test_eng = _occ("test engineer", 1)
    chemist = _occ("chemist", 2, alts=["chemical engineer"])
    data_eng = _occ("data engineer", 3)
    result = AgentResult(
        capability="locate", suite="esco", nodes=[test_eng, chemist, data_eng], confidence=0.7
    )
    suite = _Groups({test_eng.id: "A engineers", chemist.id: "A engineers", data_eng.id: "B data"})

    ranked = group_and_sort_locate(suite, result, "engineer", suite_name="esco")

    assert "ambiguous" in ranked.warnings
    assert len(ranked.nodes) == 3


def test_truncated_pool_does_not_auto_select_a_token_hit() -> None:
    """Live repro: "engineer" auto-selected "test engineer" from a cut-off pool."""
    winner = _occ("test engineer", 1)
    noise = _occ("roboticist", 2, alts=["robotics engineer"])
    nodes = [winner, noise]

    full = AgentResult(capability="locate", suite="esco", nodes=nodes, confidence=0.7)
    cut = full.model_copy(update={"warnings": ["truncated"]})

    assert group_and_sort_locate(_NoGroups(), full, "engineer", suite_name="esco").nodes == [winner]
    ranked = group_and_sort_locate(_NoGroups(), cut, "engineer", suite_name="esco")
    assert "ambiguous" in ranked.warnings
    assert len(ranked.nodes) == 2


def test_truncated_pool_still_auto_selects_an_exact_title() -> None:
    exact = _occ("software developer", 1)
    noise = _occ("web developer", 2)
    result = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[noise, exact],
        confidence=0.95,
        warnings=["truncated"],
    )

    ranked = group_and_sort_locate(_NoGroups(), result, "software developer", suite_name="esco")

    assert ranked.nodes == [exact]
    assert ranked.confidence == 0.95


def test_query_inside_a_word_is_not_a_title_match() -> None:
    """Live repro: "IT manager" auto-selected "credit manager"."""
    credit = _occ("credit manager", 1)
    assert lexical_rank("IT manager", credit)[0] == 5
    assert lexical_rank("manager", credit)[0] == 1
    assert lexical_rank("develop", _occ("software developer", 2))[0] == 2


def test_alias_the_suite_doubts_is_never_auto_selected() -> None:
    """Live repro: "AI engineer" auto-selected an insemination technician."""
    vet = _occ("animal artificial insemination technician", 1, alts=["AI engineer"])
    ai = _occ("artificial intelligence engineer", 2)
    result = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[vet, ai],
        confidence=0.9,
        warnings=["alias_unconfirmed", "ambiguous"],
    )

    ranked = group_and_sort_locate(_NoGroups(), result, "AI engineer", suite_name="esco")

    assert len(ranked.nodes) == 2
    assert "ambiguous" in ranked.warnings


def test_meaning_hits_keep_the_suite_order() -> None:
    """With no lexical match, the suite's relevance order wins over label length."""
    best = _occ("artificial intelligence engineer", 1)
    short = _occ("patent engineer", 2)
    result = AgentResult(
        capability="locate", suite="esco", nodes=[best, short], warnings=["ambiguous"]
    )

    ranked = group_and_sort_locate(_NoGroups(), result, "AI engineer", suite_name="esco")

    assert [node.pref_label for node in ranked.nodes] == [best.pref_label, short.pref_label]


def test_a_short_word_ranks_only_as_a_whole_word() -> None:
    from talent_angels.contracts import NodeRef
    from talent_angels.skills.locate.rank import lexical_rank

    def node(label: str) -> NodeRef:
        return NodeRef(
            id=label,
            suite="esco",
            source="esco",
            source_id=label,
            kind="Occupation",
            pref_label=label,
        )

    assert lexical_rank("swe", node("chimney sweep"))[0] == 5
    assert lexical_rank("SWE", node("SWE lead"))[0] == 1
    assert lexical_rank("data scien", node("data scientist"))[0] == 2


def test_a_single_alias_substring_hit_is_offered_not_selected() -> None:
    from talent_angels.contracts import AgentResult, NodeRef
    from talent_angels.skills.locate.rank import group_and_sort_locate

    localiser = NodeRef(
        id="esco:loc",
        suite="esco",
        source="esco",
        source_id="loc",
        kind="Occupation",
        pref_label="localiser",
        alt_labels=["localisation QA tester"],
    )
    hr = NodeRef(
        id="esco:hr",
        suite="esco",
        source="esco",
        source_id="hr",
        kind="Occupation",
        pref_label="human resources manager",
        alt_labels=["hr manager"],
    )

    class NoNeighbours:
        def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> object:
            raise AssertionError("not needed")

    one = AgentResult(capability="locate", suite="esco", nodes=[localiser])
    assert (
        "ambiguous"
        in group_and_sort_locate(NoNeighbours(), one, "qa tester", suite_name="esco").warnings
    )  # type: ignore[arg-type]
    exact_alias = AgentResult(capability="locate", suite="esco", nodes=[hr])
    assert (
        "ambiguous"
        not in group_and_sort_locate(
            NoNeighbours(), exact_alias, "HR manager", suite_name="esco"
        ).warnings
    )  # type: ignore[arg-type]
