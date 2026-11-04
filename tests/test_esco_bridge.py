"""An English question reaches a Swedish JobTech record through JobTech's own ESCO matches."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from talent_angels.assistant.crosswalk import bridge_via_esco, esco_records_named, wants_bridge
from talent_angels.contracts import AgentResult
from talent_angels.skills.locate.rank import code_facts
from talent_angels.suites import SuiteRegistry
from talent_angels.suites.registry import SuiteRuntime
from tests.fakes.tagging_suite import FakeCandidate, FakeNode, FakeToolResult

BAKER_URI = "http://data.europa.eu/esco/occupation/1aadb308"


class _Esco:
    def __init__(self, confidence: float = 0.95) -> None:
        self.confidence = confidence
        self.calls: list[tuple[str, str | None]] = []

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        self.calls.append((text, kind))
        if text != "baker":
            return FakeToolResult(warnings=["not_found"])
        node = FakeNode(
            "esco:occupation:1aadb308", "Occupation", "baker", source="esco", source_id=BAKER_URI
        )
        return FakeToolResult(
            candidates=[FakeCandidate(node, self.confidence, "exact_pref")],
            evidence=["esco:search:exact_pref:baker"],
        )


class _Jobtech:
    def __init__(self) -> None:
        self.asked: list[list[str]] = []

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        return FakeToolResult(warnings=["not_found"])

    def search_by_esco(self, uris: list[str], kind: str = "occupation") -> FakeToolResult:
        self.asked.append(uris)
        if BAKER_URI not in uris:
            return FakeToolResult(warnings=["not_found"])
        node = FakeNode(
            "jobtech:occupation:XugJ_6hw_wVV",
            "Occupation",
            "Bagare/Konditor",
            source="jobtech",
            source_id="XugJ_6hw_wVV",
        )
        return FakeToolResult(
            candidates=[FakeCandidate(node, 0.65, "esco_exact_match")],
            evidence=[f"jobtech:esco_match:{BAKER_URI}"],
        )


def _registry(esco: object, jobtech: object) -> SuiteRegistry:
    def factory(name: str, suite: object):  # type: ignore[no-untyped-def]
        @contextmanager
        def _open() -> Iterator[SuiteRuntime]:
            yield SuiteRuntime(name=name, suite=suite, health_check=lambda: True)  # type: ignore[arg-type]

        return _open

    return SuiteRegistry(
        {"esco": factory("esco", esco), "jobtech": factory("jobtech", jobtech)}, default="esco"
    )


def _not_found(suite: str = "jobtech") -> AgentResult:
    return AgentResult(capability="locate", suite=suite, warnings=["not_found"])


def test_english_reaches_the_swedish_record() -> None:
    jobtech = _Jobtech()
    bridged = bridge_via_esco(_registry(_Esco(), jobtech), _not_found(), "baker")
    assert bridged is not None
    assert [n.pref_label for n in bridged.nodes] == ["Bagare/Konditor"]
    assert bridged.suite == "jobtech"
    assert bridged.confidence == 0.65
    assert jobtech.asked == [[BAKER_URI]]
    assert esco_records_named(bridged) == ["baker"]
    assert {p.suite for p in bridged.evidence} == {"jobtech", "esco"}
    lines = code_facts(bridged, bridged.nodes[0])
    assert any("ESCO record(s): baker" in line for line in lines)


def test_only_a_suite_that_found_nothing_is_bridged() -> None:
    assert wants_bridge(_not_found())
    assert not wants_bridge(_not_found("esco"))
    assert not wants_bridge(AgentResult(capability="locate", suite="jobtech", warnings=[]))
    assert bridge_via_esco(_registry(_Esco(), _Jobtech()), _not_found("bls"), "baker") is None


def test_a_weak_esco_match_is_not_followed() -> None:
    assert bridge_via_esco(_registry(_Esco(0.6), _Jobtech()), _not_found(), "baker") is None


def test_an_ambiguous_esco_answer_is_not_followed() -> None:
    # ESCO offering a list must not become one confident JobTech record (which
    # would bind it and save a goal the user never chose).
    class _AmbiguousEsco(_Esco):
        def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
            result = super().search_nodes("baker", kind)
            second = FakeNode(
                "esco:occupation:2", "Occupation", "baker 2", source="esco", source_id=BAKER_URI
            )
            result.candidates.append(FakeCandidate(second, 0.95, "exact_pref"))
            result.warnings.append("ambiguous")
            return result

    assert bridge_via_esco(_registry(_AmbiguousEsco(), _Jobtech()), _not_found(), "baker") is None


def test_nothing_in_esco_means_no_bridge() -> None:
    assert bridge_via_esco(_registry(_Esco(), _Jobtech()), _not_found(), "plonkwurst") is None


def test_no_esco_attached_means_no_bridge() -> None:
    @contextmanager
    def _jobtech() -> Iterator[SuiteRuntime]:
        yield SuiteRuntime(name="jobtech", suite=_Jobtech(), health_check=lambda: True)  # type: ignore[arg-type]

    registry = SuiteRegistry({"jobtech": _jobtech}, default="jobtech")
    assert bridge_via_esco(registry, _not_found(), "baker") is None


def test_a_suite_without_the_lookup_is_never_bridged() -> None:
    class _Plain:
        def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
            return FakeToolResult(warnings=["not_found"])

    assert bridge_via_esco(_registry(_Esco(), _Plain()), _not_found(), "baker") is None


def test_phrasing_is_told_to_keep_the_users_language() -> None:
    # JobTech cards are Swedish; an English question must get an English reply.
    from talent_angels.session.i18n import set_language
    from talent_angels.session.phrase import _MAP_SYSTEM, _reply_language_line

    set_language("en")
    line = _reply_language_line()
    assert line.startswith("Reply language: the language of the User line above")
    assert "English" in line
    assert "never switch to the language of the card" in line
    assert "JobTech is" in _MAP_SYSTEM


# -- batch tagging: the suite-level bridge -----------------------------------------


def test_tagging_bridge_follows_esco_when_jobtech_finds_nothing() -> None:
    from talent_angels.assistant.crosswalk import EscoBridgedSuite, with_esco_bridge

    jobtech = _Jobtech()
    bridged = with_esco_bridge("jobtech", jobtech, _Esco())
    assert isinstance(bridged, EscoBridgedSuite)
    result = bridged.search_nodes("baker", kind="occupation")
    assert [(c.method, c.node.label) for c in result.candidates] == [
        ("esco_exact_match", "Bagare/Konditor")
    ]
    # Nothing in ESCO either: JobTech's own (empty) answer comes back.
    assert bridged.search_nodes("plonkwurst", kind="occupation").warnings == ["not_found"]


def test_tagging_bridge_leaves_other_suites_and_kinds_alone() -> None:
    from talent_angels.assistant.crosswalk import with_esco_bridge

    plain = _Jobtech()
    assert with_esco_bridge("bls", plain, _Esco()) is plain
    assert with_esco_bridge("jobtech", plain, None) is plain
    bridged = with_esco_bridge("jobtech", plain, _Esco())
    assert bridged.search_nodes("baker", kind="task").warnings == ["not_found"]
    assert plain.asked == []  # a task row never goes through ESCO


def test_tagging_bridge_survives_an_esco_outage() -> None:
    from talent_angels.assistant.crosswalk import with_esco_bridge

    class _Down:
        def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
            raise ConnectionError("esco down")

    bridged = with_esco_bridge("jobtech", _Jobtech(), _Down())
    assert bridged.search_nodes("baker", kind="occupation").warnings == ["not_found"]


# -- chat rescue: own words, strict bridge, skills -----------------------------


class _SwedishJobtech(_Jobtech):
    """Knows "Elevassistent" exactly and one weak English-word title."""

    suite_schema = __import__("talent_angels.suites.schema", fromlist=["SuiteSchema"]).SuiteSchema(
        skill_rel_types=("HAS_SKILL",),
        optional_rel_values=frozenset({"optional", "group"}),
        group_rel_type="CLASSIFIED_UNDER",
        group_node_kinds=frozenset({"SSYKGroup"}),
    )

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        if text.casefold() == "elevassistent":
            node = FakeNode(
                "jobtech:occupation:Sbzf",
                "Occupation",
                "Elevassistent",
                source="jobtech",
                source_id="Sbzf",
            )
            return FakeToolResult(candidates=[FakeCandidate(node, 0.95, "exact_pref")])
        if text == "chef":
            node = FakeNode(
                "jobtech:occupation:HR", "Occupation", "HR-chef", source="jobtech", source_id="HR"
            )
            return FakeToolResult(candidates=[FakeCandidate(node, 0.70, "contains")])
        return FakeToolResult(warnings=["not_found"])

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        skill = FakeNode("jobtech:skill:x", "Skill", "Matbröd", source="jobtech", source_id="x")
        center = FakeNode(node_id, "Occupation", "Bagare/Konditor", source="jobtech", source_id="c")
        result = FakeToolResult(candidates=[])
        result.nodes = [center, skill]  # type: ignore[attr-defined]
        from tests.fakes.taxonomy import FakeEdge

        result.edges = [  # type: ignore[attr-defined]
            FakeEdge(
                type="HAS_SKILL",
                from_id=node_id,
                to_id="jobtech:skill:x",
                properties={"relation_type": "group"},
            )
        ]
        return result


def _rescue(registry: SuiteRegistry, result: AgentResult, **kw: object) -> AgentResult | None:
    from talent_angels.assistant.crosswalk import rescue_jobtech

    args: dict[str, object] = {
        "subject": "baker",
        "question": "baker in jobtech",
        "kind": "occupation",
        "language": "en",
        "wants_skills": False,
    }
    args.update(kw)
    return rescue_jobtech(registry, result, **args)  # type: ignore[arg-type]


def test_rescue_tries_the_users_own_swedish_word_first() -> None:
    # The planner wrote "elevator operator"; the user wrote "Elevassistent".
    rescued = _rescue(
        _registry(_Esco(), _SwedishJobtech()),
        _not_found(),
        subject="elevator operator",
        question="what can an Elevassistent move to?",
    )
    assert rescued is not None
    assert [n.pref_label for n in rescued.nodes] == ["Elevassistent"]
    assert not any(w.startswith("via_esco:") for w in rescued.warnings)


def test_rescue_never_follows_a_partial_esco_match() -> None:
    # ESCO "gift" -> "teacher of talented and gifted students" is a contains hit (0.70).
    assert _rescue(_registry(_Esco(0.70), _Jobtech()), _not_found()) is None


def test_a_weak_swedish_hit_is_replaced_for_an_english_question_only() -> None:
    weak = AgentResult(
        capability="locate",
        suite="jobtech",
        confidence=0.70,
        warnings=["ambiguous"],
        nodes=[
            __import__("talent_angels.contracts", fromlist=["NodeRef"]).NodeRef(
                id="jobtech:occupation:HR",
                suite="jobtech",
                source="jobtech",
                source_id="HR",
                kind="Occupation",
                pref_label="HR-chef",
            )
        ],
    )
    english = _rescue(_registry(_Esco(), _Jobtech()), weak, language="en")
    assert english is not None and english.nodes[0].pref_label == "Bagare/Konditor"
    # A Swedish speaker typing "chef" means a manager: keep JobTech's own hits.
    assert _rescue(_registry(_Esco(), _Jobtech()), weak, language="sv") is None


def test_a_clear_answer_is_never_rescued() -> None:
    clear = AgentResult(
        capability="locate",
        suite="jobtech",
        confidence=0.95,
        nodes=[
            __import__("talent_angels.contracts", fromlist=["NodeRef"]).NodeRef(
                id="jobtech:occupation:D",
                suite="jobtech",
                source="jobtech",
                source_id="D",
                kind="Occupation",
                pref_label="Barnmorska",
            )
        ],
    )
    assert _rescue(_registry(_Esco(), _Jobtech()), clear) is None


def test_a_rescued_record_gets_its_skills_when_asked() -> None:
    class _BakerWithSkills(_SwedishJobtech):
        def search_by_esco(self, uris: list[str], kind: str = "occupation") -> FakeToolResult:
            return _Jobtech().search_by_esco(uris, kind)

    rescued = _rescue(
        _registry(_Esco(), _BakerWithSkills()),
        _not_found(),
        question="what skills does a baker need in jobtech?",
        wants_skills=True,
    )
    assert rescued is not None
    assert rescued.capability == "connect"
    assert "Matbröd" in [n.pref_label for n in rescued.nodes]
    assert "via_esco:baker" in rescued.warnings
