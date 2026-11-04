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
