"""The code-written answer shown before a pick list."""

from __future__ import annotations

from talent_angels.assistant.llm_plan import PlanDraft
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.session.lead import found, lead, understood


def _result(suite: str, *labels: str, warnings: tuple[str, ...] = ("ambiguous",)) -> AgentResult:
    nodes = [
        NodeRef(
            id=f"{suite}:{x}",
            suite=suite,
            source=suite,
            source_id=x,
            kind="Occupation",
            pref_label=x,
        )
        for x in labels
    ]
    return AgentResult(capability="locate", suite=suite, nodes=nodes, warnings=list(warnings))


def _draft(subject: str, *candidates: str, intent: str | None = None) -> PlanDraft:
    return PlanDraft(target="locate", subject=subject, candidates=candidates, profile_intent=intent)


def test_an_abbreviation_says_how_it_was_read() -> None:
    draft = _draft("SWE", "software engineer", "software developer", "web developer", "x")
    assert understood("SWE", draft) == (
        'I read "SWE" as software engineer, software developer or web developer.'
    )


def test_a_description_is_read_as_the_request() -> None:
    draft = _draft("engineer", "civil engineer")
    assert understood("I want to become an engineer who builds buildings", draft) == (
        "I read your request as civil engineer."
    )


def test_a_plain_lookup_names_its_search() -> None:
    assert understood("what does a nurse do?", _draft("nurse")) == 'I looked up "nurse".'
    assert understood("nurse", None) == 'I looked up "nurse".'


def test_found_names_what_every_map_did() -> None:
    results = [
        _result("esco", "software developer", warnings=()),
        _result("onet", "A", "B", warnings=("ambiguous", "truncated")),
        _result("sfia", warnings=("not_found",)),
    ]
    assert found(results) == (
        "ESCO matched software developer, O*NET has 2+ matching titles and SFIA has no match."
    )


def test_next_step_follows_the_profile_intent() -> None:
    text = lead(
        "I want to be an engineer", _draft("engineer", intent="goal"), [_result("esco", "a", "b")]
    )
    assert text.endswith("Pick the one you mean and I'll save it as your goal.")
    plain = lead("engineer", _draft("engineer"), [_result("esco", "a", "b")])
    assert plain.endswith("Pick the one you mean, or tell me more and I'll narrow it.")
