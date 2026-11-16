"""'How much do they earn?' about a bound occupation is answered from its published facts."""

from __future__ import annotations

import pytest

from talent_angels.assistant.connect_request import is_facts_followup
from talent_angels.contracts import NodeRef
from talent_angels.session.kernel import handle_line
from talent_angels.session.store import new_session

BAKERS = NodeRef(
    id="bls:occupation:51-3011",
    suite="bls",
    source="bls",
    source_id="51-3011",
    kind="Occupation",
    pref_label="Bakers",
    facts={
        "median_wage_annual": 37160.0,
        "wage_period": "May 2025",
        "entry_education": "No formal educational credential",
    },
)
BAKER_ESCO = NodeRef(
    id="esco:occupation:1",
    suite="esco",
    source="esco",
    source_id="u",
    kind="Occupation",
    pref_label="baker",
)


@pytest.mark.parametrize(
    "question",
    [
        "how much do they earn?",
        "what's the salary?",
        "what is the job outlook?",
        "what education do they need?",
        "how much do bakers earn?",
    ],
)
def test_facts_questions_about_the_bound_job(question: str) -> None:
    assert is_facts_followup(question, {"bls": BAKERS})


@pytest.mark.parametrize(
    "question", ["how much does a nurse earn?", "what skills do they need?", "bakers"]
)
def test_other_questions_are_not(question: str) -> None:
    assert not is_facts_followup(question, {"bls": BAKERS})


def test_nothing_bound_means_no_followup() -> None:
    assert not is_facts_followup("how much do they earn?", {})


def test_answered_in_code_from_the_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    state = new_session()
    state.bindings = {"esco": BAKER_ESCO, "bls": BAKERS}

    def runner(*_a: object, **_k: object) -> object:
        pytest.fail("a facts follow-up must not search again")

    reply = handle_line(state, "how much do they earn?", runner=runner)  # type: ignore[arg-type]
    assert "Median annual wage: $37,160 (BLS, May 2025)" in reply.text
    assert "ESCO publishes no pay, outlook or education figures" in reply.text


def test_a_bound_job_without_figures_says_so() -> None:
    # Legislators (no EP data) or a JobTech job: never generic advice.
    state = new_session()
    state.bindings = {"esco": BAKER_ESCO}

    def runner(*_a: object, **_k: object) -> object:
        pytest.fail("a facts follow-up must not search again")

    reply = handle_line(state, "is it a growing job?", runner=runner)  # type: ignore[arg-type]
    assert "ESCO publishes no pay, outlook or education figures" in reply.text
