"""The fail-closed grounding filter (ADR-0014 rule 3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from talent_angels.session import grounding
from talent_angels.session.grounding import grounded

_DATA = json.loads((Path(__file__).parent / "guardrails" / "cases.json").read_text("utf-8"))
CARD = _DATA["card_C"]


def _allowed(spec: str) -> str:
    return CARD + spec[1:] if spec.startswith("C") else spec


@pytest.mark.parametrize("case", _DATA["grounding"], ids=lambda c: c["id"])
def test_grounding_cases(case: dict[str, str]) -> None:
    assert grounded(case["text"], _allowed(case["allowed"])) == case["keep"]


def test_an_exception_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object) -> bool:
        raise RuntimeError("boom")

    monkeypatch.setattr(grounding, "_sentence_ok", boom)
    assert grounded("A data analyst models data.", CARD) == ""


def test_split_keeps_decimals() -> None:
    card = "rating: 3.5 stars for the data analyst"
    assert grounded("The data analyst has 3.5 stars.", card) == "The data analyst has 3.5 stars."


def test_thousands_commas_compare_as_numbers() -> None:
    assert grounded("It pays 85,000.", "median: 85000") == "It pays 85,000."
    assert grounded("It pays 85,001.", "median: 85000") == ""


def test_lowercase_example_lists_are_checked() -> None:
    card = "skills: SQL; Python"
    assert grounded("Skills such as sql and python.", card)
    assert grounded("Skills such as sql and tableau.", card) == ""


def test_would_like_is_not_an_example_list() -> None:
    assert grounded("I'd like to help you look up a job.", "name a job") != ""


def test_an_order_word_must_be_on_the_card() -> None:
    assert grounded("Python comes after SQL.", "skills: SQL; Python") == ""
    assert grounded("Python comes after SQL.", "SQL then Python comes after SQL") != ""


class _Model:
    provider = "scripted"
    model = "scripted-1"

    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, messages: list[object], **_kw: object) -> object:
        from talent_angels.llm.protocol import LLMResult

        return LLMResult(text=self.text, provider=self.provider, model=self.model)


def test_phrase_map_returns_the_fallback_for_an_invented_sentence() -> None:
    from talent_angels.contracts import AgentResult, NodeRef
    from talent_angels.session.phrase import locate_card, phrase_map

    node = NodeRef(
        id="esco:occupation:1", suite="esco", source="esco", source_id="1",
        kind="Occupation", pref_label="data analyst",
    )  # fmt: skip
    result = AgentResult(capability="locate", suite="esco", nodes=[node], confidence=0.9)
    text = phrase_map(
        _Model("Data analysts earn $85,000 a year."), question="data analyst", result=result,
        fallback="FALLBACK", card=locate_card(result),
    )  # fmt: skip
    assert text == "FALLBACK"


def test_phrase_chat_drops_a_number_that_is_not_in_the_fallback() -> None:
    from talent_angels.session.phrase import phrase_chat

    said = phrase_chat(
        _Model("Hello there. I know 4000 jobs."), user_text="hi", fallback="Name a job.",
        hint="greet",
    )  # fmt: skip
    assert "4000" not in said
