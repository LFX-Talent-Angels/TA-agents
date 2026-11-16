"""Over-long input keeps its question (BX-17)."""

from __future__ import annotations

from talent_angels.api.schemas import MAX_QUESTION_CHARS as API_MAX
from talent_angels.assistant.synthesize import synthesize_structured
from talent_angels.assistant.turn import MAX_QUESTION_CHARS, normalize_question
from talent_angels.contracts import AgentResult


def test_the_question_at_the_end_survives() -> None:
    text = "I want to know about nurses. " + "lorem ipsum dolor sit amet " * 120
    text += "what skills does a baker need?"
    kept, warnings = normalize_question(text)
    assert warnings == ["question_truncated"]
    assert len(kept) <= MAX_QUESTION_CHARS
    assert kept.endswith("what skills does a baker need?")
    assert kept.startswith("I want to know about nurses.")


def test_the_api_and_the_runtime_share_one_limit() -> None:
    assert API_MAX == MAX_QUESTION_CHARS


def test_an_unknown_kind_is_not_called_a_miss() -> None:
    # BX-18: "kind": "banana" for a code that exists answered "a miss".
    results = [
        AgentResult(capability="locate", suite="bls", warnings=["unknown_kind:banana"]),
        AgentResult(capability="locate", suite="esco", warnings=["unknown_kind:banana"]),
    ]
    text = synthesize_structured(results)
    assert "banana" in text and "miss" not in text
