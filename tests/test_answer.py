"""User-facing answer packaging tests."""

from talent_angels.assistant.answer import build_answer
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.llm.stub_client import StubLLMClient


def _node(index: int) -> NodeRef:
    return NodeRef(
        id=f"esco:occupation:{index}",
        suite="esco",
        source="esco",
        source_id=f"source-{index}",
        kind="Occupation",
        pref_label=f"candidate {index}",
    )


def test_structured_answer_surfaces_ambiguous_choices() -> None:
    result = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[_node(i) for i in range(1, 6)],
        confidence=0.7,
        warnings=["ambiguous"],
    )

    answer, usage = build_answer(result, llm_client=StubLLMClient(), mode="structured")

    assert answer.startswith("Ambiguous locate result")
    assert "candidate 1" in answer
    assert "candidate 3" in answer
    assert "candidate 4" not in answer
    assert "2 more candidate(s)" in answer
    assert "clarify" in answer.lower()
    assert usage is None


def test_structured_answer_keeps_unique_match_summary() -> None:
    result = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[_node(1)],
        confidence=0.95,
    )

    answer, usage = build_answer(result, llm_client=StubLLMClient(), mode="structured")

    assert answer == "candidate 1 (Occupation, id=esco:occupation:1) — confidence 95%"
    assert usage is None
