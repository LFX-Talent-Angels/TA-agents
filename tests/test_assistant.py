"""Assistant loop tests: intent routing, dispatch, answer packaging."""

from __future__ import annotations

import pytest

from talent_angels.assistant import (  # noqa: E402
    CAPABILITY_LOCATE,
    build_graph,
    classify_capability,
)
from talent_angels.llm.stub_client import StubLLMClient  # noqa: E402
from tests.fakes.taxonomy import FakeCandidate, FakeNode, FakeToolResult  # noqa: E402


class FakeSuite:
    def __init__(self, result: FakeToolResult) -> None:
        self._result = result

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        return self._result


def _occupation_node() -> FakeNode:
    return FakeNode(
        id="esco:occupation:fixture-1",
        kind="Occupation",
        label="software developer",
        source="esco",
        source_id="http://data.europa.eu/esco/occupation/fixture-1",
        properties={},
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Where is software developer in ESCO?", "locate"),
        ("What essential skills does a software developer need?", "connect"),
        ("What is the skill gap from data analyst to data scientist?", "pathfind"),
    ],
)
def test_classify_capability_heuristics(question: str, expected: str) -> None:
    assert classify_capability(question) == expected


def test_graph_answers_locate_question_structured() -> None:
    node = _occupation_node()
    result = FakeToolResult(
        candidates=[FakeCandidate(node=node, confidence=0.95, method="exact_pref")],
        nodes=[node],
        evidence=["esco:search:exact_pref:software developer"],
    )
    suite = FakeSuite(result)
    graph = build_graph(suite=suite, llm_client=StubLLMClient(), answer_mode="structured")

    final_state = graph.invoke({"question": "software developer"})

    assert final_state["capability"] == CAPABILITY_LOCATE
    assert final_state["result"].confidence == 0.95
    assert "software developer" in final_state["answer"]
    assert "95%" in final_state["answer"]


def test_graph_reports_unimplemented_capability_honestly() -> None:
    suite = FakeSuite(FakeToolResult(warnings=["not_found"]))
    graph = build_graph(suite=suite, llm_client=StubLLMClient(), answer_mode="structured")

    final_state = graph.invoke(
        {"question": "What essential skills does a software developer need?"}
    )

    assert final_state["capability"] == "connect"
    assert final_state["result"].nodes == []
    assert "capability_not_implemented:connect" in final_state["result"].warnings
    assert "no match found" in final_state["answer"].lower()
