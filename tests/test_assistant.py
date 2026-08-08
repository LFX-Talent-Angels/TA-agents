"""Assistant loop tests: intent routing, dispatch, answer packaging."""

from __future__ import annotations

import pytest
from ta_taxonomies.contract.models import Candidate, Node, ToolResult

from talent_angels.assistant import (
    CAPABILITY_LOCATE,
    build_graph,
    classify_capability,
)
from talent_angels.llm.stub_client import StubLLMClient
from talent_angels.skills.locate import ESCO_SUITE_NAME, open_esco_suite


class FakeSuite:
    def __init__(self, result: ToolResult) -> None:
        self._result = result

    def search_nodes(self, text: str, kind: str | None = None) -> ToolResult:
        return self._result


def _occupation_node() -> Node:
    return Node(
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
    result = ToolResult(
        candidates=[Candidate(node=node, confidence=0.95, method="exact_pref")],
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
    suite = FakeSuite(ToolResult(warnings=["not_found"]))
    graph = build_graph(suite=suite, llm_client=StubLLMClient(), answer_mode="structured")

    final_state = graph.invoke(
        {"question": "What essential skills does a software developer need?"}
    )

    assert final_state["capability"] == "connect"
    assert final_state["result"].nodes == []
    assert "capability_not_implemented:connect" in final_state["result"].warnings
    assert "no match found" in final_state["answer"].lower()


def _neo4j_reachable() -> bool:
    try:
        with open_esco_suite() as suite:
            suite._driver.verify_connectivity()  # noqa: SLF001 - availability probe only
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _neo4j_reachable(), reason="Neo4j not reachable; see TA-taxonomies NOTES.md"
)
def test_graph_end_to_end_against_live_esco_fixture() -> None:
    with open_esco_suite() as suite:
        graph = build_graph(suite=suite, llm_client=StubLLMClient(), answer_mode="structured")
        final_state = graph.invoke({"question": "software developer", "kind": "occupation"})

    assert final_state["result"].confidence == 0.95
    assert "software developer" in final_state["answer"].lower()
    assert final_state["result"].suite == ESCO_SUITE_NAME
