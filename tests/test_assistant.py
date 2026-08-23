"""Assistant loop tests: intent routing, dispatch, answer packaging."""

from __future__ import annotations

import json

import pytest

from talent_angels.assistant import (  # noqa: E402
    CAPABILITY_LOCATE,
    ResultCache,
    build_graph,
    classify_capability,
    run_turn,
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


def test_graph_propagates_selected_suite_name() -> None:
    suite = FakeSuite(FakeToolResult(warnings=["not_found"]))
    graph = build_graph(
        suite=suite,
        suite_name="onet",
        llm_client=StubLLMClient(),
        answer_mode="structured",
    )

    locate_state = graph.invoke({"question": "accountant"})
    connect_state = graph.invoke({"question": "What skills does an accountant need?"})

    assert locate_state["result"].suite == "onet"
    assert connect_state["result"].suite == "onet"


def test_turn_separates_cache_and_runlog_by_suite(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    runlog_path = tmp_path / "runlog.jsonl"
    monkeypatch.setenv("RUNLOG_PATH", str(runlog_path))
    monkeypatch.setenv("LLM_PROVIDER", "none")
    suite = FakeSuite(FakeToolResult(warnings=["not_found"]))
    cache = ResultCache()

    onet_first = run_turn(
        suite=suite,
        suite_name="onet",
        llm_client=StubLLMClient(),
        question="accountant",
        force_locate=True,
        cache=cache,
    )
    esco_first = run_turn(
        suite=suite,
        suite_name="esco",
        llm_client=StubLLMClient(),
        question="accountant",
        force_locate=True,
        cache=cache,
    )
    onet_cached = run_turn(
        suite=suite,
        suite_name="onet",
        llm_client=StubLLMClient(),
        question="accountant",
        force_locate=True,
        cache=cache,
    )

    assert onet_first.result.suite == "onet"
    assert esco_first.result.suite == "esco"
    assert onet_first.record.efficiency.result_cache_hit is False
    assert esco_first.record.efficiency.result_cache_hit is False
    assert onet_cached.record.efficiency.result_cache_hit is True
    records = [json.loads(line) for line in runlog_path.read_text().splitlines()]
    assert [record["suite"] for record in records] == ["onet", "esco", "onet"]
