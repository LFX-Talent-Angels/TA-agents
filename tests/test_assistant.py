"""Assistant loop tests: intent routing, dispatch, answer packaging."""

from __future__ import annotations

import json

import pytest

from talent_angels.assistant import (  # noqa: E402
    CAPABILITY_LOCATE,
    ResultCache,
    build_graph,
    classify_capability,
    extract_locate_subject,
    run_turn,
)
from talent_angels.assistant.agent_loop import LOOP_SYSTEM
from talent_angels.assistant.llm_plan import PLAN_SYSTEM
from talent_angels.llm.protocol import LLMResult, LLMUsage, Message
from talent_angels.llm.stub_client import StubLLMClient  # noqa: E402
from tests.fakes.taxonomy import (  # noqa: E402
    FakeCandidate,
    FakeEdge,
    FakeNode,
    FakeToolResult,
)


class ScriptedPlanClient:
    provider = "litellm"
    model = "azure_ai/claude-sonnet-4-6"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> LLMResult:
        self.calls.append(messages)
        return LLMResult(
            text=self.text,
            provider=self.provider,
            model=self.model,
            usage=LLMUsage(input_tokens=8, output_tokens=4),
        )


class FakeSuite:
    def __init__(
        self, result: FakeToolResult, neighbor_result: FakeToolResult | None = None
    ) -> None:
        self._result = result
        self._neighbor_result = neighbor_result or FakeToolResult(warnings=["no_neighbors"])
        self.search_calls: list[tuple[str, str | None]] = []
        self.neighbor_calls: list[tuple[str, list[str] | None]] = []

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        self.search_calls.append((text, kind))
        return self._result

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        self.neighbor_calls.append((node_id, rel_types))
        return self._neighbor_result


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
        ("skill path from data analyst to data scientist", "pathfind"),
        ("path analyst → scientist", "pathfind"),
        ("what skills I need to become a nurse", "connect"),
    ],
)
def test_classify_capability_heuristics(question: str, expected: str) -> None:
    assert classify_capability(question) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Where is nurse in ESCO?", "nurse"),
        ("software developer", "software developer"),
        ("Find accountant", "accountant"),
        (
            "What essential skills does a nurse responsible for general care need?",
            "nurse responsible for general care",
        ),
        ("what skills I need to become a nurse", "nurse"),
        ("what skills I need to be a software developer", "software developer"),
        ("skills I need to be a software developer", "software developer"),
        ("skills I need to become a software developer", "software developer"),
        ("I want to become a nurse", "nurse"),
        ("I want to be a software developer", "software developer"),
    ],
)
def test_extract_locate_subject(question: str, expected: str) -> None:
    assert extract_locate_subject(question) == expected


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
    assert final_state["plan"].capabilities == ("locate",)
    assert final_state["result"].confidence == 0.95
    assert "software developer" in final_state["answer"]
    assert "95%" in final_state["answer"]


def test_graph_executes_locate_then_connect_for_skill_question() -> None:
    occupation = _occupation_node()
    skill = FakeNode(
        id="esco:skill:fixture-2",
        kind="Skill",
        label="computer programming",
        source="esco",
        source_id="http://data.europa.eu/esco/skill/fixture-2",
        properties={},
    )
    suite = FakeSuite(
        FakeToolResult(
            candidates=[FakeCandidate(node=occupation, confidence=0.95, method="exact_pref")],
            nodes=[occupation],
            evidence=["esco:search:exact_pref:software developer"],
        ),
        FakeToolResult(
            nodes=[occupation, skill],
            edges=[
                FakeEdge(
                    type="HAS_SKILL",
                    from_id=occupation.id,
                    to_id=skill.id,
                    properties={"relation_type": "essential"},
                )
            ],
            evidence=[f"esco:neighbors:{occupation.id}"],
        ),
    )
    graph = build_graph(suite=suite, llm_client=StubLLMClient(), answer_mode="structured")

    final_state = graph.invoke(
        {"question": "What essential skills does a software developer need?"}
    )

    assert final_state["capability"] == "connect"
    assert final_state["plan"].capabilities == ("locate", "connect")
    assert suite.search_calls == [("software developer", "occupation")]
    assert suite.neighbor_calls == [(occupation.id, ["HAS_SKILL"])]
    assert [node.pref_label for node in final_state["result"].nodes] == [
        "software developer",
        "computer programming",
    ]
    assert final_state["result"].edges[0].properties["relation_type"] == "essential"
    assert [tool.name for tool in final_state["tool_calls"]] == [
        "search_nodes",
        "get_neighbors",
    ]
    assert "computer programming" in final_state["answer"]


def test_graph_does_not_connect_an_ambiguous_subject() -> None:
    first = _occupation_node()
    second = FakeNode(
        id="esco:occupation:fixture-2",
        kind="Occupation",
        label="web developer",
        source="esco",
        source_id="http://data.europa.eu/esco/occupation/fixture-2",
        properties={},
    )
    suite = FakeSuite(
        FakeToolResult(
            candidates=[
                FakeCandidate(node=first, confidence=0.7, method="contains"),
                FakeCandidate(node=second, confidence=0.7, method="contains"),
            ],
            nodes=[first, second],
            warnings=["ambiguous"],
        )
    )
    graph = build_graph(suite=suite, llm_client=StubLLMClient(), answer_mode="structured")

    final_state = graph.invoke({"question": "Show skills for developer"})

    assert final_state["result"].capability == "connect"
    assert "ambiguous" in final_state["result"].warnings
    assert suite.neighbor_calls == []
    assert [tool.name for tool in final_state["tool_calls"]] == ["search_nodes"]
    assert "please clarify" in final_state["answer"].lower()


def test_graph_planner_owns_subject_before_search() -> None:
    node = _occupation_node()
    skill = FakeNode(
        id="esco:skill:fixture-2",
        kind="Skill",
        label="computer programming",
        source="esco",
        source_id="http://data.europa.eu/esco/skill/fixture-2",
        properties={},
    )
    suite = FakeSuite(
        FakeToolResult(
            candidates=[FakeCandidate(node=node, confidence=0.95, method="exact_pref")],
            nodes=[node],
            evidence=["esco:search:exact_pref:software developer"],
        ),
        FakeToolResult(
            nodes=[node, skill],
            edges=[
                FakeEdge(
                    type="HAS_SKILL",
                    from_id=node.id,
                    to_id=skill.id,
                    properties={"relation_type": "essential"},
                )
            ],
            evidence=[f"esco:neighbors:{node.id}"],
        ),
    )
    client = ScriptedPlanClient(
        '{"target":"connect","subject":"software developer","kind":"occupation",'
        '"rel_types":["HAS_SKILL"]}'
    )
    graph = build_graph(suite=suite, llm_client=client, answer_mode="structured")

    final_state = graph.invoke({"question": "what skills I need to be a software developer"})

    assert final_state["capability"] == "connect"
    assert suite.search_calls == [("software developer", "occupation")]
    assert suite.neighbor_calls == [(node.id, ["HAS_SKILL"])]
    assert PLAN_SYSTEM in client.calls[0][0].content
    assert LOOP_SYSTEM not in client.calls[0][0].content


def test_graph_planner_locate_does_not_fetch_neighbors() -> None:
    node = FakeNode(
        id="esco:occupation:firefighter",
        kind="Occupation",
        label="firefighter",
        source="esco",
        source_id="firefighter",
        properties={},
    )
    suite = FakeSuite(
        FakeToolResult(
            candidates=[FakeCandidate(node=node, confidence=0.95, method="exact_pref")],
            nodes=[node],
        )
    )
    client = ScriptedPlanClient('{"target":"locate","subject":"firefighter","kind":"occupation"}')
    graph = build_graph(suite=suite, llm_client=client, answer_mode="structured")

    final_state = graph.invoke({"question": "what does a firefighter do"})

    assert final_state["capability"] == "locate"
    assert suite.search_calls == [("firefighter", "occupation")]
    assert suite.neighbor_calls == []


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


@pytest.mark.parametrize(
    ("question", "expected_plan"),
    [
        (
            "What essential skills does a software developer need?",
            ("locate", "connect"),
        ),
        (
            "What is the skill gap from data analyst to data scientist?",
            ("locate", "connect", "pathfind"),
        ),
    ],
)
def test_turn_records_cumulative_plan(
    question: str,
    expected_plan: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("RUNLOG_PATH", str(tmp_path / "runlog.jsonl"))
    monkeypatch.setenv("LLM_PROVIDER", "none")

    outcome = run_turn(
        suite=FakeSuite(FakeToolResult()),
        suite_name="esco",
        llm_client=StubLLMClient(),
        question=question,
    )

    assert outcome.plan.capabilities == expected_plan
    assert tuple(outcome.record.plan) == expected_plan
