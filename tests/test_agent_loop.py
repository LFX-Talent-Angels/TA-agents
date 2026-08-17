"""Main-assistant tool loop tests."""

from __future__ import annotations

from talent_angels.assistant.agent_loop import (
    _compact_result,
    parse_loop_turn,
    run_tool_loop,
)
from talent_angels.assistant.answer import PATHFIND_UNAVAILABLE, PATHFIND_UNIMPLEMENTED_WARNING
from talent_angels.assistant.graph import build_graph
from talent_angels.contracts import AgentResult, EdgeRef, NodeRef
from talent_angels.llm.protocol import LLMResult, LLMUsage, Message
from tests.fakes.taxonomy import FakeCandidate, FakeEdge, FakeNode, FakeToolResult


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


class ScriptedToolClient:
    provider = "litellm"
    model = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"

    def __init__(self, replies: list[LLMResult]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[list[Message], list[dict[str, object]] | None]] = []

    def complete(
        self, messages: list[Message], *, tools: list[dict[str, object]] | None = None
    ) -> LLMResult:
        self.calls.append((messages, tools))
        return self.replies.pop(0)


def _occupation() -> FakeNode:
    return FakeNode(
        id="esco:occupation:fixture-1",
        kind="Occupation",
        label="software developer",
        source="esco",
        source_id="http://data.europa.eu/esco/occupation/fixture-1",
        properties={},
    )


def test_parse_loop_turn_reads_laguna_xml_tool_call() -> None:
    text = (
        "<tool_call>get_neighbors"
        "<arg_key>node_id</arg_key>"
        "<arg_value>esco:occupation:f2b15a0e-e65a-438a-affb-29b9d50b77d1</arg_value>"
        "<arg_key>rel_types</arg_key>"
        '<arg_value>["HAS_SKILL"]</arg_value>'
        "<arg_key>relation_filter</arg_key>"
        "<arg_value>essential</arg_value>"
        "</tool_call>"
    )

    invocations, final = parse_loop_turn(text)

    assert final is None
    assert invocations[0].name == "get_neighbors"
    assert invocations[0].arguments["node_id"] == (
        "esco:occupation:f2b15a0e-e65a-438a-affb-29b9d50b77d1"
    )
    assert invocations[0].arguments["rel_types"] == ["HAS_SKILL"]
    assert invocations[0].arguments["relation_filter"] == "essential"


def test_tool_loop_executes_xml_get_neighbors_after_search() -> None:
    occupation = _occupation()
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
    client = ScriptedToolClient(
        [
            LLMResult(
                text='{"tool":"search_nodes","text":"software developer","kind":"occupation"}',
                provider="litellm",
                model="actual",
                usage=LLMUsage(input_tokens=10, output_tokens=4),
            ),
            LLMResult(
                text=(
                    '{"tool":"get_neighbors","node_id":"'
                    + occupation.id
                    + '","rel_types":["HAS_SKILL"],"relation_filter":"essential"}'
                ),
                provider="litellm",
                model="actual",
                usage=LLMUsage(input_tokens=12, output_tokens=4),
            ),
            LLMResult(
                text='{"final":"A software developer needs computer programming."}',
                provider="litellm",
                model="actual",
                usage=LLMUsage(input_tokens=20, output_tokens=8),
            ),
        ]
    )

    outcome = run_tool_loop(
        question="What essential skills does a software developer need?",
        suite=suite,
        suite_name="esco",
        llm_client=client,
    )

    assert outcome.plan.capabilities == ("locate", "connect")
    assert outcome.result.capability == "connect"
    assert suite.search_calls == [("software developer", "occupation")]
    assert suite.neighbor_calls == [(occupation.id, ["HAS_SKILL"])]
    assert [stage.stage for stage in outcome.stages] == ["act", "act", "answer"]
    assert outcome.answer == "A software developer needs computer programming."
    assert [tool.name for tool in outcome.tool_calls] == ["search_nodes", "get_neighbors"]


def test_tool_loop_runs_laguna_xml_get_neighbors() -> None:
    occupation = _occupation()
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
    xml = (
        "<tool_call>get_neighbors"
        "<arg_key>node_id</arg_key>"
        f"<arg_value>{occupation.id}</arg_value>"
        "<arg_key>rel_types</arg_key>"
        '<arg_value>["HAS_SKILL"]</arg_value>'
        "<arg_key>relation_filter</arg_key>"
        "<arg_value>essential</arg_value>"
        "</tool_call>"
    )
    client = ScriptedToolClient(
        [
            LLMResult(
                text='{"tool":"search_nodes","text":"software developer","kind":"occupation"}',
                provider="litellm",
                model="actual",
                usage=LLMUsage(input_tokens=8, output_tokens=4),
            ),
            LLMResult(
                text=xml,
                provider="litellm",
                model="actual",
                usage=LLMUsage(input_tokens=12, output_tokens=6),
            ),
            LLMResult(
                text='{"final":"A software developer needs computer programming."}',
                provider="litellm",
                model="actual",
                usage=LLMUsage(input_tokens=15, output_tokens=5),
            ),
        ]
    )

    outcome = run_tool_loop(
        question="What essential skills does a software developer need?",
        suite=suite,
        suite_name="esco",
        llm_client=client,
    )

    assert outcome.result.capability == "connect"
    assert suite.neighbor_calls == [(occupation.id, ["HAS_SKILL"])]
    assert "computer programming" in outcome.answer
    assert "<tool_call>" not in outcome.answer


class ExplodingToolClient:
    provider = "litellm"
    model = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"

    def complete(
        self, messages: list[Message], *, tools: list[dict[str, object]] | None = None
    ) -> LLMResult:
        raise RuntimeError("LiteLLM provider request failed: missing field `function`")


def test_graph_falls_back_when_provider_rejects_tools() -> None:
    occupation = _occupation()
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
    graph = build_graph(suite=suite, llm_client=ExplodingToolClient(), answer_mode="structured")

    final_state = graph.invoke(
        {"question": "What essential skills does a software developer need?"}
    )

    assert final_state["capability"] == "connect"
    assert suite.search_calls == [("software developer", "occupation")]
    assert suite.neighbor_calls == [(occupation.id, ["HAS_SKILL"])]


def test_graph_uses_planner_when_provider_is_not_stub() -> None:
    occupation = _occupation()
    suite = FakeSuite(
        FakeToolResult(
            candidates=[FakeCandidate(node=occupation, confidence=0.95, method="exact_pref")],
            nodes=[occupation],
            evidence=["esco:search:exact_pref:software developer"],
        )
    )
    client = ScriptedToolClient(
        [
            LLMResult(
                text='{"target":"locate","subject":"software developer","kind":"occupation"}',
                provider="litellm",
                model="actual",
                usage=LLMUsage(input_tokens=5, output_tokens=2),
            ),
        ]
    )
    graph = build_graph(suite=suite, llm_client=client, answer_mode="structured")

    final_state = graph.invoke({"question": "Where is software developer in ESCO?"})

    assert final_state["capability"] == "locate"
    assert "software developer" in final_state["answer"]
    assert suite.search_calls == [("software developer", "occupation")]
    assert [stage.stage for stage in final_state["llm_stages"]] == ["intent"]


def test_path_question_does_not_list_neighbors() -> None:
    occupation = _occupation()
    suite = FakeSuite(
        FakeToolResult(
            candidates=[FakeCandidate(node=occupation, confidence=0.95, method="exact_pref")],
            nodes=[occupation],
        )
    )
    client = ScriptedToolClient(
        [
            LLMResult(
                text='{"tool":"search_nodes","text":"data analyst"}',
                provider="litellm",
                model="actual",
                usage=LLMUsage(input_tokens=4, output_tokens=2),
            )
        ]
    )

    outcome = run_tool_loop(
        question="What is the skill path from data analyst to data scientist?",
        suite=suite,
        suite_name="esco",
        llm_client=client,
    )

    assert PATHFIND_UNIMPLEMENTED_WARNING in outcome.result.warnings
    assert outcome.plan.capabilities == ("locate", "connect", "pathfind")
    assert outcome.result.capability == "pathfind"
    assert suite.search_calls == []
    assert suite.neighbor_calls == []
    assert client.calls == []
    assert outcome.tool_calls == []
    assert outcome.answer == PATHFIND_UNAVAILABLE


def test_graph_path_question_does_not_fallback_to_connect() -> None:
    occupation = _occupation()
    suite = FakeSuite(
        FakeToolResult(
            candidates=[FakeCandidate(node=occupation, confidence=0.95, method="exact_pref")],
            nodes=[occupation],
        )
    )
    graph = build_graph(suite=suite, llm_client=ExplodingToolClient(), answer_mode="structured")

    final_state = graph.invoke({"question": "skill path from data analyst to data scientist"})

    assert final_state["capability"] == "pathfind"
    assert PATHFIND_UNIMPLEMENTED_WARNING in final_state["result"].warnings
    assert suite.search_calls == []
    assert suite.neighbor_calls == []
    assert final_state["answer"] == PATHFIND_UNAVAILABLE


def test_ambiguous_search_stops_without_another_lookup() -> None:
    first = FakeNode(
        id="esco:occupation:cook",
        kind="Occupation",
        label="diet cook",
        source="esco",
        source_id="cook",
        properties={},
    )
    second = FakeNode(
        id="esco:occupation:nanny",
        kind="Occupation",
        label="nanny",
        source="esco",
        source_id="nanny",
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
    client = ScriptedToolClient(
        [
            LLMResult(
                text='{"tool":"search_nodes","text":"what skills I need to become a nurse"}',
                provider="litellm",
                model="actual",
                usage=LLMUsage(input_tokens=4, output_tokens=2),
            ),
            LLMResult(
                text='{"tool":"search_nodes","text":"nursing"}',
                provider="litellm",
                model="actual",
                usage=LLMUsage(input_tokens=4, output_tokens=2),
            ),
        ]
    )

    outcome = run_tool_loop(
        question="what skills I need to become a nurse",
        suite=suite,
        suite_name="esco",
        llm_client=client,
    )

    assert suite.search_calls in ([("nurse", "occupation")], [("nurse", None)])
    assert len(suite.search_calls) == 1
    assert "ambiguous" in outcome.result.warnings
    assert "diet cook" in outcome.answer and "clarify" in outcome.answer.lower()
    assert outcome.answer.lower().count("diet cook") == 1
    assert "Candidates:" in outcome.answer
    assert client.calls  # first act ran
    assert len(client.calls) == 1


def test_unique_skills_question_connects_after_locate() -> None:
    occupation = _occupation()
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
    client = ScriptedToolClient(
        [
            LLMResult(
                text=(
                    '{"tool":"search_nodes","text":"what skills I need to be a software developer"}'
                ),
                provider="litellm",
                model="actual",
                usage=LLMUsage(input_tokens=4, output_tokens=2),
            ),
            LLMResult(
                text='{"final":"done"}',
                provider="litellm",
                model="actual",
                usage=LLMUsage(input_tokens=4, output_tokens=2),
            ),
        ]
    )

    outcome = run_tool_loop(
        question="what skills I need to be a software developer",
        suite=suite,
        suite_name="esco",
        llm_client=client,
    )

    assert suite.search_calls in (
        [("software developer", "occupation")],
        [("software developer", None)],
    )
    assert outcome.result.capability == "connect"
    assert suite.neighbor_calls == [(occupation.id, ["HAS_SKILL"])]
    assert "computer programming" in outcome.answer


def test_compact_result_truncates_neighbors_but_keeps_counts() -> None:
    center = NodeRef(
        id="esco:occupation:1",
        suite="esco",
        source="esco",
        source_id="1",
        kind="Occupation",
        pref_label="software developer",
    )
    skills = [
        NodeRef(
            id=f"esco:skill:{index}",
            suite="esco",
            source="esco",
            source_id=str(index),
            kind="Skill",
            pref_label=f"skill {index}",
        )
        for index in range(20)
    ]
    edges = [
        EdgeRef(
            type="HAS_SKILL",
            suite="esco",
            source_node_id=center.id,
            target_node_id=skill.id,
            properties={"relation_type": "essential"},
        )
        for skill in skills
    ]
    result = AgentResult(
        capability="connect",
        suite="esco",
        nodes=[center, *skills],
        edges=edges,
        confidence=0.95,
    )

    payload = _compact_result(result)

    assert payload["truncated"] is True
    assert payload["node_count"] == 21
    assert payload["edge_count"] == 20
    assert payload["omitted_nodes"] == 13
    assert payload["omitted_edges"] == 12
    assert len(payload["nodes"]) == 8
    assert len(payload["edges"]) == 8
