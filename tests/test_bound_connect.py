"""Bound-node Connect seam: follow-ups skip locate/search_nodes."""

from __future__ import annotations

import pytest

from talent_angels.assistant.connect_request import followup_connect_request
from talent_angels.assistant.turn import run_turn
from talent_angels.contracts import NodeRef
from talent_angels.llm.stub_client import StubLLMClient
from talent_angels.skills.connect.models import ConnectRequest
from talent_angels.suites.schema import SuiteSchema

_DEFAULT_SCHEMA = SuiteSchema(
    skill_rel_types=("HAS_SKILL", "USES_SOFTWARE"),
    optional_rel_values=frozenset({"optional", "transferable"}),
    group_rel_type=None,
    group_node_kinds=frozenset(),
)


class RecordingSuite:
    def __init__(self) -> None:
        self.searches: list[str] = []
        self.neighbor_ids: list[str] = []

    @property
    def suite_schema(self) -> SuiteSchema:
        return _DEFAULT_SCHEMA

    def search_nodes(self, text: str, kind: str | None = None):
        self.searches.append(text)
        raise AssertionError(f"search_nodes must not run on bound follow-up, got {text!r}")

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None):
        from tests.fakes.taxonomy import FakeEdge, FakeNode, FakeToolResult

        self.neighbor_ids.append(node_id)
        center = FakeNode(node_id, "Occupation", "software developer", "esco", "src")
        skill = FakeNode("esco:skill:1", "Skill", "computer programming", "esco", "sk")
        return FakeToolResult(
            nodes=[center, skill],
            edges=[FakeEdge("HAS_SKILL", node_id, skill.id, {"relation_type": "essential"})],
            evidence=["test:neighbors"],
        )


def _bound() -> NodeRef:
    return NodeRef(
        id="esco:occupation:sd",
        suite="esco",
        source="esco",
        source_id="src",
        kind="Occupation",
        pref_label="software developer",
    )


def test_bound_followup_connects_without_search(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RUNLOG_PATH", str(tmp_path / "runlog.jsonl"))
    suite = RecordingSuite()
    bound = _bound()
    outcome = run_turn(
        suite=suite,
        suite_name="esco",
        llm_client=StubLLMClient(),
        question="what are the essential skills?",
        force_capability="connect",
        bound_node=bound,
        answer_mode="structured",
    )
    assert suite.searches == []
    assert suite.neighbor_ids == ["esco:occupation:sd"]
    assert outcome.capability == "connect"
    assert "computer programming" in outcome.answer


@pytest.mark.parametrize(
    ("question", "rel_types", "relation_kind"),
    [
        ("essential skills", ("HAS_SKILL", "USES_SOFTWARE"), "essential"),
        ("what are the essential skills?", ("HAS_SKILL", "USES_SOFTWARE"), "essential"),
        ("list only essential skills", ("HAS_SKILL", "USES_SOFTWARE"), "essential"),
        ("list essential skills", ("HAS_SKILL", "USES_SOFTWARE"), "essential"),
        ("show me only essential skills", ("HAS_SKILL", "USES_SOFTWARE"), "essential"),
        ("list the skills", ("HAS_SKILL", "USES_SOFTWARE"), None),
        ("optional skills", ("HAS_SKILL", "USES_SOFTWARE"), "optional"),
        ("what do I actually need?", ("HAS_SKILL", "USES_SOFTWARE"), "essential"),
        ("what skills do I actually need?", ("HAS_SKILL", "USES_SOFTWARE"), "essential"),
        ("neighbors", (), None),
        ("neighbors of that", (), None),
        ("how to become one", ("HAS_SKILL", "USES_SOFTWARE"), "essential"),
        ("how do I become one?", ("HAS_SKILL", "USES_SOFTWARE"), "essential"),
        ("how to become that", ("HAS_SKILL", "USES_SOFTWARE"), "essential"),
        ("what software tools does it use?", ("USES_SOFTWARE",), None),
        ("which technologies do they use", ("USES_SOFTWARE",), None),
    ],
)
def test_followup_connect_request_short_phrases(
    question: str,
    rel_types: tuple[str, ...],
    relation_kind: str | None,
) -> None:
    request = followup_connect_request(question, _bound())
    assert request == ConnectRequest(
        subject="software developer",
        rel_types=rel_types,
        relation_kind=relation_kind,
    )


def test_followup_connect_request_same_occupation_keeps_extract() -> None:
    request = followup_connect_request(
        "What essential skills does a software developer need?",
        _bound(),
    )
    assert request == ConnectRequest(
        subject="software developer",
        rel_types=("HAS_SKILL", "USES_SOFTWARE"),
        relation_kind="essential",
    )


def test_followup_connect_request_new_occupation_returns_none() -> None:
    assert followup_connect_request("What essential skills does a nurse need?", _bound()) is None


class _ScriptedModel:
    provider = "litellm"
    model = "test-model"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)

    def complete(self, messages, **_: object):
        from talent_angels.llm.protocol import LLMResult

        text = self.replies.pop(0) if self.replies else '{"final":"ok"}'
        return LLMResult(text=text, provider=self.provider, model=self.model)


class GoalSuite:
    """Bound to web developer; the user names a new goal title."""

    def __init__(self) -> None:
        self.searches: list[str] = []
        self.neighbor_ids: list[str] = []

    @property
    def suite_schema(self) -> SuiteSchema:
        return _DEFAULT_SCHEMA

    def search_nodes(self, text: str, kind: str | None = None):
        from tests.fakes.taxonomy import FakeCandidate, FakeNode, FakeToolResult

        self.searches.append(text)
        hit = FakeNode("esco:occupation:sd", "Occupation", "software developer", "esco", "src")
        return FakeToolResult(candidates=[FakeCandidate(hit, 0.95, "exact")])

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None):
        from tests.fakes.taxonomy import FakeNode, FakeToolResult

        self.neighbor_ids.append(node_id)
        return FakeToolResult(nodes=[FakeNode(node_id, "Occupation", "x", "esco", "src")])


def test_a_goal_statement_looks_up_the_named_title_not_the_bound_job(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RUNLOG_PATH", str(tmp_path / "runlog.jsonl"))
    suite = GoalSuite()
    web_dev = NodeRef(
        id="esco:occupation:wd",
        suite="esco",
        source="esco",
        source_id="wd",
        kind="Occupation",
        pref_label="web developer",
    )
    # The model reads the goal, then (the live bug) asks for the bound job's neighbors.
    client = _ScriptedModel(
        [
            '{"target":"locate","subject":"software developer","kind":"occupation",'
            '"profile_intent":"goal"}',
            '{"tool":"get_neighbors","node_id":"esco:occupation:wd"}',
        ]
    )
    from talent_angels.suites import SuiteRegistry
    from tests.fakes.suite import suite_factory

    outcome = run_turn(
        registry=SuiteRegistry({"esco": suite_factory("esco", suite)}, default="esco"),
        llm_client=client,
        question="my goal is software developer",
        bound_nodes={"esco": web_dev},
        answer_mode="structured",
    )
    assert "software developer" in suite.searches
    assert "esco:occupation:wd" not in suite.neighbor_ids
    assert outcome.result.nodes[0].pref_label == "software developer"


def test_a_tools_question_on_a_bound_job_asks_for_tools_only(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RUNLOG_PATH", str(tmp_path / "runlog.jsonl"))
    from talent_angels.suites import SuiteRegistry
    from tests.fakes.suite import suite_factory

    asked: list[list[str] | None] = []

    class ToolsSuite(GoalSuite):
        def get_neighbors(self, node_id: str, rel_types: list[str] | None = None):
            asked.append(list(rel_types) if rel_types else None)
            return super().get_neighbors(node_id, rel_types)

    # The live model planned "locate" and then asked for skills and tools together.
    client = _ScriptedModel(
        [
            '{"target":"locate","subject":null}',
            '{"tool":"get_neighbors","node_id":"esco:occupation:wd",'
            '"rel_types":["HAS_SKILL","USES_SOFTWARE"]}',
        ]
    )
    bound = NodeRef(
        id="esco:occupation:wd",
        suite="esco",
        source="esco",
        source_id="wd",
        kind="Occupation",
        pref_label="web developer",
    )
    run_turn(
        registry=SuiteRegistry({"esco": suite_factory("esco", ToolsSuite())}, default="esco"),
        llm_client=client,
        question="what software tools does it use?",
        bound_nodes={"esco": bound},
        answer_mode="structured",
    )
    assert asked and all(types == ["USES_SOFTWARE"] for types in asked)
