"""A compare turn: both titles located and connected in code, then split."""

from __future__ import annotations

from typing import Any

from talent_angels.assistant import run_turn
from talent_angels.llm import LLMResult, Message
from talent_angels.llm.stub_client import StubLLMClient
from talent_angels.skills.connect.compare import skill_overlap
from talent_angels.suites import SuiteRegistry
from tests.fakes.suite import FakeSuite, suite_factory
from tests.fakes.taxonomy import FakeCandidate, FakeEdge, FakeNode, FakeToolResult


def _node(kind: str, label: str) -> FakeNode:
    return FakeNode(
        id=f"fake:{kind.casefold()}:{label}",
        kind=kind,
        label=label,
        source="fake",
        source_id=label,
        properties={},
    )


CHEF, BAKER = _node("Occupation", "chef"), _node("Occupation", "baker")
COOK, MENU, DOUGH = (
    _node("Skill", "cook food"),
    _node("Skill", "plan menus"),
    _node("Skill", "knead dough"),
)
SKILLS = {CHEF.id: [COOK, MENU], BAKER.id: [COOK, DOUGH]}
TEACHERS = [_node("Occupation", "music teacher"), _node("Occupation", "maths teacher")]


class _KitchenSuite(FakeSuite):
    def search_nodes(self, text: str, kind: str | None = None) -> Any:
        if text in ("chef", "baker"):
            node = CHEF if text == "chef" else BAKER
            return FakeToolResult(
                candidates=[FakeCandidate(node=node, confidence=0.95, method="exact_pref")],
                nodes=[node],
            )
        if text == "teacher":
            return FakeToolResult(
                candidates=[
                    FakeCandidate(node=n, confidence=0.7, method="contains") for n in TEACHERS
                ],
                nodes=TEACHERS,
            )
        return FakeToolResult(warnings=["not_found"])

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> Any:
        skills = SKILLS.get(node_id, [])
        centre = CHEF if node_id == CHEF.id else BAKER
        edges = [
            FakeEdge(type="HAS_SKILL", from_id=node_id, to_id=s.id, properties={}) for s in skills
        ]
        return FakeToolResult(nodes=[centre, *skills], edges=edges)


def _registry() -> SuiteRegistry:
    return SuiteRegistry({"fake": suite_factory("fake", _KitchenSuite())}, default="fake")


class _Planner:
    """Answers the planner once; any further model call would be the tool loop."""

    provider = "litellm"
    model = "test-model"

    def __init__(self, plan: str) -> None:
        self.plan = plan
        self.calls = 0

    def complete(self, messages: list[Message], **_: object) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self.plan, provider=self.provider, model="m")


def test_compare_runs_in_code_without_the_tool_loop() -> None:
    client = _Planner('{"target":"connect","subject":"chef","secondary_subject":"baker"}')

    outcome = run_turn(registry=_registry(), llm_client=client, question="chef vs baker")

    assert client.calls == 1
    overlap = skill_overlap(outcome.result)
    assert [n.pref_label for n in overlap.shared] == ["cook food"]
    assert [n.pref_label for n in overlap.only_a] == ["plan menus"]
    assert [n.pref_label for n in overlap.only_b] == ["knead dough"]
    assert "chef vs baker — 1 shared, 1 only chef, 1 only baker" in outcome.answer


def test_offline_compare_uses_the_heuristic_plan() -> None:
    outcome = run_turn(
        registry=_registry(), llm_client=StubLLMClient(), question="compare chef and baker"
    )
    assert outcome.result.capability == "compare"


def test_a_side_with_several_matches_is_returned_for_the_user_to_pick() -> None:
    client = _Planner('{"target":"connect","subject":"chef","secondary_subject":"teacher"}')

    outcome = run_turn(registry=_registry(), llm_client=client, question="chef vs teacher")

    assert outcome.result.capability == "locate"
    assert "ambiguous" in outcome.result.warnings
    assert [n.pref_label for n in outcome.result.nodes] == ["maths teacher", "music teacher"]
