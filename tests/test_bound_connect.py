"""Bound-node Connect seam: follow-ups skip locate/search_nodes."""

from __future__ import annotations

import pytest

from talent_angels.assistant.connect_request import followup_connect_request
from talent_angels.assistant.turn import run_turn
from talent_angels.contracts import NodeRef
from talent_angels.llm.stub_client import StubLLMClient
from talent_angels.skills.connect.models import ConnectRequest


class RecordingSuite:
    def __init__(self) -> None:
        self.searches: list[str] = []
        self.neighbor_ids: list[str] = []

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
        ("essential skills", ("HAS_SKILL",), "essential"),
        ("what are the essential skills?", ("HAS_SKILL",), "essential"),
        ("list only essential skills", ("HAS_SKILL",), "essential"),
        ("list essential skills", ("HAS_SKILL",), "essential"),
        ("show me only essential skills", ("HAS_SKILL",), "essential"),
        ("list the skills", ("HAS_SKILL",), None),
        ("optional skills", ("HAS_SKILL",), "optional"),
        ("what do I actually need?", ("HAS_SKILL",), "essential"),
        ("what skills do I actually need?", ("HAS_SKILL",), "essential"),
        ("neighbors", (), None),
        ("neighbors of that", (), None),
        ("how to become one", ("HAS_SKILL",), "essential"),
        ("how do I become one?", ("HAS_SKILL",), "essential"),
        ("how to become that", ("HAS_SKILL",), "essential"),
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
        rel_types=("HAS_SKILL",),
        relation_kind="essential",
    )


def test_followup_connect_request_new_occupation_returns_none() -> None:
    assert followup_connect_request("What essential skills does a nurse need?", _bound()) is None
