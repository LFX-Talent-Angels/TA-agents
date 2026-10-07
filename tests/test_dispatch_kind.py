"""The planner's kind reaches the tool loop, so a skill question searches skills."""

from __future__ import annotations

from typing import Any, cast

import pytest

from talent_angels.assistant import graph
from talent_angels.assistant.llm_plan import PlanDraft
from talent_angels.assistant.planning import build_plan_for_capability
from talent_angels.assistant.state import AssistantState
from talent_angels.contracts import NodeRef
from talent_angels.llm import LLMResult, Message
from tests.fakes.suite import FakeSuite


class _Live:
    provider = "litellm"
    model = "m"

    def complete(self, messages: list[Message], **_: object) -> LLMResult:
        raise AssertionError("the loop is replaced in this test")


def _dispatch(monkeypatch: pytest.MonkeyPatch, *, kind: str | None) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_loop(**kwargs: Any) -> Any:
        seen.update(kwargs)
        raise RuntimeError("stop after capturing the arguments")

    monkeypatch.setattr(graph, "run_tool_loop", fake_loop)
    state = {
        "question": "what is machine learning?",
        "kind": kind,
        "plan": build_plan_for_capability("locate", suites=("fake",)),
        "plan_draft": PlanDraft(target="locate", subject="machine learning", kind="skill"),
        "heuristic_intent": False,
    }
    graph.dispatch_plan(
        cast(AssistantState, state), suite=FakeSuite(), suite_name="fake", llm_client=_Live()
    )
    return seen


def test_planner_kind_is_passed_to_the_tool_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _dispatch(monkeypatch, kind=None)["kind"] == "skill"


def test_a_caller_kind_wins_over_the_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _dispatch(monkeypatch, kind="occupation")["kind"] == "occupation"


def test_a_follow_up_on_a_bound_node_keeps_the_loop_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_loop(**kwargs: Any) -> Any:
        seen.update(kwargs)
        raise RuntimeError("stop after capturing the arguments")

    monkeypatch.setattr(graph, "run_tool_loop", fake_loop)
    chef = NodeRef(
        id="fake:occupation:chef",
        suite="fake",
        source="fake",
        source_id="chef",
        kind="Occupation",
        pref_label="chef",
    )
    state = {
        "question": "what skills does it need?",
        "kind": None,
        "plan": build_plan_for_capability("connect", suites=("fake",)),
        "plan_draft": PlanDraft(target="connect", subject="chef", kind="skill"),
        "heuristic_intent": False,
    }
    graph.dispatch_plan(
        cast(AssistantState, state),
        suite=FakeSuite(),
        suite_name="fake",
        llm_client=_Live(),
        bound_node=chef,
    )
    assert seen["kind"] is None
