"""Wording of multi-step replies: what the picker names, what a pending suite says."""

from __future__ import annotations

import pytest

from talent_angels.assistant.llm_plan import PlanDraft
from talent_angels.assistant.planning import build_plan_for_capability
from talent_angels.assistant.turn import TurnOutcome
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.runlog import RunLogRecord
from talent_angels.session.copy import LOCATE_MISS
from talent_angels.session.kernel import handle_line
from talent_angels.session.store import new_session

_QUESTION = "what knowledge does a nurse need?"


def _node(suite: str, label: str, kind: str = "Occupation") -> NodeRef:
    return NodeRef(
        id=f"{suite}:{kind.casefold()}:{label}",
        suite=suite,
        source=suite,
        source_id=label,
        kind=kind,
        pref_label=label,
    )


def _nurses(suite: str) -> AgentResult:
    return AgentResult(
        capability="locate",
        suite=suite,
        nodes=[_node(suite, "nurse assistant"), _node(suite, "specialist nurse")],
        warnings=["ambiguous"],
        confidence=0.7,
    )


def _runner(*results: AgentResult, draft: PlanDraft | None = None):
    def runner(question: str, **_kwargs: object) -> TurnOutcome:
        return TurnOutcome(
            capability=results[0].capability,  # type: ignore[arg-type]
            plan=build_plan_for_capability("locate", suites=tuple(r.suite for r in results)),
            result=results[0],
            results=results,
            answer="ignored",
            plan_draft=draft,
            record=RunLogRecord(suite="esco,onet", plan=["locate"], question=question),
        )

    return runner


def _draft() -> PlanDraft:
    return PlanDraft(target="connect", subject="nurse", kind="occupation")


def test_single_suite_picker_names_the_subject_not_the_question() -> None:
    reply = handle_line(new_session(), _QUESTION, runner=_runner(_nurses("esco"), draft=_draft()))
    assert '"nurse"' in reply.text
    assert _QUESTION not in reply.text


def test_multi_suite_picker_names_the_subject_not_the_question() -> None:
    runner = _runner(_nurses("esco"), _nurses("onet"), draft=_draft())
    reply = handle_line(new_session(), _QUESTION, runner=runner)
    assert '"nurse"' in reply.text
    assert _QUESTION not in reply.text


def test_picker_falls_back_to_the_question_without_a_plan() -> None:
    reply = handle_line(new_session(), "nurse", runner=_runner(_nurses("esco")))
    assert '"nurse"' in reply.text


@pytest.mark.parametrize("warning", ["bind_required", "no_subject"])
def test_suite_waiting_for_a_pick_is_not_reported_as_a_miss(warning: str) -> None:
    chef = _node("esco", "chef")
    connect = AgentResult(
        capability="connect",
        suite="esco",
        nodes=[chef, _node("esco", "plan menus", kind="Skill")],
    )
    pending = AgentResult(capability="connect", suite="onet", warnings=[warning])
    state = new_session()
    state.bindings["esco"] = chef

    reply = handle_line(state, "what skills does it need?", runner=_runner(connect, pending))

    assert "No O*NET occupation is chosen yet" in reply.text
    assert LOCATE_MISS not in reply.text
