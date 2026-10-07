"""What a conversation may write to USER.md: only what the user said about themselves."""

from __future__ import annotations

from talent_angels.assistant.llm_plan import PlanDraft
from talent_angels.assistant.planning import build_plan_for_capability
from talent_angels.assistant.turn import TurnOutcome
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.memory.paths import user_md
from talent_angels.runlog import RunLogRecord
from talent_angels.session.kernel import handle_line
from talent_angels.session.store import new_session


def _node(suite: str, label: str) -> NodeRef:
    return NodeRef(
        id=f"{suite}:occupation:{label}",
        suite=suite,
        source=suite,
        source_id=label,
        kind="Occupation",
        pref_label=label,
    )


def _runner(*results: AgentResult, draft: PlanDraft | None = None):
    def runner(question: str, **_kwargs: object) -> TurnOutcome:
        return TurnOutcome(
            capability="locate",
            plan=build_plan_for_capability("locate", suites=tuple(r.suite for r in results)),
            result=results[0],
            results=results,
            answer="ignored",
            plan_draft=draft,
            record=RunLogRecord(suite="esco,onet", plan=["locate"], question=question),
        )

    return runner


def _profile() -> str:
    return user_md().read_text(encoding="utf-8") if user_md().exists() else ""


def test_looking_up_an_occupation_is_not_a_statement_about_the_user() -> None:
    esco = AgentResult(capability="locate", suite="esco", nodes=[_node("esco", "firefighter")])
    onet = AgentResult(capability="locate", suite="onet", nodes=[_node("onet", "Firefighters")])
    state = new_session()

    handle_line(state, "firefighter", runner=_runner(esco, onet))

    assert state.bindings["esco"].pref_label == "firefighter"
    assert "STANDING" not in _profile()


def test_single_suite_lookup_does_not_write_standing() -> None:
    esco = AgentResult(capability="locate", suite="esco", nodes=[_node("esco", "chef")])
    state = new_session()

    handle_line(state, "chef", runner=_runner(esco))

    assert "STANDING" not in _profile()


def test_picking_from_a_list_does_not_write_standing() -> None:
    nodes = [_node("esco", "nurse assistant"), _node("esco", "specialist nurse")]
    esco = AgentResult(
        capability="locate", suite="esco", nodes=nodes, warnings=["ambiguous"], confidence=0.7
    )
    state = new_session()

    handle_line(state, "nurse", runner=_runner(esco))
    handle_line(state, "2", runner=_runner(esco))

    assert state.bindings["esco"].pref_label == "specialist nurse"
    assert "STANDING" not in _profile()


def test_goal_is_saved_without_making_it_the_current_job() -> None:
    esco = AgentResult(capability="locate", suite="esco", nodes=[_node("esco", "data analyst")])
    onet = AgentResult(capability="locate", suite="onet", nodes=[_node("onet", "Data Analysts")])
    draft = PlanDraft(
        target="locate", subject="data analyst", kind="occupation", profile_intent="goal"
    )
    state = new_session()

    handle_line(state, "my goal is data analyst", runner=_runner(esco, onet, draft=draft))

    profile = _profile()
    assert "GOAL: data analyst" in profile
    assert "STANDING" not in profile
