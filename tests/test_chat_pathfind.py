"""A pathfind waiting on a pick resumes after it (PF-3)."""

from __future__ import annotations

from talent_angels.assistant.llm_plan import PlanDraft
from talent_angels.assistant.planning import build_plan_for_capability
from talent_angels.assistant.turn import TurnOutcome
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.runlog import RunLogRecord
from talent_angels.session.kernel import handle_line
from talent_angels.session.store import new_session


def _node(suite: str, label: str, kind: str = "Occupation") -> NodeRef:
    return NodeRef(
        id=f"{suite}:{kind.casefold()}:{label}",
        suite=suite,
        source=suite,
        source_id=label,
        kind=kind,
        pref_label=label,
    )


def _resolved_pathfind(suite: str, start: str, target: str) -> AgentResult:
    return AgentResult(
        capability="pathfind",
        suite=suite,
        nodes=[_node(suite, start), _node(suite, target)],
    )


def test_a_pathfind_waiting_on_a_pick_runs_after_it() -> None:
    nurses = AgentResult(
        capability="pathfind",
        suite="esco",
        nodes=[_node("esco", "nurse"), _node("esco", "paediatric nurse")],
        warnings=["ambiguous", "pathfind_side:start"],
    )
    asked: list[str] = []

    def runner(question: str, **_kwargs: object) -> TurnOutcome:
        asked.append(question)
        result = nurses if len(asked) == 1 else _resolved_pathfind("esco", "nurse", "midwife")
        return TurnOutcome(
            capability="pathfind",
            plan=build_plan_for_capability("pathfind", suites=("esco",)),
            result=result,
            results=(result,),
            answer="ignored",
            record=RunLogRecord(suite="esco", plan=["pathfind"], question=question),
            plan_draft=PlanDraft(target="pathfind", subject="nurse", secondary_subject="midwife"),
        )

    state = new_session()
    handle_line(state, "path from nurse to midwife", runner=runner)
    assert state.pending_pathfind == {"esco": ["nurse", "midwife", "start"]}

    reply = handle_line(state, "1", runner=runner)

    assert asked[-1] == "path from nurse to midwife"
    assert state.pending_pathfind == {}
    assert reply.text


def test_a_pathfind_waiting_on_the_target_side_resumes_with_the_chosen_target() -> None:
    midwives = AgentResult(
        capability="pathfind",
        suite="esco",
        nodes=[_node("esco", "midwife"), _node("esco", "paediatric midwife")],
        warnings=["ambiguous", "pathfind_side:target"],
    )
    asked: list[str] = []

    def runner(question: str, **_kwargs: object) -> TurnOutcome:
        asked.append(question)
        result = midwives if len(asked) == 1 else _resolved_pathfind("esco", "nurse", "midwife")
        return TurnOutcome(
            capability="pathfind",
            plan=build_plan_for_capability("pathfind", suites=("esco",)),
            result=result,
            results=(result,),
            answer="ignored",
            record=RunLogRecord(suite="esco", plan=["pathfind"], question=question),
            plan_draft=PlanDraft(target="pathfind", subject="nurse", secondary_subject="midwife"),
        )

    state = new_session()
    handle_line(state, "path from nurse to midwife", runner=runner)
    assert state.pending_pathfind == {"esco": ["nurse", "midwife", "target"]}

    handle_line(state, "1", runner=runner)

    assert asked[-1] == "path from nurse to midwife"
    assert state.pending_pathfind == {}
