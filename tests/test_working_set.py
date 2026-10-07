"""The working set: recent titles, and pair follow-ups rewritten in code."""

from __future__ import annotations

import pytest

from talent_angels.assistant.llm_plan import PlanDraft
from talent_angels.assistant.planning import build_plan_for_capability
from talent_angels.assistant.turn import TurnOutcome
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.runlog import RunLogRecord
from talent_angels.session.kernel import handle_line
from talent_angels.session.store import new_session
from talent_angels.session.working_set import RECENT_LIMIT, pair_followup, remember


def test_remember_moves_a_repeat_to_newest_and_caps() -> None:
    recent: list[str] = []
    for title in ["chef", "baker", "Chef"]:
        recent = remember(recent, title)
    assert recent == ["baker", "Chef"]
    for i in range(RECENT_LIMIT + 2):
        recent = remember(recent, f"job {i}")
    assert len(recent) == RECENT_LIMIT
    assert recent[-1] == f"job {RECENT_LIMIT + 1}"


@pytest.mark.parametrize(
    "text",
    [
        "compare the two",
        "compare them",
        "what do they have in common?",
        "how do they differ?",
        "what is the difference between them?",
        "help me choose between them",
        "which of the two needs more programming?",
        "what do both of them share",
    ],
)
def test_pair_followups_name_the_last_two_titles(text: str) -> None:
    assert pair_followup(text, ["nurse", "chef", "baker"]) == "compare chef and baker"


def test_compare_it_with_a_new_title() -> None:
    assert pair_followup("compare it with a baker", ["chef"]) == "compare chef and baker"


@pytest.mark.parametrize("text", ["what skills does it need?", "chef", "compare chef and baker"])
def test_other_lines_are_left_alone(text: str) -> None:
    assert pair_followup(text, ["chef", "baker"]) is None


def test_a_pair_needs_two_titles() -> None:
    assert pair_followup("compare the two", ["chef"]) is None


def _occupation(label: str) -> NodeRef:
    return NodeRef(
        id=f"esco:occupation:{label}",
        suite="esco",
        source="esco",
        source_id=label,
        kind="Occupation",
        pref_label=label,
    )


class _Recorder:
    """Locates whatever subject the planner would extract; keeps every question."""

    def __init__(self) -> None:
        self.questions: list[str] = []

    def __call__(self, question: str, **_kwargs: object) -> TurnOutcome:
        self.questions.append(question)
        subject = question.removeprefix("I looked up ")
        result = AgentResult(capability="locate", suite="esco", nodes=[_occupation(subject)])
        return TurnOutcome(
            capability="locate",
            plan=build_plan_for_capability("locate", suites=("esco",)),
            result=result,
            answer="ignored",
            plan_draft=PlanDraft(target="locate", subject=subject),
            record=RunLogRecord(suite="esco", plan=["locate"], question=question),
        )


def test_compare_the_two_after_two_lookups_asks_for_both_titles() -> None:
    state = new_session()
    runner = _Recorder()

    handle_line(state, "accountant", runner=runner)
    handle_line(state, "software developer", runner=runner)
    assert state.recent == ["accountant", "software developer"]
    handle_line(state, "compare the two", runner=runner)

    assert runner.questions[-1] == "compare accountant and software developer"
    assert state.transcript[-2].text == "compare the two"
