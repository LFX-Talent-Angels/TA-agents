"""Advice questions grounded in the profile and the map, never a choice made for the user."""

from __future__ import annotations

import pytest

from talent_angels.assistant.planning import build_plan_for_capability
from talent_angels.assistant.turn import TurnOutcome
from talent_angels.contracts import AgentResult
from talent_angels.memory.paths import user_md
from talent_angels.memory.profile import profile_titles
from talent_angels.runlog import RunLogRecord
from talent_angels.session.advice import advice_plan
from talent_angels.session.kernel import handle_line
from talent_angels.session.router import route_line
from talent_angels.session.store import new_session


@pytest.mark.parametrize(
    ("text", "question"),
    [
        ("should I be a nurse or a midwife?", "compare nurse and midwife"),
        (
            "should I become an accountant or a software developer",
            "compare accountant and software developer",
        ),
    ],
)
def test_either_or_becomes_a_compare(text: str, question: str) -> None:
    plan = advice_plan(text, current=None, goal=None)
    assert plan is not None and plan.question == question
    assert "can't choose for you" in plan.preface


def test_learn_first_with_job_and_goal_compares_them() -> None:
    plan = advice_plan("what should I learn first?", current="dance teacher", goal="data analyst")
    assert plan is not None
    assert plan.question == "compare dance teacher and data analyst"
    assert '"Only data analyst"' in plan.preface


def test_learn_first_with_only_a_goal_lists_the_goal_skills() -> None:
    plan = advice_plan("what should I learn first?", current=None, goal="web developer")
    assert plan is not None
    assert plan.question == "what are the essential skills of a web developer?"
    assert "I am a" in plan.preface


@pytest.mark.parametrize("text", ["what should I learn first?", "should I quit my job?"])
def test_nothing_to_ground_keeps_refusing(text: str) -> None:
    assert advice_plan(text, current=None, goal=None) is None


def test_missing_skills_for_my_goal_is_advice() -> None:
    assert route_line("what skills am I missing for my goal?").kind == "advice"


def test_profile_titles_reads_current_job_and_goal() -> None:
    user_md().parent.mkdir(parents=True, exist_ok=True)
    user_md().write_text(
        "STANDING[onet]: Dancers  [onet:27-2031.00]   since: 2026-10-07\n"
        "STANDING[esco]: dance teacher  [esco:1eb5]   since: 2026-10-07\n"
        "GOAL: data analyst  [esco:d3ed]\n",
        encoding="utf-8",
    )
    assert profile_titles() == ("dance teacher", "data analyst")


def test_chat_compares_current_job_and_goal_for_learn_first() -> None:
    user_md().parent.mkdir(parents=True, exist_ok=True)
    user_md().write_text(
        "STANDING[esco]: dance teacher  [esco:1eb5]   since: 2026-10-07\n"
        "GOAL: data analyst  [esco:d3ed]\n",
        encoding="utf-8",
    )
    asked: list[str] = []

    def runner(question: str, **_kwargs: object) -> TurnOutcome:
        asked.append(question)
        result = AgentResult(capability="locate", suite="esco", warnings=["not_found"])
        return TurnOutcome(
            capability="locate",
            plan=build_plan_for_capability("locate", suites=("esco",)),
            result=result,
            answer="ignored",
            record=RunLogRecord(suite="esco", plan=["locate"], question=question),
        )

    reply = handle_line(new_session(), "what should I learn first?", runner=runner)

    assert asked == ["compare dance teacher and data analyst"]
    assert reply.text.startswith("Your profile says you are a **dance teacher**")


def test_which_one_after_a_compare_compares_that_pair_again() -> None:
    from talent_angels.session.advice import advice_plan

    plan = advice_plan(
        "which one should I choose?", current=None, goal=None, compared=("accountant", "chef")
    )
    assert plan is not None
    assert plan.question == "compare accountant and chef"
    assert "can't choose" in plan.preface
    assert advice_plan("which one should I choose?", current=None, goal=None) is None
