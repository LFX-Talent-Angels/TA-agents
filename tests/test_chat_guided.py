"""Guided search in chat: areas under a broad list, area letters, hint narrowing."""

from __future__ import annotations

import json

from talent_angels.assistant.planning import build_plan_for_capability
from talent_angels.assistant.turn import TurnOutcome
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.llm.protocol import LLMResult, LLMUsage, Message
from talent_angels.runlog import RunLogRecord
from talent_angels.session.kernel import handle_line
from talent_angels.session.models import PendingChoice
from talent_angels.session.narrow import (
    area_by_letter,
    area_choices,
    narrow_decision,
    render_areas,
)
from talent_angels.session.store import load_session, new_session, save_session
from talent_angels.skills.locate.areas import Area


def _node(label: str, suite: str = "esco") -> NodeRef:
    return NodeRef(
        id=f"{suite}:occ:{label}",
        suite=suite,
        source=suite,
        source_id=label,
        kind="Occupation",
        pref_label=label,
    )


TITLES = ["test engineer", "civil engineer", "mechanical engineer", "marine engineer"]
AREAS = {
    "esco": [
        Area(suite="esco", code="2144", label="Mechanical engineers", count=24),
        Area(suite="esco", code="2142", label="Civil engineers", count=11),
    ]
}


class _Runner:
    """Answers "engineer" broadly with areas; an area request with its titles."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, question: str, **kwargs: object) -> TurnOutcome:
        self.calls.append((question, kwargs))
        area = kwargs.get("area")
        if area is not None:
            result = AgentResult(
                capability="locate",
                suite="esco",
                nodes=[_node("civil engineer"), _node("bridge engineer")],
                warnings=["ambiguous"],
            )
            areas: dict[str, list[Area]] = {}
        else:
            result = AgentResult(
                capability="locate",
                suite="esco",
                nodes=[_node(t) for t in TITLES],
                warnings=["ambiguous", "truncated"],
            )
            areas = AREAS
        return TurnOutcome(
            capability="locate",
            plan=build_plan_for_capability("locate", suites=("esco",)),
            result=result,
            answer="ignored",
            record=RunLogRecord(suite="esco", plan=["locate"], question=question),
            areas=areas,
        )


def _broad_list() -> tuple[object, _Runner]:
    state = new_session()
    runner = _Runner()
    reply = handle_line(state, "engineer", runner=runner)
    return (state, reply), runner


# -- rendering and letters -------------------------------------------------------


def test_area_choices_letter_the_first_suite_with_areas() -> None:
    choices = area_choices({"onet": [], **AREAS}, query="engineer", kind="occupation")
    assert [(c.letter, c.code, c.query) for c in choices] == [
        ("A", "2144", "engineer"),
        ("B", "2142", "engineer"),
    ]
    text = render_areas(choices)
    assert "A. Mechanical engineers (24 titles)" in text
    assert "Reply with a letter" in text


def test_area_by_letter_accepts_a_letter_or_area_letter() -> None:
    choices = area_choices(AREAS, query="engineer", kind="occupation")
    assert area_by_letter("b", choices) == choices[1]
    assert area_by_letter("area A.", choices) == choices[0]
    assert area_by_letter("Z", choices) is None
    assert area_by_letter("bridges", choices) is None


# -- deciding what a message means while a list is open ---------------------------


class _Scripted:
    provider = "litellm"
    model = "test"

    def __init__(self, payload: dict[str, object] | str) -> None:
        self.text = payload if isinstance(payload, str) else json.dumps(payload)
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> LLMResult:
        self.calls.append(messages)
        return LLMResult(text=self.text, provider="litellm", model="m", usage=LLMUsage())


_OPTIONS = [PendingChoice(number=i, node=_node(t)) for i, t in enumerate(TITLES, start=1)]


def test_model_may_only_point_at_what_is_shown() -> None:
    choices = area_choices(AREAS, query="engineer", kind="occupation")
    client = _Scripted({"action": "narrow", "options": [2, 99, "3"], "areas": ["b", "Z"]})
    decision = narrow_decision(client, "the ones that build bridges", _OPTIONS, choices)  # type: ignore[arg-type]
    assert decision.options == [2]
    assert decision.areas == ["B"]
    assert "1. test engineer" in client.calls[0][1].content


def test_unreadable_model_answer_falls_back_to_word_overlap() -> None:
    decision = narrow_decision(_Scripted("sure!"), "the civil ones", _OPTIONS, [])  # type: ignore[arg-type]
    assert (decision.action, decision.options) == ("narrow", [2])


def test_without_a_model_only_hint_shaped_text_narrows() -> None:
    assert narrow_decision(None, "the marine ones", _OPTIONS, []).options == [4]
    assert (
        narrow_decision(None, "what skills does a civil engineer need", _OPTIONS, []).action
        == "new"
    )


# -- the chat flow ------------------------------------------------------------------


def test_broad_list_offers_areas_and_remembers_them() -> None:
    (state, reply), _ = _broad_list()
    assert "Or narrow it down by area" in reply.text
    assert [a.letter for a in state.areas] == ["A", "B"]
    assert state.areas[0].query == "engineer"


def test_area_letter_reruns_the_search_inside_the_area() -> None:
    (state, _), runner = _broad_list()
    reply = handle_line(state, "B", runner=runner)
    question, kwargs = runner.calls[-1]
    assert question == "engineer"
    assert kwargs["area"].code == "2142"  # type: ignore[union-attr]
    assert [c.node.pref_label for c in state.pending] == ["civil engineer", "bridge engineer"]
    assert state.areas == []
    assert "Civil engineers" in reply.text
    assert not state.bindings


def test_hint_narrows_the_open_list_without_a_new_search() -> None:
    (state, _), runner = _broad_list()
    reply = handle_line(state, "the marine ones", runner=runner)
    assert len(runner.calls) == 1
    assert [c.node.pref_label for c in state.pending] == ["marine engineer"]
    assert [c.number for c in state.pending] == [1]
    assert "reply with 1" in reply.text
    assert not state.bindings


def test_hint_naming_an_area_adds_that_area_s_titles() -> None:
    (state, _), runner = _broad_list()
    handle_line(state, "the civil ones", runner=runner)
    assert runner.calls[-1][1]["area"].code == "2142"  # type: ignore[union-attr]
    # The listed match first, then the rest of the area, once each.
    assert [c.node.pref_label for c in state.pending] == ["civil engineer", "bridge engineer"]


def test_a_new_question_still_searches() -> None:
    (state, _), runner = _broad_list()
    handle_line(state, "what does a nurse do", runner=runner)
    assert runner.calls[-1][0] == "what does a nurse do"
    assert "area" not in runner.calls[-1][1] or runner.calls[-1][1]["area"] is None


def test_picking_clears_the_areas() -> None:
    (state, _), runner = _broad_list()
    handle_line(state, "2", runner=runner)
    assert state.bindings["esco"].pref_label == "civil engineer"
    assert state.areas == []


def test_areas_survive_save_and_resume() -> None:
    (state, _), _ = _broad_list()
    state.name = "guided"
    save_session(state)
    assert [a.label for a in load_session("guided").areas] == [
        "Mechanical engineers",
        "Civil engineers",
    ]


def test_hint_that_fits_nothing_searches_again_keeping_the_topic() -> None:
    (state, _), runner = _broad_list()
    state.areas = []  # a list without areas still has a topic
    client = _Scripted({"action": "narrow", "options": [], "areas": []})
    handle_line(state, "the zoo ones", runner=runner, llm_client=client)  # type: ignore[arg-type]
    assert runner.calls[-1][0] == "the zoo ones, in engineer"


def test_a_statement_about_the_user_is_never_a_hint() -> None:
    (state, _), runner = _broad_list()
    handle_line(state, "I am not a marine engineer", runner=runner)
    assert runner.calls[-1][0] == "I am not a marine engineer"


def test_after_a_pick_a_sentence_is_a_new_request() -> None:
    (state, _), runner = _broad_list()
    handle_line(state, "2", runner=runner)
    handle_line(state, "the marine ones", runner=runner)
    assert runner.calls[-1][0] == "the marine ones"


def test_typing_a_listed_title_picks_it() -> None:
    (state, _), runner = _broad_list()
    client = _Scripted({"action": "narrow", "options": [4], "areas": []})
    reply = handle_line(state, "marine engineers", runner=runner, llm_client=client)  # type: ignore[arg-type]
    assert reply.text == "Bound marine engineer."
    assert state.bindings["esco"].pref_label == "marine engineer"


def test_thanks_with_a_tail_is_not_a_search() -> None:
    from talent_angels.session.router import route_line

    assert route_line("thanks, that was helpful").kind == "greet"
    assert route_line("thank you so much!").kind == "greet"
    assert route_line("thanks, what skills does a nurse need?").kind != "greet"


def test_profile_question_is_answered_from_the_profile_in_code() -> None:
    from talent_angels.memory.profile import write_goal, write_rejected

    state = new_session()
    empty = handle_line(state, "what do you know about me?", runner=_Runner())
    assert "don't have much about you yet" in empty.text
    write_goal(_node("data scientist"))
    write_rejected(_node("nurse assistant"))
    reply = handle_line(state, "what do you know about me?", runner=_Runner())
    assert "- Goal: data scientist" in reply.text
    assert "- Not your job (you said so): nurse assistant" in reply.text
    assert "esco:" not in reply.text
