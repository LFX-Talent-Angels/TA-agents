"""Recall of what was looked at: this conversation first, then earlier ones."""

from __future__ import annotations

import pytest

from talent_angels.memory.episodes import Episode
from talent_angels.session.kernel import handle_line
from talent_angels.session.recall import EARLIER_LIMIT, recall_reply
from talent_angels.session.router import route_line
from talent_angels.session.store import new_session


def _episode(label: str, *, satisfied: bool = True) -> Episode:
    return Episode(
        run_id=label,
        ts="2026-10-05T10:00:00+00:00",
        suite="esco",
        capability="locate",
        plan=("locate",),
        question=label,
        node_ids=(f"esco:occupation:{label}",),
        node_labels=(label,),
        warnings=(),
        satisfied=satisfied,
    )


@pytest.mark.parametrize(
    "text",
    [
        "what did we talk about?",
        "what did we talk about before?",
        "remind me what we discussed",
        "what have we looked at",
        "what jobs did I look at?",
        "what was the first job I asked about in this conversation?",
        "what was the last occupation we looked at",
    ],
)
def test_recall_questions_are_routed_to_recall(text: str) -> None:
    assert route_line(text).kind == "recall"


@pytest.mark.parametrize("text", ["what does a chef do?", "what did a nurse need", "chef"])
def test_other_questions_are_not_recall(text: str) -> None:
    assert route_line(text).kind != "recall"


def test_lists_this_conversation_then_earlier_titles_once() -> None:
    episodes = [
        _episode("chef"),
        _episode("marine biologist"),
        _episode("teacher", satisfied=False),
    ]
    reply = recall_reply("what did we talk about?", ["chef", "baker"], episodes)
    assert "In this conversation you looked at: chef, baker." in reply
    assert "Earlier you looked at: marine biologist." in reply
    assert "teacher" not in reply


def test_first_and_last_title_of_this_conversation() -> None:
    recent = ["data scientist", "statistician"]
    assert "**data scientist**" in recall_reply("what was the first job I asked about", recent, [])
    assert "**statistician**" in recall_reply("what was the last job we looked at", recent, [])


def test_earlier_titles_are_capped() -> None:
    episodes = [_episode(f"job {i}") for i in range(EARLIER_LIMIT + 3)]
    reply = recall_reply("what did we talk about?", [], episodes)
    assert reply.count("job ") == EARLIER_LIMIT


def test_nothing_to_recall() -> None:
    assert "haven't looked at any" in recall_reply("what did we talk about?", [], [])


def test_chat_answers_recall_from_the_working_set_without_a_search() -> None:
    state = new_session()
    state.recent = ["accountant", "software developer"]

    def no_search(question: str, **_kwargs: object):
        raise AssertionError("recall must not search the graph")

    reply = handle_line(state, "what did we talk about?", runner=no_search)

    assert "accountant, software developer" in reply.text
