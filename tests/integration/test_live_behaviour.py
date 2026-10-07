"""Live behaviour of the chat kernel: full graphs, a real model, isolated memory.

Each case drives ``session.kernel.handle_line`` the way the TUI does, one line
at a time, and asserts what a user would notice: what got bound, what the
profile now says, whether a saved session survives.

Needs a reachable Neo4j with the full graphs and a configured LLM provider;
skips with a reason otherwise. Memory paths come from the autouse isolation in
``tests/conftest.py``, so nothing here touches the developer's real profile.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from talent_angels.assistant.turn import persist_turn_record, run_turn
from talent_angels.env import load_local_dotenv
from talent_angels.llm import LLMClient
from talent_angels.llm.factory import get_llm_client
from talent_angels.llm.protocol import uses_chat_phrasing
from talent_angels.memory.paths import user_md
from talent_angels.session.kernel import ChatReply, handle_line
from talent_angels.session.models import SessionState
from talent_angels.session.store import new_session, save_session
from talent_angels.suites import SuiteRegistry, default_suite_registry
from tests.integration.support import neo4j_reachable

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def live_env() -> Iterator[tuple[SuiteRegistry, LLMClient]]:
    saved = dict(os.environ)
    load_local_dotenv()
    try:
        if not neo4j_reachable():
            pytest.skip("Neo4j is not reachable; start it and load the full graphs")
        client = get_llm_client()
        if not uses_chat_phrasing(client):
            pytest.skip("No LLM provider configured (LLM_PROVIDER=none)")
        yield default_suite_registry(), client
    finally:
        os.environ.clear()
        os.environ.update(saved)


class Chat:
    """One TUI-like conversation: same runner, same persistence, no console."""

    def __init__(self, registry: SuiteRegistry, client: LLMClient) -> None:
        self.registry = registry
        self.client = client
        self.state: SessionState = new_session()

    def say(self, line: str) -> ChatReply:
        def runner(question, *, bound_node=None, bound_nodes=None, force_capability=None):
            outcome = run_turn(
                registry=self.registry,
                llm_client=self.client,
                question=question,
                answer_mode="structured",
                bound_node=bound_node,
                bound_nodes=bound_nodes,
                force_capability=force_capability,
                persist=False,
            )
            persist_turn_record(outcome.record)
            return outcome

        reply = handle_line(self.state, line, runner=runner, llm_client=self.client)
        save_session(self.state)
        return reply

    def bound(self, suite: str) -> str | None:
        node = self.state.bindings.get(suite)
        return node.pref_label if node else None


def profile() -> str:
    path = user_md()
    return path.read_text(encoding="utf-8") if path.exists() else ""


@pytest.fixture
def chat(live_env: tuple[SuiteRegistry, LLMClient]) -> Chat:
    return Chat(*live_env)


# --- Baseline: behaviour that works today and must keep working ------------


def test_exact_title_binds_in_both_suites(chat: Chat) -> None:
    chat.say("chef")
    assert chat.bound("esco") == "chef"
    assert chat.bound("onet") == "Chefs and Head Cooks"


def test_follow_up_lists_skills_of_the_bound_occupation(chat: Chat) -> None:
    chat.say("chef")
    reply = chat.say("what skills does it need?")
    assert "food" in reply.text.casefold()


def test_gibberish_binds_nothing(chat: Chat) -> None:
    chat.say("asdfgh qwerty")
    assert chat.state.bindings == {}


# --- Bugs found in live testing on 2026-10-07 ---------------------------------


def test_lookup_is_not_a_statement_about_the_user(chat: Chat) -> None:
    chat.say("firefighter")
    assert "STANDING" not in profile()


def test_goal_does_not_set_current_job(chat: Chat) -> None:
    chat.say("my goal is data analyst")
    text = profile()
    assert "GOAL: data analyst" in text
    assert "STANDING" not in text


def test_i_am_statement_sets_current_job(chat: Chat) -> None:
    chat.say("chef")
    chat.say("I am a plumber")
    text = profile()
    assert "STANDING[esco]: plumber" in text
    assert "Chefs and Head Cooks" not in text


def test_ambiguous_i_am_statement_is_recorded_on_pick(chat: Chat) -> None:
    chat.say("I am a teacher")
    assert "STANDING" not in profile()
    chat.say("1")
    assert "STANDING[esco]:" in profile()


def test_want_to_become_saves_goal(chat: Chat) -> None:
    chat.say("I want to become a web developer")
    assert "GOAL: web developer" in profile()


def test_asking_for_skills_to_become_x_still_lists_skills(chat: Chat) -> None:
    reply = chat.say("what skills do I need to become a data analyst?")
    assert chat.bound("esco") == "data analyst"
    assert "data" in reply.text.casefold()


def test_reset_keeps_a_saved_session(chat: Chat) -> None:
    chat.say("chef")
    chat.say("/save chefwork")
    chat.say("/reset")
    reply = chat.say("/resume chefwork")
    assert "Resumed" in reply.text
    assert chat.bound("esco") == "chef"


def test_ai_engineer_is_not_an_insemination_technician(chat: Chat) -> None:
    chat.say("AI engineer")
    assert chat.bound("esco") != "animal artificial insemination technician"


def test_bare_engineer_asks_which_one(chat: Chat) -> None:
    chat.say("engineer")
    assert chat.bound("esco") is None


def test_it_manager_is_not_a_credit_manager(chat: Chat) -> None:
    chat.say("IT manager")
    assert chat.bound("esco") != "credit manager"
