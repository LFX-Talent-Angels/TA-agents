"""Tests for injecting the saved USER.md profile into chat phrasing hints."""

from __future__ import annotations

from pathlib import Path

import pytest

from talent_angels.llm.protocol import LLMResult, LLMUsage, Message
from talent_angels.memory import confirm_goal, confirm_standing
from talent_angels.session.kernel import _profile_hint, handle_line
from talent_angels.session.store import new_session


def _boom(question: str, **kwargs: object):
    raise AssertionError(f"runner must not be called, got {question!r}")


@pytest.fixture(autouse=True)
def _local_memory_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TA_MEMORY_DIR", str(tmp_path / "memory"))


def test_profile_hint_is_empty_when_nothing_is_saved() -> None:
    assert _profile_hint() == ""


def test_profile_hint_mentions_standing_and_goal() -> None:
    confirm_standing("Software Developers", node_id="onet:15-1252.00")
    confirm_goal("Data Scientists", node_id="onet:15-2051.00")

    hint = _profile_hint()

    assert "current role Software Developers" in hint
    assert "target role Data Scientists" in hint


def test_profile_hint_labels_itself_as_self_reported_not_a_taxonomy_fact() -> None:
    confirm_standing("Software Developers")

    hint = _profile_hint()

    assert "not a taxonomy fact" in hint


class _RecordingClient:
    provider = "litellm"

    def __init__(self) -> None:
        self.messages: list[Message] = []

    def complete(self, messages: list[Message], **_kwargs: object) -> LLMResult:
        self.messages.extend(messages)
        return LLMResult(text="Hi there.", provider=self.provider, model="test", usage=LLMUsage())


def test_greeting_carries_the_profile_hint_into_the_system_message() -> None:
    confirm_standing("Software Developers", node_id="onet:15-1252.00")
    client = _RecordingClient()

    handle_line(new_session(), "hi", runner=_boom, llm_client=client)

    system_messages = [m.content for m in client.messages if m.role == "system"]
    assert any("current role Software Developers" in text for text in system_messages)


def test_greeting_has_no_profile_text_when_nothing_is_saved() -> None:
    client = _RecordingClient()

    handle_line(new_session(), "hi", runner=_boom, llm_client=client)

    system_messages = [m.content for m in client.messages if m.role == "system"]
    assert not any("current role" in text for text in system_messages)
