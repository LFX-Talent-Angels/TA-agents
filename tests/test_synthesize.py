"""One answer from N attached suites; a third suite appears without new branches."""

from __future__ import annotations

from pathlib import Path

import pytest

from talent_angels.assistant import synthesize as synth
from talent_angels.assistant.synthesize import sources_line, synthesize, synthesize_structured
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.llm.protocol import LLMResult, LLMUsage
from talent_angels.memory.episodes import record_episode
from talent_angels.runlog.models import ResultSummary, RunLogRecord


def _occ(suite: str, node_id: str, label: str) -> AgentResult:
    return AgentResult(
        capability="locate",
        suite=suite,
        nodes=[
            NodeRef(
                id=node_id,
                suite=suite,
                source=suite,
                source_id=node_id,
                kind="Occupation",
                pref_label=label,
            )
        ],
        confidence=0.95,
    )


def test_two_suites_one_answer_and_sources() -> None:
    answer = synthesize_structured(
        (
            _occ("esco", "esco:occupation:dev", "software developer"),
            _occ("onet", "onet:occupation:15-1252.00", "Software Developers"),
        )
    )
    assert "software developer" in answer
    assert "Software Developers" in answer
    assert "Sources used: ESCO · O*NET" in answer
    assert "not one shared id" in answer


def test_third_suite_appears_in_sources_without_hardcoding() -> None:
    answer = synthesize_structured(
        (
            _occ("esco", "esco:occupation:dev", "software developer"),
            _occ("onet", "onet:occupation:15-1252.00", "Software Developers"),
            _occ("sfia", "sfia:skill:prog", "Programming"),
        )
    )
    line = sources_line(
        (
            _occ("esco", "esco:occupation:dev", "software developer"),
            _occ("onet", "onet:occupation:15-1252.00", "Software Developers"),
            _occ("sfia", "sfia:skill:prog", "Programming"),
        )
    )
    assert "SFIA" in line
    assert "Programming" in answer
    assert "Sources used: ESCO · O*NET · SFIA" in answer


# --------------------------------------------------------------------------
# The fourth site that renders `recall_prefix`.
#
# The other three (agent_loop, phrase_chat, phrase_map) were wired when recall
# landed and this one was missed: `synthesize` builds a system prompt from
# `profile_prefix() + notes_prefix() + _SYNTH_SYSTEM` with nothing in between, so
# a user whose question overlaps an earlier turn got their own history in the
# prompt everywhere except the one place their words are restated back to them.
#
# These tests exist because the gap was invisible from the code — the function
# looked complete, and the suite was green, because nothing asserted that
# recall appears in the prompt it builds.
# --------------------------------------------------------------------------


class _Scripted:
    provider = "litellm"
    model = "test-synth"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[list] = []

    def complete(self, messages: list, **_kwargs: object) -> LLMResult:
        self.calls.append(messages)
        return LLMResult(text=self.text, provider=self.provider, model=self.model, usage=LLMUsage())


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A throwaway episode database, so the retriever under test is the real one."""
    return tmp_path / "memory.db"


def _episode(db: Path, run_id: str, question: str, *labels: str) -> None:
    record_episode(
        RunLogRecord(
            run_id=run_id,
            ts="2026-01-01T00:00:00+00:00",
            suite="esco",
            plan=["locate"],
            question=question,
            result=ResultSummary(node_ids=[f"esco:occupation:{run_id}"], node_labels=list(labels)),
        ),
        db_path=db,
    )


def _system_prompt(client: _Scripted) -> str:
    assert client.calls, "the model was never called, so there is no prompt to check"
    return client.calls[0][0].content


def test_synthesize_never_shows_past_turns_to_the_model(db_path: Path) -> None:
    """A reply prompt carries no earlier turns, even with recall on.

    Given them, the model told a user "you asked which is most important and
    the answer was X" about an exchange that never happened (2026-10-08).
    "What did we talk about?" is answered in code (session.recall) instead.
    """
    _episode(db_path, "old-1", "what skills does a nurse need?", "nurse")
    client = _Scripted("A nurse needs a licence and a training certificate.")
    answer = synthesize(
        (_occ("esco", "esco:occupation:dev", "nurse"),),
        question="what training does a nurse need",
        llm_client=client,
    )
    assert answer.startswith("A nurse needs")
    system = _system_prompt(client)
    assert "skills does a nurse need" not in system
    assert system.endswith(synth._SYNTH_SYSTEM)


def test_synthesize_short_circuits_before_the_model(db_path: Path) -> None:
    """The no-results fallback must not reach the model."""
    client = _Scripted("should not be called")
    answer = synthesize((), question="anything", llm_client=client)
    assert client.calls == []
    assert answer == synthesize_structured((), extra_warnings=())


def test_synthesize_with_no_question_still_answers(db_path: Path) -> None:
    """`question` defaults to `""`, and callers that omit it are unaffected."""
    client = _Scripted("A nurse needs a licence.")
    answer = synthesize((_occ("esco", "esco:occupation:dev", "nurse"),), llm_client=client)
    assert answer.startswith("A nurse needs a licence.")
    assert _system_prompt(client).endswith(synth._SYNTH_SYSTEM)
