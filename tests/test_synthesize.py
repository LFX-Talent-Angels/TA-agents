"""One answer from N attached suites; a third suite appears without new branches."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from talent_angels.assistant import synthesize as synth
from talent_angels.assistant.synthesize import sources_line, synthesize, synthesize_structured
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.llm.protocol import LLMResult, LLMUsage
from talent_angels.memory.episodes import record_episode
from talent_angels.memory.fts_retriever import Fts5EpisodeRetriever
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


def test_synthesize_recalls_earlier_turns(db_path: Path) -> None:
    """The prompt carries what the user asked before, between the notes and the brief.

    Ordering is part of the contract, not decoration: recall has to sit with the
    other user-context blocks, *before* `_SYNTH_SYSTEM`, so the instructions read
    as instructions about content the model has already been given. Asserted
    against the real retriever over a real database — a stubbed `recall_prefix`
    would pass here and fail in production, which is how the gap survived the
    first three wirings.
    """
    _episode(db_path, "old-1", "what skills does a nurse need?", "nurse")
    client = _Scripted("A nurse needs a licence and a training certificate.")

    with patch.object(synth, "episode_retriever", lambda: Fts5EpisodeRetriever(db_path=db_path)):
        answer = synthesize(
            (_occ("esco", "esco:occupation:dev", "nurse"),),
            question="what training does a nurse need",
            llm_client=client,
        )

    assert answer.startswith("A nurse needs")
    system = _system_prompt(client)
    assert "nurse" in system.split(synth._SYNTH_SYSTEM)[0], "no recalled turn in the prompt"
    assert "old-1" not in system, "the prefix renders a summary, not raw run ids"
    # The brief still comes last, after the context blocks.
    assert system.index("skills does a nurse need") < system.index(synth._SYNTH_SYSTEM)


def test_synthesize_recalls_on_the_question_not_the_fact_card(db_path: Path) -> None:
    """Retrieval is keyed to the user's words, not to the taxonomy labels we matched.

    The fact card is full of node ids, `kind=occupation` and pref_labels, all of
    which are in the index. Recalling on it would return the turns that happen to
    share a label with the answer instead of the turn about the question — the
    same mistake already documented on the agent loop's call, asserted here so
    the two sites cannot drift apart.
    """
    _episode(db_path, "old-1", "how do i become a plumber?", "plumber")
    client = _Scripted("Plumbing needs an apprenticeship.")

    with patch.object(synth, "episode_retriever", lambda: Fts5EpisodeRetriever(db_path=db_path)):
        synthesize(
            (_occ("esco", "esco:occupation:dev", "plumber"),),
            question="how do i become a plumber",
            llm_client=client,
        )

    recalled = _system_prompt(client).split(synth._SYNTH_SYSTEM)[0]
    assert "how do i become a plumber?" in recalled
    assert "esco:occupation:dev" not in recalled, "recalled on the card, not the question"
    assert "nurse" not in recalled, "a turn about an unrelated topic leaked in"


def test_synthesize_prompt_is_unchanged_when_recall_is_off(db_path: Path) -> None:
    """No retriever means no recall block, so the prompt is byte-identical to before.

    `episode_retriever()` returns a `NullRetriever` when recall is not configured,
    and `recall_prefix` then returns `""`. The opt-in has to stay opt-in: a user
    who has not enabled memory should not be sent a block that says so.
    """
    _episode(db_path, "old-1", "what skills does a nurse need?", "nurse")
    client = _Scripted("A nurse needs a licence.")

    with patch.object(synth, "episode_retriever", lambda: None):
        synthesize(
            (_occ("esco", "esco:occupation:dev", "nurse"),),
            question="what training does a nurse need",
            llm_client=client,
        )

    system = _system_prompt(client)
    assert "nurse needs" not in system.split(synth._SYNTH_SYSTEM)[0]
    assert system.endswith(synth._SYNTH_SYSTEM)


def test_synthesize_short_circuits_before_recalling(db_path: Path) -> None:
    """The no-results fallback must not reach the retriever or the model.

    Asserted by the absence of a call rather than by the returned string,
    because the interesting failure is a *new* code path running the model on an
    empty fact card, which no assertion about the answer text would catch.
    """
    _episode(db_path, "old-1", "what skills does a nurse need?", "nurse")
    client = _Scripted("should not be called")
    with patch.object(synth, "episode_retriever", lambda: Fts5EpisodeRetriever(db_path=db_path)):
        answer = synthesize((), question="anything", llm_client=client)

    assert client.calls == []
    assert answer == synthesize_structured((), extra_warnings=())


def test_synthesize_with_no_question_still_answers(db_path: Path) -> None:
    """`question` defaults to `""`, and callers that omit it must be unaffected.

    Worth pinning because the recall wiring made the prompt depend on a parameter
    that did not exist on this function before. An empty question retrieves
    nothing — correctly, since there is nothing to retrieve on — and the prompt
    falls back to being exactly what it was before this change, with the model
    still called.
    """
    _episode(db_path, "old-1", "what skills does a nurse need?", "nurse")
    client = _Scripted("A nurse needs a licence.")
    with patch.object(synth, "episode_retriever", lambda: Fts5EpisodeRetriever(db_path=db_path)):
        answer = synthesize((_occ("esco", "esco:occupation:dev", "nurse"),), llm_client=client)

    assert answer.startswith("A nurse needs a licence.")
    system = _system_prompt(client)
    assert system.endswith(synth._SYNTH_SYSTEM)
    # Nothing recalled: an empty question has no terms, so the prefix is empty and
    # the memory blocks sit directly against the brief, exactly as before.
    assert "nurse needs" not in system.split(synth._SYNTH_SYSTEM)[0]
