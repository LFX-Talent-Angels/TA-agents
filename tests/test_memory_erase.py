"""Offline tests for talent_angels.memory.erase — forgetting means all three stores.

Regression cover for the gap where ``/reset`` deleted ``USER.md`` and
``MEMORY.md`` but left the episode index in ``memory.db`` intact, so the
questions a person asked and the node ids cited on their behalf survived the
"Memory and session cleared" message.

These tests seed files through the autouse ``MemoryHome`` fixture, never
through ``memory.paths`` — see ``tests/conftest.py`` for why.
"""

from __future__ import annotations

import pytest

from talent_angels.memory.episodes import recent_episodes, record_episode
from talent_angels.memory.erase import EraseResult, erase_person, erase_summary
from talent_angels.runlog.models import ResultSummary, RunLogRecord
from tests.conftest import MemoryHome


def _record(run_id: str, question: str) -> RunLogRecord:
    return RunLogRecord(
        run_id=run_id,
        ts="2026-01-01T00:00:00+00:00",
        suite="esco",
        plan=["locate", "connect"],
        question=question,
        result=ResultSummary(
            node_ids=["esco:occupation:web-developer"],
            node_labels=["web developer"],
            warnings=[],
        ),
    )


def test_erase_removes_profile_notes_and_episodes(memory_home: MemoryHome) -> None:
    memory_home.user_md.write_text("STANDING[esco]: Web Developer  [esco:web-dev]\n")
    memory_home.memory_md.write_text("- prefers concise answers\n")
    record_episode(_record("run-1", "web developer skills"))
    record_episode(_record("run-2", "data scientist path"))

    result = erase_person()

    assert result.files_deleted == 2
    assert result.episodes_deleted == 2
    assert not memory_home.user_md.exists()
    assert not memory_home.memory_md.exists()
    assert recent_episodes() == []


def test_erase_purges_the_question_text_too(memory_home: MemoryHome) -> None:
    """The episode row is where the free text lives — it must not survive."""
    record_episode(_record("run-1", "how do I become a nurse in Germany?"))

    erase_person()

    assert recent_episodes(limit=50) == []


def test_erase_keeps_the_neighbor_cache(memory_home: MemoryHome) -> None:
    """Cache rows are keyed by taxonomy node, carry no user dimension, and TTL out.

    There is nothing personal in them, so a forget request must not throw away
    answers that are identical for every user.
    """
    from talent_angels.memory.cache import get_cached_neighbors, set_cached_neighbors
    from tests.fakes.taxonomy import FakeEdge, FakeNode, FakeToolResult

    memory_home.user_md.write_text("GOAL: Data Scientist  [onet:15-2051.00]\n")
    set_cached_neighbors(
        "esco:occupation:1",
        ["HAS_SKILL"],
        FakeToolResult(
            nodes=[
                FakeNode(
                    id="esco:skill:1",
                    kind="Skill",
                    label="computer programming",
                    source="esco",
                    source_id="src-1",
                    properties={},
                )
            ],
            edges=[
                FakeEdge(
                    type="HAS_SKILL",
                    from_id="esco:occupation:1",
                    to_id="esco:skill:1",
                    properties={},
                )
            ],
            evidence=["esco:neighbors:1"],
        ),
    )

    result = erase_person()

    assert result.files_deleted == 1
    assert result.episodes_deleted == 0
    assert not memory_home.user_md.exists()
    # The derived, non-personal cache row survives the erase.
    assert get_cached_neighbors("esco:occupation:1", ["HAS_SKILL"]) is not None


def test_erase_is_idempotent_and_reports_nothing_to_do() -> None:
    first = erase_person()
    second = erase_person()

    assert (first.files_deleted, first.episodes_deleted) == (0, 0)
    assert (second.files_deleted, second.episodes_deleted) == (0, 0)
    assert "Nothing to erase" in erase_summary(second)


def test_erase_summary_names_what_it_removed() -> None:
    both = erase_summary(EraseResult(files_deleted=2, episodes_deleted=7))
    assert "2 profile file(s)" in both
    assert "7 recorded turn(s)" in both

    only_files = erase_summary(EraseResult(files_deleted=1, episodes_deleted=0))
    assert "1 profile file(s)" in only_files
    assert "recorded turn" not in only_files


@pytest.mark.parametrize("count", [1, 5])
def test_erase_purges_every_episode_not_just_recent(count: int) -> None:
    for i in range(count):
        record_episode(_record(f"run-{i}", f"question {i}"))

    result = erase_person()

    assert result.episodes_deleted == count
    assert recent_episodes(limit=50) == []
