"""Offline tests for the SQLite episodes index."""

from __future__ import annotations

from pathlib import Path

import pytest

from talent_angels.memory.episodes import (
    clear_episodes,
    episodes_citing,
    recent_episodes,
    record_episode,
)
from talent_angels.runlog.models import ResultSummary, RunLogRecord


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "memory.db"


def _record(
    *,
    run_id: str,
    ts: str = "2026-01-01T00:00:00+00:00",
    plan: list[str] | None = None,
    question: str = "web developer",
    node_ids: list[str] | None = None,
    node_labels: list[str] | None = None,
    warnings: list[str] | None = None,
) -> RunLogRecord:
    return RunLogRecord(
        run_id=run_id,
        ts=ts,
        suite="esco",
        plan=plan or ["locate", "connect"],
        question=question,
        result=ResultSummary(
            node_ids=node_ids if node_ids is not None else ["esco:occupation:1"],
            node_labels=node_labels if node_labels is not None else ["web developer"],
            warnings=warnings or [],
        ),
    )


def test_no_episodes_before_anything_is_recorded(db_path: Path) -> None:
    assert recent_episodes(db_path=db_path) == []


def test_record_then_recent_round_trips(db_path: Path) -> None:
    record_episode(_record(run_id="r1"), db_path=db_path)

    episodes = recent_episodes(db_path=db_path)

    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.run_id == "r1"
    assert ep.suite == "esco"
    assert ep.capability == "connect"
    assert ep.plan == ("locate", "connect")
    assert ep.question == "web developer"
    assert ep.node_ids == ("esco:occupation:1",)
    assert ep.node_labels == ("web developer",)
    assert ep.satisfied is True


@pytest.mark.parametrize(
    ("node_ids", "warnings", "expected_satisfied"),
    [
        ([], [], False),
        (["esco:occupation:1"], ["ambiguous"], False),
        (["esco:occupation:1"], ["not_found"], False),
        (["esco:occupation:1"], ["capability_not_implemented:pathfind"], False),
        (["esco:occupation:1"], ["unsupported_connect_query"], False),
        (["esco:occupation:1"], [], True),
    ],
)
def test_satisfied_reflects_warnings_and_whether_anything_was_found(
    db_path: Path, node_ids: list[str], warnings: list[str], expected_satisfied: bool
) -> None:
    record_episode(_record(run_id="r1", node_ids=node_ids, warnings=warnings), db_path=db_path)

    assert recent_episodes(db_path=db_path)[0].satisfied is expected_satisfied


def test_recent_episodes_orders_newest_first(db_path: Path) -> None:
    record_episode(_record(run_id="r1", ts="2026-01-01T00:00:00+00:00"), db_path=db_path)
    record_episode(_record(run_id="r2", ts="2026-01-03T00:00:00+00:00"), db_path=db_path)
    record_episode(_record(run_id="r3", ts="2026-01-02T00:00:00+00:00"), db_path=db_path)

    ids = [ep.run_id for ep in recent_episodes(db_path=db_path)]

    assert ids == ["r2", "r3", "r1"]


def test_recent_episodes_respects_limit(db_path: Path) -> None:
    for i in range(5):
        ts = f"2026-01-0{i + 1}T00:00:00+00:00"
        record_episode(_record(run_id=f"r{i}", ts=ts), db_path=db_path)

    assert len(recent_episodes(limit=2, db_path=db_path)) == 2


def test_episodes_citing_finds_only_matching_turns(db_path: Path) -> None:
    record_episode(_record(run_id="r1", node_ids=["esco:occupation:1"]), db_path=db_path)
    record_episode(_record(run_id="r2", node_ids=["esco:occupation:2"]), db_path=db_path)
    record_episode(
        _record(run_id="r3", node_ids=["esco:occupation:1", "esco:skill:9"]), db_path=db_path
    )

    hits = episodes_citing("esco:occupation:1", db_path=db_path)

    assert {ep.run_id for ep in hits} == {"r1", "r3"}


def test_episodes_citing_finds_nothing_for_an_uncited_node(db_path: Path) -> None:
    record_episode(_record(run_id="r1", node_ids=["esco:occupation:1"]), db_path=db_path)

    assert episodes_citing("esco:occupation:404", db_path=db_path) == []


def test_re_recording_the_same_run_id_replaces_it_cleanly(db_path: Path) -> None:
    record_episode(_record(run_id="r1", node_ids=["esco:occupation:1"]), db_path=db_path)
    record_episode(_record(run_id="r1", node_ids=["esco:occupation:2"]), db_path=db_path)

    episodes = recent_episodes(db_path=db_path)

    assert len(episodes) == 1
    assert episodes[0].node_ids == ("esco:occupation:2",)
    assert episodes_citing("esco:occupation:1", db_path=db_path) == []
    assert [ep.run_id for ep in episodes_citing("esco:occupation:2", db_path=db_path)] == ["r1"]


def test_clear_episodes_removes_everything_and_reports_the_count(db_path: Path) -> None:
    assert clear_episodes(db_path=db_path) == 0

    record_episode(_record(run_id="r1"), db_path=db_path)
    record_episode(_record(run_id="r2", ts="2026-01-02T00:00:00+00:00"), db_path=db_path)

    removed = clear_episodes(db_path=db_path)

    assert removed == 2
    assert recent_episodes(db_path=db_path) == []
