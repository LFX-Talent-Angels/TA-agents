"""Offline tests for the SQLite episodes index."""

from __future__ import annotations

from pathlib import Path

import pytest

from talent_angels.memory.episodes import (
    clear_episodes,
    episodes_citing,
    episodes_db_path,
    recent_episodes,
    record_episode,
)
from talent_angels.runlog.models import ResultSummary, RunLogRecord


@pytest.fixture(autouse=True)
def _local_memory_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TA_MEMORY_DIR", str(tmp_path / "memory"))


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


def test_no_episodes_before_anything_is_recorded() -> None:
    assert recent_episodes() == []
    assert not episodes_db_path().exists()


def test_record_then_recent_round_trips() -> None:
    record_episode(_record(run_id="r1"))

    episodes = recent_episodes()

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
    node_ids: list[str], warnings: list[str], expected_satisfied: bool
) -> None:
    record_episode(_record(run_id="r1", node_ids=node_ids, warnings=warnings))

    assert recent_episodes()[0].satisfied is expected_satisfied


def test_recent_episodes_orders_newest_first() -> None:
    record_episode(_record(run_id="r1", ts="2026-01-01T00:00:00+00:00"))
    record_episode(_record(run_id="r2", ts="2026-01-03T00:00:00+00:00"))
    record_episode(_record(run_id="r3", ts="2026-01-02T00:00:00+00:00"))

    ids = [ep.run_id for ep in recent_episodes()]

    assert ids == ["r2", "r3", "r1"]


def test_recent_episodes_respects_limit() -> None:
    for i in range(5):
        record_episode(_record(run_id=f"r{i}", ts=f"2026-01-0{i + 1}T00:00:00+00:00"))

    assert len(recent_episodes(limit=2)) == 2


def test_episodes_citing_finds_only_matching_turns() -> None:
    record_episode(_record(run_id="r1", node_ids=["esco:occupation:1"]))
    record_episode(_record(run_id="r2", node_ids=["esco:occupation:2"]))
    record_episode(_record(run_id="r3", node_ids=["esco:occupation:1", "esco:skill:9"]))

    hits = episodes_citing("esco:occupation:1")

    assert {ep.run_id for ep in hits} == {"r1", "r3"}


def test_episodes_citing_finds_nothing_for_an_uncited_node() -> None:
    record_episode(_record(run_id="r1", node_ids=["esco:occupation:1"]))

    assert episodes_citing("esco:occupation:404") == []


def test_re_recording_the_same_run_id_replaces_it_cleanly() -> None:
    record_episode(_record(run_id="r1", node_ids=["esco:occupation:1"]))
    record_episode(_record(run_id="r1", node_ids=["esco:occupation:2"]))

    episodes = recent_episodes()

    assert len(episodes) == 1
    assert episodes[0].node_ids == ("esco:occupation:2",)
    assert episodes_citing("esco:occupation:1") == []
    assert [ep.run_id for ep in episodes_citing("esco:occupation:2")] == ["r1"]


def test_clear_episodes_removes_everything_and_reports_the_count() -> None:
    assert clear_episodes() == 0

    record_episode(_record(run_id="r1"))
    record_episode(_record(run_id="r2", ts="2026-01-02T00:00:00+00:00"))

    removed = clear_episodes()

    assert removed == 2
    assert recent_episodes() == []
