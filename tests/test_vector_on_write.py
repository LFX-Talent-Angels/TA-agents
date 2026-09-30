"""Vector recall stays current without a manual rebuild, and stays on topic."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence

import pytest

from talent_angels.assistant import run_turn
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.llm.stub_client import StubLLMClient
from talent_angels.memory.embeddings import LocalEmbedder, StaticEmbedder
from talent_angels.memory.episodes import record_episode, suite_satisfied
from talent_angels.memory.vector_index import SqliteVecIndex
from talent_angels.runlog.models import ResultSummary, RunLogRecord
from tests.fakes.suite import fake_registry

pytest.importorskip("sqlite_vec")


class _OfflineLocal(LocalEmbedder):
    """Behaves as the local embedder (free, on-device) without loading a model."""

    def __init__(self) -> None:
        super().__init__()
        self._static = StaticEmbedder(dimensions=16)

    @property
    def configured(self) -> bool:
        return True

    @property
    def dimensions(self) -> int:
        return 16

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._static.embed(texts)


@pytest.fixture(autouse=True)
def _local_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TA_RECALL", "hybrid")
    monkeypatch.setattr(
        "talent_angels.memory.embeddings.default_embedder", lambda: _OfflineLocal()
    )


def _turn(question: str) -> None:
    run_turn(registry=fake_registry(), llm_client=StubLLMClient(), question=question)


def test_every_recorded_turn_is_indexed_without_a_rebuild() -> None:
    _turn("where is software developer")
    _turn("what skills does a software developer need")

    assert SqliteVecIndex(dimensions=16).count() == 2


def test_the_first_indexed_turn_backfills_earlier_history() -> None:
    for n in range(3):
        record_episode(
            RunLogRecord(
                run_id=f"old-{n}",
                suite="esco",
                plan=["locate"],
                question=f"old question {n}",
                result=ResultSummary(node_ids=["esco:x"], node_labels=["nurse"]),
            )
        )
    _turn("where is software developer")

    assert SqliteVecIndex(dimensions=16).count() == 4


def test_a_paid_embedder_is_never_used_on_write(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "talent_angels.memory.embeddings.default_embedder", lambda: StaticEmbedder(dimensions=16)
    )
    _turn("where is software developer")

    assert SqliteVecIndex(dimensions=16).count() == 0


def test_episodes_keep_the_topic_not_every_neighbour(memory_home) -> None:
    _turn("what skills does a software developer need")

    conn = sqlite3.connect(memory_home.db)
    [(labels,)] = conn.execute("SELECT node_labels FROM episodes").fetchall()
    [(fts_labels,)] = conn.execute("SELECT node_labels FROM episodes_fts").fetchall()
    conn.close()
    assert json.loads(labels) == ["software developer"]
    assert json.loads(fts_labels) == ["software developer"]


def _node(suite: str) -> NodeRef:
    return NodeRef(
        id=f"{suite}:occupation:1",
        suite=suite,
        source=suite,
        source_id="1",
        kind="Occupation",
        pref_label="nurse",
    )


def test_one_suites_miss_does_not_make_the_turn_unsatisfied() -> None:
    hit = AgentResult(capability="locate", suite="esco", nodes=[_node("esco")])
    miss = AgentResult(capability="locate", suite="onet", warnings=["not_found"])

    assert suite_satisfied(hit) and not suite_satisfied(miss)
    outcome = run_turn(
        registry=fake_registry(), llm_client=StubLLMClient(), question="software developer"
    )
    assert outcome.record.result.satisfied is True
